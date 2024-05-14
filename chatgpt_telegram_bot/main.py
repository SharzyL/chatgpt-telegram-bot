#!/usr/bin/env python3
import atexit
import sys
import asyncio
import os
import shelve
import datetime
import time
import traceback
import hashlib
import base64
import copy
from collections import defaultdict
from urllib.parse import urlparse
import tomllib
from argparse import ArgumentParser

import openai
from telethon import TelegramClient, events, errors, functions, types
from loguru import logger

from chatgpt_telegram_bot.richtext import RichText


def parse_proxy():
    proxy_env = os.getenv("ALL_PROXY")
    if proxy_env:
        proxy_url = urlparse(proxy_env)
        return {
            'proxy_type': proxy_url.scheme,
            'addr': proxy_url.hostname,
            'port': proxy_url.port,
        }
    else:
        return None


def retry(max_retry=30, interval=10):
    def decorator(func):
        async def new_func(*args, **kwargs):
            for _ in range(max_retry - 1):
                try:
                    return await func(*args, **kwargs)
                except errors.FloodWaitError as e:
                    logger.exception(e)
                    await asyncio.sleep(interval)
            return await func(*args, **kwargs)

        return new_func

    return decorator


class PendingReplyManager:
    def __init__(self):
        self.messages = {}

    def add(self, reply_id):
        assert reply_id not in self.messages
        self.messages[reply_id] = asyncio.Event()

    def remove(self, reply_id):
        if reply_id not in self.messages:
            return
        self.messages[reply_id].set()
        del self.messages[reply_id]

    async def wait_for(self, reply_id):
        if reply_id not in self.messages:
            return
        logger.info('PendingReplyManager waiting for %r', reply_id)
        await self.messages[reply_id].wait()
        logger.info('PendingReplyManager waiting for %r finished', reply_id)


class ChatGPTTelegramBot:
    def __init__(self, config_path):
        # parse env
        self.TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
        self.TELEGRAM_API_ID = int(os.environ["TELEGRAM_API_ID"])
        self.TELEGRAM_API_HASH = os.environ["TELEGRAM_API_HASH"]
        self.API_KEY = os.environ["OPENAI_API_KEY"]

        self.TELEGRAM_API_ID = int(self.TELEGRAM_API_ID)

        with open(config_path, "rb") as f:
            _config = tomllib.load(f)
        self.admin_id = _config['admin_id']
        self.models = _config['models']
        self.api_endpoint = _config['api_endpoint']
        self.vision_model = _config['vision_model']
        self.default_model = _config['default_model']

        self.telegram_last_timestamp = defaultdict(lambda: None)

        self.aclient = openai.AsyncOpenAI(
            api_key=self.API_KEY,
            base_url=self.api_endpoint,
            max_retries=0,
            timeout=15,
        )

        self.TELEGRAM_LENGTH_LIMIT = 4096
        self.TELEGRAM_MIN_INTERVAL = 3
        self.OPENAI_MAX_RETRY = 3
        self.OPENAI_RETRY_INTERVAL = 10
        self.FIRST_BATCH_DELAY = 1
        self.TEXT_FILE_SIZE_LIMIT = 100_000

        self.telegram_last_timestamp = dict()
        self.telegram_rate_limit_lock = defaultdict(asyncio.Lock)

        self.pending_reply_manager = PendingReplyManager()

        self.db = shelve.open('db')
        atexit.register(self.db.close)
        # db[(chat_id, msg_id)] = (is_bot, text, reply_id, model)
        # compatible old db format: db[(chat_id, msg_id)] = (is_bot, text, reply_id)
        # db['whitelist'] = set(whitelist_chat_ids)
        if 'whitelist' not in self.db:
            self.db['whitelist'] = {self.admin_id}

        self.bot_id = int(self.TELEGRAM_BOT_TOKEN.split(':')[0])
        self.pending_reply_manager = PendingReplyManager()
        self.bot = TelegramClient('bot', self.TELEGRAM_API_ID, self.TELEGRAM_API_HASH, proxy=parse_proxy())

    async def start(self):
        await self.bot.start(bot_token=self.TELEGRAM_BOT_TOKEN)
        logger.info('Bot started')
        self.bot.parse_mode = None
        me = await self.bot.get_me()

        @self.bot.on(events.NewMessage)
        async def process(event):
            if event.message.chat_id is None:
                return
            if event.message.sender_id is None:
                return
            if event.message.message is None:
                return
            text = event.message.message
            if text == '/ping' or text == f'/ping@{me.username}':
                await self.ping(event.message)
            elif text == '/list_models' or text == f'/list_models@{me.username}':
                await self.list_models_handler(event.message)
            elif text == '/add_whitelist' or text == f'/add_whitelist@{me.username}':
                await self.add_whitelist_handler(event.message)
            elif text == '/del_whitelist' or text == f'/del_whitelist@{me.username}':
                await self.del_whitelist_handler(event.message)
            elif text == '/get_whitelist' or text == f'/get_whitelist@{me.username}':
                await self.get_whitelist_handler(event.message)
            else:
                await self.reply_handler(event.message)

        admin_input_peer = await self.bot.get_input_entity(self.admin_id)
        await self.bot(functions.bots.SetBotCommandsRequest(
            scope=types.BotCommandScopePeer(admin_input_peer),
            lang_code='',
            commands=[types.BotCommand(command, description) for command, description in [
                ('ping', 'Test bot connectivity'),
                ('list_models', 'List supported models'),
                ('add_whitelist', 'Add this group to whitelist (only admin)'),
                ('del_whitelist', 'Delete this group from whitelist (only admin)'),
                ('get_whitelist', 'List groups in whitelist (only admin)'),
            ]]
        ))

        await self.bot(functions.bots.SetBotCommandsRequest(
            scope=types.BotCommandScopeDefault(),
            lang_code='',
            commands=[types.BotCommand(command, description) for command, description in [
                ('ping', 'Test bot connectivity'),
                ('list_models', 'List supported models'),
            ]]
        ))
        logger.info('Bot commands registered')

        await self.bot.run_until_disconnected()

    @staticmethod
    def get_prompt(model):
        current_time = (datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=8)).strftime('%Y-%m-%d %H:%M:%S')
        return f'''
    You are ChatGPT Telegram bot running {model} model. Knowledge cutoff: Sep 2021. Current Beijing Time: {current_time}
    '''

    def within_interval(self, chat_id):
        last_timestamp = self.telegram_last_timestamp.get(chat_id, None)
        if last_timestamp is None:
            return False
        else:
            remaining_time = last_timestamp + self.TELEGRAM_MIN_INTERVAL - time.time()
        return remaining_time > 0

    @staticmethod
    def ensure_interval(func):
        async def new_func(self, *args, **kwargs):
            chat_id = args[0]
            async with self.telegram_rate_limit_lock[chat_id]:
                last_timestamp = self.telegram_last_timestamp.get(chat_id, None)
                if last_timestamp is not None:
                    remaining_time = last_timestamp + self.TELEGRAM_MIN_INTERVAL - time.time()
                    if remaining_time > 0:
                        await asyncio.sleep(remaining_time)
                result = await func(self, *args, **kwargs)
                self.telegram_last_timestamp[chat_id] = time.time()
                return result

        return new_func

    def is_whitelist(self, chat_id):
        whitelist = self.db['whitelist']
        return chat_id in whitelist

    def add_whitelist(self, chat_id):
        whitelist = self.db['whitelist']
        whitelist.add(chat_id)
        self.db['whitelist'] = whitelist

    def del_whitelist(self, chat_id):
        whitelist = self.db['whitelist']
        whitelist.discard(chat_id)
        self.db['whitelist'] = whitelist

    def get_whitelist(self):
        return self.db['whitelist']

    @staticmethod
    def only_admin(func):
        async def new_func(self, message):
            if message.sender_id != self.admin_id:
                await self.send_message(message.chat_id, 'Only admin can use this command', message.id)
                return
            await func(self, message)

        return new_func

    @staticmethod
    def only_private(func):
        async def new_func(self, message):
            if message.chat_id != message.sender_id:
                await self.send_message(message.chat_id, 'This command only works in private chat', message.id)
                return
            await func(self, message)

        return new_func

    @staticmethod
    def only_whitelist(func):
        async def new_func(self, message):
            if not self.is_whitelist(message.chat_id):
                if message.chat_id == message.sender_id:
                    await self.send_message(message.chat_id, 'This chat is not in whitelist', message.id)
                return
            await func(self, message)

        return new_func

    @staticmethod
    def save_photo(photo_blob):  # TODO: change to async
        h = hashlib.sha256(photo_blob).hexdigest()
        save_dir = f'photos/{h[:2]}/{h[2:4]}'
        path = f'{save_dir}/{h}'
        if not os.path.isfile(path):
            os.makedirs(save_dir, exist_ok=True)
            with open(path, 'wb') as f:
                f.write(photo_blob)
        return h

    @staticmethod
    def load_photo(h):
        save_dir = f'photos/{h[:2]}/{h[2:4]}'
        path = f'{save_dir}/{h}'
        with open(path, 'rb') as f:
            return f.read()

    async def completion(self, chat_history, model, chat_id, msg_id):  # chat_history = [user, ai, user, ai, ..., user]
        assert len(chat_history) % 2 == 1
        system_prompt = self.get_prompt(model)
        messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
        roles = ["user", "assistant"]
        role_id = 0
        for msg in chat_history:
            messages.append({"role": roles[role_id], "content": msg})
            role_id = 1 - role_id

        def remove_image(messages_):
            new_messages = copy.deepcopy(messages_)
            for message in new_messages:
                if 'content' in message:
                    if isinstance(message['content'], list):
                        for obj_ in message['content']:
                            if obj_['type'] == 'image_url':
                                obj_['image_url']['url'] = obj_['image_url']['url'][:50] + '...'
            return new_messages

        logger.info(f'Request ({chat_id=}, {msg_id=}): {remove_image(messages)}')
        if model == self.vision_model:
            stream = await self.aclient.chat.completions.create(model=model, messages=messages, stream=True,
                                                                max_tokens=4096)
        else:
            stream = await self.aclient.chat.completions.create(model=model, messages=messages, stream=True)
        finished = False
        async for response in stream:
            logger.debug(f'Response ({chat_id=}, {msg_id=}): {response}')
            assert not finished or len(
                response.choices) == 0  # OpenAI sometimes returns a empty response even when finished
            if len(response.choices) == 0:
                continue

            obj = response.choices[0]
            if obj.delta.role is not None:
                if obj.delta.role != 'assistant':
                    raise ValueError("Role error")
            if obj.delta.content is not None:
                yield obj.delta.content

            # handle the finish
            if obj.finish_reason is not None or (
                    'finish_details' in obj.model_extra and obj.finish_details is not None):
                assert all(item is None for item in [
                    obj.delta.content,
                    obj.delta.function_call,
                    obj.delta.role,
                    obj.delta.tool_calls,
                ]) or obj.delta.content == ''
                finish_reason = obj.finish_reason
                if 'finish_details' in obj.model_extra and obj.finish_details is not None:
                    assert finish_reason is None
                    finish_reason = obj.finish_details['type']
                if finish_reason == 'length':
                    yield '\n\n[!] Error: Output truncated due to limit'
                elif finish_reason == 'stop':
                    pass
                elif finish_reason is not None:
                    if obj.finish_reason is not None:
                        yield f'\n\n[!] Error: finish_reason="{finish_reason}"'
                    else:
                        yield f'\n\n[!] Error: finish_details="{obj.finish_details}"'
                finished = True

    def construct_chat_history(self, chat_id, msg_id):
        messages = []
        should_be_bot = False
        model = self.default_model
        has_image = False
        while True:
            key = repr((chat_id, msg_id))
            if key not in self.db:
                logger.error(f'History message not found ({chat_id=}, {msg_id=})')
                return None, None
            is_bot, message, reply_id, *params = self.db[key]
            if params:
                model = params[0]
            if is_bot != should_be_bot:
                logger.error(f'Role does not match ({chat_id=}, {msg_id=})')
                return None, None
            if isinstance(message, list):
                new_message = []
                for obj in message:
                    if obj['type'] == 'text':
                        new_message.append(obj)
                    elif obj['type'] == 'image':
                        blob = self.load_photo(obj['hash'])
                        blob_base64 = base64.b64encode(blob).decode()
                        image_url = 'data:image/jpeg;base64,' + blob_base64
                        new_message.append({'type': 'image_url', 'image_url': {'url': image_url, 'detail': 'high'}})
                        has_image = True
                    else:
                        raise ValueError('Unknown message type in chat history')
                message = new_message
            messages.append(message)
            should_be_bot = not should_be_bot
            if reply_id is None:
                break
            msg_id = reply_id
        if len(messages) % 2 != 1:
            logger.error(f'First message not from user ({chat_id=}, {msg_id=})')
            return None, None
        if has_image:
            model = self.vision_model
        return messages[::-1], model

    @only_admin
    async def add_whitelist_handler(self, message):
        if self.is_whitelist(message.chat_id):
            await self.send_message(message.chat_id, 'Already in whitelist', message.id)
            return
        self.add_whitelist(message.chat_id)
        await self.send_message(message.chat_id, 'Whitelist added', message.id)

    @only_admin
    async def del_whitelist_handler(self, message):
        if not self.is_whitelist(message.chat_id):
            await self.send_message(message.chat_id, 'Not in whitelist', message.id)
            return
        self.del_whitelist(message.chat_id)
        await self.send_message(message.chat_id, 'Whitelist deleted', message.id)

    @only_admin
    @only_private
    async def get_whitelist_handler(self, message):
        await self.send_message(message.chat_id, str(self.get_whitelist()), message.id)

    @only_whitelist
    async def list_models_handler(self, message):
        text = ''
        for m in self.models:
            text += f'Prefix: "{m["prefix"]}", model: {m["model"]}\n'
        await self.send_message(message.chat_id, text, message.id)

    @retry()
    @ensure_interval
    async def send_message(self, chat_id, text, reply_to_message_id):
        logger.debug(f'Sending message: {chat_id=}, {reply_to_message_id=}, {text=}')
        text = RichText(text)
        text, entities = text.to_telegram()
        msg = await self.bot.send_message(
            chat_id,
            text,
            reply_to=reply_to_message_id,
            link_preview=False,
            formatting_entities=entities,
        )
        logger.info(f'Message sent: {chat_id=}, {reply_to_message_id=}, {msg.id=}')
        return msg.id

    @retry()
    @ensure_interval
    async def edit_message(self, chat_id, text, message_id):
        logger.info(f'Editing message: {chat_id=}, {message_id=}, {text=}')
        text = RichText(text)
        text, entities = text.to_telegram()
        try:
            await self.bot.edit_message(
                chat_id,
                message_id,
                text,
                link_preview=False,
                formatting_entities=entities,
            )
        except errors.MessageNotModifiedError:
            logger.info(f'Message not modified: {chat_id=}, {message_id=}')
        else:
            logger.info(f'Message edited: {chat_id=}, {message_id=}')

    @retry()
    @ensure_interval
    async def delete_message(self, chat_id, message_id):
        logger.info(f'Deleting message: {chat_id=}, {message_id=}')
        await self.bot.delete_messages(
            chat_id,
            message_id,
        )
        logger.info(f'Message deleted: {chat_id=}, {message_id=}')

    @only_whitelist
    async def reply_handler(self, message):
        chat_id = message.chat_id
        sender_id = message.sender_id
        msg_id = message.id
        text = message.message
        logger.info(f'New message to reply: '
                    f'{chat_id=}, {sender_id=}, {msg_id=}, {text=}, {message.photo=}, {message.document=}')
        reply_to_id = None
        model = self.default_model
        extra_photo_message = None
        extra_document_message = None
        if not text and message.photo is None and message.document is None:  # unknown media types
            return
        if message.is_reply:
            if message.reply_to.quote_text is not None:
                return
            reply_to_message = await message.get_reply_message()
            if reply_to_message.sender_id == self.bot_id:  # user reply to bot message
                reply_to_id = message.reply_to.reply_to_msg_id
                await self.pending_reply_manager.wait_for((chat_id, reply_to_id))
            elif reply_to_message.photo is not None:  # user reply to a photo
                extra_photo_message = reply_to_message
            elif reply_to_message.document is not None:  # user reply to a document
                extra_document_message = reply_to_message
            else:
                return
        if not message.is_reply or extra_photo_message is not None or extra_document_message is not None:  # new message
            for m in self.models:
                if text.startswith(m['prefix']):
                    text = text[len(m['prefix']):]
                    model = m['model']
                    break
            else:  # not reply or new message to bot
                if chat_id == sender_id:  # if in private chat, send hint
                    await self.send_message(chat_id, 'Please start a new conversation with $ or reply to a bot message',
                                            msg_id)
                return

        photo_message = message if message.photo is not None else extra_photo_message
        photo_hash = None
        if photo_message is not None:
            if photo_message.grouped_id is not None:
                await self.send_message(chat_id, 'Grouped photos are not yet supported, but will be supported soon',
                                        msg_id)
                return
            photo_blob = await photo_message.download_media(bytes)
            photo_hash = self.save_photo(photo_blob)

        document_message = message if message.document is not None else extra_document_message
        document_text = None
        if document_message is not None:
            if document_message.grouped_id is not None:
                await self.send_message(chat_id, 'Grouped files are not yet supported, but will be supported soon',
                                        msg_id)
                return
            if document_message.document.size > self.TEXT_FILE_SIZE_LIMIT:
                await self.send_message(chat_id, 'File too large', msg_id)
                return
            document_blob = await document_message.download_media(bytes)
            try:
                document_text = document_blob.decode()
                assert all(c != '\x00' for c in document_text)
            except UnicodeDecodeError:
                await self.send_message(chat_id, 'File is not text file or not valid UTF-8', msg_id)
                return

        if photo_hash:
            new_message = [{'type': 'text', 'text': text}, {'type': 'image', 'hash': photo_hash}]
        elif document_text:
            if text:
                new_message = document_text + '\n\n' + text
            else:
                new_message = document_text
        else:
            new_message = text

        self.db[repr((chat_id, msg_id))] = (False, new_message, reply_to_id, model)

        chat_history, model = self.construct_chat_history(chat_id, msg_id)
        if chat_history is None:
            await self.send_message(chat_id,
                                    f"[!] Error: Unable to proceed with this conversation. Potential "
                                    f"causes: the message replied to may be incomplete, contain an error, "
                                    f"be a system message, or not exist in the database.",
                                    msg_id)
            return

        error_cnt = 0
        while True:
            reply = ''
            prefix = '🤖 ' + RichText.Code(model) + '\n\n'
            async with BotReplyMessages(self, chat_id, msg_id, prefix) as replymsgs:
                try:
                    stream = self.completion(chat_history, model, chat_id, msg_id)
                    first_update_timestamp = None
                    async for delta in stream:
                        reply += delta
                        if first_update_timestamp is None:
                            first_update_timestamp = time.time()
                        if time.time() >= first_update_timestamp + self.FIRST_BATCH_DELAY:
                            await replymsgs.update(RichText.from_markdown(reply) + ' [!Generating...]')
                    await replymsgs.update(RichText.from_markdown(reply))
                    await replymsgs.finalize()
                    for message_id, _ in replymsgs.replied_msgs:
                        self.db[repr((chat_id, message_id))] = (True, reply, msg_id, model)
                    return

                # handling completion errors
                except Exception as e:
                    error_cnt += 1
                    logger.error(f'Error on generating exception({chat_id=}, {msg_id=}, {error_cnt=}): {e}')
                    will_retry = not isinstance(e, openai.BadRequestError) \
                                 and not isinstance(e, openai.AuthenticationError) \
                                 and error_cnt <= self.OPENAI_MAX_RETRY
                    error_msg = f'[!] Error: {traceback.format_exception_only(e)[-1].strip()}'
                    if will_retry:
                        error_msg += f'\nRetrying ({error_cnt}/{self.OPENAI_MAX_RETRY})...'
                    if reply:
                        error_msg = reply + '\n\n' + error_msg
                    await replymsgs.update(error_msg)
                    if will_retry:
                        await asyncio.sleep(self.OPENAI_RETRY_INTERVAL)
                    if not will_retry:
                        break

    async def ping(self, message):
        await self.send_message(message.chat_id, f'''
chat_id={message.chat_id}
user_id={message.sender_id}
is_whitelisted={self.is_whitelist(message.chat_id)}
''', message.id)


class BotReplyMessages:
    def __init__(self, cbot: ChatGPTTelegramBot, chat_id, orig_msg_id, prefix):
        self.cbot = cbot
        self.prefix = prefix
        self.msg_len = cbot.TELEGRAM_LENGTH_LIMIT - len(prefix)
        assert self.msg_len > 0
        self.chat_id = chat_id
        self.orig_msg_id = orig_msg_id
        self.replied_msgs = []
        self.text = ''

    async def __aenter__(self):
        return self

    async def __aexit__(self, type_, value, tb):
        await self.finalize()
        for msg_id, _ in self.replied_msgs:
            self.cbot.pending_reply_manager.remove((self.chat_id, msg_id))

    async def _force_update(self, text):
        slices = []
        while len(text) > self.msg_len:
            slices.append(text[:self.msg_len])
            text = text[self.msg_len:]
        if text:
            slices.append(text)
        if not slices:
            slices = ['']  # deal with empty message

        for i in range(min(len(slices), len(self.replied_msgs))):
            msg_id, msg_text = self.replied_msgs[i]
            if slices[i] != msg_text:
                await self.cbot.edit_message(self.chat_id, self.prefix + slices[i], msg_id)
                self.replied_msgs[i] = (msg_id, slices[i])
        if len(slices) > len(self.replied_msgs):
            for i in range(len(self.replied_msgs), len(slices)):
                if i == 0:
                    reply_to = self.orig_msg_id
                else:
                    reply_to, _ = self.replied_msgs[i - 1]
                msg_id = await self.cbot.send_message(self.chat_id, self.prefix + slices[i], reply_to)
                self.replied_msgs.append((msg_id, slices[i]))
                self.cbot.pending_reply_manager.add((self.chat_id, msg_id))
        if len(self.replied_msgs) > len(slices):
            for i in range(len(slices), len(self.replied_msgs)):
                msg_id, _ = self.replied_msgs[i]
                await self.cbot.delete_message(self.chat_id, msg_id)
                self.cbot.pending_reply_manager.remove((self.chat_id, msg_id))
            self.replied_msgs = self.replied_msgs[:len(slices)]

    async def update(self, text):
        self.text = text
        if not self.cbot.within_interval(self.chat_id):
            await self._force_update(self.text)

    async def finalize(self):
        await self._force_update(self.text)


async def async_main():
    parser = ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('-c', '--config', default='bot.toml')

    args = parser.parse_args()

    log_level = "DEBUG" if args.debug else "INFO"
    logger.remove()
    logger.add(sys.stdout, colorize=True, format="<green>{time}</green> <level>{message}</level>", level=log_level)

    cbot = ChatGPTTelegramBot(args.config)
    await cbot.start()


def main():
    asyncio.run(async_main())
