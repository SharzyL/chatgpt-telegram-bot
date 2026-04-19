import atexit
import asyncio
import os
import shelve
import datetime
import time
import traceback
import tomllib
from collections.abc import Sequence
from typing import Any

import anthropic
import openai
from telethon import TelegramClient, events, errors, functions, types
from loguru import logger

from chatgpt_telegram_bot.richtext import RichText
from chatgpt_telegram_bot.models import (
    EndPoint,
    Model,
    MsgInfo,
    MsgPartInHistory,
    StreamMeta,
    ThinkingDelta,
    ResponseDelta,
    StatusChange,
    make_image_part,
    make_text_part,
    OVERRIDE_ALIASES,
)
from chatgpt_telegram_bot.utils import (
    parse_overrides,
    parse_proxy,
    retry,
    telegram_len,
    PendingReplyManager,
    save_photo,
)
from chatgpt_telegram_bot.completion import completion
from chatgpt_telegram_bot.reply import format_reply


class ChatGPTTelegramBot:
    def __init__(self, config_path: str) -> None:
        self.config_path: str = config_path
        # parse env
        self.TELEGRAM_BOT_TOKEN: str = os.environ['TELEGRAM_BOT_TOKEN']
        self.TELEGRAM_API_ID: int = int(os.environ['TELEGRAM_API_ID'])
        self.TELEGRAM_API_HASH: str = os.environ['TELEGRAM_API_HASH']

        with open(config_path, 'rb') as f:
            _config = tomllib.load(f)
        self.admin_id: int = _config['admin_id']
        self.models: Sequence[Model] = [Model(**m) for m in _config['models']]
        self.endpoints: Sequence[EndPoint] = [EndPoint(**e) for e in _config['endpoints']]
        self.default_endpoint: str = _config['default_endpoint']
        self.system_prompt: str = _config.get(
            'system_prompt',
            'Current Beijing Time: {current_time}. Reply in the same language as the user sent you.',
        )
        self.default_image_prompt: str = _config.get('default_image_prompt', 'Describe the image')
        self.default_image_reply_prompt: str = _config.get('default_image_reply_prompt', 'Continue')

        for model in self.models:
            if ' ' in model.prefix or '$' in model.prefix or ',' in model.prefix:
                raise ValueError(f'prefix must not contain space, "$", or ",": "{model.prefix}"')
            if model.thinking == 'adaptive' and model.api_type != 'anthropic':
                raise ValueError(
                    'thinking="adaptive" is only supported for api_type="anthropic",'
                    + f' got api_type="{model.api_type}" for model "{model.prefix}"'
                )

        # map (endpoint_name, api_type, suffix) to SDK client
        self.endpoint_by_name: dict[str, EndPoint] = {e.name: e for e in self.endpoints}
        self.clients: dict[tuple[str, str, str], Any] = {}
        endpoint_by_name = self.endpoint_by_name

        # collect unique (endpoint, api_type, suffix) triples from models
        client_triples: set[tuple[str, str, str]] = set()
        for model in self.models:
            ep = model.endpoint or self.default_endpoint
            assert ep in endpoint_by_name, f'Unknown endpoint: {ep}'
            suffix = model.suffix if model.suffix is not None else endpoint_by_name[ep].default_suffix
            client_triples.add((ep, model.api_type, suffix))

        for ep_name, api_type, suffix in client_triples:
            endpoint = endpoint_by_name[ep_name]
            url = endpoint.url + suffix
            api_key = os.environ[f'OPENAI_API_KEY_{ep_name}']
            if api_type in ('openai', 'openai_legacy'):
                self.clients[(ep_name, api_type, suffix)] = openai.AsyncOpenAI(
                    api_key=api_key,
                    base_url=url,
                    max_retries=0,
                    timeout=300,
                )
            elif api_type == 'anthropic':
                self.clients[(ep_name, api_type, suffix)] = anthropic.AsyncAnthropic(
                    api_key=api_key,
                    base_url=url,
                    max_retries=0,
                    timeout=300,
                )
            else:
                raise ValueError(f'Unknown api_type: {api_type}')

        self.TELEGRAM_LENGTH_LIMIT: int = 4096
        self.TELEGRAM_MIN_INTERVAL: float = 0.5
        self.OPENAI_MAX_RETRY: int = 3
        self.OPENAI_RETRY_INTERVAL: int = 3
        self.TEXT_FILE_SIZE_LIMIT: int = 100_000

        self.pending_reply_manager: PendingReplyManager = PendingReplyManager()

        # db scheme:
        # whitelist: Set[int]
        # msg_info_{chat_id}_{msg_id}: MsgInfo
        # system_prompt_{chat_id}: str
        self.db: shelve.Shelf[Any] = shelve.open('db')

        _ = atexit.register(self.db.close)
        if 'whitelist' not in self.db:
            self.db['whitelist'] = {self.admin_id}

        self.bot_id: int = int(self.TELEGRAM_BOT_TOKEN.split(':')[0])
        self.pending_reply_manager = PendingReplyManager()
        self.bot: TelegramClient = TelegramClient(
            'bot',
            self.TELEGRAM_API_ID,
            self.TELEGRAM_API_HASH,
            proxy=parse_proxy(),  # pyright: ignore[reportArgumentType]  # telethon proxy type is broader at runtime
        )

    @staticmethod
    def match_prefix(text: str, prefix: str) -> tuple[str, dict[str, str | None]] | None:
        """Check if text matches prefix with optional overrides. Returns (remaining_text, overrides) or None."""
        # Case 1: prefix with overrides (prefix,key=val+... delim text)
        if text.startswith(prefix + ','):
            after_pipe = text[len(prefix) + 1 :]
            # Find where overrides end (space or $ delimiter, or end of string)
            for delim in (' ', '$'):
                idx = after_pipe.find(delim)
                if idx != -1:
                    return (after_pipe[idx + 1 :], parse_overrides(after_pipe[:idx]))
            # No delimiter — entire rest is overrides, no text
            return ('', parse_overrides(after_pipe))
        # Case 2: bare prefix (existing behavior)
        if text == prefix:
            return ('', {})
        for delim in (' ', '$'):
            if text.startswith(prefix + delim):
                return (text[len(prefix) + len(delim) :], {})
        return None

    THINKING_DEFAULTS: dict[str, str] = {
        'openai': 'high',
        'openai_legacy': 'high',
        'anthropic': 'adaptive',
    }

    @staticmethod
    def apply_overrides(model: Model, overrides: dict[str, str | None]) -> Model:
        """Apply inline parameter overrides to a model, returning a new Model."""
        if not overrides:
            return model
        replacements: dict[str, Any] = {}
        for key, value in overrides.items():
            field = OVERRIDE_ALIASES.get(key, key)
            if field == 'thinking':
                if value is None:
                    # bare key (e.g. |t) — use api_type default
                    replacements['thinking'] = ChatGPTTelegramBot.THINKING_DEFAULTS.get(model.api_type)
                elif value == '':
                    # explicit empty (e.g. |t=) — disable thinking
                    replacements['thinking'] = None
                else:
                    try:
                        replacements['thinking'] = int(value)
                    except ValueError:
                        replacements['thinking'] = value
            elif field == 'search':
                # bare key (|s) enables, explicit empty (|s=) disables
                replacements['search'] = value is None or (value != '' and value.lower() not in ('0', 'false', 'no'))
            else:
                raise ValueError(f'Unknown override key: {key}')
        return model._replace(**replacements)

    def get_client(self, endpoint: str, model: Model) -> Any:
        suffix = model.suffix if model.suffix is not None else self.endpoint_by_name[endpoint].default_suffix
        return self.clients[(endpoint, model.api_type, suffix)]

    def get_msg_info(self, chat_id: int, msg_id: int) -> MsgInfo | None:
        key = f'msg_info_{chat_id}_{msg_id}'
        if key in self.db:
            return self.db[key]
        else:
            return None

    def set_msg_info(self, chat_id: int, msg_id: int, msg_info: MsgInfo):
        key = f'msg_info_{chat_id}_{msg_id}'
        self.db[key] = msg_info

    def get_system_prompt_by_chat(self, chat_id: int):
        key = f'system_prompt_{chat_id}'
        if key in self.db:
            return self.db[key]
        else:
            return None

    async def start(self) -> None:
        logger.info('Pre bot start, config: {}', self.config_path)
        await self.bot.start(bot_token=self.TELEGRAM_BOT_TOKEN)  # pyright: ignore[reportGeneralTypeIssues]  # telethon's start() is awaitable at runtime
        logger.info('Bot started')
        self.bot.parse_mode = None  # pyright: ignore[reportAttributeAccessIssue]  # telethon supports this at runtime
        me: Any = await self.bot.get_me()  # telethon stubs type as InputPeerUser but returns User at runtime

        @self.bot.on(events.NewMessage)  # pyright: ignore[reportArgumentType]  # telethon accepts event class at runtime
        async def _process(event: events.NewMessage.Event) -> None:  # pyright: ignore[reportUnusedFunction]  # registered by @bot.on decorator
            if event.message.grouped_id is not None:
                return
            prompt_db_key = f'system_prompt_{event.message.chat_id}'
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

            elif text == '/get_prompt' or text == f'/get_prompt@{me.username}':
                if prompt_db_key in self.db:
                    prompt = self.db[prompt_db_key]
                    _ = await self.send_message(
                        event.message.chat_id,
                        f'system prompt:\n\n{prompt}',
                        event.message.id,
                    )
                else:
                    _ = await self.send_message(event.message.chat_id, 'no prompt set yet', event.message.id)
            elif text.startswith('/set_prompt'):
                space_pos = text.find(' ')
                if space_pos == -1:
                    space_pos = len(text) - 1
                prompt = text[space_pos + 1 :]
                self.db[prompt_db_key] = prompt
                _ = await self.send_message(
                    event.message.chat_id,
                    f'system prompt set to:\n\n{prompt}',
                    event.message.id,
                )
            elif text == '/clear_prompt' or text == f'/clear_prompt@{me.username}':
                if prompt_db_key in self.db:
                    del self.db[prompt_db_key]
                _ = await self.send_message(event.message.chat_id, 'system prompt cleared', event.message.id)
            else:
                await self.reply_handler(event.message)

        @self.bot.on(events.Album)  # pyright: ignore[reportArgumentType]  # telethon accepts event class at runtime
        async def _process_album(event: events.Album.Event) -> None:  # pyright: ignore[reportUnusedFunction]  # registered by @bot.on decorator
            if event.chat_id is None or event.sender_id is None:
                return
            await self.album_handler(event)

        admin_input_peer = await self.bot.get_input_entity(self.admin_id)
        _ = await self.bot(
            functions.bots.SetBotCommandsRequest(
                scope=types.BotCommandScopePeer(admin_input_peer),
                lang_code='',
                commands=[
                    types.BotCommand(command, description)
                    for command, description in [
                        ('ping', 'Test bot connectivity'),
                        ('list_models', 'List supported models'),
                        ('add_whitelist', 'Add this group to whitelist (only admin)'),
                        (
                            'del_whitelist',
                            'Delete this group from whitelist (only admin)',
                        ),
                        ('get_whitelist', 'List groups in whitelist (only admin)'),
                        ('get_prompt', 'Get system prompt'),
                        ('set_prompt', 'Set system prompt'),
                        ('clear_prompt', 'Clear system prompt'),
                    ]
                ],
            )
        )

        _ = await self.bot(
            functions.bots.SetBotCommandsRequest(
                scope=types.BotCommandScopeDefault(),
                lang_code='',
                commands=[
                    types.BotCommand(command, description)
                    for command, description in [
                        ('ping', 'Test bot connectivity'),
                        ('list_models', 'List supported models'),
                    ]
                ],
            )
        )
        logger.info('Bot commands registered')

        _ = await self.bot.run_until_disconnected()  # pyright: ignore[reportGeneralTypeIssues]  # telethon's run_until_disconnected() is awaitable at runtime

    def get_prompt(self, model: str) -> str:
        current_time = (datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=8)).strftime('%Y-%m-%d %H:%M:%S')
        return self.system_prompt.format_map({'current_time': current_time, 'model': model})

    def is_whitelist(self, chat_id: int) -> bool:
        whitelist = self.db['whitelist']
        return chat_id in whitelist

    def add_whitelist(self, chat_id: int) -> None:
        whitelist = self.db['whitelist']
        whitelist.add(chat_id)
        self.db['whitelist'] = whitelist

    def del_whitelist(self, chat_id: int) -> None:
        whitelist = self.db['whitelist']
        whitelist.discard(chat_id)
        self.db['whitelist'] = whitelist

    def get_whitelist(self) -> Any:
        return self.db['whitelist']

    @staticmethod
    def only_admin(func):  # pyright: ignore[reportMissingParameterType]  # generic decorator wrapper
        async def new_func(self, message):  # pyright: ignore[reportMissingParameterType]  # generic decorator wrapper
            if message.sender_id != self.admin_id:
                _ = await self.send_message(message.chat_id, 'Only admin can use this command', message.id)
                return
            await func(self, message)

        return new_func

    @staticmethod
    def only_private(func):  # pyright: ignore[reportMissingParameterType]  # generic decorator wrapper
        async def new_func(self, message):  # pyright: ignore[reportMissingParameterType]  # generic decorator wrapper
            if message.chat_id != message.sender_id:
                _ = await self.send_message(
                    message.chat_id,
                    'This command only works in private chat',
                    message.id,
                )
                return
            await func(self, message)

        return new_func

    @staticmethod
    def only_whitelist(func):  # pyright: ignore[reportMissingParameterType]  # generic decorator wrapper
        async def new_func(self, message):  # pyright: ignore[reportMissingParameterType]  # generic decorator wrapper
            if not self.is_whitelist(message.chat_id):
                if message.chat_id == message.sender_id:
                    _ = await self.send_message(message.chat_id, 'This chat is not in whitelist', message.id)
                return
            await func(self, message)

        return new_func

    def construct_chat_history(self, chat_id: int, msg_id: int) -> tuple[list[list[MsgPartInHistory]], Model, str]:
        """Returns (history, model, system_prompt). History is a list of messages in neutral format (MsgPartInHistory)."""
        history: list[list[MsgPartInHistory]] = []
        should_be_bot = False
        model_of_history: Model | None = None
        system_prompt = None

        # trace through the replay chain and construct the message history
        cur_msg_id = msg_id
        while True:
            msg_info = self.get_msg_info(chat_id, cur_msg_id)
            if msg_info is None:
                raise RuntimeError(f'MsgInfo not found ({chat_id=}, {cur_msg_id=}, {msg_id=})')

            # infer the model and endpoint from the first replied msg
            if msg_info.prefix:
                for model in self.models:
                    if model.prefix == msg_info.prefix:
                        overrides = getattr(msg_info, 'overrides', None) or {}
                        model_of_history = self.apply_overrides(model, overrides)

            if msg_info.system_prompt:
                system_prompt = msg_info.system_prompt

            if msg_info.sent_by_bot != should_be_bot:
                raise RuntimeError(f'Role does not match ({chat_id=}, {cur_msg_id=}, {msg_id=}, {should_be_bot=})')

            history.append(list(msg_info.message))
            should_be_bot = not should_be_bot
            if msg_info.reply_id is None:
                break
            cur_msg_id = msg_info.reply_id

        if len(history) % 2 != 1:
            raise RuntimeError(f'First message not from user ({chat_id=}, {msg_id=})')

        assert model_of_history
        if system_prompt is None:
            system_prompt = self.get_prompt(model_of_history.name)

        assert model_of_history
        return history[::-1], model_of_history, system_prompt

    @only_admin
    async def add_whitelist_handler(self, message: Any) -> None:
        if self.is_whitelist(message.chat_id):
            _ = await self.send_message(message.chat_id, 'Already in whitelist', message.id)
            return
        self.add_whitelist(message.chat_id)
        _ = await self.send_message(message.chat_id, 'Whitelist added', message.id)

    @only_admin
    async def del_whitelist_handler(self, message: Any) -> None:
        if not self.is_whitelist(message.chat_id):
            _ = await self.send_message(message.chat_id, 'Not in whitelist', message.id)
            return
        self.del_whitelist(message.chat_id)
        _ = await self.send_message(message.chat_id, 'Whitelist deleted', message.id)

    @only_admin
    @only_private
    async def get_whitelist_handler(self, message: Any) -> None:
        _ = await self.send_message(message.chat_id, str(self.get_whitelist()), message.id)

    @only_whitelist
    async def list_models_handler(self, message: Any) -> None:
        text = ''
        for m in self.models:
            if 'endpoint' in m:
                text += f'"<code>{m.prefix}</code>": <code>{m.name}</code> (from {m.endpoint})\n'
            else:
                text += f'"<code>{m.prefix}</code>": <code>{m.name}</code>\n'
        _ = await self.send_message_html(message.chat_id, text, message.id)

    @retry()
    async def send_message(self, chat_id: int, text: str | RichText, reply_to_message_id: int) -> int:
        logger.debug(f'Sending message: {chat_id=}, {reply_to_message_id=}, {text=}')
        text = RichText(text)
        text, entities = text.to_telegram()
        entity_info = [(type(e).__name__, e.offset, e.length) for e in entities]
        logger.debug(f'Sending message entities: {chat_id=}, text_len={len(text)}, entities={entity_info}')
        msg = await self.bot.send_message(
            chat_id,
            text,
            reply_to=reply_to_message_id,
            link_preview=False,
            formatting_entities=entities,
        )
        logger.debug(f'Message sent: {chat_id=}, {reply_to_message_id=}, {msg.id=}')
        return msg.id

    @retry()
    async def send_message_html(self, chat_id: int, text: str, reply_to_message_id: int) -> int:
        logger.debug(f'Sending message html: {chat_id=}, {reply_to_message_id=}, {text=}')
        msg = await self.bot.send_message(
            chat_id,
            text,
            reply_to=reply_to_message_id,
            link_preview=False,
            parse_mode='html',
        )
        logger.debug(f'Message sent: {chat_id=}, {reply_to_message_id=}, {msg.id=}')
        return msg.id

    @retry()
    async def edit_message(self, chat_id: int, text: str | RichText, message_id: int) -> None:
        logger.debug(f'Editing message: {chat_id=}, {message_id=}, {text=}')
        text = RichText(text)
        text, entities = text.to_telegram()
        entity_info = [(type(e).__name__, e.offset, e.length) for e in entities]
        logger.debug(
            f'Editing message entities: {chat_id=}, {message_id=}, text_len={len(text)}, entities={entity_info}'
        )
        try:
            _ = await self.bot.edit_message(
                chat_id,
                message_id,  # pyright: ignore[reportArgumentType]  # telethon accepts int message_id at runtime
                text,
                link_preview=False,
                formatting_entities=entities,
            )
        except errors.MessageNotModifiedError:
            logger.debug(f'Message not modified: {chat_id=}, {message_id=}')
        else:
            logger.debug(f'Message edited: {chat_id=}, {message_id=}')

    @retry()
    async def delete_message(self, chat_id: int, message_id: int) -> None:
        logger.debug(f'Deleting message: {chat_id=}, {message_id=}')
        _ = await self.bot.delete_messages(
            chat_id,
            message_id,
        )
        logger.debug(f'Message deleted: {chat_id=}, {message_id=}')

    @only_whitelist
    async def reply_handler(self, message: Any) -> None:
        chat_id = message.chat_id
        sender_id = message.sender_id
        msg_id = message.id
        text = message.message
        logger.info(
            f'New message to reply: {chat_id=}, {sender_id=}, {msg_id=}, {text=}, {message.photo=}, {message.document=}'
        )
        reply_to_id: int | None = None
        model_by_prefix: Model | None = None

        extra_photo_message = None
        extra_document_message = None
        if not text and message.photo is None and message.document is None:
            logger.debug(f'Unknown media types {chat_id=}, {msg_id=}')
            return
        if message.is_reply:
            if message.reply_to.quote_text is not None:
                logger.debug(f'Reply contains quote text {chat_id=}, {msg_id=}')
                return
            reply_to_message = await message.get_reply_message()
            if reply_to_message.sender_id == self.bot_id:  # user reply to a bot message
                reply_to_id = message.reply_to.reply_to_msg_id
                assert isinstance(reply_to_id, int)
                await self.pending_reply_manager.wait_for((chat_id, reply_to_id))
            elif reply_to_message.photo is not None:  # user reply to a photo
                extra_photo_message = reply_to_message
            elif reply_to_message.document is not None:  # user reply to a document
                extra_document_message = reply_to_message
            else:
                return

        overrides: dict[str, str | None] = {}
        if not message.is_reply or extra_photo_message is not None or extra_document_message is not None:  # new message
            for m in self.models:
                match = self.match_prefix(text, m.prefix)
                if match is not None:
                    text, overrides = match
                    try:
                        model_by_prefix = self.apply_overrides(m, overrides)
                    except ValueError as e:
                        _ = await self.send_message(chat_id, f'[!] {e}', msg_id)
                        return
                    break
            else:  # not reply or new message to bot
                if chat_id == sender_id:  # if in private chat, send hint
                    _ = await self.send_message(
                        chat_id,
                        'Please start a new conversation with specified prefixes or reply to a bot message',
                        msg_id,
                    )
                return

        photo_message = message if message.photo is not None else extra_photo_message
        photo_hash = None
        if photo_message is not None:
            photo_blob = await photo_message.download_media(bytes)
            photo_hash = save_photo(photo_blob)

        document_message = message if message.document is not None else extra_document_message
        document_text = None
        if document_message is not None:
            if document_message.document.size > self.TEXT_FILE_SIZE_LIMIT:
                _ = await self.send_message(chat_id, 'File too large', msg_id)
                return
            document_blob = await document_message.download_media(bytes)
            try:
                document_text = document_blob.decode()
                assert all(c != '\x00' for c in document_text)
            except UnicodeDecodeError:
                _ = await self.send_message(chat_id, 'File is not text file or not valid UTF-8', msg_id)
                return

        if photo_hash and not text:
            text = self.default_image_reply_prompt if reply_to_id is not None else self.default_image_prompt

        if photo_hash:
            new_message: list[MsgPartInHistory] = [
                make_text_part(text),
                make_image_part(photo_hash),
            ]
        elif document_text:
            if text:
                new_message = [make_text_part(document_text + '\n\n' + text)]
            else:
                new_message = [make_text_part(document_text)]
        else:
            new_message = [make_text_part(text)]

        system_prompt: str | None = (
            self.get_system_prompt_by_chat(chat_id)
            or (model_by_prefix and model_by_prefix.system_prompt)
            or (model_by_prefix and self.get_prompt(model_by_prefix.name))
        )

        # note that prefix and system_prompt are None when reply_id is not None
        self.set_msg_info(
            chat_id,
            msg_id,
            MsgInfo(
                sent_by_bot=False,
                message=new_message,
                reply_id=reply_to_id,
                prefix=model_by_prefix and model_by_prefix.prefix,
                system_prompt=system_prompt,
                overrides=overrides or None,
            ),
        )

        await self._run_completion(chat_id, msg_id)

    async def _run_completion(self, chat_id: int, msg_id: int):
        try:
            chat_history, model, system_prompt = self.construct_chat_history(chat_id, msg_id)
        except RuntimeError as e:
            logger.exception(e)
            _ = await self.send_message(chat_id, f'[!] Error on resolving conversation: {e}', msg_id)
            return

        error_cnt = 0
        while True:
            thinking = ''
            reply = ''
            thinking_done = False
            stream_meta = StreamMeta()
            model_flags: list[str] = []
            if model.thinking is not None:
                model_flags.append(f'thinking={model.thinking}')
            if model.search:
                model_flags.append('search')
            prefix = '🤖 ' + RichText.Code(model.name)
            if model_flags:
                prefix += ' (' + ', '.join(model_flags) + ')'
            prefix += '\n\n'
            async with BotReplyMessages(self, chat_id, msg_id, prefix) as replymsgs:
                try:
                    endpoint = model.endpoint or self.default_endpoint
                    status: str | None = 'Generating...'
                    await replymsgs.update(f'[{status}]')
                    client = self.get_client(endpoint, model)
                    stream = completion(client, chat_history, model, system_prompt, chat_id, msg_id)
                    stream_start = time.time()
                    first_token_time: float | None = None
                    async for event in stream:
                        if isinstance(event, ThinkingDelta):
                            if first_token_time is None:
                                first_token_time = time.time()
                            thinking += event.text
                        elif isinstance(event, ResponseDelta):
                            if first_token_time is None:
                                first_token_time = time.time()
                            thinking_done = True
                            reply += event.text
                        elif isinstance(event, StatusChange):
                            status = event.status
                        elif isinstance(event, StreamMeta):
                            stream_meta = event
                            continue
                        await replymsgs.update(format_reply(thinking, reply, thinking_done, status))
                    stream_end = time.time()
                    usage = stream_meta.usage
                    if usage:
                        ttft = (first_token_time or stream_end) - stream_start
                        usage['TTFT'] = f'{ttft:.1f}s'
                        out_tokens = (usage.get('out') or 0) + (usage.get('reasoning') or 0)
                        if out_tokens and first_token_time:
                            gen_duration = stream_end - first_token_time
                            if gen_duration > 0:
                                usage['TPS'] = f'{out_tokens / gen_duration:.1f}'
                    await replymsgs.update(
                        format_reply(
                            thinking,
                            reply,
                            True,
                            tool_calls=stream_meta.tool_calls or None,
                            usage=usage or None,
                        )
                    )
                    await replymsgs.finalize()
                    full_reply = reply
                    for bot_msg_id, _ in replymsgs.replied_msgs:
                        self.set_msg_info(
                            chat_id,
                            bot_msg_id,
                            MsgInfo(
                                sent_by_bot=True,
                                message=[make_text_part(full_reply)],
                                reply_id=msg_id,
                                prefix=None,
                                system_prompt=None,
                            ),
                        )
                    return

                # handling completion errors
                except Exception as e:
                    error_cnt += 1
                    logger.exception(f'Error on generating exception({chat_id=}, {msg_id=}, {error_cnt=})')
                    retryable_errors = (
                        openai.APITimeoutError,
                        openai.InternalServerError,
                        anthropic.APITimeoutError,
                        anthropic.InternalServerError,
                        TimeoutError,
                    )
                    will_retry = isinstance(e, retryable_errors) and error_cnt <= self.OPENAI_MAX_RETRY
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

    async def album_handler(self, event: events.Album.Event) -> None:
        chat_id: int = event.chat_id  # pyright: ignore[reportAssignmentType]  # checked for None in process_album
        sender_id = event.sender_id

        # Inline whitelist check (Album.Event lacks .id, so @only_whitelist cannot be used)
        if not self.is_whitelist(chat_id):
            if chat_id == sender_id:
                _ = await self.send_message(chat_id, 'This chat is not in whitelist', event.messages[0].id)
            return

        msg_id = event.messages[0].id
        all_msg_ids = [m.id for m in event.messages]
        text = next((m.message for m in event.messages if m.message), '')

        logger.info(
            f'New album to reply: {chat_id=}, {sender_id=}, {msg_id=}, {text=}, num_messages={len(event.messages)}'
        )

        reply_to_id: int | None = None
        model_by_prefix: Model | None = None

        if event.is_reply:
            first_msg = event.messages[0]
            if first_msg.reply_to and first_msg.reply_to.quote_text is not None:
                logger.debug(f'Album reply contains quote text {chat_id=}, {msg_id=}')
                return
            reply_to_message = await event.get_reply_message()
            if reply_to_message.sender_id == self.bot_id:
                reply_to_id = first_msg.reply_to.reply_to_msg_id
                assert isinstance(reply_to_id, int)
                await self.pending_reply_manager.wait_for((chat_id, reply_to_id))
            else:
                return

        overrides: dict[str, str | None] = {}
        if not event.is_reply:
            for m in self.models:
                match = self.match_prefix(text, m.prefix)
                if match is not None:
                    text, overrides = match
                    try:
                        model_by_prefix = self.apply_overrides(m, overrides)
                    except ValueError as e:
                        _ = await self.send_message(chat_id, f'[!] {e}', msg_id)
                        return
                    break
            else:
                if chat_id == sender_id:
                    _ = await self.send_message(
                        chat_id,
                        'Please start a new conversation with specified prefixes or reply to a bot message',
                        msg_id,
                    )
                return

        photo_messages = [m for m in event.messages if m.photo is not None]
        photo_blobs = await asyncio.gather(*[m.download_media(bytes) for m in photo_messages])
        photo_hashes = [save_photo(blob) for blob in photo_blobs]

        if not photo_hashes:
            logger.debug(f'Album has no photos {chat_id=}, {msg_id=}')
            return

        new_message: list[MsgPartInHistory] = [make_text_part(text)]
        for h in photo_hashes:
            new_message.append(make_image_part(h))

        system_prompt: str | None = (
            self.get_system_prompt_by_chat(chat_id)
            or (model_by_prefix and model_by_prefix.system_prompt)
            or (model_by_prefix and self.get_prompt(model_by_prefix.name))
        )

        msg_info = MsgInfo(
            sent_by_bot=False,
            message=new_message,
            reply_id=reply_to_id,
            prefix=model_by_prefix and model_by_prefix.prefix,
            system_prompt=system_prompt,
            overrides=overrides or None,
        )

        for mid in all_msg_ids:
            self.set_msg_info(chat_id, mid, msg_info)

        await self._run_completion(chat_id, msg_id)

    async def ping(self, message: Any) -> None:
        _ = await self.send_message(
            message.chat_id,
            f"""
chat_id={message.chat_id}
user_id={message.sender_id}
is_whitelisted={self.is_whitelist(message.chat_id)}
""",
            message.id,
        )


class BotReplyMessages:
    def __init__(self, cbot: ChatGPTTelegramBot, chat_id: int, orig_msg_id: int, prefix: str | RichText) -> None:
        self.cbot: ChatGPTTelegramBot = cbot
        self.prefix: str | RichText = prefix
        self.msg_len: int = cbot.TELEGRAM_LENGTH_LIMIT - telegram_len(prefix)
        assert self.msg_len > 0
        self.chat_id: int = chat_id
        self.orig_msg_id: int = orig_msg_id
        self.replied_msgs: list[tuple[int, str | RichText]] = []
        self.text: str | RichText = ''
        self.last_update_time: float = 0.0

    async def __aenter__(self) -> 'BotReplyMessages':
        return self

    async def __aexit__(self, type_: type[BaseException] | None, value: BaseException | None, tb: Any) -> None:
        await self.finalize()
        for msg_id, _ in self.replied_msgs:
            self.cbot.pending_reply_manager.remove((self.chat_id, msg_id))

    async def _force_update(self, text: str | RichText) -> None:
        slices: list[str | RichText] = []
        limit = self.msg_len  # first slice accounts for prefix
        while len(text) > limit:
            slices.append(text[:limit])
            text = text[limit:]
            limit = self.cbot.TELEGRAM_LENGTH_LIMIT  # continuation slices get full limit
        if text:
            slices.append(text)
        if not slices:
            slices = ['']  # deal with empty message

        for i in range(min(len(slices), len(self.replied_msgs))):
            msg_id, msg_text = self.replied_msgs[i]
            if slices[i] != msg_text:
                content = (self.prefix + slices[i]) if i == 0 else slices[i]
                await self.cbot.edit_message(self.chat_id, content, msg_id)
                self.replied_msgs[i] = (msg_id, slices[i])
        if len(slices) > len(self.replied_msgs):
            for i in range(len(self.replied_msgs), len(slices)):
                if i == 0:
                    reply_to = self.orig_msg_id
                else:
                    reply_to, _ = self.replied_msgs[i - 1]
                content = (self.prefix + slices[i]) if i == 0 else slices[i]
                msg_id = await self.cbot.send_message(self.chat_id, content, reply_to)
                self.replied_msgs.append((msg_id, slices[i]))
                self.cbot.pending_reply_manager.add((self.chat_id, msg_id))
        if len(self.replied_msgs) > len(slices):
            for i in range(len(slices), len(self.replied_msgs)):
                msg_id, _ = self.replied_msgs[i]
                await self.cbot.delete_message(self.chat_id, msg_id)
                self.cbot.pending_reply_manager.remove((self.chat_id, msg_id))
            self.replied_msgs = self.replied_msgs[: len(slices)]

    async def update(self, text: str | RichText) -> None:
        self.text = text
        now = time.time()
        if now - self.last_update_time >= self.cbot.TELEGRAM_MIN_INTERVAL:
            self.last_update_time = now
            await self._force_update(self.text)

    async def finalize(self) -> None:
        await self._force_update(self.text)
