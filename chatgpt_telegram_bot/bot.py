import atexit
import asyncio
import os
import shelve
import datetime
import time
import traceback
import tomllib
from html import escape as html_escape
from zoneinfo import ZoneInfo
from collections.abc import Sequence
from typing import Any

import anthropic
import diskcache
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
)
from chatgpt_telegram_bot.utils import (
    apply_overrides,
    load_photo,
    match_prefix,
    parse_proxy,
    retry,
    save_photo,
    telegram_len,
    PendingReplyManager,
)
from chatgpt_telegram_bot.completion import completion
from chatgpt_telegram_bot.reply import format_reply


class ChatGPTTelegramBot:
    def __init__(self, config_path: str, data_dir: str) -> None:
        self.config_path: str = config_path
        self.data_dir: str = data_dir
        os.makedirs(data_dir, exist_ok=True)
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
            'You are {model} model. Current date: {current_date}. Reply in the same language as the user sent you. Format the reply in MarkdownV2 but do not use markdown headings, tables, separator lines and TeX math.',
        )
        self.allowed_chats: set[int] = set(_config.get('allowed_chats', []))
        self.allowed_chats.add(self.admin_id)
        self.timezone: ZoneInfo = ZoneInfo(_config.get('timezone', 'UTC'))
        self.default_image_prompt: str = _config.get('default_image_prompt', 'Describe the image')
        self.default_image_reply_prompt: str = _config.get('default_image_reply_prompt', 'Continue')

        for model in self.models:
            if ' ' in model.prefix or ',' in model.prefix:
                raise ValueError(f'prefix must not contain space or ",": "{model.prefix}"')
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
        # msg_info_{chat_id}_{msg_id}: MsgInfo
        self.db: shelve.Shelf[Any] = shelve.open(os.path.join(data_dir, 'db'))
        image_cache_size = _config.get('image_cache_size', 50 * 1024 * 1024)
        if not isinstance(image_cache_size, int):
            raise ValueError(f'image_cache_size must be an integer, got {type(image_cache_size).__name__}')
        self.image_cache: diskcache.Cache = diskcache.Cache(
            os.path.join(data_dir, 'image_cache'), size_limit=image_cache_size
        )

        _ = atexit.register(self.db.close)
        _ = atexit.register(self.image_cache.close)

        self.bot_id: int = int(self.TELEGRAM_BOT_TOKEN.split(':')[0])
        self.pending_reply_manager = PendingReplyManager()
        self.bot: TelegramClient = TelegramClient(
            os.path.join(data_dir, 'bot'),
            self.TELEGRAM_API_ID,
            self.TELEGRAM_API_HASH,
            proxy=parse_proxy(),  # pyright: ignore[reportArgumentType]  # telethon proxy type is broader at runtime
        )

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

    def is_allowed(self, chat_id: int) -> bool:
        return chat_id in self.allowed_chats

    async def start(self) -> None:
        await self.bot.start(bot_token=self.TELEGRAM_BOT_TOKEN)  # pyright: ignore[reportGeneralTypeIssues]  # telethon's start() is awaitable at runtime
        logger.info('Bot started')
        self.bot.parse_mode = None  # pyright: ignore[reportAttributeAccessIssue]  # telethon supports this at runtime
        me: Any = await self.bot.get_me()  # telethon stubs type as InputPeerUser but returns User at runtime

        @self.bot.on(events.NewMessage)  # pyright: ignore[reportArgumentType]  # telethon accepts event class at runtime
        async def _process(event: events.NewMessage.Event) -> None:  # pyright: ignore[reportUnusedFunction]  # registered by @bot.on decorator
            if event.message.grouped_id is not None:
                return
            if event.message.chat_id is None:
                return
            if event.message.sender_id is None:
                return
            if event.message.message is None:
                return
            text = event.message.message
            if text == '/ping' or text == f'/ping@{me.username}':
                await self.ping(event.message)
            elif text == '/help' or text == f'/help@{me.username}':
                await self.help_handler(event.message)
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
                        ('help', 'Show available models and usage'),
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
                        ('help', 'Show available models and usage'),
                    ]
                ],
            )
        )
        logger.info('Bot commands registered')

        _ = await self.bot.run_until_disconnected()  # pyright: ignore[reportGeneralTypeIssues]  # telethon's run_until_disconnected() is awaitable at runtime

    def _format_system_prompt(self, template: str, model: str) -> str:
        current_date = datetime.datetime.now(self.timezone).strftime('%Y-%m-%d')
        return template.format_map({'current_date': current_date, 'model': model})

    def get_prompt(self, model: str) -> str:
        return self._format_system_prompt(self.system_prompt, model)

    def _model_flags(self, model: Model) -> list[str]:
        flags: list[str] = []
        if model.endpoint is not None and model.endpoint != self.default_endpoint:
            flags.append(f'endpoint={model.endpoint}')
        if model.thinking is not None:
            flags.append(f'thinking={model.thinking}')
        if model.search:
            flags.append('search')
        return flags

    @staticmethod
    def only_admin(func):  # pyright: ignore[reportMissingParameterType]  # generic decorator wrapper
        async def new_func(self, message):  # pyright: ignore[reportMissingParameterType]  # generic decorator wrapper
            if message.sender_id != self.admin_id:
                _ = await self.send_message(message.chat_id, 'Only admin can use this command', message.id)
                return
            await func(self, message)

        return new_func

    @staticmethod
    def only_allowed(func):  # pyright: ignore[reportMissingParameterType]  # generic decorator wrapper
        async def new_func(self, message, *args, **kwargs):  # pyright: ignore[reportMissingParameterType]  # generic decorator wrapper
            if not self.is_allowed(message.chat_id):
                if message.chat_id == message.sender_id:
                    _ = await self.send_message(message.chat_id, 'This chat is not allowed', message.id)
                return
            await func(self, message, *args, **kwargs)

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
                        model_of_history = apply_overrides(model, overrides)

            if msg_info.system_prompt is not None:
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

    def _build_help_text(self, error: str | None = None) -> str:
        lines: list[str] = []
        if error:
            lines.append(f'⚠️ {error}\n')
        lines.append('<b>Models</b>')
        for m in self.models:
            flags = ', '.join(self._model_flags(m))
            suffix = f' ({flags})' if flags else ''
            lines.append(f'  <code>{m.prefix}</code> → <code>{m.name}</code>{suffix}')
        lines.append('')
        lines.append('<b>Usage</b>')
        lines.append('  <code>&lt;prefix&gt; &lt;message&gt;</code>')
        lines.append('  <code>&lt;prefix&gt;,t &lt;message&gt;</code>  — enable thinking')
        lines.append('  <code>&lt;prefix&gt;,t=high &lt;message&gt;</code>  — set thinking effort')
        lines.append('  <code>&lt;prefix&gt;,t= &lt;message&gt;</code>  — disable thinking')
        lines.append('  <code>&lt;prefix&gt;,s &lt;message&gt;</code>  — enable search')
        lines.append('  <code>&lt;prefix&gt;,[custom prompt] &lt;message&gt;</code>  — custom system prompt')
        lines.append('  <code>&lt;prefix&gt;,+[extra prompt] &lt;message&gt;</code>  — append to default prompt')
        lines.append('  <code>&lt;prefix&gt;,[] &lt;message&gt;</code>  — clear system prompt')
        lines.append('')
        lines.append('Reply to a bot message to continue the conversation.')
        lines.append('')
        lines.append(f'<b>Default system prompt</b>\n<code>{html_escape(self.system_prompt)}</code>')
        return '\n'.join(lines)

    @only_allowed
    async def help_handler(self, message: Any, error: str | None = None) -> None:
        _ = await self.send_message_html(message.chat_id, self._build_help_text(error), message.id)

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

    @only_allowed
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
                match = match_prefix(text, m.prefix)
                if match is not None:
                    text, overrides = match
                    try:
                        model_by_prefix = apply_overrides(m, overrides)
                    except ValueError as e:
                        _ = await self.send_message(chat_id, f'[!] {e}', msg_id)
                        return
                    break
            else:  # no matching prefix
                if chat_id == sender_id:  # in private chat, show help with error
                    await self.help_handler(message, error='Unknown prefix. Use one of the prefixes below.')
                return

        photo_message = message if message.photo is not None else extra_photo_message
        photo_hash = None
        if photo_message is not None:
            photo_blob = await photo_message.download_media(bytes)
            photo_hash = save_photo(self.image_cache, photo_blob, chat_id, photo_message.id)

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

        if model_by_prefix and model_by_prefix.system_prompt is not None:
            system_prompt: str | None = self._format_system_prompt(model_by_prefix.system_prompt, model_by_prefix.name)
        elif model_by_prefix:
            system_prompt = self.get_prompt(model_by_prefix.name)
        else:
            system_prompt = None
        if model_by_prefix and model_by_prefix.system_prompt_append is not None:
            base = system_prompt if system_prompt is not None else self.get_prompt(model_by_prefix.name)
            system_prompt = (
                base + '\n' + self._format_system_prompt(model_by_prefix.system_prompt_append, model_by_prefix.name)
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
            model_flags = self._model_flags(model)
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

                    async def fetch_image(cid: int, mid: int) -> bytes:
                        msg = await self.bot.get_messages(cid, ids=mid)
                        return await msg.download_media(bytes)  # pyright: ignore[reportAttributeAccessIssue]  # ids=int returns single Message at runtime

                    async def load_image(key: str) -> bytes | None:
                        return await load_photo(self.image_cache, key, fetch_image)

                    stream = completion(client, chat_history, model, system_prompt, chat_id, msg_id, load_image)
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

        # Inline allowed check (Album.Event lacks .id, so @only_allowed cannot be used)
        if not self.is_allowed(chat_id):
            if chat_id == sender_id:
                _ = await self.send_message(chat_id, 'This chat is not allowed', event.messages[0].id)
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
                match = match_prefix(text, m.prefix)
                if match is not None:
                    text, overrides = match
                    try:
                        model_by_prefix = apply_overrides(m, overrides)
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
        photo_hashes = [
            save_photo(self.image_cache, blob, chat_id, msg.id) for blob, msg in zip(photo_blobs, photo_messages)
        ]

        if not photo_hashes:
            logger.debug(f'Album has no photos {chat_id=}, {msg_id=}')
            return

        new_message: list[MsgPartInHistory] = [make_text_part(text)]
        for h in photo_hashes:
            new_message.append(make_image_part(h))

        if model_by_prefix and model_by_prefix.system_prompt is not None:
            system_prompt: str | None = self._format_system_prompt(model_by_prefix.system_prompt, model_by_prefix.name)
        elif model_by_prefix:
            system_prompt = self.get_prompt(model_by_prefix.name)
        else:
            system_prompt = None
        if model_by_prefix and model_by_prefix.system_prompt_append is not None:
            base = system_prompt if system_prompt is not None else self.get_prompt(model_by_prefix.name)
            system_prompt = (
                base + '\n' + self._format_system_prompt(model_by_prefix.system_prompt_append, model_by_prefix.name)
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
is_allowed={self.is_allowed(message.chat_id)}
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
