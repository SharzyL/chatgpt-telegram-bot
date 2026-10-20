import asyncio
import atexit
import dataclasses
import datetime
import dbm
import os
import time
import tomllib
import traceback
from collections.abc import Sequence
from typing import Any
from zoneinfo import ZoneInfo

import anthropic
import diskcache
import httpx
import httpx2
import openai
from loguru import logger
from telethon import Button, TelegramClient, errors, events, functions, types
from telethon.tl.custom import Message

from chatgpt_telegram_bot.commands import (
    COMMANDS_WITH_BODY,
    COMMANDS_WITHOUT_BODY,
    apply_overrides,
    is_foreign_command,
    match_prefix,
    parse_chat_spec,
    parse_command,
    parse_manipulation,
)
from chatgpt_telegram_bot.completion import completion, reasoning_text_of, thinking_text_of
from chatgpt_telegram_bot.markdown import preview_line
from chatgpt_telegram_bot.models import (
    REGENERATION_EDITION,
    EndPoint,
    Model,
    MsgInfo,
    MsgPartInHistory,
    ResponseDelta,
    StatusChange,
    StreamMeta,
    ThinkingDelta,
    ThreadTurn,
    make_image_part,
    make_reasoning_part,
    make_text_part,
    make_thinking_text_part,
    regeneration_edition,
    text_of,
)
from chatgpt_telegram_bot.render import (
    ENTITY_RENDERER,
    PLAIN_LENGTH_LIMIT,
    SERVER_MARKDOWN_RENDERER,
    Rendered,
    Renderer,
)
from chatgpt_telegram_bot.reply import (
    format_imported_prompt,
    format_imported_turn,
    format_linearize_listing,
    format_navigation_page,
    format_reply,
)
from chatgpt_telegram_bot.reply_messages import BotReplyMessages
from chatgpt_telegram_bot.store import Store, migrate_shelf
from chatgpt_telegram_bot.texts import (
    BOT_COMMANDS,
    CHAT_USAGE,
    ENTITY_PROMPT_APPEND,
    EXPORT_USAGE,
    IMPORT_USAGE,
    LINEARIZE_USAGE,
    MANIPULATE_USAGE,
    NAVIGATE_USAGE,
    NO_STORED_TEXT,
    REGENERATE_USAGE,
    RICHTEXT_PROMPT_APPEND,
    help_sections,
)
from chatgpt_telegram_bot.throttle import Throttle
from chatgpt_telegram_bot.transfer import export_filename, format_export, parse_import
from chatgpt_telegram_bot.utils import (
    PendingReplyManager,
    decode_text_file,
    linearize_plan,
    load_photo,
    message_link,
    pack_html_sections,
    parse_proxy,
    plan_digest,
    replace_text_part,
    replied_message_id,
    retry,
    save_photo,
)

# failures the same request may well not meet again: the connection (timeouts included, which
# subclass APIConnectionError), the provider's load, and its rate limit. A stream that drops
# mid-reply surfaces as the SDK's HTTP library's own error, unwrapped: httpx for openai,
# httpx2 for anthropic.
RETRYABLE_ERRORS = (
    openai.APIConnectionError,
    openai.InternalServerError,
    openai.RateLimitError,
    anthropic.APIConnectionError,
    anthropic.InternalServerError,
    anthropic.RateLimitError,
    httpx.TransportError,
    httpx2.TransportError,
    TimeoutError,
)
# a rate limit with no Retry-After is waited out for this long, and a Retry-After is capped
RATE_LIMIT_WAIT = 20.0
RATE_LIMIT_MAX_WAIT = 60.0

# under every part of a live thread's reply; a press finds its turn by the message it is under
LIVE_REPLY_MARKUP = types.ReplyInlineMarkup(
    rows=[
        types.KeyboardInlineButtonRow(
            buttons=[Button.inline('Regenerate', b'live:re'), Button.inline('Delete', b'live:del')]
        )
    ]
)


def _retry_after(response: Any) -> float | None:
    """The seconds a response's Retry-After header asks for, if it gives a number."""
    try:
        return float(response.headers['retry-after'])
    except (AttributeError, KeyError, TypeError, ValueError):
        return None


def retry_delay(e: BaseException, interval: float) -> float | None:
    """How long to wait before generating again after *e*; None when a retry would fail the same way."""
    if isinstance(e, (openai.RateLimitError, anthropic.RateLimitError)):
        return min(_retry_after(e.response) or RATE_LIMIT_WAIT, RATE_LIMIT_MAX_WAIT)
    if isinstance(e, RETRYABLE_ERRORS):
        return interval
    # Anthropic reports overload met mid-stream as an `error` event, which the SDK raises as a
    # bare APIStatusError carrying the stream's 200 rather than as the 529 it stands for
    if isinstance(e, anthropic.APIStatusError) and isinstance(e.body, dict):
        error: Any = e.body.get('error')
        if isinstance(error, dict) and error.get('type') in ('overloaded_error', 'api_error'):
            return interval
    return None


class ChatGPTTelegramBot:
    # this bot's username, read once connected; None until then
    username: str | None = None

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
        default_prompt = (
            'You are {model} model. Current date: {current_date}. Reply in the same language as the user sent you.'
            + ' Format the reply in Markdown; headings, tables, horizontal rules and LaTeX math are all supported.'
            + ' Delimit math with $...$ inline and $$...$$ on its own line.'
        )
        self.system_prompt: str = _config.get('system_prompt', default_prompt)
        self.allowed_chats: set[int] = set(_config.get('allowed_chats', []))
        self.allowed_chats.add(self.admin_id)
        self.timezone: ZoneInfo = ZoneInfo(_config.get('timezone', 'UTC'))
        self.default_image_prompt: str = _config.get('default_image_prompt', 'Describe the image')
        self.default_image_reply_prompt: str = _config.get('default_image_reply_prompt', 'Continue')

        for model in self.models:
            if ' ' in model.prefix or ',' in model.prefix:
                raise ValueError(f'prefix must not contain space or ",": "{model.prefix}"')
            if model.dummy_tool and model.api_type != 'openai':
                raise ValueError(f'dummy_tool needs api_type openai, got model "{model.prefix}"')
            if model.thinking == 'adaptive' and model.api_type != 'anthropic':
                raise ValueError(
                    'thinking="adaptive" is only supported for api_type="anthropic",'
                    + f' got api_type="{model.api_type}" for model "{model.prefix}"'
                )

        # SDK clients by (endpoint, api_type, suffix); the configured models' are made now, so
        # a missing API key stops startup rather than a first generation
        self.endpoint_by_name: dict[str, EndPoint] = {e.name: e for e in self.endpoints}
        self.clients: dict[tuple[str, str, str], Any] = {}
        for model in self.models:
            ep = model.endpoint or self.default_endpoint
            if ep not in self.endpoint_by_name:
                raise ValueError(f'Unknown endpoint "{ep}" for model "{model.prefix}"')
            _ = self.get_client(model)

        # a streaming reply settles at one edit per TELEGRAM_MIN_INTERVAL; the first
        # TELEGRAM_BURST_WINDOW seconds may go at TELEGRAM_BURST_INTERVAL instead
        self.TELEGRAM_MIN_INTERVAL: float = 3.0
        self.TELEGRAM_BURST_INTERVAL: float = 1.0
        self.TELEGRAM_BURST_WINDOW: float = 15.0
        # an import sends a run of messages rather than editing one, and Telegram's guidance
        # for a group is around 20 messages a minute; a short import still goes out at once
        self.IMPORT_MIN_INTERVAL: float = 3.0
        self.IMPORT_BURST_INTERVAL: float = 1.0
        self.IMPORT_BURST_WINDOW: float = 10.0
        self.OPENAI_MAX_RETRY: int = 3
        self.OPENAI_RETRY_INTERVAL: int = 3
        self.TEXT_FILE_SIZE_LIMIT: int = 900_000
        # an image sent as a file is not recompressed by Telegram; the models' own caps are lower
        self.IMAGE_FILE_SIZE_LIMIT: int = 20_000_000
        # /navigate: turns per page, and the width of a preview line in columns -- about what a
        # phone shows on one line, with a wide (CJK) character counting two
        self.NAVIGATE_PER_PAGE: int = 8
        self.NAVIGATE_PREVIEW_WIDTH: int = 36
        # /linearize lists at most this many messages in its confirmation
        self.LINEARIZE_LIST_LIMIT: int = 30

        self.pending_reply_manager: PendingReplyManager = PendingReplyManager()
        # prompts being answered right now, by (chat_id, msg_id). The flag turns True when
        # /abort asks that generation to stop, which it does at its next stream event.
        self.generating: dict[tuple[int, int], bool] = {}
        # set when the generation of the same key is over and its reply stored, for a live
        # edit that has to stop one before answering again
        self.generation_done: dict[tuple[int, int], asyncio.Event] = {}
        # (chat_id, anchor) of a /linearize being carried out, so a second press cannot run it twice
        self.linearizing: set[tuple[int, int]] = set()
        # (chat_id, prompt_id) of a reply whose Delete button is being carried out
        self.deleting_replies: set[tuple[int, int]] = set()

        db_compression = _config.get('db_compression', True)
        if not isinstance(db_compression, bool):
            raise TypeError(f'db_compression must be true or false, got {type(db_compression).__name__}')
        # the store lives for the whole process and is closed via atexit below. A data dir still
        # holding the `shelve` database of earlier versions, `db`, is migrated on first start and
        # the shelf kept beside it, so the store is named v2 to tell the two apart
        store_path = os.path.join(data_dir, 'db-v2.sqlite3')
        shelf_path = os.path.join(data_dir, 'db')
        if not os.path.exists(store_path) and dbm.whichdb(shelf_path):
            migrate_shelf(shelf_path, store_path, db_compression)
        self.store: Store = Store(store_path, db_compression)
        image_cache_size = _config.get('image_cache_size', 50 * 1024 * 1024)
        if not isinstance(image_cache_size, int):
            raise TypeError(f'image_cache_size must be an integer, got {type(image_cache_size).__name__}')
        self.image_cache: diskcache.Cache = diskcache.Cache(
            os.path.join(data_dir, 'image_cache'), size_limit=image_cache_size
        )

        _ = atexit.register(self.store.close)
        _ = atexit.register(self.image_cache.close)

        self.bot_id: int = int(self.TELEGRAM_BOT_TOKEN.split(':')[0])
        self.bot: TelegramClient = TelegramClient(
            os.path.join(data_dir, 'bot'),
            self.TELEGRAM_API_ID,
            self.TELEGRAM_API_HASH,
            proxy=parse_proxy(),  # pyright: ignore[reportArgumentType]  # telethon proxy type is broader at runtime
        )

    def _make_client(self, ep_name: str, api_type: str, suffix: str) -> Any:
        url = self.endpoint_by_name[ep_name].url + suffix
        api_key = os.environ.get(f'OPENAI_API_KEY_{ep_name}')
        if api_key is None:
            raise ValueError(f'No API key for endpoint "{ep_name}": set OPENAI_API_KEY_{ep_name}')
        if api_type in ('openai', 'openai_legacy'):
            return openai.AsyncOpenAI(api_key=api_key, base_url=url, max_retries=0, timeout=300)
        if api_type == 'anthropic':
            return anthropic.AsyncAnthropic(api_key=api_key, base_url=url, max_retries=0, timeout=300)
        raise ValueError(f'Unknown api_type: {api_type}')

    def get_client(self, model: Model) -> Any:
        """The SDK client for *model*, made on first use for a combination no configured model has."""
        endpoint = model.endpoint or self.default_endpoint
        suffix = model.suffix if model.suffix is not None else self.endpoint_by_name[endpoint].default_suffix
        key = (endpoint, model.api_type, suffix)
        if key not in self.clients:
            self.clients[key] = self._make_client(*key)
        return self.clients[key]

    def _configured_model(self, prefix: str) -> Model | None:
        return next((m for m in self.models if m.prefix == prefix), None)

    def _match_model(self, text: str) -> tuple[Model, dict[str, str | None], str] | None:
        """The configured model *text* opens with, its overrides, and the text after them.

        None when *text* opens with no prefix. Raises ValueError, for the user, on an override
        the model cannot take.
        """
        for m in self.models:
            match = match_prefix(text, m.prefix)
            if match is not None:
                rest, overrides = match
                return apply_overrides(m, overrides), overrides, rest
        return None

    def _spec_model(self, spec: str) -> Model:
        """The model behind an ``<endpoint>/<name>`` spec. Raises ValueError, for the user."""
        endpoint, sep, name = spec.partition('/')
        if not sep or not endpoint or not name:
            raise ValueError(f'`{spec}` is neither a model prefix nor `<endpoint>/<model>`; /models lists both')
        if endpoint not in self.endpoint_by_name:
            raise ValueError(f'Unknown endpoint `{endpoint}`; /models lists the endpoints')
        return Model(prefix=spec, name=name, endpoint=endpoint)

    def _stored_model(self, msg_info: MsgInfo) -> Model | None:
        """The model a stored turn names, overrides applied, or None if it names none we know."""
        base: Model | None = None
        if msg_info.prefix:
            base = self._configured_model(msg_info.prefix)
        elif spec := msg_info.model_spec:
            try:
                base = self._spec_model(spec)
            except ValueError:  # its endpoint has left the config since
                return None
        if base is None:
            return None
        return apply_overrides(base, msg_info.overrides or {})

    def _chat_model(self, head: str, overrides: dict[str, str | None]) -> tuple[Model, str | None]:
        """The model a /chat names, and the spec to store for it when no prefix stands for it.

        A configured prefix is read first, so one written like a spec (``o/gf``) means the
        model it is configured as. Raises ValueError, for the user.
        """
        model = self._configured_model(head)
        spec = None
        if model is None:
            model, spec = self._spec_model(head), head
        model = apply_overrides(model, overrides)
        # made now, so a missing API key is reported here rather than as a failed generation
        _ = self.get_client(model)
        return model, spec

    def get_msg_info(self, chat_id: int, msg_id: int) -> MsgInfo | None:
        return self.store.get(chat_id, msg_id)

    def set_msg_info(self, chat_id: int, msg_id: int, msg_info: MsgInfo) -> None:
        """Rewrite the turn *msg_id* shows -- every part of it -- or store a new one-message turn."""
        self.store.update(chat_id, msg_id, msg_info)

    def is_allowed(self, chat_id: int) -> bool:
        return chat_id in self.allowed_chats

    async def start(self) -> None:
        await self.bot.start(bot_token=self.TELEGRAM_BOT_TOKEN)  # pyright: ignore[reportGeneralTypeIssues]  # telethon's start() is awaitable at runtime
        logger.info('Bot started')
        self.bot.parse_mode = None  # pyright: ignore[reportAttributeAccessIssue]  # telethon supports this at runtime
        me: Any = await self.bot.get_me()  # telethon stubs type as InputPeerUser but returns User at runtime
        self.username = me.username

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
            if is_foreign_command(text, me.username):
                return
            command = parse_command(text, me.username)
            name, body = command if command is not None else ('', '')
            if name in COMMANDS_WITHOUT_BODY and body:
                await self.command_argument_error(event.message, name)
            elif name == 'ping':
                await self.ping(event.message)
            elif name == 'help':
                await self.help_handler(event.message)
            elif name == 'models':
                await self.models_handler(event.message)
            elif name == 'chat':
                await self.chat_handler(event.message, body)
            elif name == 'manipulate':
                await self.manipulate_handler(event.message, body)
            elif name == 'import':
                await self.import_handler(event.message, body)
            elif name == 'regenerate':
                await self.regenerate_handler(event.message, body)
            elif name == 'abort':
                await self.abort_handler(event.message)
            elif name == 'navigate':
                await self.navigate_handler(event.message)
            elif name == 'linearize':
                await self.linearize_handler(event.message)
            elif name == 'export':
                await self.export_handler(event.message, body)
            else:
                await self.reply_handler(event.message)

        @self.bot.on(events.MessageEdited)  # pyright: ignore[reportArgumentType]  # telethon accepts event class at runtime
        async def _process_edit(event: events.MessageEdited.Event) -> None:  # pyright: ignore[reportUnusedFunction]  # registered by @bot.on decorator
            if event.message.chat_id is None or event.message.sender_id is None:
                return
            await self.edit_handler(event.message)

        @self.bot.on(events.CallbackQuery(pattern=rb'^lin:'))
        async def _linearize_confirm(event: events.CallbackQuery.Event) -> None:  # pyright: ignore[reportUnusedFunction]  # registered by @bot.on decorator
            await self.linearize_callback(event)

        @self.bot.on(events.CallbackQuery(pattern=rb'^live:'))
        async def _live_reply_button(event: events.CallbackQuery.Event) -> None:  # pyright: ignore[reportUnusedFunction]  # registered by @bot.on decorator
            await self.live_reply_callback(event)

        @self.bot.on(events.CallbackQuery(pattern=rb'^nav:'))
        async def _navigate_page(event: events.CallbackQuery.Event) -> None:  # pyright: ignore[reportUnusedFunction]  # registered by @bot.on decorator
            await self.navigate_callback(event)

        @self.bot.on(events.Album)  # pyright: ignore[reportArgumentType]  # telethon accepts event class at runtime
        async def _process_album(event: events.Album.Event) -> None:  # pyright: ignore[reportUnusedFunction]  # registered by @bot.on decorator
            if event.chat_id is None or event.sender_id is None:
                return
            await self.album_handler(event)

        admin_input_peer = await self.bot.get_input_entity(self.admin_id)
        commands = [types.BotCommand(command, description) for command, description in BOT_COMMANDS]
        for scope in (types.BotCommandScopePeer(admin_input_peer), types.BotCommandScopeDefault()):
            _ = await self.bot(functions.bots.SetBotCommandsRequest(scope=scope, lang_code='', commands=commands))
        logger.info('Bot commands registered')

        _ = await self.bot.run_until_disconnected()  # pyright: ignore[reportGeneralTypeIssues]  # telethon's run_until_disconnected() is awaitable at runtime

    def _format_system_prompt(self, template: str, model: str) -> str:
        current_date = datetime.datetime.now(self.timezone).strftime('%Y-%m-%d')
        return template.format_map({'current_date': current_date, 'model': model})

    def _system_prompt_for(self, model: Model | None) -> str | None:
        """The prompt a message that named *model* records, or None if it named no model."""
        if model is None:
            return None
        if model.system_prompt is not None:
            prompt = self._format_system_prompt(model.system_prompt, model.name)
        else:
            prompt = self.get_prompt(model)
        if model.system_prompt_append is not None:
            prompt = prompt + '\n' + self._format_system_prompt(model.system_prompt_append, model.name)
        return prompt

    def get_prompt(self, model: Model) -> str:
        """The default system prompt, with formatting guidance for this model's renderer."""
        append = RICHTEXT_PROMPT_APPEND if model.richtext else ENTITY_PROMPT_APPEND
        return self._format_system_prompt(self.system_prompt + '\n' + append, model.name)

    @staticmethod
    def get_renderer(model: Model | None) -> Renderer:
        """The renderer a model asks for; entities when no model has been chosen yet."""
        return SERVER_MARKDOWN_RENDERER if model is not None and model.richtext else ENTITY_RENDERER

    def _reply_prefix(self, model: Model, manipulated: bool = False, edition: int = 1) -> str:
        """The header of a reply; *edition* above 1 marks a regenerated one with its version."""
        prefix = f'🤖 `{model.name}`'
        flags = self._model_flags(model)
        if edition > 1:
            flags.append(f'version={edition}')
        if flags:
            prefix += ' (' + ', '.join(flags) + ')'
        if manipulated:
            prefix += ' (manipulated)'
        return prefix + '\n\n'

    def _import_prefix(self, model: Model | None, index: int, total: int) -> str:
        name = f' `{model.name}`' if model is not None else ''
        return f'📥{name} ({index}/{total})\n\n'

    def _model_flags(self, model: Model) -> list[str]:
        flags: list[str] = []
        if model.endpoint is not None and model.endpoint != self.default_endpoint:
            flags.append(f'endpoint={model.endpoint}')
        if model.thinking is not None:
            flags.append(f'thinking={model.thinking}')
        if model.search:
            flags.append('search')
        if model.richtext:
            flags.append('richtext')
        if model.dummy_tool:
            flags.append('dummy_tool')
        if model.live:
            flags.append('live')
        return flags

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

            # the walk runs newest to oldest, so the first prefix found is the most recent
            # one the user named; an earlier prefix only applies until a reply replaces it
            if model_of_history is None:
                model_of_history = self._stored_model(msg_info)

            # likewise for the prompt, which a switch of model brings along with it
            if system_prompt is None and msg_info.system_prompt is not None:
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

        if model_of_history is None:
            # an imported chat starts without one; replying with a prefix is what picks it
            raise RuntimeError('no model for this conversation, reply with a model prefix to choose one')
        if system_prompt is None:
            system_prompt = self.get_prompt(model_of_history)

        return history[::-1], model_of_history, system_prompt

    def _build_models_text(self) -> str:
        """The /models page: every prefix, the model behind it and what it overrides."""
        lines = ['<b>Models</b>  — start a message with a prefix to talk to one']
        for m in self.models:
            flags = ', '.join(self._model_flags(m))
            suffix = f' ({flags})' if flags else ''
            lines.append(f'  <code>{m.prefix}</code> → <code>{m.name}</code>{suffix}')
        endpoints = ', '.join(f'<code>{e.name}</code>' for e in self.endpoints)
        lines.append(f'\nEndpoints, for <code>/chat &lt;endpoint&gt;/&lt;model&gt;</code>: {endpoints}')
        return '\n'.join(lines)

    @only_allowed
    async def help_handler(self, message: Any) -> None:
        reply_to = message.id
        for text in pack_html_sections(help_sections(self.system_prompt), PLAIN_LENGTH_LIMIT):
            reply_to = await self.send_message_html(message.chat_id, text, reply_to)

    @only_allowed
    async def models_handler(self, message: Any) -> None:
        _ = await self.send_message_html(message.chat_id, self._build_models_text(), message.id)

    def _no_prefix_text(self) -> str:
        """The short answer to a message that opens a thread without naming a model."""
        example = self.models[0].prefix if self.models else 'prefix'
        return (
            f'[!] Start a conversation with a model prefix, such as `{example} hello`.'
            + ' /models lists the prefixes; /help covers the rest.'
        )

    @staticmethod
    def _sent_message_id(result: Any) -> int:
        """Pull the new message id out of any of the shapes sendMessage can answer with."""
        # a private-chat send can come back as UpdateShortSentMessage, which has no .updates
        if isinstance(result, types.UpdateShortSentMessage):
            return result.id
        for update in getattr(result, 'updates', ()):
            if isinstance(update, types.UpdateMessageID):
                return update.id
            if isinstance(update, (types.UpdateNewMessage, types.UpdateNewChannelMessage)):
                return update.message.id
        raise RuntimeError(f'sendMessage returned no message id: {type(result).__name__}')

    @staticmethod
    def _rich_message(body: Rendered) -> types.InputRichMessageMarkdown | None:
        if body.rich_markdown is None:
            return None
        return types.InputRichMessageMarkdown(markdown=body.rich_markdown)

    @staticmethod
    def _as_body(body: Rendered | str) -> Rendered:
        """The bot's own short messages are given as markdown and always rendered server-side."""
        return body if isinstance(body, Rendered) else SERVER_MARKDOWN_RENDERER.render_one(body)

    @retry()
    async def send_message(self, chat_id: int, body: Rendered | str, reply_to_message_id: int) -> int:
        body = self._as_body(body)
        logger.debug(f'Sending message: {chat_id=}, {reply_to_message_id=}, text={body.text!r}')
        # telethon has no high-level wrapper for rich messages, so sendMessage is invoked directly
        peer = await self.bot.get_input_entity(chat_id)
        result: Any = await self.bot(
            functions.messages.SendMessageRequest(
                peer=peer,
                message=body.text,
                reply_to=types.InputReplyToMessage(reply_to_msg_id=reply_to_message_id),
                no_webpage=True,
                entities=body.entities or None,
                rich_message=self._rich_message(body),
                reply_markup=body.markup,
            )
        )
        msg_id = self._sent_message_id(result)
        logger.debug(f'Message sent: {chat_id=}, {reply_to_message_id=}, {msg_id=}')
        return msg_id

    @retry()
    async def send_message_html(self, chat_id: int, text: str, reply_to_message_id: int, buttons: Any = None) -> int:
        logger.debug(f'Sending message html: {chat_id=}, {reply_to_message_id=}, {text=}')
        msg = await self.bot.send_message(
            chat_id,
            text,
            reply_to=reply_to_message_id,
            link_preview=False,
            parse_mode='html',
            buttons=buttons,
        )
        logger.debug(f'Message sent: {chat_id=}, {reply_to_message_id=}, {msg.id=}')
        return msg.id

    @retry()
    async def send_document(
        self, chat_id: int, data: bytes, filename: str, caption: str, reply_to_message_id: int
    ) -> int:
        logger.debug(f'Sending document: {chat_id=}, {reply_to_message_id=}, {filename=}, bytes={len(data)}')
        # a single file comes back as one Message; only an album would be a list
        msg: Any = await self.bot.send_file(
            chat_id,
            file=data,
            caption=caption,
            reply_to=reply_to_message_id,
            force_document=True,
            mime_type='application/json',
            attributes=[types.DocumentAttributeFilename(filename)],
        )
        return msg.id

    @retry()
    async def edit_message(self, chat_id: int, body: Rendered | str, message_id: int) -> None:
        body = self._as_body(body)
        logger.trace(f'Editing message: {chat_id=}, {message_id=}, text={body.text!r}')
        peer = await self.bot.get_input_entity(chat_id)
        try:
            _ = await self.bot(
                functions.messages.EditMessageRequest(
                    peer=peer,
                    id=message_id,
                    message=body.text,
                    no_webpage=True,
                    entities=body.entities or None,
                    rich_message=self._rich_message(body),
                    reply_markup=body.markup,
                )
            )
        except errors.MessageNotModifiedError:
            logger.trace(f'Message not modified: {chat_id=}, {message_id=}')
        else:
            logger.trace(f'Message edited: {chat_id=}, {message_id=}')

    @retry()
    async def delete_message(self, chat_id: int, message_id: int) -> None:
        logger.debug(f'Deleting message: {chat_id=}, {message_id=}')
        _ = await self.bot.delete_messages(
            chat_id,
            message_id,
        )
        logger.debug(f'Message deleted: {chat_id=}, {message_id=}')

    async def _replied_bot_turn(self, message: Any, usage: str) -> tuple[int, MsgInfo, int] | None:
        """The bot reply *message* answers, its stored turn and the prompt it answered.

        None, having told the user why, when *message* replies to no stored bot reply.
        """
        chat_id = message.chat_id
        target_id = replied_message_id(message)
        replied = await message.get_reply_message() if target_id is not None else None
        if target_id is None or replied is None or replied.sender_id != self.bot_id:
            _ = await self.send_message(chat_id, usage, message.id)
            return None
        # the reply may still be streaming, and its MsgInfo is only written once it finishes
        await self.pending_reply_manager.wait_for((chat_id, target_id))
        info = self.get_msg_info(chat_id, target_id)
        if info is None or not info.sent_by_bot or info.reply_id is None:
            _ = await self.send_message(chat_id, NO_STORED_TEXT, message.id)
            return None
        return target_id, info, info.reply_id

    async def _replied_anchor(self, message: Any, usage: str) -> int | None:
        """The message *message* replies to, once any reply streaming into it is done; None, having said why, if none."""
        anchor_id = replied_message_id(message)
        if anchor_id is None:
            _ = await self.send_message(message.chat_id, usage, message.id)
            return None
        # a reply still streaming has no MsgInfo yet
        await self.pending_reply_manager.wait_for((message.chat_id, anchor_id))
        return anchor_id

    @only_allowed
    async def manipulate_handler(self, message: Any, body: str) -> None:
        """Replace a stored bot reply with user-supplied text (and optionally a chain of thought)."""
        chat_id = message.chat_id
        msg_id = message.id
        logger.info(f'Manipulate requested: {chat_id=}, {msg_id=}, {body=}')

        target = await self._replied_bot_turn(message, MANIPULATE_USAGE)
        if target is None:
            return
        target_id, target_info, prompt_id = target

        thinking, text = parse_manipulation(body)
        if not text:
            _ = await self.send_message(chat_id, MANIPULATE_USAGE, msg_id)
            return

        # the model may have been switched part-way down the chain, so resolve it there
        try:
            history, model, _ = self.construct_chat_history(chat_id, prompt_id)
        except RuntimeError as e:
            logger.exception(e)
            _ = await self.send_message(chat_id, f'[!] Error on resolving conversation: {e}', msg_id)
            return

        # the first message of an imported turn shows its prompt too, keyed by its negated id
        lead = format_imported_prompt(text_of(history[-1])) if prompt_id == -target_id else ''
        # only the message replied to is rewritten, so a body that no longer fits is refused
        bodies = self.get_renderer(model).render(
            lead + format_reply(thinking or '', text, True),
            self._reply_prefix(model, manipulated=True, edition=regeneration_edition(target_info)),
            thinking is not None,
        )
        if len(bodies) > 1:
            _ = await self.send_message(chat_id, '[!] Manipulated message is too long', msg_id)
            return

        # any stored reasoning items describe the output being replaced, so they are dropped
        new_parts: list[MsgPartInHistory] = []
        if thinking:
            new_parts.append(make_thinking_text_part(thinking))
        new_parts.append(make_text_part(text))
        self.set_msg_info(
            chat_id,
            target_id,
            target_info._replace(message=new_parts),
        )
        # an edit without its buttons would take a live reply's away
        edited = dataclasses.replace(bodies[0], markup=LIVE_REPLY_MARKUP) if model.live else bodies[0]
        await self.edit_message(chat_id, edited, target_id)
        logger.info(f'Manipulated {chat_id=}, {target_id=}, has_thinking={thinking is not None}')

    async def _import_document(self, message: Any) -> Any:
        """The message holding the file to import: this one, or the one it replies to."""
        if message.document is not None:
            return message
        if replied_message_id(message) is not None:
            replied = await message.get_reply_message()
            if replied is not None and replied.document is not None:
                return replied
        return None

    def _import_model(self, spec: str) -> tuple[Model | None, dict[str, str | None]]:
        """The model an `/import` argument names, or None when it names none.

        An import with no prefix records no model, exactly as a reply that names none does;
        the chat is continued by replying to a turn with one.
        """
        spec = spec.strip()
        if not spec:
            return None, {}
        matched = self._match_model(spec)
        if matched is None:
            raise ValueError(f'Unknown prefix: `{spec}`')
        model, overrides, _ = matched
        return model, overrides

    @only_allowed
    async def import_handler(self, message: Any, body: str) -> None:
        chat_id = message.chat_id
        msg_id = message.id

        document_message = await self._import_document(message)
        if document_message is None:
            _ = await self.send_message(chat_id, IMPORT_USAGE, msg_id)
            return

        try:
            model, overrides = self._import_model(body)
        except ValueError as e:  # an unknown prefix, or an override the model cannot take
            _ = await self.send_message(chat_id, f'[!] {e}', msg_id)
            return

        # no size limit here: TEXT_FILE_SIZE_LIMIT exists because an attached text file is inlined
        # into one prompt, and an import never reaches a model whole -- each turn is its own message
        text = decode_text_file(await document_message.download_media(bytes))
        if text is None:
            _ = await self.send_message(chat_id, '[!] File is not text file or not valid UTF-8', msg_id)
            return
        try:
            chat = parse_import(text)
        except ValueError as e:
            _ = await self.send_message(chat_id, f'[!] {e}', msg_id)
            return

        throttle = Throttle(self.IMPORT_MIN_INTERVAL, self.IMPORT_BURST_INTERVAL, self.IMPORT_BURST_WINDOW)
        turns = f'{len(chat.turns)} turn' + ('s' if len(chat.turns) != 1 else '')
        note = f'Importing a chat titled "{chat.title}" — {turns}, '
        note += (
            f'continuing with `{model.name}`'
            if model is not None
            else 'reply to a turn with a model prefix to continue it'
        )
        if chat.trailing_prompt is not None:
            note += '\nThe last prompt has no response and was left out.'
        # the pacing below makes a long export take minutes, which is worth saying up front
        eta = (len(chat.turns) - throttle.capacity) * self.IMPORT_MIN_INTERVAL
        if eta > 60:
            note += f'\nSending is paced, so this takes about {round(eta / 60)} minutes.'
        _ = await self.send_message(chat_id, note, msg_id)

        # each turn is shown as one message but stored as two nodes, because the history has
        # to alternate user and assistant; the user half is keyed by the negated id of the
        # message that displays the turn, which no real message can collide with
        renderer = self.get_renderer(model)
        system_prompt = self._system_prompt_for(model)
        reply_to = msg_id
        previous_turn_id: int | None = None
        for i, (prompt, response, reasoning) in enumerate(chat.turns, start=1):
            bodies = renderer.render(
                format_imported_turn(prompt, response, reasoning),
                self._import_prefix(model, i, len(chat.turns)),
                collapse_first_quote=bool(reasoning),
            )
            msg_ids: list[int] = []
            for part in bodies:
                # a long export is a burst of sends that would otherwise trip flood control,
                # and a flood wait is a penalty that grows each time it is tripped, so the
                # run is paced instead of being left for `retry` to absorb
                await asyncio.sleep(throttle.reserve(time.time()))
                reply_to = await self.send_message(chat_id, part, reply_to)
                msg_ids.append(reply_to)
            prompt_id = -msg_ids[0]
            # only the first turn carries the model, as the head of the chain does elsewhere
            self.set_msg_info(
                chat_id,
                prompt_id,
                MsgInfo(
                    sent_by_bot=False,
                    message=[make_text_part(prompt)],
                    reply_id=previous_turn_id,
                    prefix=model.prefix if model is not None and i == 1 else None,
                    system_prompt=system_prompt if i == 1 else None,
                    overrides=(overrides or None) if i == 1 else None,
                ),
            )
            # the chain of thought is stored as a reply's would be, so dt can send it back
            response_parts = [make_thinking_text_part(reasoning)] if reasoning else []
            response_parts.append(make_text_part(response))
            response_info = MsgInfo(
                sent_by_bot=True, message=response_parts, reply_id=prompt_id, prefix=None, system_prompt=None
            )
            # a turn too long for one message is one turn shown by every part, so replying to
            # any of them continues from the same point
            self.store.put_turn(chat_id, msg_ids, response_info)
            previous_turn_id = msg_ids[0]

        logger.info(f'Imported {chat_id=}, {msg_id=}, title={chat.title!r}, turns={len(chat.turns)}')

    def _thread_chain(self, chat_id: int, anchor_id: int) -> list[tuple[int, MsgInfo]] | None:
        """The stored nodes of the thread ending at *anchor_id*, oldest first; None if it is not stored.

        The walk goes backwards through reply_id, so it sees only the branch that leads to the
        anchor: anchored on a prompt, that prompt's own answer is not part of it. A missing link
        or a loop in damaged data ends the walk early, leaving a chain whose first node still
        has a reply_id -- which is how a caller that needs the whole thread can tell.
        """
        chain: list[tuple[int, MsgInfo]] = []
        seen: set[int] = set()
        cur: int | None = anchor_id
        while cur is not None:
            info = self.get_msg_info(chat_id, cur)
            if info is None or cur in seen:
                break
            seen.add(cur)
            chain.append((cur, info))
            cur = info.reply_id
        return chain[::-1] or None

    def _thread_turns(self, chat_id: int, anchor_id: int) -> list[ThreadTurn] | None:
        """The turns of the thread ending at *anchor_id*, oldest first, or None if it is not stored."""
        chain = self._thread_chain(chat_id, anchor_id)
        return self._turns_of(chain) if chain is not None else None

    @staticmethod
    def _turns_of(chain: list[tuple[int, MsgInfo]]) -> list[ThreadTurn]:
        """Pair a chain of nodes, oldest first, into prompts and their answers."""
        turns: list[ThreadTurn] = []
        for mid, info in chain:
            text = text_of(info.message)
            if not info.sent_by_bot:
                if any(p.type_ == 'image' for p in info.message):
                    text = ('🖼 ' + text).strip()
                turns.append(ThreadTurn(prompt=text, response=None, msg_id=mid))
            elif turns and turns[-1].response is None:
                # jump to the start of a split answer rather than whichever part the chain names
                turns[-1] = turns[-1]._replace(response=text, msg_id=info.part_ids[0] if info.part_ids else mid)
        return turns

    def _navigation(self, chat_id: int, anchor_id: int, page: int | None) -> tuple[str, Any] | None:
        """A /navigate page and its buttons; None for a thread that is not stored. page=None is the last."""
        turns = self._thread_turns(chat_id, anchor_id)
        if not turns:
            return None
        per_page = self.NAVIGATE_PER_PAGE
        if page is None:
            page = (len(turns) - 1) // per_page
        body, page, pages = format_navigation_page(
            turns, page, per_page, self.NAVIGATE_PREVIEW_WIDTH, lambda mid: message_link(chat_id, mid)
        )
        noun = 'turn' if len(turns) == 1 else 'turns'
        text = f'🧭 <b>Thread</b> · {len(turns)} {noun}\n\n{body}'
        if pages == 1:
            return text, None
        # the anchor travels in the button, so a press re-reads the thread as it is now
        row: list[Any] = []
        if page > 0:
            row.append(Button.inline('‹', f'nav:{anchor_id}:{page - 1}'.encode()))
        row.append(Button.inline(f'{page + 1}/{pages}', b'nav:noop'))
        if page < pages - 1:
            row.append(Button.inline('›', f'nav:{anchor_id}:{page + 1}'.encode()))
        return text, [row]

    @only_allowed
    async def navigate_handler(self, message: Any) -> None:
        """List the turns of the replied-to thread, a page at a time."""
        chat_id = message.chat_id
        msg_id = message.id
        anchor_id = await self._replied_anchor(message, NAVIGATE_USAGE)
        if anchor_id is None:
            return
        rendered = self._navigation(chat_id, anchor_id, None)
        if rendered is None:
            _ = await self.send_message(chat_id, NO_STORED_TEXT, msg_id)
            return
        text, buttons = rendered
        _ = await self.send_message_html(chat_id, text, msg_id, buttons=buttons)

    async def navigate_callback(self, event: Any) -> None:
        """Turn the page of a /navigate listing."""
        chat_id: int | None = event.chat_id
        data: bytes = event.data
        if chat_id is None or not self.is_allowed(chat_id) or data == b'nav:noop':
            await event.answer()
            return
        try:
            _, anchor, page = data.decode().split(':')
            anchor_id, page_no = int(anchor), int(page)
        except ValueError:
            await event.answer()
            return
        rendered = self._navigation(chat_id, anchor_id, page_no)
        if rendered is None:
            await event.answer('That thread is no longer stored', alert=True)
            return
        text, buttons = rendered
        try:
            await event.edit(text, parse_mode='html', buttons=buttons, link_preview=False)
        except errors.MessageNotModifiedError:
            pass
        await event.answer()

    @staticmethod
    def _export_content(info: MsgInfo) -> str:
        """A node's text as an export carries it: the answer, and a marker for an image.

        A reply's plain chain of thought goes in the entry's `reasoning` field beside this;
        an encrypted reasoning item means nothing outside this bot and is left behind.
        """
        text = text_of(info.message)
        if not info.sent_by_bot and any(p.type_ == 'image' for p in info.message):
            return '[image]' + ('\n\n' + text if text else '')
        return text

    async def _message_dates(self, chat_id: int, msg_ids: list[int]) -> dict[int, datetime.datetime]:
        """When each message was sent, for those Telegram still has; an export's timestamps."""
        if not msg_ids:
            return {}
        try:
            found: Any = await self.bot.get_messages(chat_id, ids=msg_ids)
        except Exception:  # noqa: BLE001  # timestamps are optional; a failed lookup only drops them
            logger.warning(f'Could not fetch message dates for export ({chat_id=})')
            return {}
        return {m.id: m.date for m in found if m is not None and m.date is not None}

    @only_allowed
    async def export_handler(self, message: Any, body: str) -> None:
        """Send the replied-to thread as a JSON file that /import can read back."""
        chat_id = message.chat_id
        msg_id = message.id
        anchor_id = await self._replied_anchor(message, EXPORT_USAGE)
        if anchor_id is None:
            return
        chain = self._thread_chain(chat_id, anchor_id)
        if chain is None:
            _ = await self.send_message(chat_id, NO_STORED_TEXT, msg_id)
            return
        first = chain[0][1]
        if first.reply_id is not None or first.sent_by_bot:
            # a partial thread would start mid-conversation, which /import refuses
            _ = await self.send_message(chat_id, '[!] Part of that thread is no longer stored', msg_id)
            return

        # an imported prompt lives under a negative id and was never a message, so it has no date
        dates = await self._message_dates(chat_id, [mid for mid, _ in chain if mid > 0])
        messages = [
            (
                'assistant' if info.sent_by_bot else 'user',
                self._export_content(info),
                dates.get(mid),
                thinking_text_of(info.message) if info.sent_by_bot else None,
            )
            for mid, info in chain
        ]
        title = body.strip() or preview_line(messages[0][1], 40) or 'Untitled'
        document = format_export(title, messages, datetime.datetime.now(datetime.UTC)).encode()

        caption = f'Exported "{title}" — {len(messages)} messages'
        _ = await self.send_document(chat_id, document, export_filename(title), caption, msg_id)
        logger.info(f'Exported {chat_id=}, {anchor_id=}, messages={len(messages)}, bytes={len(document)}')

    def _linearize_plan(self, chat_id: int, anchor_id: int) -> tuple[dict[int, MsgInfo], set[int]] | None:
        """The anchor's tree and the ids a /linearize at *anchor_id* would delete; None if not stored."""
        anchor = self.get_msg_info(chat_id, anchor_id)
        if anchor is None or anchor.root_msg_id is None:
            return None
        nodes = self.store.tree_nodes(chat_id, anchor.root_msg_id)
        plan = linearize_plan(nodes, anchor_id)
        if plan is None:
            return None
        _, delete = plan
        return nodes, delete

    def _generating_in(self, chat_id: int, nodes: dict[int, MsgInfo], ids: set[int]) -> bool:
        """Whether a generation is answering one of *ids*, or writing a reply into it."""
        for gen_chat, prompt_id in self.generating:
            if gen_chat == chat_id and (prompt_id in ids or prompt_id in nodes and nodes[prompt_id].reply_id in ids):
                return True
        return False

    @only_allowed
    async def linearize_handler(self, message: Any) -> None:
        """Offer to delete every branch of a thread except the one leading to the replied message."""
        chat_id = message.chat_id
        msg_id = message.id
        anchor_id = await self._replied_anchor(message, LINEARIZE_USAGE)
        if anchor_id is None:
            return
        planned = self._linearize_plan(chat_id, anchor_id)
        if planned is None:
            _ = await self.send_message(chat_id, NO_STORED_TEXT, msg_id)
            return
        nodes, delete = planned
        listing, count = format_linearize_listing(nodes, delete, self.NAVIGATE_PREVIEW_WIDTH, self.LINEARIZE_LIST_LIMIT)
        if count == 0:
            _ = await self.send_message(chat_id, 'Nothing to delete: that thread is already a single line.', msg_id)
            return
        if self._generating_in(chat_id, nodes, delete):
            _ = await self.send_message(chat_id, '[!] Part of that thread is still generating; /abort it first', msg_id)
            return
        # the digest pins the confirmation to this exact plan, and the sender id to whoever asked
        tag = f'lin:{anchor_id}:{plan_digest(delete)}:{message.sender_id}'
        noun = 'message' if count == 1 else 'messages'
        text = f'🧹 <b>Delete {count} {noun}?</b>\nEverything branching off this thread except the line to the replied message.\n\n{listing}'
        buttons = [
            [Button.inline(f'🗑 Delete {count}', f'{tag}:go'.encode()), Button.inline('Cancel', f'{tag}:no'.encode())]
        ]
        _ = await self.send_message_html(chat_id, text, msg_id, buttons=buttons)

    @retry()
    async def delete_messages(self, chat_id: int, message_ids: list[int]) -> None:
        logger.debug(f'Deleting messages: {chat_id=}, count={len(message_ids)}')
        _ = await self.bot.delete_messages(chat_id, message_ids, revoke=True)

    async def _still_there(self, chat_id: int, message_ids: list[int]) -> set[int]:
        """Which of *message_ids* still exist; on a failed lookup, all of them, to be safe."""
        try:
            found: Any = await self.bot.get_messages(chat_id, ids=message_ids)
        except Exception:  # noqa: BLE001  # unknown means keep the rows, never orphan a live message
            return set(message_ids)
        return {m.id for m in found if m is not None}

    async def linearize_callback(self, event: Any) -> None:
        """Carry out, or call off, a confirmed /linearize."""
        chat_id: int | None = event.chat_id
        try:
            _, anchor, digest, asker, action = event.data.decode().split(':')
            anchor_id, asker_id = int(anchor), int(asker)
        except ValueError:
            await event.answer()
            return
        if chat_id is None or not self.is_allowed(chat_id):
            await event.answer()
            return
        if event.sender_id != asker_id:
            await event.answer('Only whoever asked can confirm this', alert=True)
            return
        if action == 'no':
            await event.edit('🧹 Linearize cancelled.', buttons=None)
            await event.answer()
            return
        if (chat_id, anchor_id) in self.linearizing:
            await event.answer('Already deleting')
            return

        planned = self._linearize_plan(chat_id, anchor_id)
        if planned is None or plan_digest(planned[1]) != digest:
            await event.edit('🧹 That thread has changed since; run /linearize again.', buttons=None)
            await event.answer()
            return
        nodes, delete = planned
        if self._generating_in(chat_id, nodes, delete):
            await event.answer('Part of that thread is still generating; /abort it first', alert=True)
            return

        self.linearizing.add((chat_id, anchor_id))
        try:
            await event.edit('🧹 Deleting…', buttons=None)
            await event.answer()
            targets = sorted(n for n in delete if n > 0)
            for start in range(0, len(targets), 100):
                batch = targets[start : start + 100]
                try:
                    await self.delete_messages(chat_id, batch)
                except Exception:  # noqa: BLE001  # whatever failed is found by the check below
                    logger.exception(f'Linearize could not delete a batch ({chat_id=}, {anchor_id=})')
            # Telegram can decline part of a batch without saying which part, so the rows are
            # dropped only for messages confirmed gone; a survivor keeps its history and stays
            # replyable
            remaining: set[int] = set()
            for start in range(0, len(targets), 100):
                remaining |= await self._still_there(chat_id, targets[start : start + 100])
            gone = set(targets) - remaining
            # an imported prompt has no message of its own; it goes with the turn that showed it
            self.store.delete(chat_id, [mid for mid in delete if mid in gone or mid < 0 and -mid in gone])
            summary = f'🧹 Deleted {len(gone)} {"message" if len(gone) == 1 else "messages"}.'
            if remaining:
                summary += (
                    f'\n{len(remaining)} could not be deleted: a bot needs delete rights for other'
                    " people's messages in a group, and Telegram refuses some older messages."
                )
            await event.edit(summary, buttons=None)
            logger.info(f'Linearized {chat_id=}, {anchor_id=}, deleted={len(gone)}, kept={len(remaining)}')
        finally:
            self.linearizing.discard((chat_id, anchor_id))

    async def live_reply_callback(self, event: Any) -> None:
        """Regenerate or delete the live reply whose button was pressed."""
        chat_id: int | None = event.chat_id
        if chat_id is None or not self.is_allowed(chat_id):
            await event.answer()
            return
        target_id: int = event.message_id
        info = self.get_msg_info(chat_id, target_id)
        if info is None or not info.sent_by_bot or info.reply_id is None:
            await event.answer('That reply is no longer stored', alert=True)
            return
        key = (chat_id, info.reply_id)
        if key in self.generating:
            await event.answer('That reply is already being generated', alert=True)
            return
        if key in self.deleting_replies:
            await event.answer('That reply is being deleted', alert=True)
            return
        if event.data == b'live:re':
            logger.info(f'Regenerate pressed: {chat_id=}, {target_id=}')
            # answered once the generation has claimed the prompt, which it does before its
            # first await, so a second press of the button finds it claimed
            answered = asyncio.ensure_future(event.answer())
            await self._rewrite_reply(chat_id, info.reply_id, target_id, info, error_reply_to=target_id)
            await answered
        elif event.data == b'live:del':
            await self._delete_reply(event, chat_id, target_id, info)
        else:
            await event.answer()

    async def _delete_reply(self, event: Any, chat_id: int, target_id: int, info: MsgInfo) -> None:
        """Delete every message of the reply *target_id* belongs to, if no later turn follows it."""
        assert info.reply_id is not None
        # parts a shrunken regeneration left mapped count too: a turn may have replied to one
        msg_ids = self.store.turn_messages(chat_id, target_id)
        if self.store.has_replies(chat_id, msg_ids):
            await event.answer('A later turn follows this reply, so it cannot be deleted', alert=True)
            return
        key = (chat_id, info.reply_id)
        self.deleting_replies.add(key)
        try:
            shown = list(info.part_ids or [target_id])
            logger.info(f'Delete pressed: {chat_id=}, {target_id=}, {shown=}')
            try:
                await self.delete_messages(chat_id, shown)
            except Exception:  # noqa: BLE001  # whatever failed is found by the check below
                logger.exception(f'Could not delete a reply ({chat_id=}, {target_id=})')
            # as under /linearize, only rows of messages confirmed gone are dropped
            remaining = await self._still_there(chat_id, shown)
            drop = [mid for mid in msg_ids if mid not in remaining]
            # the prompt of an imported turn was shown by the reply's first message, and goes with it
            if info.reply_id < 0 and -info.reply_id in drop:
                drop.append(info.reply_id)
            self.store.delete(chat_id, drop)
        finally:
            self.deleting_replies.discard(key)
        if remaining:
            await event.answer(f'{len(remaining)} of the messages could not be deleted', alert=True)
        else:
            await event.answer()

    @only_allowed
    async def abort_handler(self, message: Any) -> None:
        """Ask whatever is generating in this chat to stop."""
        chat_id = message.chat_id
        msg_id = message.id
        running = [key for key in self.generating if key[0] == chat_id]
        logger.info(f'Abort requested: {chat_id=}, {msg_id=}, running={len(running)}')
        if not running:
            _ = await self.send_message(chat_id, '[!] Nothing is generating in this chat', msg_id)
            return
        for key in running:
            self.generating[key] = True
        # no confirmation: each reply marks itself `[!aborted]` as it stops, which is the receipt

    @only_allowed
    async def regenerate_handler(self, message: Any, body: str) -> None:
        """Answer a prompt again, writing over the reply that is there."""
        chat_id = message.chat_id
        msg_id = message.id
        logger.info(f'Regenerate requested: {chat_id=}, {msg_id=}, {body=}')

        target = await self._replied_bot_turn(message, REGENERATE_USAGE)
        if target is None:
            return
        target_id, target_info, prompt_id = target
        prompt_info = self.get_msg_info(chat_id, prompt_id)
        if prompt_info is None or prompt_info.sent_by_bot:
            _ = await self.send_message(chat_id, '[!] No stored prompt for that message', msg_id)
            return
        # wait_for above only covers a generation that has already claimed its messages; two
        # /regenerate arriving together would both get past it and the second would collide
        # on claiming them
        if (chat_id, prompt_id) in self.generating:
            _ = await self.send_message(chat_id, '[!] That reply is already being generated', msg_id)
            return

        # the argument parses exactly as a message prefix does, so the text left over after
        # the prefix and its overrides is the new prompt; failing a prefix, a first word of
        # the form <endpoint>/<model> names a model as /chat does. An argument that opens
        # with neither is all prompt, and keeps the model the turn was on
        model: Model | None = None
        model_spec: str | None = None
        overrides: dict[str, str | None] = {}
        new_prompt = body.strip()
        try:
            if new_prompt and (matched := self._match_model(body)) is not None:
                model, overrides, new_prompt = matched
            elif new_prompt:
                head, spec_overrides, rest = parse_chat_spec(new_prompt)
                endpoint, sep, _ = head.partition('/')
                if sep and endpoint in self.endpoint_by_name:
                    model, model_spec = self._chat_model(head, spec_overrides)
                    overrides, new_prompt = spec_overrides, rest
        except ValueError as e:
            _ = await self.send_message(chat_id, f'[!] {e}', msg_id)
            return

        # a named model and a new prompt both belong to the turn being answered, so they are
        # recorded on its prompt exactly as sending it afresh with that prefix would
        if model is not None or new_prompt:
            parts = list(prompt_info.message)
            if new_prompt:
                parts = replace_text_part(parts, new_prompt)
            info = prompt_info._replace(message=parts)
            if model is not None:
                info = info._replace(**self._model_fields(model, model_spec, overrides))
            self.set_msg_info(chat_id, prompt_id, info)

        logger.info(f'Regenerating {chat_id=}, {prompt_id=}, reprompted={bool(new_prompt)}')
        # an error that leaves no reply to write over answers the command: the prompt of an
        # imported turn is stored under a negative id, which is no message to reply to
        await self._rewrite_reply(chat_id, prompt_id, target_id, target_info, error_reply_to=msg_id)

    async def _rewrite_reply(
        self, chat_id: int, prompt_id: int, target_id: int, target_info: MsgInfo, error_reply_to: int
    ) -> None:
        """Answer *prompt_id* again over its reply, *target_id*, as the next edition of it."""
        # every part of the old reply is written over, and a shorter one deletes the rest
        adopt = list(target_info.part_ids or [target_id])
        edition = regeneration_edition(target_info) + 1
        logger.info(f'Rewriting the reply to {chat_id=}, {prompt_id=}, {adopt=}, {edition=}')
        await self._run_completion(chat_id, prompt_id, adopt=adopt, error_reply_to=error_reply_to, edition=edition)

    async def _stop_generation(self, chat_id: int, prompt_id: int) -> None:
        """Abort an answer to *prompt_id* in progress, and wait until it is over and stored."""
        key = (chat_id, prompt_id)
        # another generation may claim the key while this one waits, so it is asked again
        while key in self.generating:
            self.generating[key] = True
            _ = await self.generation_done[key].wait()

    async def edit_handler(self, message: Any) -> None:
        """Answer an edited prompt again, over its reply, if the thread it belongs to is live.

        The prompt is rebuilt from the edited message as sending it afresh would, its media
        read again and a prefix in it re-read, but it keeps its place in the thread. Whether
        the thread is live is judged before the edit, so an edit cannot switch it on for itself.
        An edit that changes nothing stored -- Telegram adding a link preview -- is left alone.
        """
        chat_id = message.chat_id
        edited_id = message.id
        if not self.is_allowed(chat_id) or message.sender_id == self.bot_id:
            return
        info = self.get_msg_info(chat_id, edited_id)
        if info is None or info.sent_by_bot:
            return
        # an album's caption may sit on any of its messages; the turn is answered under the first
        prompt_id = info.part_ids[0] if info.part_ids else edited_id
        try:
            _, thread_model, _ = self.construct_chat_history(chat_id, prompt_id)
        except RuntimeError:
            return
        if not thread_model.live:
            return

        messages = await self._album_of(message)
        text = next((m.message for m in messages if m.message), '')
        if is_foreign_command(text, self.username):
            return
        command = parse_command(text, self.username)
        chat = command is not None and command[0] == 'chat'
        if command is not None and not chat and command[0] in COMMANDS_WITH_BODY | COMMANDS_WITHOUT_BODY:
            return  # edited into a command, which is no prompt to answer
        try:
            model, spec, overrides, text = self._turn_model(command[1] if chat and command else text, chat)
            # media a thread-opening prompt took from the message it replied to comes along
            # again; a prompt continuing a thread took none
            replied_media: list[Any] = []
            if info.reply_id is None and replied_message_id(message) is not None:
                replied = await message.get_reply_message()
                if replied is not None and self._media_kind(replied) is not None:
                    replied_media = await self._album_of(replied)
            own_media = [m for m in messages if self._media_kind(m) is not None]
            parts = await self._prompt_parts(chat_id, text, own_media, replied_media, info.reply_id is not None)
        except ValueError as e:
            _ = await self.send_message(chat_id, f'[!] {e}', edited_id)
            return
        new_info = info._replace(message=parts)
        if model is not None:
            new_info = new_info._replace(**self._model_fields(model, spec, overrides))
        # the system prompt is left out of the comparison: it is rendered with the date
        if new_info._replace(system_prompt=None) == info._replace(system_prompt=None):
            return

        logger.info(f'Live edit: {chat_id=}, {prompt_id=}, {edited_id=}')
        # the latest edit wins: an answer still streaming for an earlier text is stopped, and
        # nothing awaits between the store write and the new generation claiming the prompt
        await self._stop_generation(chat_id, prompt_id)
        self.set_msg_info(chat_id, prompt_id, new_info)
        answers = self.store.answers_to(chat_id, prompt_id)
        target_info = self.get_msg_info(chat_id, answers[-1]) if answers else None
        if target_info is None:  # nothing answered it yet, as after a failed generation
            await self._run_completion(chat_id, prompt_id)
            return
        await self._rewrite_reply(chat_id, prompt_id, answers[-1], target_info, error_reply_to=prompt_id)

    @staticmethod
    def _media_kind(message: Any) -> str | None:
        """``'image'`` for a photo or an image sent as a file, ``'file'`` for any other file."""
        if message.photo is not None:
            return 'image'
        document = message.document
        if document is None:
            return None
        if (getattr(document, 'mime_type', None) or '').startswith('image/'):
            return 'image'
        return 'file'

    async def _album_of(self, message: Any) -> list[Any]:
        """The album *message* belongs to, in order, or the message alone."""
        if message.grouped_id is None:
            return [message]
        # Telegram has no call for an album, but one holds at most ten consecutive messages
        try:
            around: Any = await self.bot.get_messages(message.chat_id, ids=list(range(message.id - 9, message.id + 10)))
        except Exception:  # noqa: BLE001  # without its siblings the replied message is still usable
            logger.warning(f'Failed to fetch the album of {message.chat_id=}, {message.id=}')
            return [message]
        album = [m for m in around if m is not None and m.grouped_id == message.grouped_id]
        return album or [message]

    async def _read_media(self, chat_id: int, messages: list[Any]) -> tuple[list[str], list[str]]:
        """The images (cache keys) and text files (decoded) *messages* carry.

        Raises ValueError, with a message for the user, on a file that cannot be used.
        """
        images: list[str] = []
        files: list[str] = []
        for m in messages:
            kind = self._media_kind(m)
            if kind == 'image':
                if m.document is not None and m.document.size > self.IMAGE_FILE_SIZE_LIMIT:
                    raise ValueError('Image too large')
                blob = await m.download_media(bytes)
                images.append(save_photo(self.image_cache, blob, chat_id, m.id))
            elif kind == 'file':
                if m.document.size > self.TEXT_FILE_SIZE_LIMIT:
                    raise ValueError('File too large')
                text = decode_text_file(await m.download_media(bytes))
                if text is None:
                    raise ValueError('File is not text file or not valid UTF-8')
                files.append(text)
        return images, files

    @only_allowed
    async def chat_handler(self, message: Any, body: str) -> None:
        """Start a thread with the model *body* names, which need not be configured."""
        logger.info(f'New chat: {message.chat_id=}, {message.sender_id=}, {message.id=}, {body=}')
        if not body.strip():
            _ = await self.send_message(message.chat_id, CHAT_USAGE, message.id)
            return
        own_media = [message] if self._media_kind(message) is not None else []
        await self._user_turn(message, [message.id], body, own_media, chat=True)

    @only_allowed
    async def reply_handler(self, message: Any) -> None:
        logger.info(
            f'New message to reply: {message.chat_id=}, {message.sender_id=}, {message.id=}, {message.message=},'
            + f' {message.photo=}, {message.document=}'
        )
        if not message.message and message.photo is None and message.document is None:
            logger.debug(f'Unknown media types {message.chat_id=}, {message.id=}')
            return
        own_media = [message] if self._media_kind(message) is not None else []
        await self._user_turn(message, [message.id], message.message or '', own_media)

    async def _reply_context(self, first: Any, chat: bool) -> tuple[int | None, list[Any]] | None:
        """What *first* replies to: the turn it continues, and the messages whose media it takes.

        None for a reply the bot leaves alone. With *chat* nothing is continued, and a replied
        message only lends its media.
        """
        # a message sitting in a forum topic carries a reply header too, so what counts as a
        # reply is decided by replied_message_id rather than by the header being present
        replied_id = replied_message_id(first)
        if replied_id is None:
            return None, []
        chat_id = first.chat_id
        if chat:
            # a bot turn with media included, which is what /chat is for: replying normally
            # would continue that turn's thread
            replied = await first.get_reply_message()
            if replied is not None and self._media_kind(replied) is not None:
                return None, await self._album_of(replied)
            return None, []
        if first.reply_to.quote_text is not None:
            logger.debug(f'Reply contains quote text {chat_id=}, {first.id=}')
            return None
        replied = await first.get_reply_message()
        if replied is None:
            logger.debug(f'Replied message is unavailable {chat_id=}, {first.id=}')
            return None
        has_media = self._media_kind(replied) is not None
        # a bot message continues its thread when it is a stored turn, whatever it carries;
        # one with media and no turn behind it is a file the bot sent for keeping, such as an
        # /export, and is talked about like anyone else's. A turn still streaming has no
        # MsgInfo until it finishes, hence the wait before looking.
        if replied.sender_id == self.bot_id:
            await self.pending_reply_manager.wait_for((chat_id, replied_id))
            if self.get_msg_info(chat_id, replied_id) is not None or not has_media:
                return replied_id, []
        if has_media:
            return None, await self._album_of(replied)
        return None

    def _turn_model(self, text: str, chat: bool) -> tuple[Model | None, str | None, dict[str, str | None], str]:
        """The model a prompt names, the spec stored for it, its overrides, and the prompt left.

        With *chat*, *text* is a /chat body, which always names one. Otherwise a leading prefix
        does, or nothing does and the model is None. Raises ValueError, for the user.
        """
        if chat:
            head, overrides, text = parse_chat_spec(text)
            model, spec = self._chat_model(head, overrides)
            return model, spec, overrides, text
        matched = self._match_model(text)
        if matched is None:
            return None, None, {}, text
        model, overrides, text = matched
        return model, None, overrides, text

    def _model_fields(self, model: Model | None, spec: str | None, overrides: dict[str, str | None]) -> dict[str, Any]:
        """What a user turn records of the model it names, as MsgInfo fields.

        All None on a turn that names none, which is what lets construct_chat_history fall back
        to the choice already in the chain; a model named by its endpoint is stored as that spec
        instead of a prefix.
        """
        return {
            'prefix': model.prefix if model is not None and spec is None else None,
            'system_prompt': self._system_prompt_for(model),
            'overrides': overrides or None,
            'model_spec': spec,
        }

    async def _prompt_parts(
        self, chat_id: int, text: str, own_media: list[Any], replied_media: list[Any], continuing: bool
    ) -> list[MsgPartInHistory]:
        """A prompt's stored parts: text files inlined ahead of the text, then the images.

        Raises ValueError, with a message for the user, on a file that cannot be used.
        """
        replied_images, replied_files = await self._read_media(chat_id, replied_media)
        own_images, own_files = await self._read_media(chat_id, own_media)
        images = replied_images + own_images
        if images and not text:
            text = self.default_image_reply_prompt if continuing else self.default_image_prompt
        # only the replied message's media is taken, never its caption: the prompt is what
        # the user wrote in this turn
        prompt = '\n\n'.join(t for t in (*replied_files, *own_files, text) if t)
        return [make_text_part(prompt), *(make_image_part(h) for h in images)]

    async def _user_turn(
        self, first: Any, msg_ids: list[int], text: str, own_media: list[Any], chat: bool = False
    ) -> None:
        """Record a user's turn and answer it: one message, or the messages of an album.

        *first* is the message whose reply header counts, *own_media* the messages whose photos
        and files belong to the turn. Replying to a message that carries media, rather than to
        a bot reply, brings that media in as well -- a forwarded image, a file sent on its own
        beforehand, or a file the bot sent outside any thread -- but not its caption, and
        opens a thread. A bot message that is a stored turn is continued, media or not.

        With *chat*, *text* is a /chat body, which names the model itself, and the turn opens
        a thread whatever it replies to; a replied message only lends its media.
        """
        chat_id = first.chat_id
        msg_id = msg_ids[0]
        context = await self._reply_context(first, chat)
        if context is None:
            return
        reply_to_id, replied_media = context

        # a message that opens a thread has no model unless its prefix names one, so a
        # missing prefix is an error there; on a reply the prefix is optional and switches
        # the model for this turn and the ones after it
        try:
            model, model_spec, overrides, text = self._turn_model(text, chat)
        except ValueError as e:
            _ = await self.send_message(chat_id, f'[!] {e}', msg_id)
            return
        # with no prefix, a reply carries over the model and prompt of the message replied to
        if model is None and reply_to_id is None:
            if chat_id == first.sender_id:  # in private chat, point at /models and /help
                _ = await self.send_message(chat_id, self._no_prefix_text(), msg_id)
            return

        try:
            new_message = await self._prompt_parts(chat_id, text, own_media, replied_media, reply_to_id is not None)
        except ValueError as e:
            _ = await self.send_message(chat_id, str(e), msg_id)
            return
        msg_info = MsgInfo(
            sent_by_bot=False,
            message=new_message,
            reply_id=reply_to_id,
            **self._model_fields(model, model_spec, overrides),
        )
        # an album is one turn, so a reply to any of its messages continues it
        self.store.put_turn(chat_id, msg_ids, msg_info)

        await self._run_completion(chat_id, msg_id)

    async def _run_completion(
        self,
        chat_id: int,
        msg_id: int,
        adopt: list[int] | None = None,
        error_reply_to: int | None = None,
        edition: int = 1,
    ) -> None:
        """Answer *msg_id*, recorded as in flight so /abort can reach it.

        *error_reply_to* is where a failure before any reply exists is reported; *msg_id* by
        default. *edition* is the version of the answer a regeneration writes, 1 otherwise.
        """
        key = (chat_id, msg_id)
        self.generating[key] = False
        done = self.generation_done[key] = asyncio.Event()
        try:
            await self._generate(chat_id, msg_id, adopt, error_reply_to or msg_id, edition)
        finally:
            _ = self.generating.pop(key, None)
            _ = self.generation_done.pop(key, None)
            done.set()

    async def _fetch_image(self, chat_id: int, msg_id: int) -> bytes:
        """Download an image again from its message, for one evicted from the cache."""
        msg = await self.bot.get_messages(chat_id, ids=msg_id)
        # ids=int returns a single Message at runtime, or None if it no longer exists
        if not isinstance(msg, Message):
            # a vanished message is a lookup failure, not a type error
            raise ValueError(f'cannot fetch message {msg_id} in chat {chat_id}')  # noqa: TRY004
        blob = await msg.download_media(bytes)
        if not isinstance(blob, bytes):
            # absent media is a lookup failure, not a type error
            raise ValueError(f'message {msg_id} in chat {chat_id} has no downloadable media')  # noqa: TRY004
        return blob

    async def _load_image(self, key: str) -> bytes | None:
        return await load_photo(self.image_cache, key, self._fetch_image)

    @staticmethod
    def _add_timing(usage: dict[str, Any], start: float, first_token: float | None, end: float) -> None:
        """Add time to first token and tokens per second to a reply's usage, if it has any."""
        if not usage:
            return
        usage['TTFT'] = f'{(first_token or end) - start:.1f}s'
        # reasoning tokens are a breakdown of the output total, not an addition to it
        out_tokens = usage.get('out') or 0
        if out_tokens and first_token and end > first_token:
            usage['TPS'] = f'{out_tokens / (end - first_token):.1f}'

    def _store_reply(
        self,
        chat_id: int,
        prompt_id: int,
        model: Model,
        part_ids: list[int],
        reply: str,
        carry: list[dict[str, Any]],
        edition: int,
    ) -> None:
        """Record a finished reply as the turn its messages show, with what it carries forward.

        A regeneration keeps the first part of the reply it replaces, so the same turn is
        rewritten; parts it no longer needs stay mapped to it, so a later turn that replied to
        one of them goes on from the new answer.
        """
        # reasoning first: replaying it requires the order the provider emitted
        parts: list[MsgPartInHistory] = []
        for item in carry:
            chain_of_thought = reasoning_text_of(item)
            if chain_of_thought:
                # A chain of thought we can read is carried as text rather than as a reasoning
                # item. DeepSeek returns the text beside a handle into a response `store=false`
                # never kept, and drops the item on replay whichever way it is sent -- observed
                # by the reasoning tokens never reaching the next turn's input, and by the model
                # reinventing numbers it had picked. As text it goes back as a plain reasoning
                # item under dt, and is dropped otherwise.
                parts.append(make_thinking_text_part(chain_of_thought))
            else:
                parts.append(make_reasoning_part(model.name, item))
        parts.append(make_text_part(reply))
        info = MsgInfo(
            sent_by_bot=True,
            message=parts,
            reply_id=prompt_id,
            prefix=None,
            system_prompt=None,
            extra_metadata={REGENERATION_EDITION: edition} if edition > 1 else None,
        )
        self.store.put_turn(chat_id, part_ids, info)

    async def _generate(
        self, chat_id: int, msg_id: int, adopt: list[int] | None, error_reply_to: int, edition: int
    ) -> None:
        try:
            chat_history, model, system_prompt = self.construct_chat_history(chat_id, msg_id)
        except RuntimeError as e:
            logger.exception(e)
            _ = await self.send_message(chat_id, f'[!] Error on resolving conversation: {e}', error_reply_to)
            return

        # an imported turn is shown by one message holding its prompt above the answer, the
        # prompt having no message of its own (it is stored under a negative id, which only an
        # import writes); an answer written over that message keeps the prompt on top
        lead = format_imported_prompt(text_of(chat_history[-1])) if msg_id < 0 else ''

        error_cnt = 0
        while True:
            thinking = ''
            reply = ''
            thinking_done = False
            stream_meta = StreamMeta()
            prefix = self._reply_prefix(model, edition=edition)
            # a live reply shows its buttons from the first message on; a press while it streams
            # is refused, and one on a reply that failed finds no new turn to act on
            markup = LIVE_REPLY_MARKUP if model.live else None
            async with BotReplyMessages(
                self, chat_id, msg_id, prefix, self.get_renderer(model), adopt, markup
            ) as replymsgs:
                try:
                    status: str | None = 'Generating...'
                    await replymsgs.update(lead + f'[{status}]')
                    client = self.get_client(model)
                    stream = completion(client, chat_history, model, system_prompt, chat_id, msg_id, self._load_image)
                    stream_start = time.time()
                    first_token_time: float | None = None
                    aborted = False
                    async for event in stream:
                        if self.generating.get((chat_id, msg_id)):
                            # closing the generator unwinds the backend and drops the HTTP
                            # stream, rather than leaving it to be collected later
                            logger.info(f'Aborting generation ({chat_id=}, {msg_id=})')
                            await stream.aclose()
                            aborted = True
                            break
                        if isinstance(event, ThinkingDelta):
                            first_token_time = first_token_time or time.time()
                            thinking += event.text
                        elif isinstance(event, ResponseDelta):
                            first_token_time = first_token_time or time.time()
                            thinking_done = True
                            reply += event.text
                        elif isinstance(event, StatusChange):
                            status = event.status
                        elif isinstance(event, StreamMeta):
                            stream_meta = event
                            continue
                        await replymsgs.update(
                            lead + format_reply(thinking, reply, thinking_done, status), bool(thinking)
                        )
                    usage = stream_meta.usage
                    self._add_timing(usage, stream_start, first_token_time, time.time())
                    final = format_reply(
                        thinking,
                        reply,
                        True,
                        status='aborted' if aborted else None,
                        tool_calls=stream_meta.tool_calls or None,
                        usage=usage or None,
                    )
                    await replymsgs.update(lead + final, bool(thinking))
                    await replymsgs.finalize()
                    part_ids = [bot_msg_id for bot_msg_id, _ in replymsgs.replied_msgs]
                    self._store_reply(chat_id, msg_id, model, part_ids, reply, stream_meta.carry_items, edition)
                    return

                # handling completion errors
                except Exception as e:  # noqa: BLE001  # the retry loop handles every completion failure
                    error_cnt += 1
                    logger.exception(f'Error on generating exception({chat_id=}, {msg_id=}, {error_cnt=})')
                    delay = retry_delay(e, self.OPENAI_RETRY_INTERVAL)
                    if error_cnt > self.OPENAI_MAX_RETRY:
                        delay = None
                    error_msg = f'[!] Error: {traceback.format_exception_only(e)[-1].strip()}'
                    if delay is not None:
                        error_msg += f'\nRetrying in {delay:.0f}s ({error_cnt}/{self.OPENAI_MAX_RETRY})...'
                    if reply:
                        error_msg = reply + '\n\n' + error_msg
                    await replymsgs.update(lead + error_msg)
                    if delay is None:
                        break
                    await asyncio.sleep(delay)

    async def album_handler(self, event: events.Album.Event) -> None:
        chat_id: int = event.chat_id  # pyright: ignore[reportAssignmentType]  # checked for None in process_album
        first = event.messages[0]

        # Inline allowed check (Album.Event lacks .id, so @only_allowed cannot be used)
        if not self.is_allowed(chat_id):
            if chat_id == event.sender_id:
                _ = await self.send_message(chat_id, 'This chat is not allowed', first.id)
            return

        text = next((m.message for m in event.messages if m.message), '')
        logger.info(
            f'New album to reply: {chat_id=}, {event.sender_id=}, {first.id=}, {text=},'
            + f' num_messages={len(event.messages)}'
        )
        own_media = [m for m in event.messages if self._media_kind(m) is not None]
        if not own_media:
            logger.debug(f'Album has no photos or files {chat_id=}, {first.id=}')
            return
        # every message of the album is stored, so a reply to any of them continues the thread
        msg_ids = [m.id for m in event.messages]
        command = parse_command(text, self.username)
        if command is not None and command[0] == 'chat':
            await self._user_turn(first, msg_ids, command[1], own_media, chat=True)
        else:
            await self._user_turn(first, msg_ids, text, own_media)

    async def command_argument_error(self, message: Any, name: str) -> None:
        """Refuse an argument given to a command that takes none, such as ``/ping now``."""
        # /ping answers in any chat, so its misuse is reported in any chat too; the other
        # commands are allowlisted, and stay as silent elsewhere as they would when used right
        if name != 'ping' and not self.is_allowed(message.chat_id):
            return
        _ = await self.send_message(message.chat_id, f'[!] `/{name}` takes no argument', message.id)

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
