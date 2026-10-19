import asyncio
import atexit
import datetime
import os
import shelve
import time
import tomllib
import traceback
from collections.abc import Sequence
from html import escape as html_escape
from types import TracebackType
from typing import Any, Self
from zoneinfo import ZoneInfo

import anthropic
import diskcache
import openai
from loguru import logger
from telethon import Button, TelegramClient, errors, events, functions, types
from telethon.tl.custom import Message

from chatgpt_telegram_bot.completion import completion, reasoning_text_of
from chatgpt_telegram_bot.models import (
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
)
from chatgpt_telegram_bot.render import (
    ENTITY_RENDERER,
    PLAIN_LENGTH_LIMIT,
    SERVER_MARKDOWN_RENDERER,
    Rendered,
    Renderer,
)
from chatgpt_telegram_bot.reply import (
    format_imported_turn,
    format_linearize_listing,
    format_navigation_page,
    format_reply,
)
from chatgpt_telegram_bot.utils import (
    COMMANDS_WITHOUT_BODY,
    PendingReplyManager,
    Throttle,
    apply_overrides,
    decode_text_file,
    export_filename,
    format_export,
    is_foreign_command,
    linearize_plan,
    load_photo,
    match_prefix,
    message_link,
    pack_html_sections,
    parse_command,
    parse_import,
    parse_manipulation,
    parse_proxy,
    plan_digest,
    preview_line,
    replace_text_part,
    replied_message_id,
    retry,
    save_photo,
)

# Formatting guidance depends on which renderer the reply goes through, so it is appended to
# the default system prompt rather than baked into it. A custom prompt (`,[…]`) is left exactly
# as the user wrote it.
RICHTEXT_PROMPT_APPEND = (
    'Format the reply in Markdown; headings, tables, horizontal rules and LaTeX math all render.'
    + ' Delimit math with $...$ inline and $$...$$ on its own line.'
)
ENTITY_PROMPT_APPEND = (
    'Format the reply in Markdown, but do not use tables or LaTeX math: they do not render and'
    + ' would reach the user as raw source. Bold, italic, strikethrough, inline code, code blocks,'
    + ' links, blockquotes and lists all render.'
)


# Commands offered in the palette: only those that are safe to send bare, since picking one
# sends it at once. /export qualifies, its title being optional and the export harmless; a bare
# /regenerate would write over a reply before the user could add the argument they meant, and
# /manipulate or /import could only fail. Those are typed, and /help describes them.
BOT_COMMANDS = [
    ('ping', 'Test bot connectivity'),
    ('help', 'Show available models and commands'),
    ('abort', 'Stop whatever is generating in this chat'),
    ('navigate', 'List the turns of the replied-to thread'),
    ('linearize', 'Delete every branch except the line to the replied message'),
    ('export', 'Save the replied-to thread as JSON'),
]

LINEARIZE_USAGE = """[!] `/linearize` prunes a thread, so it has to reply to a message in one.

It deletes every branch that grew from the same first message, keeping only the line down to the replied message — so anything after that message goes too. You confirm the list first."""

EXPORT_USAGE = """[!] `/export` saves a thread as JSON, so it has to reply to a message in one.

`/export` — titled after the first prompt
`/export <title>` — with that title

The file is in the format `/import` reads. Reply to a bot message to include its answer."""

NAVIGATE_USAGE = """[!] `/navigate` lists the turns of a thread, so it has to reply to a message in one.

Reply to a bot message to include its answer; the listing opens on the last page."""

REGENERATE_USAGE = """[!] `/regenerate` answers a prompt again in place, so it has to reply to a bot message:

`/regenerate` — the same prompt and model
`/regenerate <prefix>` — the same prompt, another model
`/regenerate <new prompt>` — another prompt, the same model
`/regenerate <prefix> <new prompt>` — both

`/re` is short for `/regenerate`.

A first word that is a model prefix is read as one, so a new prompt that happens to start with a prefix needs the model named in front of it."""

IMPORT_USAGE = """[!] `/import` needs an exported conversation as a JSON file, either attached to the command or replied to.

`/import` — record no model; the first reply names one with a prefix
`/import <prefix>` — continue it with that model, overrides allowed

The file holds a `messages` array of `{role, content}` objects, alternating user and assistant."""

MANIPULATE_USAGE = """[!] `/manipulate` rewrites a bot reply, so it has to reply to one:

`/manipulate <text>`

or, to supply a chain of thought too:

`/manipulate <think>your cot</think> <text>`

`/mani` is short for `/manipulate`."""


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
        # (chat_id, anchor) of a /linearize being carried out, so a second press cannot run it twice
        self.linearizing: set[tuple[int, int]] = set()

        # db scheme:
        # msg_info_{chat_id}_{msg_id}: MsgInfo
        # the shelf lives for the whole process and is closed via atexit below
        self.db: shelve.Shelf[Any] = shelve.open(os.path.join(data_dir, 'db'))  # noqa: SIM115
        image_cache_size = _config.get('image_cache_size', 50 * 1024 * 1024)
        if not isinstance(image_cache_size, int):
            raise TypeError(f'image_cache_size must be an integer, got {type(image_cache_size).__name__}')
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

        @self.bot.on(events.CallbackQuery(pattern=rb'^lin:'))
        async def _linearize_confirm(event: events.CallbackQuery.Event) -> None:  # pyright: ignore[reportUnusedFunction]  # registered by @bot.on decorator
            await self.linearize_callback(event)

        @self.bot.on(events.CallbackQuery(pattern=rb'^nav:'))
        async def _navigate_page(event: events.CallbackQuery.Event) -> None:  # pyright: ignore[reportUnusedFunction]  # registered by @bot.on decorator
            await self.navigate_callback(event)

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
                commands=[types.BotCommand(command, description) for command, description in BOT_COMMANDS],
            )
        )

        _ = await self.bot(
            functions.bots.SetBotCommandsRequest(
                scope=types.BotCommandScopeDefault(),
                lang_code='',
                commands=[types.BotCommand(command, description) for command, description in BOT_COMMANDS],
            )
        )
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

    def _reply_prefix(self, model: Model, manipulated: bool = False) -> str:
        prefix = f'🤖 `{model.name}`'
        flags = self._model_flags(model)
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

            # the walk runs newest to oldest, so the first prefix found is the most recent
            # one the user named; an earlier prefix only applies until a reply replaces it
            if model_of_history is None and msg_info.prefix:
                for model in self.models:
                    if model.prefix == msg_info.prefix:
                        overrides = getattr(msg_info, 'overrides', None) or {}
                        model_of_history = apply_overrides(model, overrides)

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

    def _build_help_sections(self, error: str | None = None) -> list[str]:
        """The help page as HTML sections, which `help_handler` packs into as few messages as fit."""
        models: list[str] = []
        if error:
            models.append(f'⚠️ {error}\n')
        models.append('<b>Models</b>')
        for m in self.models:
            flags = ', '.join(self._model_flags(m))
            suffix = f' ({flags})' if flags else ''
            models.append(f'  <code>{m.prefix}</code> → <code>{m.name}</code>{suffix}')

        usage = [
            '<b>Usage</b>',
            '  <code>&lt;prefix&gt; &lt;message&gt;</code>  — start a conversation',
            '  <code>&lt;prefix&gt;,t &lt;message&gt;</code>  — enable thinking',
            '  <code>&lt;prefix&gt;,t=high &lt;message&gt;</code>  — set thinking effort',
            '  <code>&lt;prefix&gt;,t= &lt;message&gt;</code>  — disable thinking',
            '  <code>&lt;prefix&gt;,s &lt;message&gt;</code>  — enable search',
            '  <code>&lt;prefix&gt;,r &lt;message&gt;</code>  — let Telegram render the markdown',
            '  <code>&lt;prefix&gt;,r= &lt;message&gt;</code>  — render to entities instead',
            '  <code>&lt;prefix&gt;,[custom prompt] &lt;message&gt;</code>  — custom system prompt',
            '  <code>&lt;prefix&gt;,+[extra prompt] &lt;message&gt;</code>  — append to default prompt',
            '  <code>&lt;prefix&gt;,[] &lt;message&gt;</code>  — clear system prompt',
            'Reply to a bot message to continue; reply to an older one to fork.',
            'Reply to a photo or file with a prefix to talk about it.',
            'A prefix on a reply switches the model from there on.',
        ]

        commands = [
            '<b>Commands</b>',
            '  <code>/help</code>  — this page',
            '  <code>/ping</code>  — chat id, and whether the bot answers here',
            '  <code>/abort</code>  — stop whatever is generating in this chat',
            'Sent as a reply to a message in a conversation:',
            '  <code>/navigate</code>  — list its turns, one line each, paged',
            '  <code>/linearize</code>  — delete every other branch, after confirming',
            '  <code>/export [title]</code>  — save it as JSON that <code>/import</code> reads',
            'Typed, not in the command menu, as they act on their arguments:',
            '  <code>/regenerate [prefix] [new prompt]</code>, or <code>/re</code>',
            '    Reply to a bot message to answer its prompt again, over the old reply.'
            + ' A prefix switches the model and new text replaces the prompt;'
            + ' a first word that is a prefix is read as one.',
            '  <code>/manipulate &lt;text&gt;</code>, or <code>/mani</code>',
            '    Reply to a bot message to rewrite what it said; later turns build on yours.'
            + ' Lead with <code>&lt;think&gt;…&lt;/think&gt;</code> to supply its reasoning.',
            '  <code>/import [prefix]</code>',
            '    Attach an exported JSON file, or reply to one, to replay it as messages to reply to.'
            + ' Without a prefix, the first reply chooses the model.',
        ]

        prompt = [
            f'<b>Default system prompt</b>\n<code>{html_escape(self.system_prompt)}</code>',
            '',
            'Formatting guidance is appended to it, depending on <code>richtext</code>:',
            f'  on → <code>{html_escape(RICHTEXT_PROMPT_APPEND)}</code>',
            f'  off → <code>{html_escape(ENTITY_PROMPT_APPEND)}</code>',
        ]
        return ['\n'.join(section) for section in (models, usage, commands, prompt)]

    @only_allowed
    async def help_handler(self, message: Any, error: str | None = None) -> None:
        reply_to = message.id
        for text in pack_html_sections(self._build_help_sections(error), PLAIN_LENGTH_LIMIT):
            reply_to = await self.send_message_html(message.chat_id, text, reply_to)

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

    @only_allowed
    async def manipulate_handler(self, message: Any, body: str) -> None:
        """Replace a stored bot reply with user-supplied text (and optionally a chain of thought)."""
        chat_id = message.chat_id
        msg_id = message.id
        logger.info(f'Manipulate requested: {chat_id=}, {msg_id=}, {body=}')

        target_id = replied_message_id(message)
        if target_id is None:
            _ = await self.send_message(chat_id, MANIPULATE_USAGE, msg_id)
            return
        reply_to_message = await message.get_reply_message()
        if reply_to_message is None or reply_to_message.sender_id != self.bot_id:
            _ = await self.send_message(chat_id, MANIPULATE_USAGE, msg_id)
            return
        # the reply may still be streaming, and its MsgInfo is only written once it finishes
        await self.pending_reply_manager.wait_for((chat_id, target_id))

        target_info = self.get_msg_info(chat_id, target_id)
        if target_info is None or not target_info.sent_by_bot or target_info.reply_id is None:
            _ = await self.send_message(chat_id, '[!] No stored conversation for that message', msg_id)
            return

        thinking, text = parse_manipulation(body)
        if not text:
            _ = await self.send_message(chat_id, MANIPULATE_USAGE, msg_id)
            return

        # the model may have been switched part-way down the chain, so resolve it there
        try:
            _, model, _ = self.construct_chat_history(chat_id, target_info.reply_id)
        except RuntimeError as e:
            logger.exception(e)
            _ = await self.send_message(chat_id, f'[!] Error on resolving conversation: {e}', msg_id)
            return

        # only the message replied to is rewritten, so a body that no longer fits is refused
        bodies = self.get_renderer(model).render(
            format_reply(thinking or '', text, True),
            self._reply_prefix(model, manipulated=True),
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
            MsgInfo(
                sent_by_bot=True,
                message=new_parts,
                reply_id=target_info.reply_id,
                prefix=target_info.prefix,
                system_prompt=target_info.system_prompt,
                overrides=target_info.overrides,
                part_ids=getattr(target_info, 'part_ids', None),
            ),
        )
        await self.edit_message(chat_id, bodies[0], target_id)
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
        for m in self.models:
            match = match_prefix(spec, m.prefix)
            if match is not None:
                _, overrides = match
                return apply_overrides(m, overrides), overrides
        raise ValueError(f'Unknown prefix: `{spec}`')

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
        for i, (prompt, response) in enumerate(chat.turns, start=1):
            bodies = renderer.render(
                format_imported_turn(prompt, response), self._import_prefix(model, i, len(chat.turns))
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
            response_info = MsgInfo(
                sent_by_bot=True,
                message=[make_text_part(response)],
                reply_id=prompt_id,
                prefix=None,
                system_prompt=None,
                part_ids=list(msg_ids),
            )
            # a turn too long for one message is stored under every part, so replying to any
            # of them continues from the same point
            for mid in msg_ids:
                self.set_msg_info(chat_id, mid, response_info)
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
            text = next((p.text for p in info.message if p.type_ == 'text' and p.text), '')
            if not info.sent_by_bot:
                if any(p.type_ == 'image' for p in info.message):
                    text = ('🖼 ' + text).strip()
                turns.append(ThreadTurn(prompt=text, response=None, msg_id=mid))
            elif turns and turns[-1].response is None:
                # jump to the start of a split answer rather than whichever part the chain names
                part_ids = getattr(info, 'part_ids', None)
                turns[-1] = turns[-1]._replace(response=text, msg_id=part_ids[0] if part_ids else mid)
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
        anchor_id = replied_message_id(message)
        if anchor_id is None:
            _ = await self.send_message(chat_id, NAVIGATE_USAGE, msg_id)
            return
        # a reply still streaming has no MsgInfo yet
        await self.pending_reply_manager.wait_for((chat_id, anchor_id))
        rendered = self._navigation(chat_id, anchor_id, None)
        if rendered is None:
            _ = await self.send_message(chat_id, '[!] No stored conversation for that message', msg_id)
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
        """A node's text as an export carries it: the answer only, and a marker for an image."""
        text = next((p.text for p in info.message if p.type_ == 'text' and p.text), '')
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
        anchor_id = replied_message_id(message)
        if anchor_id is None:
            _ = await self.send_message(chat_id, EXPORT_USAGE, msg_id)
            return
        # a reply still streaming has no MsgInfo yet
        await self.pending_reply_manager.wait_for((chat_id, anchor_id))
        chain = self._thread_chain(chat_id, anchor_id)
        if chain is None:
            _ = await self.send_message(chat_id, '[!] No stored conversation for that message', msg_id)
            return
        first = chain[0][1]
        if first.reply_id is not None or first.sent_by_bot:
            # a partial thread would start mid-conversation, which /import refuses
            _ = await self.send_message(chat_id, '[!] Part of that thread is no longer stored', msg_id)
            return

        # an imported prompt lives under a negative id and was never a message, so it has no date
        dates = await self._message_dates(chat_id, [mid for mid, _ in chain if mid > 0])
        messages = [
            ('assistant' if info.sent_by_bot else 'user', self._export_content(info), dates.get(mid))
            for mid, info in chain
        ]
        title = body.strip() or preview_line(messages[0][1], 40) or 'Untitled'
        document = format_export(title, messages, datetime.datetime.now(datetime.UTC)).encode()

        caption = f'Exported "{title}" — {len(messages)} messages'
        _ = await self.send_document(chat_id, document, export_filename(title), caption, msg_id)
        logger.info(f'Exported {chat_id=}, {anchor_id=}, messages={len(messages)}, bytes={len(document)}')

    def _chat_nodes(self, chat_id: int) -> dict[int, MsgInfo]:
        """Every stored node of one chat, by id -- a scan, since MsgInfo links only backwards."""
        prefix = f'msg_info_{chat_id}_'
        nodes: dict[int, MsgInfo] = {}
        for key in self.db:
            if key.startswith(prefix):
                try:
                    nodes[int(key[len(prefix) :])] = self.db[key]
                except ValueError:
                    continue
        return nodes

    def _linearize_plan(self, chat_id: int, anchor_id: int) -> tuple[dict[int, MsgInfo], set[int]] | None:
        """The chat's nodes and the ids a /linearize at *anchor_id* would delete; None if not stored."""
        nodes = self._chat_nodes(chat_id)
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
        anchor_id = replied_message_id(message)
        if anchor_id is None:
            _ = await self.send_message(chat_id, LINEARIZE_USAGE, msg_id)
            return
        await self.pending_reply_manager.wait_for((chat_id, anchor_id))
        planned = self._linearize_plan(chat_id, anchor_id)
        if planned is None:
            _ = await self.send_message(chat_id, '[!] No stored conversation for that message', msg_id)
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
            for mid in gone:
                _ = self.db.pop(f'msg_info_{chat_id}_{mid}', None)
            # an imported prompt has no message of its own; it goes with the turn that showed it
            for mid in delete:
                if mid < 0 and -mid in gone:
                    _ = self.db.pop(f'msg_info_{chat_id}_{mid}', None)
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

        target_id = replied_message_id(message)
        if target_id is None:
            _ = await self.send_message(chat_id, REGENERATE_USAGE, msg_id)
            return
        reply_to_message = await message.get_reply_message()
        if reply_to_message is None or reply_to_message.sender_id != self.bot_id:
            _ = await self.send_message(chat_id, REGENERATE_USAGE, msg_id)
            return
        # the reply may still be streaming, and its MsgInfo is only written once it finishes
        await self.pending_reply_manager.wait_for((chat_id, target_id))

        target_info = self.get_msg_info(chat_id, target_id)
        if target_info is None or not target_info.sent_by_bot or target_info.reply_id is None:
            _ = await self.send_message(chat_id, '[!] No stored conversation for that message', msg_id)
            return
        prompt_id = target_info.reply_id
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
        # the prefix and its overrides is the new prompt; an argument that opens with no
        # prefix is all prompt, and keeps the model the turn was on
        model: Model | None = None
        overrides: dict[str, str | None] = {}
        new_prompt = body.strip()
        if new_prompt:
            for m in self.models:
                match = match_prefix(body, m.prefix)
                if match is not None:
                    new_prompt, overrides = match
                    try:
                        model = apply_overrides(m, overrides)
                    except ValueError as e:
                        _ = await self.send_message(chat_id, f'[!] {e}', msg_id)
                        return
                    break

        # a named model and a new prompt both belong to the turn being answered, so they are
        # recorded on its prompt exactly as sending it afresh with that prefix would
        if model is not None or new_prompt:
            parts = list(prompt_info.message)
            if new_prompt:
                parts = replace_text_part(parts, new_prompt)
            self.set_msg_info(
                chat_id,
                prompt_id,
                MsgInfo(
                    sent_by_bot=False,
                    message=parts,
                    reply_id=prompt_info.reply_id,
                    prefix=model.prefix if model is not None else prompt_info.prefix,
                    system_prompt=self._system_prompt_for(model) if model is not None else prompt_info.system_prompt,
                    overrides=(overrides or None) if model is not None else prompt_info.overrides,
                ),
            )

        # every part of the old reply is written over, and a shorter one deletes the rest
        adopt = list(getattr(target_info, 'part_ids', None) or [target_id])
        logger.info(f'Regenerating {chat_id=}, {prompt_id=}, {adopt=}, reprompted={bool(new_prompt)}')
        await self._run_completion(chat_id, prompt_id, adopt=adopt)

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

    async def _user_turn(self, first: Any, msg_ids: list[int], text: str, own_media: list[Any]) -> None:
        """Record a user's turn and answer it: one message, or the messages of an album.

        *first* is the message whose reply header counts, *own_media* the messages whose photos
        and files belong to the turn. Replying to a message that carries media, rather than to
        a bot reply, brings that media in as well -- a forwarded image, a file sent on its own
        beforehand, or a file the bot sent outside any thread -- but not its caption, and
        opens a thread. A bot message that is a stored turn is continued, media or not.
        """
        chat_id = first.chat_id
        sender_id = first.sender_id
        msg_id = msg_ids[0]
        reply_to_id: int | None = None
        model_by_prefix: Model | None = None
        replied_media: list[Any] = []

        # a message sitting in a forum topic carries a reply header too, so what counts as a
        # reply is decided by replied_message_id rather than by the header being present
        replied_id = replied_message_id(first)
        if replied_id is not None:
            if first.reply_to.quote_text is not None:
                logger.debug(f'Reply contains quote text {chat_id=}, {msg_id=}')
                return
            reply_to_message = await first.get_reply_message()
            if reply_to_message is None:
                logger.debug(f'Replied message is unavailable {chat_id=}, {msg_id=}')
                return
            # a bot message continues its thread when it is a stored turn, whatever it carries;
            # one with media and no turn behind it is a file the bot sent for keeping, such as
            # an /export, and is talked about like anyone else's. A turn still streaming has
            # no MsgInfo until it finishes, hence the wait before looking.
            continues_thread = False
            if reply_to_message.sender_id == self.bot_id:
                await self.pending_reply_manager.wait_for((chat_id, replied_id))
                stored = self.get_msg_info(chat_id, replied_id)
                continues_thread = stored is not None or self._media_kind(reply_to_message) is None
            if continues_thread:  # user reply to a bot message
                reply_to_id = replied_id
            elif self._media_kind(reply_to_message) is not None:
                replied_media = await self._album_of(reply_to_message)
            else:
                return

        # a message that opens a thread has no model unless its prefix names one, so a
        # missing prefix is an error there; on a reply the prefix is optional and switches
        # the model for this turn and the ones after it
        overrides: dict[str, str | None] = {}
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
            if reply_to_id is None:
                if chat_id == sender_id:  # in private chat, show help with error
                    await self.help_handler(first, error='Unknown prefix. Use one of the prefixes below.')
                return
            # continuing a thread: model and prompt carry over from the message replied to

        try:
            replied_images, replied_files = await self._read_media(chat_id, replied_media)
            own_images, own_files = await self._read_media(chat_id, own_media)
        except ValueError as e:
            _ = await self.send_message(chat_id, str(e), msg_id)
            return
        images = replied_images + own_images

        if images and not text:
            text = self.default_image_reply_prompt if reply_to_id is not None else self.default_image_prompt
        # only the replied message's media is taken, never its caption: the prompt is what
        # the user wrote in this turn
        prompt = '\n\n'.join(t for t in (*replied_files, *own_files, text) if t)
        new_message: list[MsgPartInHistory] = [make_text_part(prompt), *(make_image_part(h) for h in images)]

        system_prompt = self._system_prompt_for(model_by_prefix)

        # prefix and system_prompt stay None on a reply that named no model, which is what
        # lets construct_chat_history fall back to the ones already in the chain
        msg_info = MsgInfo(
            sent_by_bot=False,
            message=new_message,
            reply_id=reply_to_id,
            prefix=model_by_prefix and model_by_prefix.prefix,
            system_prompt=system_prompt,
            overrides=overrides or None,
        )
        for mid in msg_ids:
            self.set_msg_info(chat_id, mid, msg_info)

        await self._run_completion(chat_id, msg_id)

    async def _run_completion(self, chat_id: int, msg_id: int, adopt: list[int] | None = None) -> None:
        """Answer *msg_id*, recorded as in flight so /abort can reach it."""
        self.generating[(chat_id, msg_id)] = False
        try:
            await self._generate(chat_id, msg_id, adopt)
        finally:
            _ = self.generating.pop((chat_id, msg_id), None)

    async def _generate(self, chat_id: int, msg_id: int, adopt: list[int] | None = None):
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
            prefix = self._reply_prefix(model)
            async with BotReplyMessages(self, chat_id, msg_id, prefix, self.get_renderer(model), adopt) as replymsgs:
                try:
                    endpoint = model.endpoint or self.default_endpoint
                    status: str | None = 'Generating...'
                    await replymsgs.update(f'[{status}]')
                    client = self.get_client(endpoint, model)

                    async def fetch_image(cid: int, mid: int) -> bytes:
                        msg = await self.bot.get_messages(cid, ids=mid)
                        # ids=int returns a single Message at runtime, or None if it no longer exists
                        if not isinstance(msg, Message):
                            # a vanished message is a lookup failure, not a type error
                            raise ValueError(f'cannot fetch message {mid} in chat {cid}')  # noqa: TRY004
                        blob = await msg.download_media(bytes)
                        if not isinstance(blob, bytes):
                            # absent media is a lookup failure, not a type error
                            raise ValueError(f'message {mid} in chat {cid} has no downloadable media')  # noqa: TRY004
                        return blob

                    async def load_image(key: str) -> bytes | None:
                        return await load_photo(self.image_cache, key, fetch_image)

                    stream = completion(client, chat_history, model, system_prompt, chat_id, msg_id, load_image)
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
                        await replymsgs.update(format_reply(thinking, reply, thinking_done, status), bool(thinking))
                    stream_end = time.time()
                    usage = stream_meta.usage
                    if usage:
                        ttft = (first_token_time or stream_end) - stream_start
                        usage['TTFT'] = f'{ttft:.1f}s'
                        # reasoning tokens are a breakdown of the output total, not an addition to it
                        out_tokens = usage.get('out') or 0
                        if out_tokens and first_token_time:
                            gen_duration = stream_end - first_token_time
                            if gen_duration > 0:
                                usage['TPS'] = f'{out_tokens / gen_duration:.1f}'
                    await replymsgs.update(
                        format_reply(
                            thinking,
                            reply,
                            True,
                            status='aborted' if aborted else None,
                            tool_calls=stream_meta.tool_calls or None,
                            usage=usage or None,
                        ),
                        bool(thinking),
                    )
                    await replymsgs.finalize()
                    full_reply = reply
                    # reasoning first: replaying it requires the order the provider emitted
                    reply_parts: list[MsgPartInHistory] = []
                    for item in stream_meta.carry_items:
                        chain_of_thought = reasoning_text_of(item)
                        if chain_of_thought:
                            # A chain of thought we can read is carried as text rather than
                            # as a reasoning item. DeepSeek returns the text beside a handle
                            # into a response `store=false` never kept, and drops the item on
                            # replay whichever way it is sent -- observed by the reasoning
                            # tokens never reaching the next turn's input, and by the model
                            # reinventing numbers it had picked. A <think> block in the
                            # message is ordinary input and cannot be ignored.
                            reply_parts.append(make_thinking_text_part(chain_of_thought))
                        else:
                            reply_parts.append(make_reasoning_part(model.name, item))
                    reply_parts.append(make_text_part(full_reply))
                    part_ids = [bot_msg_id for bot_msg_id, _ in replymsgs.replied_msgs]
                    for bot_msg_id in part_ids:
                        self.set_msg_info(
                            chat_id,
                            bot_msg_id,
                            MsgInfo(
                                sent_by_bot=True,
                                message=list(reply_parts),
                                reply_id=msg_id,
                                prefix=None,
                                system_prompt=None,
                                part_ids=list(part_ids),
                            ),
                        )
                    return

                # handling completion errors
                except Exception as e:  # noqa: BLE001  # the retry loop handles every completion failure
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
        await self._user_turn(first, [m.id for m in event.messages], text, own_media)

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


class BotReplyMessages:
    def __init__(
        self,
        cbot: ChatGPTTelegramBot,
        chat_id: int,
        orig_msg_id: int,
        prefix: str,
        renderer: Renderer,
        adopt: list[int] | None = None,
    ) -> None:
        self.cbot: ChatGPTTelegramBot = cbot
        self.prefix: str = prefix
        self.renderer: Renderer = renderer
        self.chat_id: int = chat_id
        self.orig_msg_id: int = orig_msg_id
        # messages a regeneration writes over: seeding them as already-sent parts makes
        # `_apply` edit them, and delete the ones a shorter reply no longer needs. Their
        # recorded text is empty so the first flush always counts as a change.
        self.replied_msgs: list[tuple[int, Rendered]] = [(mid, Rendered(text='')) for mid in adopt or []]
        for mid in adopt or []:
            cbot.pending_reply_manager.add((chat_id, mid))
        self.text: str = ''
        self.collapse_first_quote: bool = False
        self.throttle: Throttle = Throttle(
            cbot.TELEGRAM_MIN_INTERVAL, cbot.TELEGRAM_BURST_INTERVAL, cbot.TELEGRAM_BURST_WINDOW
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self, type_: type[BaseException] | None, value: BaseException | None, tb: TracebackType | None
    ) -> None:
        await self.finalize()
        for msg_id, _ in self.replied_msgs:
            self.cbot.pending_reply_manager.remove((self.chat_id, msg_id))

    async def _force_update(self, text: str) -> None:
        # how the reply is cut, and how long a part may be, are the renderer's business
        try:
            await self._apply(self.renderer.render(text, self.prefix, self.collapse_first_quote))
        except errors.BadRequestError as e:
            # the server renderer rejects constructs it cannot lay out (an image with no
            # media, an unsupported block); rendering to entities instead keeps the reply
            if self.renderer is ENTITY_RENDERER or 'RICH_MESSAGE' not in str(e):
                raise
            logger.warning(f'Rich rendering rejected, falling back to entities ({self.chat_id=}): {e}')
            self.renderer = ENTITY_RENDERER
            await self._apply(self.renderer.render(text, self.prefix, self.collapse_first_quote))

    async def _apply(self, bodies: list[Rendered]) -> None:
        for i in range(min(len(bodies), len(self.replied_msgs))):
            msg_id, previous = self.replied_msgs[i]
            if bodies[i] != previous:
                await self.cbot.edit_message(self.chat_id, bodies[i], msg_id)
                self.replied_msgs[i] = (msg_id, bodies[i])
        if len(bodies) > len(self.replied_msgs):
            for i in range(len(self.replied_msgs), len(bodies)):
                if i == 0:
                    reply_to = self.orig_msg_id
                else:
                    reply_to, _ = self.replied_msgs[i - 1]
                msg_id = await self.cbot.send_message(self.chat_id, bodies[i], reply_to)
                self.replied_msgs.append((msg_id, bodies[i]))
                self.cbot.pending_reply_manager.add((self.chat_id, msg_id))
        if len(self.replied_msgs) > len(bodies):
            for i in range(len(bodies), len(self.replied_msgs)):
                msg_id, _ = self.replied_msgs[i]
                await self.cbot.delete_message(self.chat_id, msg_id)
                self.cbot.pending_reply_manager.remove((self.chat_id, msg_id))
            self.replied_msgs = self.replied_msgs[: len(bodies)]

    async def update(self, text: str, collapse_first_quote: bool = False) -> None:
        self.text = text
        self.collapse_first_quote = collapse_first_quote
        if self.throttle.take(time.time()):
            await self._force_update(self.text)
            # timed from the end of the flush: one tick can edit several messages when a
            # cut shifts, and starting the clock before that would shorten the next gap
            self.throttle.flushed(time.time())

    async def finalize(self) -> None:
        await self._force_update(self.text)
