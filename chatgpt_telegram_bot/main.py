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
from collections import defaultdict
from urllib.parse import urlparse
import tomllib
from argparse import ArgumentParser
from collections.abc import Sequence
from typing import Any, NamedTuple

import anthropic
import openai
from google import genai
from google.genai import types as genai_types
from telethon import TelegramClient, events, errors, functions, types
from loguru import logger

from chatgpt_telegram_bot.richtext import RichText


class Model(NamedTuple):
    prefix: str
    name: str
    endpoint: str | None = None
    no_system_prompt: bool = False
    system_prompt: str | None = None
    api_type: str = 'openai'  # 'openai', 'anthropic', or 'gemini'
    suffix: str | None = None  # appended to endpoint URL, None = use endpoint's default_suffix
    thinking: int | str | None = None  # None = disabled, int = budget, str = effort level or 'adaptive'
    search: bool = False


class EndPoint(NamedTuple):
    name: str
    url: str
    default_suffix: str = ''


class MsgPartInHistory(NamedTuple):
    """
    either "text" or "image"
    """

    type_: str
    hash: str | None  # must present when str == "img"
    text: str | None  # must present when str == "text"


def make_image_part(_hash: str) -> MsgPartInHistory:
    return MsgPartInHistory(type_='image', hash=_hash, text=None)


def make_text_part(text: str) -> MsgPartInHistory:
    return MsgPartInHistory(type_='text', hash=None, text=text)


class MsgInfo(NamedTuple):
    sent_by_bot: bool
    message: list[MsgPartInHistory]
    reply_id: int | None

    """only present for head of conversation"""
    prefix: str | None

    """only present for head of conversation"""
    system_prompt: str | None

    """only present for head of conversation, stores inline parameter overrides"""
    overrides: dict[str, str | None] | None = None


OVERRIDE_ALIASES: dict[str, str] = {'t': 'thinking', 's': 'search'}


def _parse_overrides(s: str) -> dict[str, str | None]:
    """Parse 'key=val,key2=val2,key3' into dict. Bare key (no '=') maps to None (use default)."""
    overrides: dict[str, str | None] = {}
    for part in s.split(','):
        part = part.strip()
        if not part:
            continue
        if '=' in part:
            k, v = part.split('=', 1)
            overrides[k.strip()] = v.strip()
        else:
            overrides[part] = None
    return overrides


def parse_proxy():
    proxy_env = os.getenv('ALL_PROXY')
    if proxy_env:
        proxy_url = urlparse(proxy_env)
        return {
            'proxy_type': proxy_url.scheme,
            'addr': proxy_url.hostname,
            'port': proxy_url.port,
        }
    else:
        return None


def retry(max_retry: int = 30, interval: int = 10):
    def decorator(func):  # pyright: ignore[reportMissingParameterType]  # generic decorator wrapper
        async def new_func(*args, **kwargs):  # pyright: ignore[reportMissingParameterType]  # generic decorator wrapper
            for _ in range(max_retry - 1):
                try:
                    return await func(*args, **kwargs)
                except AssertionError as e:
                    logger.exception(e)
                except ValueError as e:
                    logger.exception(e)
                except errors.FloodWaitError as e:
                    logger.exception(e)
                    await asyncio.sleep(interval)
            return await func(*args, **kwargs)

        return new_func

    return decorator


class PendingReplyManager:
    def __init__(self) -> None:
        self.messages: dict[tuple[int, int], asyncio.Event] = {}

    def add(self, reply_id: tuple[int, int]) -> None:
        assert reply_id not in self.messages
        self.messages[reply_id] = asyncio.Event()

    def remove(self, reply_id: tuple[int, int]) -> None:
        if reply_id not in self.messages:
            return
        self.messages[reply_id].set()
        del self.messages[reply_id]

    async def wait_for(self, reply_id: tuple[int, int]) -> None:
        if reply_id not in self.messages:
            return
        logger.info('PendingReplyManager waiting for %r', reply_id)
        _ = await self.messages[reply_id].wait()
        logger.info('PendingReplyManager waiting for %r finished', reply_id)


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

        for model in self.models:
            if ' ' in model.prefix or '$' in model.prefix or '|' in model.prefix:
                raise ValueError(f'prefix must not contain space, "$", or "|": "{model.prefix}"')
            if model.thinking == 'adaptive' and model.api_type != 'anthropic':
                raise ValueError(
                    f'thinking="adaptive" is only supported for api_type="anthropic",'
                    + f' got api_type="{model.api_type}" for model "{model.prefix}"'
                )

        self.telegram_last_timestamp: defaultdict[int, int | None] = defaultdict(lambda: None)
        self.telegram_rate_limit_lock: defaultdict[int, asyncio.Lock] = defaultdict(asyncio.Lock)

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
            if api_type == 'openai':
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
            elif api_type == 'gemini':
                self.clients[(ep_name, api_type, suffix)] = genai.Client(
                    api_key=api_key,
                    http_options={'base_url': url},
                )
            else:
                raise ValueError(f'Unknown api_type: {api_type}')

        self.TELEGRAM_LENGTH_LIMIT: int = 4096
        self.TELEGRAM_MIN_INTERVAL: int = 3
        self.OPENAI_MAX_RETRY: int = 3
        self.OPENAI_RETRY_INTERVAL: int = 3
        self.FIRST_BATCH_DELAY: int = 1
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
        # Case 1: prefix with overrides (prefix|key=val,... delim text)
        if text.startswith(prefix + '|'):
            after_pipe = text[len(prefix) + 1 :]
            # Find where overrides end (space or $ delimiter, or end of string)
            for delim in (' ', '$'):
                idx = after_pipe.find(delim)
                if idx != -1:
                    return (after_pipe[idx + 1 :], _parse_overrides(after_pipe[:idx]))
            # No delimiter — entire rest is overrides, no text
            return ('', _parse_overrides(after_pipe))
        # Case 2: bare prefix (existing behavior)
        if text == prefix:
            return ('', {})
        for delim in (' ', '$'):
            if text.startswith(prefix + delim):
                return (text[len(prefix) + len(delim) :], {})
        return None

    THINKING_DEFAULTS: dict[str, str] = {
        'openai': 'high',
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
                    _ = await self.send_message(event.message.chat_id, f'no prompt set yet', event.message.id)
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
                _ = await self.send_message(event.message.chat_id, f'system prompt cleared', event.message.id)
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

    @staticmethod
    def get_prompt(_model: str) -> str:
        current_time = (datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=8)).strftime('%Y-%m-%d %H:%M:%S')
        return f"""Current Beijing Time: {current_time}. Reply in the same language as the user sent you.
    """

    def within_interval(self, chat_id: int) -> bool:
        last_timestamp = self.telegram_last_timestamp.get(chat_id, None)
        if last_timestamp is None:
            return False
        else:
            remaining_time = last_timestamp + self.TELEGRAM_MIN_INTERVAL - time.time()
        return remaining_time > 0

    @staticmethod
    def ensure_interval(func):  # pyright: ignore[reportMissingParameterType]  # generic decorator wrapper
        async def new_func(self, *args, **kwargs):  # pyright: ignore[reportMissingParameterType]  # generic decorator wrapper
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

    @staticmethod
    def save_photo(photo_blob: bytes) -> str:
        h = hashlib.sha256(photo_blob).hexdigest()
        save_dir = f'photos/{h[:2]}/{h[2:4]}'
        path = f'{save_dir}/{h}'
        if not os.path.isfile(path):
            os.makedirs(save_dir, exist_ok=True)
            with open(path, 'wb') as f:
                _ = f.write(photo_blob)
        return h

    @staticmethod
    def load_photo(h: str) -> bytes:
        save_dir = f'photos/{h[:2]}/{h[2:4]}'
        path = f'{save_dir}/{h}'
        with open(path, 'rb') as f:
            return f.read()

    async def completion(
        self,
        chat_history: list[list[MsgPartInHistory]],
        model: Model,
        system_prompt: str,
        endpoint: str,
        chat_id: int,
        msg_id: int,
    ):  # chat_history = [user, ai, user, ai, ..., user]
        assert len(chat_history) % 2 == 1
        log_history = [[{'type': p.type_, 'text': p.text} for p in msg] for msg in chat_history]
        logger.info(f'Starting completion ({model.api_type}) for {chat_id=}, {msg_id=}: {log_history}')

        if model.api_type == 'openai':
            backend = self._completion_openai(chat_history, model, system_prompt, endpoint, chat_id, msg_id)
        elif model.api_type == 'anthropic':
            backend = self._completion_anthropic(chat_history, model, system_prompt, endpoint, chat_id, msg_id)
        elif model.api_type == 'gemini':
            backend = self._completion_gemini(chat_history, model, system_prompt, endpoint, chat_id, msg_id)
        else:
            raise ValueError(f'Unknown api_type: {model.api_type}')

        async for delta in backend:
            yield delta

    def _convert_history_openai(self, chat_history: list[list[MsgPartInHistory]]) -> list[list[dict[str, Any]]]:
        result: list[list[dict[str, Any]]] = []
        for msg_parts in chat_history:
            converted: list[dict[str, Any]] = []
            for part in msg_parts:
                if part.type_ == 'text':
                    converted.append({'type': 'input_text', 'text': part.text})
                elif part.type_ == 'image':
                    assert part.hash is not None
                    blob = self.load_photo(part.hash)
                    blob_base64 = base64.b64encode(blob).decode()
                    converted.append({'type': 'input_image', 'image_url': 'data:image/jpeg;base64,' + blob_base64})
            result.append(converted)
        return result

    async def _completion_openai(
        self,
        chat_history: list[list[MsgPartInHistory]],
        model: Model,
        system_prompt: str,
        endpoint: str,
        chat_id: int,
        msg_id: int,
    ):
        converted = self._convert_history_openai(chat_history)
        input_messages: list[Any] = []
        roles = ['user', 'assistant']
        for i, msg in enumerate(converted):
            role = roles[i % len(roles)]
            content: Any = msg
            if len(msg) == 1 and msg[0]['type'] == 'input_text':
                content = msg[0]['text']
            input_messages.append({'role': role, 'content': content})

        kwargs: dict[str, Any] = {
            'model': model.name,
            'input': input_messages,
            'stream': True,
        }
        if system_prompt and not model.no_system_prompt:
            kwargs['instructions'] = system_prompt
        if model.thinking is not None:
            effort = model.thinking if isinstance(model.thinking, str) else 'high'
            kwargs['reasoning'] = {'effort': effort, 'summary': 'detailed'}
        if model.search:
            kwargs['tools'] = [{'type': 'web_search', 'search_context_size': 'medium'}]

        aclient: openai.AsyncOpenAI = self.get_client(endpoint, model)
        stream = await aclient.responses.create(**kwargs)
        has_reasoning = False
        reasoning_ended = False
        search_queries: list[str] = []
        async for event in stream:
            logger.debug(f'Received event ({chat_id=}, {msg_id=}): {event.type}')
            if event.type == 'response.web_search_call.searching':
                yield '\x01Searching...\x01'
            elif event.type == 'response.output_item.done' and event.item.type == 'web_search_call':
                query = getattr(event.item.action, 'query', None) if hasattr(event.item, 'action') else None
                if query:
                    search_queries.append(query)
                yield '\x01Generating...\x01'
            elif event.type in ('response.reasoning_summary_text.delta', 'response.reasoning_text.delta'):
                if not has_reasoning:
                    has_reasoning = True
                yield event.delta
            elif event.type == 'response.output_text.delta':
                if has_reasoning and not reasoning_ended:
                    reasoning_ended = True
                    yield '\x00'
                yield event.delta
            elif event.type == 'response.completed':
                resp = event.response
                if resp.status == 'incomplete' and resp.incomplete_details:
                    reason = resp.incomplete_details.reason
                    if reason == 'max_output_tokens':
                        yield '\n\n[!] Error: Output truncated due to limit'
                    else:
                        yield f'\n\n[!] Error: incomplete reason="{reason}"'
        if search_queries:
            yield '\x02' + '\n'.join(search_queries)

    def _convert_history_anthropic(self, chat_history: list[list[MsgPartInHistory]]) -> list[list[dict[str, Any]]]:
        result: list[list[dict[str, Any]]] = []
        for msg_parts in chat_history:
            converted: list[dict[str, Any]] = []
            for part in msg_parts:
                if part.type_ == 'text':
                    converted.append({'type': 'text', 'text': part.text})
                elif part.type_ == 'image':
                    assert part.hash is not None
                    blob = self.load_photo(part.hash)
                    blob_base64 = base64.b64encode(blob).decode()
                    converted.append(
                        {
                            'type': 'image',
                            'source': {'type': 'base64', 'media_type': 'image/jpeg', 'data': blob_base64},
                        }
                    )
            result.append(converted)
        return result

    async def _completion_anthropic(
        self,
        chat_history: list[list[MsgPartInHistory]],
        model: Model,
        system_prompt: str,
        endpoint: str,
        _chat_id: int,
        _msg_id: int,
    ):
        converted = self._convert_history_anthropic(chat_history)
        messages: list[Any] = []
        roles = ['user', 'assistant']
        for i, msg in enumerate(converted):
            role = roles[i % len(roles)]
            content: Any = msg
            if len(msg) == 1 and msg[0]['type'] == 'text':
                content = msg[0]['text']
            messages.append({'role': role, 'content': content})

        kwargs: dict[str, Any] = {
            'model': model.name,
            'max_tokens': 16384,
            'messages': messages,
        }
        if system_prompt and not model.no_system_prompt:
            kwargs['system'] = system_prompt
        if model.thinking == 'adaptive':
            kwargs['thinking'] = {'type': 'adaptive'}
        elif isinstance(model.thinking, int) and model.thinking > 0:
            kwargs['thinking'] = {'type': 'enabled', 'budget_tokens': model.thinking}

        if model.search:
            kwargs.setdefault('tools', []).append({'type': 'web_search_20250305', 'name': 'web_search'})

        aclient: anthropic.AsyncAnthropic = self.get_client(endpoint, model)
        async with aclient.messages.stream(**kwargs) as stream:
            has_thinking = False
            async for event in stream:
                if event.type == 'content_block_start':
                    if event.content_block.type == 'thinking':
                        has_thinking = True
                    elif event.content_block.type == 'text' and has_thinking:
                        yield '\x00'
                elif event.type == 'content_block_delta':
                    if event.delta.type == 'thinking_delta':
                        yield event.delta.thinking
                    elif event.delta.type == 'text_delta':
                        yield event.delta.text
            final_message = await stream.get_final_message()
            if final_message.stop_reason == 'max_tokens':
                yield '\n\n[!] Error: Output truncated due to limit'
            elif final_message.stop_reason not in ('end_turn', 'stop_sequence'):
                yield f'\n\n[!] Error: stop_reason="{final_message.stop_reason}"'

    def _convert_history_gemini(self, chat_history: list[list[MsgPartInHistory]]) -> list[genai_types.Content]:
        contents: list[genai_types.Content] = []
        roles = ['user', 'model']
        for i, msg_parts in enumerate(chat_history):
            role = roles[i % len(roles)]
            parts: list[genai_types.Part] = []
            for part in msg_parts:
                if part.type_ == 'text':
                    assert part.text is not None
                    parts.append(genai_types.Part.from_text(text=part.text))
                elif part.type_ == 'image':
                    assert part.hash is not None
                    blob = self.load_photo(part.hash)
                    parts.append(genai_types.Part.from_bytes(data=blob, mime_type='image/jpeg'))
            contents.append(genai_types.Content(role=role, parts=parts))
        return contents

    async def _completion_gemini(
        self,
        chat_history: list[list[MsgPartInHistory]],
        model: Model,
        system_prompt: str,
        endpoint: str,
        _chat_id: int,
        _msg_id: int,
    ):
        contents = self._convert_history_gemini(chat_history)

        config_kwargs: dict[str, Any] = {}
        if system_prompt and not model.no_system_prompt:
            config_kwargs['system_instruction'] = system_prompt
        if isinstance(model.thinking, int) and model.thinking > 0:
            config_kwargs['thinking_config'] = genai_types.ThinkingConfig(thinking_budget=model.thinking)
        if model.search:
            config_kwargs['tools'] = [genai_types.Tool(google_search=genai_types.GoogleSearch())]

        gclient: genai.Client = self.get_client(endpoint, model)
        finish_reason = None
        has_thought = False
        thought_ended = False
        async for chunk in await gclient.aio.models.generate_content_stream(
            model=model.name,
            contents=contents,
            config=genai_types.GenerateContentConfig(**config_kwargs) if config_kwargs else None,
        ):
            if chunk.candidates and chunk.candidates[0].content and chunk.candidates[0].content.parts:
                for part in chunk.candidates[0].content.parts:
                    if part.thought:
                        has_thought = True
                        if part.text:
                            yield part.text
                    elif part.text:
                        if has_thought and not thought_ended:
                            thought_ended = True
                            yield '\x00'
                        yield part.text
            if chunk.candidates and chunk.candidates[0].finish_reason:
                finish_reason = chunk.candidates[0].finish_reason

        if finish_reason is not None:
            fr = str(finish_reason)
            if 'MAX_TOKENS' in fr:
                yield '\n\n[!] Error: Output truncated due to limit'
            elif 'SAFETY' in fr:
                yield '\n\n[!] Error: Response blocked by safety filter'
            elif 'STOP' not in fr:
                yield f'\n\n[!] Error: finish_reason="{fr}"'

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
    @ensure_interval
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
    @ensure_interval
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
    @ensure_interval
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
    @ensure_interval
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
            photo_hash = self.save_photo(photo_blob)

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
            text = 'Continue' if reply_to_id is not None else 'Describe the image in Chinese'

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

    @staticmethod
    def _format_reply(
        thinking: str,
        reply: str,
        thinking_done: bool,
        status: str | None = None,
        expect_thinking: bool = False,
        tool_calls: list[str] | None = None,
    ) -> str | RichText:
        suffix = f' [!{status}]' if status else ''
        if not thinking_done:
            # still in thinking phase, or no thinking at all
            if not thinking:
                return suffix.strip() if suffix else ''
            if expect_thinking:
                return RichText.Blockquote(RichText.from_markdown(thinking) + suffix)
            return RichText.from_markdown(thinking) + suffix
        # thinking is done, show blockquote + response
        result: str | RichText = ''
        if thinking:
            result = RichText.Blockquote(RichText.from_markdown(thinking.rstrip('\n')))
        if reply or suffix:
            result = result + '\n' + RichText.from_markdown(reply) + suffix
        if tool_calls:
            footer = RichText.Bold('Tool calls') + '\n' + '\n'.join(f'🔍 {q}' for q in tool_calls)
            result = result + '\n\n' + footer
        return result

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
            tool_calls: list[str] = []
            expect_thinking = model.thinking is not None or model.api_type == 'anthropic'
            model_flags: list[str] = []
            if model.thinking is not None:
                model_flags.append(f'thinking={model.thinking}')
            if model.search:
                model_flags.append('search')
            prefix = '🤖 ' + RichText.Code(model.name)
            if model_flags:
                prefix += ' ' + ' '.join(model_flags)
            prefix += '\n\n'
            async with BotReplyMessages(self, chat_id, msg_id, prefix) as replymsgs:
                try:
                    endpoint = model.endpoint or self.default_endpoint
                    status: str | None = 'Generating...'
                    await replymsgs.update(f'[{status}]')
                    stream = self.completion(chat_history, model, system_prompt, endpoint, chat_id, msg_id)
                    first_update_timestamp = None
                    async for delta in stream:
                        # \x02 sentinel carries tool call info (at end of stream)
                        if '\x02' in delta:
                            tool_calls.extend(delta.split('\x02', 1)[1].split('\n'))
                            continue
                        # \x01 sentinel signals status change
                        if '\x01' in delta:
                            parts = delta.split('\x01')
                            for i, part in enumerate(parts):
                                if i % 2 == 1:
                                    status = part if part else 'Generating...'
                                elif part:
                                    if '\x00' in part:
                                        t, r = part.split('\x00', 1)
                                        thinking += t
                                        reply += r
                                        thinking_done = True
                                    elif not thinking_done:
                                        thinking += part
                                    else:
                                        reply += part
                            await replymsgs.update(
                                self._format_reply(thinking, reply, thinking_done, status, expect_thinking)
                            )
                            continue
                        # \x00 sentinel separates thinking from response
                        if '\x00' in delta:
                            parts = delta.split('\x00', 1)
                            thinking += parts[0]
                            reply += parts[1]
                            thinking_done = True
                        elif not thinking_done:
                            thinking += delta
                        else:
                            reply += delta
                        if first_update_timestamp is None:
                            first_update_timestamp = time.time()
                        if time.time() >= first_update_timestamp + self.FIRST_BATCH_DELAY:
                            await replymsgs.update(
                                self._format_reply(thinking, reply, thinking_done, status, expect_thinking)
                            )
                    if not thinking_done:
                        # no \x00 sentinel received — all text is response, not thinking
                        reply = thinking
                        thinking = ''
                    await replymsgs.update(self._format_reply(thinking, reply, True, tool_calls=tool_calls or None))
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
        photo_hashes = [self.save_photo(blob) for blob in photo_blobs]

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


def _telegram_len(s: str | RichText) -> int:
    """Length in UTF-16 code units, matching Telegram's counting."""
    if isinstance(s, RichText):
        text, _ = s.to_telegram()
        return len(text.encode('utf-16-le')) // 2
    return len(s.encode('utf-16-le')) // 2


class BotReplyMessages:
    def __init__(self, cbot: ChatGPTTelegramBot, chat_id: int, orig_msg_id: int, prefix: str | RichText) -> None:
        self.cbot: ChatGPTTelegramBot = cbot
        self.prefix: str | RichText = prefix
        self.msg_len: int = cbot.TELEGRAM_LENGTH_LIMIT - _telegram_len(prefix)
        assert self.msg_len > 0
        self.chat_id: int = chat_id
        self.orig_msg_id: int = orig_msg_id
        self.replied_msgs: list[tuple[int, str | RichText]] = []
        self.text: str | RichText = ''

    async def __aenter__(self) -> 'BotReplyMessages':
        return self

    async def __aexit__(self, type_: type[BaseException] | None, value: BaseException | None, tb: Any) -> None:
        await self.finalize()
        for msg_id, _ in self.replied_msgs:
            self.cbot.pending_reply_manager.remove((self.chat_id, msg_id))

    async def _force_update(self, text: str | RichText) -> None:
        slices: list[str | RichText] = []
        while len(text) > self.msg_len:
            slices.append(text[: self.msg_len])
            text = text[self.msg_len :]
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
            self.replied_msgs = self.replied_msgs[: len(slices)]

    async def update(self, text: str | RichText) -> None:
        self.text = text
        if not self.cbot.within_interval(self.chat_id):
            await self._force_update(self.text)

    async def finalize(self) -> None:
        await self._force_update(self.text)


async def async_main() -> None:
    parser = ArgumentParser()
    _ = parser.add_argument('--debug', action='store_true')
    _ = parser.add_argument('-c', '--config', default='bot.toml')

    args = parser.parse_args()

    log_level = 'DEBUG' if args.debug else 'INFO'
    logger.remove()
    _ = logger.add(
        sys.stdout,
        colorize=True,
        format='<green>{time}</green> <level>{message}</level>',
        level=log_level,
    )

    cbot = ChatGPTTelegramBot(args.config)
    await cbot.start()


def main():
    asyncio.run(async_main())
