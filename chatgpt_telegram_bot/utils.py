import asyncio
import hashlib
import os
from typing import Any
from urllib.parse import urlparse

from loguru import logger
from telethon import errors

from chatgpt_telegram_bot.models import Model, OVERRIDE_ALIASES
from chatgpt_telegram_bot.richtext import RichText


def split_respecting_brackets(s: str, delimiters: str = ',') -> list[str]:
    """Split *s* on any char in *delimiters*, but skip content inside ``[…]``.

    Backslash-escaped brackets (``\\[``, ``\\]``) are treated as literal characters
    and do **not** affect nesting depth.
    """
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    i = 0
    while i < len(s):
        c = s[i]
        if c == '\\' and i + 1 < len(s) and s[i + 1] in '[]':
            current.append(c)
            current.append(s[i + 1])
            i += 2
            continue
        if c == '[':
            depth += 1
        elif c == ']':
            if depth > 0:
                depth -= 1
        if c in delimiters and depth == 0:
            parts.append(''.join(current))
            current = []
        else:
            current.append(c)
        i += 1
    if current:
        parts.append(''.join(current))
    return parts


def find_delimiter_outside_brackets(s: str) -> int:
    """Return the index of the first space outside ``[…]``, or ``-1``."""
    depth = 0
    i = 0
    while i < len(s):
        c = s[i]
        if c == '\\' and i + 1 < len(s) and s[i + 1] in '[]':
            i += 2
            continue
        if c == '[':
            depth += 1
        elif c == ']':
            if depth > 0:
                depth -= 1
        elif c == ' ' and depth == 0:
            return i
        i += 1
    return -1


def parse_overrides(s: str) -> dict[str, str | None]:
    """Parse ``key=val,key2=val2,key3,[custom system prompt]`` into a dict.

    Bare keys (no ``=``) map to ``None`` (use default).  A ``[…]``-delimited
    segment is stored under the ``system_prompt`` key; brackets inside the
    prompt can be escaped with a leading backslash (``\\[``, ``\\]``).
    """
    overrides: dict[str, str | None] = {}
    for part in split_respecting_brackets(s):
        part = part.strip()
        if not part:
            continue
        if part.startswith('[') and part.endswith(']'):
            content = part[1:-1].replace('\\[', '[').replace('\\]', ']')
            overrides['system_prompt'] = content
        elif '=' in part:
            k, v = part.split('=', 1)
            overrides[k.strip()] = v.strip()
        else:
            overrides[part] = None
    return overrides


def match_prefix(text: str, prefix: str) -> tuple[str, dict[str, str | None]] | None:
    """Check if text matches prefix with optional overrides. Returns (remaining_text, overrides) or None."""
    # Case 1: prefix with overrides (prefix,key=val+... delim text)
    if text.startswith(prefix + ','):
        after_pipe = text[len(prefix) + 1 :]
        # Find where overrides end (space delimiter, or end of string)
        # Must skip over [...] bracket sections which may contain spaces
        idx = find_delimiter_outside_brackets(after_pipe)
        if idx != -1:
            return (after_pipe[idx + 1 :], parse_overrides(after_pipe[:idx]))
        # No delimiter — entire rest is overrides, no text
        return ('', parse_overrides(after_pipe))
    # Case 2: bare prefix (existing behavior)
    if text == prefix:
        return ('', {})
    if text.startswith(prefix + ' '):
        return (text[len(prefix) + 1 :], {})
    return None


THINKING_DEFAULTS: dict[str, str] = {
    'openai': 'high',
    'openai_legacy': 'high',
    'anthropic': 'adaptive',
}


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
                replacements['thinking'] = THINKING_DEFAULTS.get(model.api_type)
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
        elif field == 'system_prompt':
            replacements['system_prompt'] = value
        else:
            raise ValueError(f'Unknown override key: {key}')
    return model._replace(**replacements)


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


def telegram_len(s: str | RichText) -> int:
    """Length in UTF-16 code units, matching Telegram's counting."""
    if isinstance(s, RichText):
        text, _ = s.to_telegram()
        return len(text.encode('utf-16-le')) // 2
    return len(s.encode('utf-16-le')) // 2


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


def save_photo(photo_blob: bytes) -> str:
    h = hashlib.sha256(photo_blob).hexdigest()
    save_dir = f'photos/{h[:2]}/{h[2:4]}'
    path = f'{save_dir}/{h}'
    if not os.path.isfile(path):
        os.makedirs(save_dir, exist_ok=True)
        with open(path, 'wb') as f:
            _ = f.write(photo_blob)
    return h


def load_photo(h: str) -> bytes:
    save_dir = f'photos/{h[:2]}/{h[2:4]}'
    path = f'{save_dir}/{h}'
    with open(path, 'rb') as f:
        return f.read()
