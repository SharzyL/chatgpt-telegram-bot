import asyncio
import os
import re
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import urlparse

import diskcache
from loguru import logger
from telethon import errors

from chatgpt_telegram_bot.models import OVERRIDE_ALIASES, Model


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
        elif c == ']' and depth > 0:
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
    segment is stored under the ``system_prompt`` key; ``+[…]`` is stored
    under ``system_prompt_append``.  Brackets inside the prompt can be
    escaped with a leading backslash (``\\[``, ``\\]``).
    """
    overrides: dict[str, str | None] = {}
    for part in split_respecting_brackets(s):
        part = part.strip()
        if not part:
            continue
        if part.startswith('+[') and part.endswith(']'):
            content = part[2:-1].replace('\\[', '[').replace('\\]', ']')
            overrides['system_prompt_append'] = content
        elif part.startswith('[') and part.endswith(']'):
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
        elif field == 'richtext':
            # bare key (|r) enables, explicit empty (|r=) disables
            replacements['richtext'] = value is None or (value != '' and value.lower() not in ('0', 'false', 'no'))
        elif field == 'system_prompt':
            replacements['system_prompt'] = value
        elif field == 'system_prompt_append':
            replacements['system_prompt_append'] = value
        else:
            raise ValueError(f'Unknown override key: {key}')
    return model._replace(**replacements)


_THINK_BLOCK = re.compile(r'\A<think>(.*?)</think>', re.DOTALL)


def parse_manipulation(body: str) -> tuple[str | None, str]:
    """Split a ``/manipulate`` body into an optional chain of thought and the reply text.

    A leading ``<think>…</think>`` block is taken as the chain of thought and everything
    after it as the reply text; without the block the whole body is the reply text.
    """
    body = body.strip()
    match = _THINK_BLOCK.match(body)
    if match is None:
        return None, body
    thinking = match.group(1).strip()
    text = body[match.end() :].strip()
    return (thinking or None), text


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


def telegram_len(s: str) -> int:
    """Length in UTF-16 code units, matching Telegram's counting."""
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


def save_photo(cache: diskcache.Cache, photo_blob: bytes, chat_id: int, msg_id: int) -> str:
    key = f'{chat_id}:{msg_id}'
    _ = cache.set(key, photo_blob)
    return key


async def load_photo(
    cache: diskcache.Cache,
    key: str,
    fetcher: Callable[[int, int], Awaitable[bytes]] | None = None,
) -> bytes | None:
    blob: bytes | None = cache.get(key)  # pyright: ignore[reportAssignmentType]  # diskcache returns stored type
    if blob is not None:
        return blob
    if fetcher and ':' in key:
        chat_id_str, msg_id_str = key.split(':', 1)
        try:
            blob = await fetcher(int(chat_id_str), int(msg_id_str))
            _ = cache.set(key, blob)
            return blob
        except Exception:  # noqa: BLE001  # any re-fetch failure just means the image is unavailable
            logger.warning(f'Failed to re-fetch image for key={key}')
            return None
    return None


# a code fence, optionally inside a blockquote (`>` prefixed, as thinking blocks are)
_FENCE_LINE_RE = re.compile(r'^(>*)\s*(```.*)$')
# room reserved for a closing fence appended to a part, in UTF-16 units
_FENCE_MARGIN = 8


def telegram_truncate(s: str, units: int) -> str:
    """Truncate *s* to at most *units* UTF-16 code units, the unit Telegram counts in."""
    return s[: utf16_prefix(s, units)]


def utf16_prefix(s: str, units: int) -> int:
    """Largest index i for which ``telegram_len(s[:i]) <= units``."""
    total = 0
    for i, ch in enumerate(s):
        width = 2 if ord(ch) > 0xFFFF else 1
        if total + width > units:
            return i
        total += width
    return len(s)


def _open_fence(text: str) -> tuple[str, str] | None:
    """The (blockquote prefix, opener) of a code fence *text* leaves unclosed, if any."""
    opener: tuple[str, str] | None = None
    for line in text.split('\n'):
        m = _FENCE_LINE_RE.match(line)
        if m:
            opener = None if opener else (m.group(1), m.group(2))
    return opener


def _cut_at(text: str, at: int) -> tuple[str, str]:
    """Split *text* at *at*, closing and reopening a fence that straddles the boundary."""
    head, rest = text[:at], text[at:].lstrip('\n')
    fence = _open_fence(head)
    if fence:
        quote, opener = fence
        head += '\n' + quote + '```'
        rest = quote + opener + '\n' + rest
    return head, rest


def split_markdown(text: str, first_limit: int, limit: int) -> list[str]:
    """Split markdown source into parts of at most *first_limit* / *limit* UTF-16 units.

    Rich messages are rendered server-side, so a part has to stay valid markdown on its
    own. Cuts are made at a paragraph break where possible, then a line break, and only
    mid-line as a last resort. A fenced code block left open by a cut is closed and
    reopened across the boundary, including inside a blockquote.
    """
    parts: list[str] = []
    cur_limit = first_limit
    while telegram_len(text) > cur_limit:
        # closing a fence appends to the part, so leave room for it
        budget = max(1, cur_limit - _FENCE_MARGIN) if '```' in text else cur_limit
        end = utf16_prefix(text, budget)
        cut = text.rfind('\n\n', 0, end + 1)
        if cut <= 0:
            cut = text.rfind('\n', 0, end + 1)
        if cut <= 0:
            cut = end
        head, rest = _cut_at(text, cut)
        # a cut landing on a fence opener reproduces `text` unchanged, which would spin
        # forever; fall back to cutting at the budget, then to no fence handling at all
        if len(rest) >= len(text):
            head, rest = _cut_at(text, end)
        if len(rest) >= len(text):
            head, rest = text[:end], text[end:]
        parts.append(head)
        text = rest
        cur_limit = limit
    if text:
        parts.append(text)
    return parts


# fenced blocks and inline code spans, whose contents must not be rewritten
_CODE_SPAN_RE = re.compile(r'```.*?```|``.*?``|`[^`\n]*`', re.DOTALL)
# a markdown image, inline `![alt](url)` or reference `![alt][ref]`
_IMAGE_RE = re.compile(r'!(\[[^\]]*\])(\([^)]*\)|\[[^\]]*\])')
_INLINE_MATH_RE = re.compile(r'\\\((.+?)\\\)', re.DOTALL)
_DISPLAY_MATH_RE = re.compile(r'\\\[(.+?)\\\]', re.DOTALL)


def _rewrite_outside_code(markdown: str, convert: Callable[[str], str]) -> str:
    """Apply *convert* to every stretch of *markdown* that is not a code span."""
    parts: list[str] = []
    pos = 0
    for m in _CODE_SPAN_RE.finditer(markdown):
        parts.append(convert(markdown[pos : m.start()]))
        parts.append(m.group(0))
        pos = m.end()
    parts.append(convert(markdown[pos:]))
    return ''.join(parts)


def normalize_math(markdown: str) -> str:
    r"""Rewrite LaTeX ``\(…\)`` and ``\[…\]`` delimiters into the ``$`` forms.

    Telegram's rich-message markdown recognises ``$…$`` and ``$$…$$`` (and a ```` ```math ````
    fence) but leaves the backslash-delimiter forms as plain text, so models that emit those
    would otherwise show raw LaTeX. Code spans are left untouched.
    """

    def convert(s: str) -> str:
        s = _INLINE_MATH_RE.sub(r'$\1$', s)
        return _DISPLAY_MATH_RE.sub(r'$$\1$$', s)

    return _rewrite_outside_code(markdown, convert)


def _image_to_link(m: re.Match[str]) -> str:
    alt, target = m.group(1), m.group(2)
    # an empty alt would leave a link with no clickable text, so show the target instead
    if alt == '[]' and target.startswith('(') and target.endswith(')'):
        return f'[{target[1:-1]}]{target}'
    return alt + target


def neutralize_images(markdown: str) -> str:
    """Turn markdown images into ordinary links.

    Telegram's rich-message renderer reads an image as a photo block and rejects the whole
    message with ``RICH_MESSAGE_PHOTO_NO_MEDIA_FOUND`` when no media is attached, which is
    always the case for a text reply. Demoting the image keeps the alt text and the URL
    reachable. Code spans are left untouched.
    """
    return _rewrite_outside_code(markdown, lambda s: _IMAGE_RE.sub(_image_to_link, s))
