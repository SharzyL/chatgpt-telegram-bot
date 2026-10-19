"""Markdown source and Telegram text: UTF-16 lengths, splitting, fences, math and previews."""

import re
import unicodedata
from collections.abc import Callable

# markdown punctuation that is noise in a one-line preview: emphasis and code markers anywhere,
# and heading, quote and list markers at the start of a line
_PREVIEW_INLINE_RE = re.compile(r'\*\*|__|`+|~~')


_PREVIEW_LINE_START_RE = re.compile(r'^\s*(?:#{1,6}\s+|>+\s?|[-*+]\s+)', re.MULTILINE)


def _column_width(ch: str) -> int:
    """Columns *ch* takes on screen: two for wide (CJK, fullwidth) characters, else one."""
    return 2 if unicodedata.east_asian_width(ch) in ('W', 'F') else 1


def preview_line(text: str, width: int) -> str:
    """*text* as a single line of at most *width* columns, with an ellipsis if it was cut.

    Columns rather than characters, because a line of Chinese holds half as many characters
    as a line of English; counting characters would make a CJK preview wrap onto a second line.
    """
    text = _PREVIEW_LINE_START_RE.sub('', text)
    text = _PREVIEW_INLINE_RE.sub('', text)
    text = ' '.join(text.split())
    used = 0
    for i, ch in enumerate(text):
        w = _column_width(ch)
        # leave a column for the ellipsis
        if used + w > width - 1 and i < len(text) - 1 or used + w > width:
            return text[:i].rstrip() + '…'
        used += w
    return text


def telegram_len(s: str) -> int:
    """Length in UTF-16 code units, matching Telegram's counting."""
    return len(s.encode('utf-16-le')) // 2


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
        if not m:
            continue
        quote, fence = m.group(1), m.group(2)
        if opener is None:
            opener = (quote, fence)
        elif opener[0] == quote:
            # only a fence at the same blockquote depth closes the block: a fence inside a
            # quoted thinking block and one in the reply below it are different code blocks,
            # and pairing them across the boundary reopens a `>` fence over the reply
            opener = None
    return opener


def balance_fences(text: str) -> str:
    """Close a code fence *text* leaves open, so it cannot swallow what follows it."""
    fence = _open_fence(text)
    if fence is None:
        return text
    quote, _ = fence
    return text + '\n' + quote + '```'


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
