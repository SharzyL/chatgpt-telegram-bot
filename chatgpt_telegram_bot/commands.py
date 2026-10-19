"""Reading what a user typed: bot commands, model prefixes and their inline overrides."""

import re
from typing import Any

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


def find_delimiter_outside_brackets(s: str, any_whitespace: bool = False) -> int:
    """Return the index of the first space outside ``[…]``, or ``-1``.

    With *any_whitespace*, a newline or tab ends the run as well.
    """
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
        elif depth == 0 and (c == ' ' or (any_whitespace and c.isspace())):
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


def parse_chat_spec(body: str) -> tuple[str, dict[str, str | None], str]:
    """Split a ``/chat`` body into the model it names, its overrides and the prompt.

    The body opens with ``<model>[,overrides]``, where the model is a prefix or
    ``<endpoint>/<name>``, and the prompt follows after a space or on the next line.
    """
    idx = find_delimiter_outside_brackets(body, any_whitespace=True)
    spec, prompt = (body, '') if idx == -1 else (body[:idx], body[idx + 1 :].lstrip())
    head, _, flags = spec.partition(',')
    return head, parse_overrides(flags), prompt


API_TYPES = ('openai', 'openai_legacy', 'anthropic')


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
    # the backend comes first, since a bare `t` takes its default effort from it
    api_type = next((v for k, v in overrides.items() if OVERRIDE_ALIASES.get(k, k) == 'api_type'), None)
    if api_type is not None and api_type not in API_TYPES:
        raise ValueError(f'api_type must be one of {", ".join(API_TYPES)}')
    if api_type is not None:
        model = model._replace(api_type=api_type)
    for key, value in overrides.items():
        field = OVERRIDE_ALIASES.get(key, key)
        if field == 'api_type':
            if value is None:
                raise ValueError('api_type needs a value')
        elif field in ('suffix', 'system_prompt', 'system_prompt_append'):
            # a bare `suffix` takes the endpoint's default, an explicit empty one (`suffix=`) none
            replacements[field] = value
        elif field == 'thinking':
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
        elif field in ('search', 'richtext', 'dummy_tool', 'live'):
            # bare key (`s`) enables, explicit empty (`s=`) or a false word disables
            replacements[field] = value is None or (value != '' and value.lower() not in ('0', 'false', 'no'))
        else:
            raise ValueError(f'Unknown override key: {key}')
    model = model._replace(**replacements)
    if model.dummy_tool and model.api_type != 'openai':
        raise ValueError('dt needs api_type openai')
    return model


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


# commands that read the rest of the message, and those that must be the whole of it
COMMANDS_WITH_BODY = frozenset({'chat', 'manipulate', 'import', 'regenerate', 'export'})


COMMANDS_WITHOUT_BODY = frozenset({'ping', 'help', 'models', 'abort', 'navigate', 'linearize'})


# short spellings, read as the command they stand for
COMMAND_ALIASES = {'re': 'regenerate', 'mani': 'manipulate'}


_COMMAND_TOKEN = re.compile(r'/(\w+)(?:@(\w+))?')


def parse_command(text: str, username: str | None) -> tuple[str, str] | None:
    """Read a bot command from the start of a message, as ``(name, body)``.

    The command is the first token, optionally addressed as ``/name@username``, and the body
    is the rest of the message, which may start on the next line. An alias in
    ``COMMAND_ALIASES`` is returned under the name it stands for. A command in
    ``COMMANDS_WITHOUT_BODY`` given a body anyway is still returned with it, so the caller can
    report the misuse. ``None`` means the message is not a command for this bot.
    """
    tokens = text.split(maxsplit=1)
    if not tokens:
        return None
    match = _COMMAND_TOKEN.fullmatch(tokens[0])
    if match is None:
        return None
    name, addressee = match.group(1), match.group(2)
    if addressee is not None and addressee != username:
        return None
    return COMMAND_ALIASES.get(name, name), tokens[1] if len(tokens) > 1 else ''


def is_foreign_command(text: str, username: str | None) -> bool:
    """Whether a message opens with a command addressed to another bot (``/start@other_bot``).

    Usernames are case-insensitive, so a mis-cased mention of this bot is not foreign.
    """
    tokens = text.split(maxsplit=1)
    if not tokens:
        return False
    match = _COMMAND_TOKEN.fullmatch(tokens[0])
    if match is None or match.group(2) is None:
        return False
    return username is None or match.group(2).lower() != username.lower()
