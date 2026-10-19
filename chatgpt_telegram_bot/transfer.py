"""Conversations in and out of the bot as JSON: what /import reads and /export writes."""

import datetime
import json
import re
from typing import Any

from chatgpt_telegram_bot.models import ImportedChat, ImportedTurn


def parse_import(text: str) -> ImportedChat:
    """Read an exported conversation, or raise ValueError saying what is wrong with it.

    The message list has to alternate user and assistant, starting with the user, because
    that is the only shape the reply chain can store. An assistant message may carry its
    chain of thought as text in an optional `reasoning` field.
    """
    try:
        data: Any = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f'File is not valid JSON: {e}') from e
    if not isinstance(data, dict):
        raise ValueError('Expected a JSON object with a "messages" array')  # noqa: TRY004  # a malformed file is bad input, not a type error
    messages: Any = data.get('messages')
    if not isinstance(messages, list) or not messages:
        raise ValueError('File has no "messages" array')
    raw_title: Any = data.get('title')
    title = raw_title.strip() if isinstance(raw_title, str) and raw_title.strip() else 'Untitled'

    turns: list[ImportedTurn] = []
    prompt: str | None = None
    expected = 'user'
    for i, message in enumerate(messages, start=1):
        if not isinstance(message, dict):
            raise ValueError(f'Message {i} is not an object')  # noqa: TRY004  # a malformed file is bad input, not a type error
        role: Any = message.get('role')
        content: Any = message.get('content')
        if role not in ('user', 'assistant'):
            raise ValueError(f'Message {i} has unsupported role {role!r}')
        if not isinstance(content, str):
            raise ValueError(f'Message {i} has no text content')  # noqa: TRY004  # a malformed file is bad input, not a type error
        if role != expected:
            raise ValueError(f'Message {i} is {role}, expected {expected}: roles must alternate, starting with user')
        reasoning: Any = message.get('reasoning')
        if reasoning is not None and not isinstance(reasoning, str):
            raise ValueError(
                f'Message {i} has a "reasoning" that is not text'
            )  # a malformed file is bad input, not a type error
        if role == 'user':
            prompt = content
            expected = 'assistant'
        else:
            assert prompt is not None  # the alternation check above guarantees it
            # a prompt's own reasoning has nowhere to go, as only a reply carries one
            turns.append(ImportedTurn(prompt, content, reasoning or None))
            prompt = None
            expected = 'user'

    if not turns:
        raise ValueError('File has no complete prompt-and-response turn')
    return ImportedChat(title=title.replace('\n', ' '), turns=turns, trailing_prompt=prompt)


def _iso_timestamp(moment: datetime.datetime) -> str:
    """UTC with milliseconds and a Z, which is how the exports /import reads write their times."""
    return moment.astimezone(datetime.UTC).isoformat(timespec='milliseconds').replace('+00:00', 'Z')


def format_export(
    title: str,
    messages: list[tuple[str, str, datetime.datetime | None, str | None]],
    exported_at: datetime.datetime,
) -> str:
    """A thread as JSON in the shape parse_import reads, so an export can be imported back.

    *messages* is (role, content, sent at, reasoning) oldest first. A message with no known
    time gets no timestamp field, which parse_import does not need anyway, and one with no
    chain of thought no reasoning field.
    """
    entries: list[dict[str, str]] = []
    for role, content, sent_at, reasoning in messages:
        entry = {'role': role, 'content': content}
        if reasoning:
            entry['reasoning'] = reasoning
        if sent_at is not None:
            entry['timestamp'] = _iso_timestamp(sent_at)
        entries.append(entry)
    document = {'title': title, 'exportedAt': _iso_timestamp(exported_at), 'messages': entries}
    return json.dumps(document, ensure_ascii=False, indent=2) + '\n'


# characters a filename cannot hold on one platform or another
_UNSAFE_FILENAME_RE = re.compile(r'[\x00-\x1f/\\:*?"<>|]+')


def export_filename(title: str, limit: int = 60) -> str:
    """A filename for an exported thread, from its title; `thread.json` if nothing usable is left."""
    stem = _UNSAFE_FILENAME_RE.sub(' ', title)
    stem = ' '.join(stem.split())[:limit].strip(' .')
    return f'{stem or "thread"}.json'
