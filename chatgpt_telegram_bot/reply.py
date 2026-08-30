from typing import Any

from chatgpt_telegram_bot.utils import normalize_math


def _blockquote_markdown(text: str) -> str:
    return '\n'.join('>' + line if line else '>' for line in text.split('\n'))


def format_reply(
    thinking: str,
    reply: str,
    thinking_done: bool,
    status: str | None = None,
    tool_calls: list[str] | None = None,
    usage: dict[str, Any] | None = None,
) -> str:
    """Assemble a reply as markdown source, for server-side rendering as a rich message.

    The model's own markdown is passed through untouched, so headings, tables, math and
    rules survive; only the thinking block and the footer are marked up here.
    """
    suffix = f' \\[!{status}]' if status else ''
    thinking = normalize_math(thinking)
    reply = normalize_math(reply)
    if not thinking_done:
        if not thinking:
            return suffix.strip()
        return _blockquote_markdown(thinking + suffix)
    result = ''
    if thinking:
        result = _blockquote_markdown(thinking.rstrip('\n'))
    if reply or suffix:
        sep = '\n\n' if thinking else ''
        result = result + sep + reply + suffix
    if tool_calls or usage:
        footer = ''
        if tool_calls:
            footer += '**Tool calls**\n' + '\n'.join(f'🔍 {q}' for q in tool_calls)
        if usage:
            if footer:
                footer += '\n\n'
            footer += '**Usage**\n' + ', '.join(f'{k}={v}' for k, v in usage.items())
        result = result + '\n\n' + _blockquote_markdown(footer)
    return result
