from typing import Any

from chatgpt_telegram_bot.utils import balance_fences, normalize_math


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
        return _blockquote_markdown(balance_fences(thinking) + suffix)
    result = ''
    if thinking:
        # a fence the thinking leaves open would run past the quote and take the answer
        # into the code block with it, and a cut reopens that fence — with its `>` prefix —
        # at the head of the next part, quoting everything below
        result = _blockquote_markdown(balance_fences(thinking.rstrip('\n')))
    if reply or suffix:
        sep = '\n\n' if thinking else ''
        # the footer sits below the reply and needs the same protection
        body = balance_fences(reply) if (tool_calls or usage) else reply
        result = result + sep + body + suffix
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
