from collections.abc import Callable
from html import escape as html_escape
from typing import Any

from chatgpt_telegram_bot.models import MsgInfo, ThreadTurn, text_of
from chatgpt_telegram_bot.utils import balance_fences, normalize_math, preview_line


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


# a literal dash line: bare, `-----` would be a horizontal rule (sent as `---` by the entity
# renderer), or turn the line above into a heading when nothing separates them
IMPORT_SEPARATOR = '\\-----'


def format_imported_turn(prompt: str, response: str, reasoning: str | None = None) -> str:
    """One turn of an imported conversation: the prompt, a `-----` line, then the answer.

    The answer is laid out as a reply is, its chain of thought quoted above it, so the quote
    is the first one in the message and the renderer folds it. `balance_fences` keeps a fence
    left open in the prompt or the reasoning from swallowing what follows.
    """
    answer = format_reply(reasoning or '', response, True)
    return balance_fences(normalize_math(prompt)) + '\n\n' + IMPORT_SEPARATOR + '\n\n' + answer


def format_navigation_page(
    turns: list[ThreadTurn], page: int, per_page: int, width: int, link: Callable[[int], str | None]
) -> tuple[str, int, int]:
    """One page of a thread's turns as HTML, with the page it settled on and the page count.

    *page* is clamped, so a stale button on a thread that has since shrunk still lands somewhere.
    Each turn is two preview lines; the answer is a link to its message where the chat has links.
    """
    pages = max(1, -(-len(turns) // per_page))
    page = min(max(page, 0), pages - 1)
    lines: list[str] = []
    for number, turn in enumerate(turns[page * per_page : (page + 1) * per_page], start=page * per_page + 1):
        lines.append(f'<b>{number}.</b> 👤 {html_escape(preview_line(turn.prompt, width))}')
        answer = html_escape(preview_line(turn.response, width)) if turn.response is not None else '<i>…</i>'
        url = link(turn.msg_id)
        if url is not None:
            answer = f'<a href="{html_escape(url)}">{answer}</a>'
        lines.append(f'      🤖 {answer}')
    return '\n'.join(lines), page, pages


def format_linearize_listing(nodes: dict[int, MsgInfo], delete: set[int], width: int, limit: int) -> tuple[str, int]:
    """What a /linearize would delete, as HTML lines, with the number of messages it comes to.

    Only real messages are listed: the negative id of an imported prompt was never one, and
    its text is inside the message that showed its turn. A split answer is one line with its
    part count, since every part holds the same stored text. At most *limit* lines are shown.
    """
    groups: dict[tuple[bool, int | None, int], list[int]] = {}
    for mid in sorted(n for n in delete if n > 0):
        info = nodes[mid]
        # parts of one answer share their prompt; anything else stands alone
        key = (True, info.reply_id, 0) if info.sent_by_bot else (False, None, mid)
        groups.setdefault(key, []).append(mid)
    lines: list[str] = []
    for (is_bot, _, _), mids in sorted(groups.items(), key=lambda item: item[1][0]):
        info = nodes[mids[0]]
        text = text_of(info.message)
        line = f'{"🤖" if is_bot else "👤"} {html_escape(preview_line(text, width))}'
        if len(mids) > 1:
            line += f' <i>({len(mids)} messages)</i>'
        lines.append(line)
    if len(lines) > limit:
        lines = lines[:limit] + [f'<i>…and {len(lines) - limit} more</i>']
    return '\n'.join(lines), sum(len(m) for m in groups.values())
