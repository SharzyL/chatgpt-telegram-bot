"""A code fence the model leaves open must not swallow what follows it."""

from typing import Any

import mistune

from chatgpt_telegram_bot.render import SERVER_MARKDOWN_RENDERER
from chatgpt_telegram_bot.reply import format_reply

_parse = mistune.create_markdown(renderer='ast')


def _ast(markdown: str) -> list[dict[str, Any]]:
    nodes = _parse(markdown)
    assert isinstance(nodes, list)
    return nodes


# a chain of thought that opens a fence and never closes it, long enough to be split
UNCLOSED_COT = 'Let me think.\n\n```python\ndef f(x):\n    return x\n\n' + '\n\n'.join(
    f'Step {i}: ' + 'w' * 300 for i in range(107)
)


def _swallowed(node: dict[str, Any], quote: bool = False, code: bool = False) -> bool:
    """Whether the answer ended up inside a blockquote or a code block."""
    if node.get('type') == 'block_quote':
        quote = True
    if node.get('type') == 'block_code':
        code = True
    if 'ANSWERTEXT' in str(node.get('raw', '')) and (quote or code):
        return True
    return any(_swallowed(c, quote, code) for c in (node.get('children') or []))


def _bodies(thinking: str, reply: str, **kw: Any) -> list[str]:
    markdown = format_reply(thinking, reply, True, **kw)
    return [b.rich_markdown or '' for b in SERVER_MARKDOWN_RENDERER.render(markdown, 'prefix\n', True)]


def test_unclosed_thinking_fence_does_not_quote_the_answer():
    bodies = _bodies(UNCLOSED_COT, 'ANSWERTEXT is the answer.', usage={'in': 1})
    assert len(bodies) > 1, 'the reply should be long enough to split'
    for body in bodies:
        fences = [ln for ln in body.split('\n') if ln.lstrip('>').lstrip().startswith('```')]
        assert len(fences) % 2 == 0, body[:200]
        assert not any(_swallowed(n) for n in _ast(body)), body[:200]


def test_thinking_fence_is_closed_inside_the_quote():
    markdown = format_reply('open it\n\n```py\nx = 1', 'the answer', True)
    quote, rest = markdown.split('\n\n', 1)
    assert quote.endswith('>```')
    assert rest == 'the answer'


def test_unclosed_reply_fence_does_not_swallow_the_footer():
    markdown = format_reply('', 'ANSWERTEXT\n\n```py\nx = 1', True, usage={'in': 1})
    assert '\n```\n\n>**Usage**' in markdown


def test_streaming_thinking_keeps_the_status_out_of_the_code_block():
    markdown = format_reply('thinking\n\n```py\nx = 1', '', False, status='searching')
    assert markdown.endswith('>``` \\[!searching]')


def test_balanced_fences_are_left_alone():
    markdown = format_reply('a\n\n```py\nx = 1\n```', 'the answer', True)
    assert markdown == '>a\n>\n>```py\n>x = 1\n>```\n\nthe answer'
