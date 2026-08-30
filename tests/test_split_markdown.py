"""Tests for splitting markdown source across several rich messages."""

from __future__ import annotations

from chatgpt_telegram_bot.utils import split_markdown, telegram_len


def test_short_text_is_one_part():
    assert split_markdown('hello', 100, 100) == ['hello']


def test_empty_text_yields_nothing():
    assert split_markdown('', 100, 100) == []


def test_respects_limits():
    text = '\n\n'.join(f'para {i} ' + 'x' * 30 for i in range(10))
    parts = split_markdown(text, 100, 100)
    assert all(len(p) <= 100 for p in parts)


def test_first_part_uses_first_limit():
    text = '\n\n'.join('x' * 20 for _ in range(6))
    parts = split_markdown(text, 30, 100)
    assert len(parts[0]) <= 30


def test_prefers_paragraph_break():
    text = 'a' * 40 + '\n\n' + 'b' * 40
    assert split_markdown(text, 50, 50) == ['a' * 40, 'b' * 40]


def test_falls_back_to_line_break():
    text = 'a' * 40 + '\n' + 'b' * 40
    assert split_markdown(text, 50, 50) == ['a' * 40, 'b' * 40]


def test_hard_cut_when_no_break_available():
    text = 'a' * 120
    parts = split_markdown(text, 50, 50)
    assert parts == ['a' * 50, 'a' * 50, 'a' * 20]


def test_reopens_fenced_code_block():
    body = '\n'.join(f'line {i}' for i in range(20))
    text = f'```python\n{body}\n```'
    parts = split_markdown(text, 60, 60)
    assert len(parts) > 1
    assert parts[0].endswith('```')
    assert parts[1].startswith('```python')
    # every part is balanced on its own
    for p in parts:
        assert len([ln for ln in p.split('\n') if ln.lstrip().startswith('```')]) % 2 == 0


def test_closed_fence_is_not_reopened():
    text = '```\nshort\n```\n\n' + 'x' * 80
    parts = split_markdown(text, 40, 40)
    assert not parts[1].startswith('```')


# --- regressions ---------------------------------------------------------


def test_unclosed_fence_with_long_first_line_terminates():
    """A cut landing on the fence opener used to reproduce the input and loop forever."""
    parts = split_markdown('```json\n' + 'x' * 300, 100, 100)
    assert len(parts) > 1
    assert ''.join(p for p in parts).count('x') == 300
    for p in parts:
        assert telegram_len(p) <= 100


def test_no_break_and_open_fence_still_terminates():
    parts = split_markdown('```\n' + 'y' * 500, 40, 40)
    assert sum(p.count('y') for p in parts) == 500
    for p in parts:
        assert telegram_len(p) <= 40


def test_limits_are_utf16_units():
    """Astral characters count as two units, matching Telegram's own counting."""
    parts = split_markdown('🤖' * 200, 100, 100)
    assert len(parts) > 1
    for p in parts:
        assert telegram_len(p) <= 100


def test_blockquoted_fence_is_balanced():
    body = '\n'.join(f'>line {i}' for i in range(20))
    text = '>```python\n' + body + '\n>```'
    parts = split_markdown(text, 80, 80)
    assert len(parts) > 1
    for p in parts:
        fences = [ln for ln in p.split('\n') if ln.lstrip('>').lstrip().startswith('```')]
        assert len(fences) % 2 == 0, p
    # the reopened fence stays inside the blockquote
    assert parts[1].startswith('>```python')
