"""Tests for rewriting LaTeX math delimiters into the forms Telegram parses."""

from __future__ import annotations

from chatgpt_telegram_bot.utils import normalize_math


def test_inline_paren_becomes_dollar():
    assert normalize_math(r'see \(e^{i\pi}\) here') == r'see $e^{i\pi}$ here'


def test_display_bracket_becomes_double_dollar():
    assert normalize_math(r'\[\int_0^1 x\]') == r'$$\int_0^1 x$$'


def test_existing_dollar_math_untouched():
    assert normalize_math('$a+b$ and $$c$$') == '$a+b$ and $$c$$'


def test_multiple_occurrences():
    assert normalize_math(r'\(a\) and \(b\)') == '$a$ and $b$'


def test_spans_newlines():
    assert normalize_math('\\[a\n+b\\]') == '$$a\n+b$$'


def test_inline_code_untouched():
    assert normalize_math(r'`\(not math\)`') == r'`\(not math\)`'


def test_fenced_code_untouched():
    text = 'text \\(x\\)\n```\n\\(y\\)\n```\nmore \\(z\\)'
    out = normalize_math(text)
    assert '```\n\\(y\\)\n```' in out
    assert '$x$' in out and '$z$' in out


def test_plain_text_unchanged():
    assert normalize_math('nothing to do here') == 'nothing to do here'


def test_unclosed_delimiter_left_alone():
    assert normalize_math(r'partial \(a') == r'partial \(a'
