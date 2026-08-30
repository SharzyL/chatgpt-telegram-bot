"""Tests for UTF-16 length accounting, the unit Telegram counts messages in."""

from __future__ import annotations

from chatgpt_telegram_bot.utils import telegram_len, telegram_truncate


def test_len_counts_astral_as_two():
    assert telegram_len('a') == 1
    assert telegram_len('🤖') == 2
    assert telegram_len('🤖a') == 3


def test_truncate_is_a_noop_when_short_enough():
    assert telegram_truncate('hello', 100) == 'hello'


def test_truncate_counts_in_utf16_units():
    assert telegram_truncate('🤖' * 10, 10) == '🤖' * 5
    assert telegram_len(telegram_truncate('🤖' * 10, 10)) == 10


def test_truncate_never_splits_a_surrogate_pair():
    # an odd budget must drop the whole character rather than half of it
    out = telegram_truncate('🤖' * 10, 9)
    assert out == '🤖' * 4
    assert telegram_len(out) <= 9


def test_truncate_respects_mixed_content():
    s = 'ab🤖cd'
    assert telegram_truncate(s, 4) == 'ab🤖'
    assert telegram_truncate(s, 3) == 'ab'
