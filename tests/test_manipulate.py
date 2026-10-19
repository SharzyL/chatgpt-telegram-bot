from chatgpt_telegram_bot.completion import drop_thinking_text, thinking_text_of
from chatgpt_telegram_bot.models import (
    MsgPartInHistory,
    make_image_part,
    make_reasoning_part,
    make_text_part,
    make_thinking_text_part,
)
from chatgpt_telegram_bot.utils import parse_manipulation


def test_plain_body_is_all_text():
    assert parse_manipulation('user provided text') == (None, 'user provided text')


def test_think_block_is_split_off():
    body = '<think>\nuser provided cot\n</think>\n\nuser provided text'
    assert parse_manipulation(body) == ('user provided cot', 'user provided text')


def test_think_block_keeps_inner_newlines():
    thinking, text = parse_manipulation('<think>\nline one\n\nline two\n</think>\nanswer')
    assert thinking == 'line one\n\nline two'
    assert text == 'answer'


def test_leading_whitespace_before_think_tolerated():
    assert parse_manipulation('\n\n<think>cot</think>\n\ntext') == ('cot', 'text')


def test_think_block_not_at_start_is_left_in_text():
    body = 'preamble <think>cot</think> text'
    assert parse_manipulation(body) == (None, body)


def test_empty_think_block_yields_no_thinking():
    assert parse_manipulation('<think></think>\n\ntext') == (None, 'text')


def test_body_with_only_a_think_block_has_empty_text():
    assert parse_manipulation('<think>cot</think>') == ('cot', '')


def test_unclosed_think_block_is_text():
    assert parse_manipulation('<think>cot') == (None, '<think>cot')


def test_drop_removes_only_thinking_text():
    item = {'type': 'reasoning', 'id': 'rs_1', 'encrypted_content': 'blob'}
    parts = [make_reasoning_part('m', item), make_thinking_text_part('cot'), make_text_part('answer')]
    assert [p.type_ for p in drop_thinking_text(parts)] == ['reasoning', 'text']
    assert drop_thinking_text(parts)[1] == make_text_part('answer'), 'the text is left as it was, no <think> block'


def test_drop_is_a_noop_without_thinking_text():
    parts = [make_text_part('answer'), make_image_part('deadbeef')]
    assert drop_thinking_text(parts) == parts


def test_thinking_text_of_joins_and_skips_blanks():
    blank = MsgPartInHistory(type_='thinking_text', hash=None, text=None)
    parts = [make_thinking_text_part('one'), blank, make_text_part('answer'), make_thinking_text_part('two')]
    assert thinking_text_of(parts) == 'one\n\ntwo'
    assert thinking_text_of([make_text_part('answer')]) == ''
