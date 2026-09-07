import asyncio
import json
from typing import Any

from chatgpt_telegram_bot.completion import build_input_openai, has_reasoning_items, strip_reasoning_items
from chatgpt_telegram_bot.models import (
    MsgPartInHistory,
    make_image_part,
    make_reasoning_part,
    make_text_part,
)

MODEL = 'grok-4.5'
ITEM: dict[str, Any] = {'type': 'reasoning', 'id': 'rs_1', 'summary': [], 'encrypted_content': 'blob'}


async def _no_image(_key: str) -> bytes | None:
    return b'\xff\xd8jpeg'


def build(history: list[list[MsgPartInHistory]], model_name: str = MODEL, replay: bool = True) -> list[dict[str, Any]]:
    return asyncio.run(build_input_openai(history, _no_image, model_name, replay))


def test_plain_history_collapses_to_strings():
    items = build([[make_text_part('hi')], [make_text_part('hello')], [make_text_part('again')]])
    assert items == [
        {'role': 'user', 'content': 'hi'},
        {'role': 'assistant', 'content': 'hello'},
        {'role': 'user', 'content': 'again'},
    ]


def test_reasoning_replayed_before_its_message():
    history = [
        [make_text_part('hi')],
        [make_reasoning_part(MODEL, ITEM), make_text_part('hello')],
        [make_text_part('again')],
    ]
    assert build(history) == [
        {'role': 'user', 'content': 'hi'},
        ITEM,
        {'role': 'assistant', 'content': 'hello'},
        {'role': 'user', 'content': 'again'},
    ]


def test_multiple_reasoning_items_keep_order():
    second = {**ITEM, 'id': 'rs_2'}
    history = [
        [make_text_part('hi')],
        [make_reasoning_part(MODEL, ITEM), make_reasoning_part(MODEL, second), make_text_part('hello')],
    ]
    assert build(history)[1:3] == [ITEM, second]


def test_reasoning_dropped_when_replay_disabled():
    history = [[make_text_part('hi')], [make_reasoning_part(MODEL, ITEM), make_text_part('hello')]]
    assert build(history, replay=False) == [
        {'role': 'user', 'content': 'hi'},
        {'role': 'assistant', 'content': 'hello'},
    ]


def test_reasoning_of_another_model_dropped():
    history = [[make_text_part('hi')], [make_reasoning_part('other-model', ITEM), make_text_part('hello')]]
    assert not has_reasoning_items(build(history))


def test_malformed_reasoning_part_dropped():
    broken = MsgPartInHistory(type_='reasoning', hash=None, text='{not json')
    no_item = MsgPartInHistory(type_='reasoning', hash=None, text=json.dumps({'model': MODEL}))
    empty = MsgPartInHistory(type_='reasoning', hash=None, text=None)
    for part in (broken, no_item, empty):
        items = build([[make_text_part('hi')], [part, make_text_part('hello')]])
        assert not has_reasoning_items(items)
        assert items[-1] == {'role': 'assistant', 'content': 'hello'}


def test_image_part_stays_content_and_blocks_collapse():
    items = build([[make_text_part('look'), make_image_part('deadbeef')]])
    assert items[0]['role'] == 'user'
    content = items[0]['content']
    assert [c['type'] for c in content] == ['input_text', 'input_image']
    assert content[1]['image_url'].startswith('data:image/jpeg;base64,')


def test_assistant_multipart_uses_output_text():
    history = [[make_text_part('hi')], [make_text_part('a'), make_text_part('b')]]
    content = build(history)[1]['content']
    assert [c['type'] for c in content] == ['output_text', 'output_text']


def test_strip_reasoning_items_leaves_messages():
    items = build([[make_text_part('hi')], [make_reasoning_part(MODEL, ITEM), make_text_part('hello')]])
    assert has_reasoning_items(items)
    stripped = strip_reasoning_items(items)
    assert not has_reasoning_items(stripped)
    assert stripped == [{'role': 'user', 'content': 'hi'}, {'role': 'assistant', 'content': 'hello'}]
