"""Saving a thread in the format /import reads."""

import asyncio
import datetime
import json
from types import SimpleNamespace
from typing import Any, final

import pytest
from telethon.tl.types import MessageReplyHeader

from chatgpt_telegram_bot.bot import ChatGPTTelegramBot
from chatgpt_telegram_bot.models import (
    MsgInfo,
    make_image_part,
    make_reasoning_part,
    make_text_part,
    make_thinking_text_part,
)
from chatgpt_telegram_bot.store import Store
from chatgpt_telegram_bot.transfer import export_filename, format_export, parse_import
from chatgpt_telegram_bot.utils import PendingReplyManager

CHAT = -1001234567890
WHEN = datetime.datetime(2026, 1, 2, 3, 4, 5, 678000, tzinfo=datetime.UTC)


# --- format_export and export_filename ------------------------------------------------------


def test_the_export_has_the_import_shape():
    out = json.loads(format_export('T', [('user', 'q', WHEN, None), ('assistant', 'a', WHEN, None)], WHEN))
    assert out == {
        'title': 'T',
        'exportedAt': '2026-01-02T03:04:05.678Z',
        'messages': [
            {'role': 'user', 'content': 'q', 'timestamp': '2026-01-02T03:04:05.678Z'},
            {'role': 'assistant', 'content': 'a', 'timestamp': '2026-01-02T03:04:05.678Z'},
        ],
    }


def test_a_message_with_no_known_time_has_no_timestamp():
    out = json.loads(format_export('T', [('user', 'q', None, None), ('assistant', 'a', WHEN, None)], WHEN))
    assert 'timestamp' not in out['messages'][0]


def test_non_ascii_is_written_as_is():
    assert '标题' in format_export('标题', [('user', 'q', None, None), ('assistant', 'a', None, None)], WHEN)


def test_a_timestamp_in_another_zone_is_written_in_utc():
    tokyo = datetime.datetime(2026, 1, 2, 12, 4, 5, tzinfo=datetime.timezone(datetime.timedelta(hours=9)))
    out = json.loads(format_export('T', [('user', 'q', tokyo, None), ('assistant', 'a', None, None)], WHEN))
    assert out['messages'][0]['timestamp'] == '2026-01-02T03:04:05.000Z'


@pytest.mark.parametrize(
    ('title', 'filename'),
    [
        ('Reading list', 'Reading list.json'),
        ('a/b:c*d?"e"<f>|g', 'a b c d e f g.json'),
        ('  spaced   out  ', 'spaced out.json'),
        ('', 'thread.json'),
        ('...', 'thread.json'),
        ('标题', '标题.json'),
    ],
)
def test_filenames_are_safe(title: str, filename: str):
    assert export_filename(title) == filename


def test_a_long_title_makes_a_bounded_filename():
    assert len(export_filename('x' * 300)) <= 60 + len('.json')


# --- the handler -------------------------------------------------------------------------------


@final
class FakeMessage:
    def __init__(self, target: int | None) -> None:
        self.chat_id: int = CHAT
        self.sender_id: int = 1
        self.id: int = 900
        self.message: str = '/export'
        self.reply_to: Any = MessageReplyHeader(reply_to_msg_id=target) if target is not None else None


def _bot(dates_fail: bool = False) -> Any:
    bot = object.__new__(ChatGPTTelegramBot)
    bot.store = Store(':memory:')
    bot.allowed_chats = {CHAT}
    bot.pending_reply_manager = PendingReplyManager()
    sent: list[Any] = []
    bot.sent = sent  # pyright: ignore[reportAttributeAccessIssue]  # test-only capture

    async def send_message(_chat_id: int, body: Any, _reply_to: int) -> int:
        sent.append(('message', body))
        return 999

    async def send_document(_chat_id: int, data: bytes, filename: str, caption: str, _reply_to: int) -> int:
        sent.append(('document', data, filename, caption))
        return 999

    async def get_messages(_chat_id: int, ids: list[int]) -> list[Any]:
        if dates_fail:
            raise ConnectionError('telegram is down')
        return [SimpleNamespace(id=i, date=WHEN + datetime.timedelta(minutes=i)) for i in ids]

    bot.send_message = send_message  # pyright: ignore[reportAttributeAccessIssue]  # test-only stub
    bot.send_document = send_document  # pyright: ignore[reportAttributeAccessIssue]  # test-only stub
    bot.bot = SimpleNamespace(get_messages=get_messages)  # pyright: ignore[reportAttributeAccessIssue]  # test-only stub
    return bot


def _thread(bot: Any, turns: int) -> int:
    reply_id: int | None = None
    for i in range(1, turns + 1):
        prompt_id, answer_id = 2 * i - 1, 2 * i
        bot.set_msg_info(
            CHAT, prompt_id, MsgInfo(False, [make_text_part(f'prompt {i}')], reply_id, 'g' if i == 1 else None, None)
        )
        bot.set_msg_info(CHAT, answer_id, MsgInfo(True, [make_text_part(f'answer {i}')], prompt_id, None, None))
        reply_id = answer_id
    assert reply_id is not None
    return reply_id


def _export(bot: Any, target: int | None, body: str = '') -> Any:
    asyncio.run(bot.export_handler(FakeMessage(target), body))
    return bot


def _document(bot: Any) -> tuple[dict[str, Any], str, str]:
    kind, data, filename, caption = bot.sent[0]
    assert kind == 'document'
    return json.loads(data), filename, caption


def test_the_thread_exports_in_order_with_timestamps():
    bot = _bot()
    _ = _export(bot, _thread(bot, 2), 'My chat')
    doc, filename, caption = _document(bot)
    assert doc['title'] == 'My chat'
    assert [(m['role'], m['content']) for m in doc['messages']] == [
        ('user', 'prompt 1'),
        ('assistant', 'answer 1'),
        ('user', 'prompt 2'),
        ('assistant', 'answer 2'),
    ]
    assert doc['messages'][0]['timestamp'] == '2026-01-02T03:05:05.678Z'
    assert filename == 'My chat.json'
    assert caption == 'Exported "My chat" — 4 messages'


def test_the_export_imports_back_to_the_same_turns():
    bot = _bot()
    _ = _export(bot, _thread(bot, 3), 'Round trip')
    _, data, _, _ = bot.sent[0]
    chat = parse_import(data.decode())
    assert chat.title == 'Round trip'
    assert chat.turns == [(f'prompt {i}', f'answer {i}', None) for i in range(1, 4)]


def test_without_a_title_the_first_prompt_names_it():
    bot = _bot()
    _ = _export(bot, _thread(bot, 1))
    doc, _, _ = _document(bot)
    assert doc['title'] == 'prompt 1'


def test_a_plain_chain_of_thought_is_exported_as_reasoning_and_ciphertext_is_not():
    bot = _bot()
    bot.set_msg_info(CHAT, 1, MsgInfo(False, [make_text_part('q')], None, 'g', None))
    item = {'type': 'reasoning', 'id': 'rs_1', 'encrypted_content': 'opaque'}
    parts = [make_reasoning_part('gpt-5', item), make_thinking_text_part('my reasoning'), make_text_part('the answer')]
    bot.set_msg_info(CHAT, 2, MsgInfo(True, parts, 1, None, None))
    _ = _export(bot, 2)
    doc, _, _ = _document(bot)
    assert doc['messages'][1]['content'] == 'the answer'
    assert doc['messages'][1]['reasoning'] == 'my reasoning'
    assert 'opaque' not in json.dumps(doc)
    assert 'reasoning' not in doc['messages'][0]


def test_a_reply_without_a_chain_of_thought_has_no_reasoning_field():
    out = json.loads(format_export('T', [('user', 'q', None, None), ('assistant', 'a', None, '')], WHEN))
    assert 'reasoning' not in out['messages'][1]


def test_reasoning_survives_a_round_trip():
    bot = _bot()
    bot.set_msg_info(CHAT, 1, MsgInfo(False, [make_text_part('q')], None, 'g', None))
    bot.set_msg_info(CHAT, 2, MsgInfo(True, [make_thinking_text_part('why'), make_text_part('a')], 1, None, None))
    _ = _export(bot, 2)
    _, data, _, _ = bot.sent[0]
    assert parse_import(data.decode()).turns == [('q', 'a', 'why')]


def test_an_image_prompt_keeps_a_marker():
    bot = _bot()
    bot.set_msg_info(CHAT, 1, MsgInfo(False, [make_text_part('look'), make_image_part('ab')], None, 'g', None))
    bot.set_msg_info(CHAT, 2, MsgInfo(True, [make_text_part('a cat')], 1, None, None))
    _ = _export(bot, 2)
    doc, _, _ = _document(bot)
    assert doc['messages'][0]['content'] == '[image]\n\nlook'


def test_an_imported_prompt_has_no_timestamp():
    # the user half of an imported turn is stored under a negative id that was never a message
    bot = _bot()
    bot.set_msg_info(CHAT, -50, MsgInfo(False, [make_text_part('q')], None, None, None))
    bot.set_msg_info(CHAT, 50, MsgInfo(True, [make_text_part('a')], -50, None, None))
    _ = _export(bot, 50)
    doc, _, _ = _document(bot)
    assert 'timestamp' not in doc['messages'][0]
    assert 'timestamp' in doc['messages'][1]


def test_a_failed_date_lookup_still_exports():
    bot = _bot(dates_fail=True)
    _ = _export(bot, _thread(bot, 1))
    doc, _, _ = _document(bot)
    assert all('timestamp' not in m for m in doc['messages'])


def test_it_needs_a_reply():
    bot = _export(_bot(), None)
    assert bot.sent[0][0] == 'message'
    assert bot.sent[0][1].startswith('[!] `/export`')


def test_an_unstored_thread_is_reported():
    bot = _export(_bot(), 404)
    assert bot.sent[0] == ('message', '[!] No stored conversation for that message')


def test_a_thread_with_a_missing_link_is_refused():
    # exported, it would start mid-conversation, which /import rejects
    bot = _bot()
    bot.set_msg_info(CHAT, 3, MsgInfo(False, [make_text_part('q')], 2, None, None))
    bot.set_msg_info(CHAT, 4, MsgInfo(True, [make_text_part('a')], 3, None, None))
    _ = _export(bot, 4)
    assert bot.sent[0] == ('message', '[!] Part of that thread is no longer stored')
