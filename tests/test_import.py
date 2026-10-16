"""Reading an exported conversation and laying it out as a reply chain."""

import asyncio
import json
from types import SimpleNamespace
from typing import Any, final
from zoneinfo import ZoneInfo

import pytest

from chatgpt_telegram_bot.bot import ChatGPTTelegramBot
from chatgpt_telegram_bot.models import Model, MsgInfo, make_text_part
from chatgpt_telegram_bot.utils import parse_import

TWO_TURNS: dict[str, Any] = {
    'title': 'A chat',
    'messages': [
        {'role': 'user', 'content': 'first prompt'},
        {'role': 'assistant', 'content': 'first response'},
        {'role': 'user', 'content': 'second prompt'},
        {'role': 'assistant', 'content': 'second response'},
    ],
}


def test_turns_are_paired_up():
    chat = parse_import(json.dumps(TWO_TURNS))
    assert chat.title == 'A chat'
    assert chat.turns == [('first prompt', 'first response'), ('second prompt', 'second response')]
    assert chat.trailing_prompt is None


def test_a_prompt_with_no_response_is_reported_not_dropped_silently():
    payload = {'messages': TWO_TURNS['messages'] + [{'role': 'user', 'content': 'unanswered'}]}
    chat = parse_import(json.dumps(payload))
    assert len(chat.turns) == 2
    assert chat.trailing_prompt == 'unanswered'


def test_a_missing_title_gets_a_placeholder():
    assert parse_import(json.dumps({'messages': TWO_TURNS['messages']})).title == 'Untitled'


def test_a_multiline_title_is_flattened():
    payload = {'title': 'two\nlines', 'messages': TWO_TURNS['messages']}
    assert parse_import(json.dumps(payload)).title == 'two lines'


@pytest.mark.parametrize(
    ('payload', 'message'),
    [
        ('{not json', 'not valid JSON'),
        ('[]', 'JSON object'),
        ('{}', 'no "messages" array'),
        ('{"messages": []}', 'no "messages" array'),
        ('{"messages": [{"role": "assistant", "content": "a"}]}', 'Message 1 is assistant, expected user'),
        ('{"messages": [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]}', 'Message 2 is user'),
        ('{"messages": [{"role": "system", "content": "a"}]}', "unsupported role 'system'"),
        ('{"messages": [{"role": "user"}]}', 'Message 1 has no text content'),
        ('{"messages": ["nope"]}', 'Message 1 is not an object'),
        ('{"messages": [{"role": "user", "content": "a"}]}', 'no complete prompt-and-response turn'),
    ],
)
def test_a_malformed_file_says_what_is_wrong(payload: str, message: str):
    with pytest.raises(ValueError, match=message):
        _ = parse_import(payload)


def _bot() -> Any:
    """A bot with just the state import_handler and construct_chat_history read."""
    bot = object.__new__(ChatGPTTelegramBot)
    bot.db = {}  # pyright: ignore[reportAttributeAccessIssue]  # a dict is enough of a shelf here
    bot.models = [Model(prefix='g', name='gpt-5'), Model(prefix='c', name='claude')]
    bot.system_prompt = 'be helpful'
    bot.timezone = ZoneInfo('UTC')
    bot.TEXT_FILE_SIZE_LIMIT = 900_000
    # the real pacing would make these tests sleep for minutes; the schedule itself is
    # covered in test_throttle.py, so here the intervals are small enough to be invisible
    bot.IMPORT_MIN_INTERVAL = 0.001
    bot.IMPORT_BURST_INTERVAL = 0.001
    bot.IMPORT_BURST_WINDOW = 0.001
    bot.allowed_chats = {1}
    sent: list[tuple[int, Any, int | None]] = []
    bot.sent = sent  # pyright: ignore[reportAttributeAccessIssue]  # test-only capture
    next_id = [100]

    async def send_message(_chat_id: int, body: Any, reply_to: int | None = None) -> int:
        next_id[0] += 1
        sent.append((next_id[0], body, reply_to))
        return next_id[0]

    bot.send_message = send_message  # pyright: ignore[reportAttributeAccessIssue]  # test-only stub
    return bot


def _import(payload: Any, arg: str = '') -> Any:
    bot = _bot()
    blob = json.dumps(payload).encode() if not isinstance(payload, str) else payload.encode()

    @final
    class Message:
        chat_id = 1
        id = 100
        sender_id = 1
        is_reply = False
        document = SimpleNamespace(size=len(blob))

        async def download_media(self, _: Any) -> bytes:
            return blob

    asyncio.run(bot.import_handler(Message(), arg))
    return bot


def _continue_from(bot: Any, msg_id: int, reply_msg_id: int = 900, prefix: str | None = None) -> Any:
    """Store a user reply to *msg_id* and resolve the history it would be sent with."""
    bot.set_msg_info(
        1, reply_msg_id, MsgInfo(False, [make_text_part('follow up')], msg_id, prefix, 'switched' if prefix else None)
    )
    return bot.construct_chat_history(1, reply_msg_id)


def test_the_note_comes_first_then_one_message_per_turn():
    bot = _import(TWO_TURNS)
    note, *turns = bot.sent
    assert note[1].startswith('Importing a chat titled "A chat" — 2 turns')
    assert len(turns) == 2


def test_each_turn_shows_the_prompt_and_the_response():
    bot = _import(TWO_TURNS, 'g')
    first = bot.sent[1][1].text
    assert 'first prompt' in first
    assert 'first response' in first
    assert '(1/2)' in first


def test_replying_to_a_turn_continues_from_that_point():
    bot = _import(TWO_TURNS, 'g')
    first_turn_id = bot.sent[1][0]
    history, model, _ = _continue_from(bot, first_turn_id)
    # the first turn plus the new prompt, and nothing from the second turn
    assert [part[0].text for part in history] == ['first prompt', 'first response', 'follow up']
    assert model.name == 'gpt-5'


def test_replying_to_the_last_turn_carries_the_whole_chat():
    bot = _import(TWO_TURNS, 'g')
    history, _, _ = _continue_from(bot, bot.sent[-1][0])
    assert [part[0].text for part in history] == [
        'first prompt',
        'first response',
        'second prompt',
        'second response',
        'follow up',
    ]


def test_every_part_of_a_split_turn_continues_from_the_same_point():
    payload = {
        'title': 'Long',
        'messages': [
            {'role': 'user', 'content': 'prompt one'},
            {'role': 'assistant', 'content': 'A' * 9000},
            {'role': 'user', 'content': 'prompt two'},
            {'role': 'assistant', 'content': 'short'},
        ],
    }
    bot = _import(payload, 'g')
    turn_ids = [m for m, _, _ in bot.sent[1:]]
    assert len(turn_ids) > 2, 'the long response should have been split'
    lengths = [len(_continue_from(bot, mid, 900 + n)[0]) for n, mid in enumerate(turn_ids)]
    # every part of turn one resolves to the same history; only the last message is turn two
    assert lengths[:-1] == [3] * (len(turn_ids) - 1)
    assert lengths[-1] == 5


def test_the_imported_chain_records_the_model_and_its_prompt():
    bot = _import(TWO_TURNS, 'g,t=high')
    _, model, system_prompt = _continue_from(bot, bot.sent[1][0])
    assert model.name == 'gpt-5'
    assert model.thinking == 'high'
    assert system_prompt.startswith('be helpful')


def test_a_prefix_on_the_reply_switches_the_model():
    bot = _import(TWO_TURNS, 'g')
    _, model, system_prompt = _continue_from(bot, bot.sent[1][0], prefix='c')
    assert model.name == 'claude'
    assert system_prompt == 'switched'


def test_an_import_with_no_prefix_records_no_model():
    bot = _import(TWO_TURNS)
    assert bot.sent[0][1].endswith('reply to a turn with a model prefix to continue it')
    # nothing in the chain names a model, and the turn header cannot show one
    assert bot.get_msg_info(1, -bot.sent[1][0]).prefix is None
    assert bot.get_msg_info(1, -bot.sent[1][0]).system_prompt is None
    assert '(1/2)' in bot.sent[1][1].text
    assert 'gpt-5' not in bot.sent[1][1].text


def test_continuing_a_modelless_import_without_a_prefix_is_reported():
    # it used to be an assert, which escaped the handler and left the user with no reply
    bot = _import(TWO_TURNS)
    with pytest.raises(RuntimeError, match='no model for this conversation'):
        _ = _continue_from(bot, bot.sent[1][0])


def test_a_prefix_on_the_reply_picks_the_model_for_a_modelless_import():
    bot = _import(TWO_TURNS)
    history, model, system_prompt = _continue_from(bot, bot.sent[1][0], prefix='c')
    assert model.name == 'claude'
    assert system_prompt == 'switched'
    assert [part[0].text for part in history] == ['first prompt', 'first response', 'follow up']


def test_an_unknown_prefix_is_refused_before_anything_is_sent():
    bot = _import(TWO_TURNS, 'nope')
    assert bot.sent == [(101, '[!] Unknown prefix: `nope`', 100)]


def test_a_malformed_file_imports_nothing():
    bot = _import('{"messages": [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]}')
    assert len(bot.sent) == 1
    assert bot.sent[0][1].startswith('[!] Message 2 is user')
