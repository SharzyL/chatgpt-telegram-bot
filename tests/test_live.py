"""Live threads: editing a prompt answers it again, over the reply that is there."""

import asyncio
from types import SimpleNamespace
from typing import Any, final
from zoneinfo import ZoneInfo

from telethon.tl.types import MessageReplyHeader

from chatgpt_telegram_bot.bot import ChatGPTTelegramBot
from chatgpt_telegram_bot.commands import apply_overrides
from chatgpt_telegram_bot.models import EndPoint, Model, MsgInfo, make_text_part
from chatgpt_telegram_bot.store import Store
from chatgpt_telegram_bot.utils import PendingReplyManager

BOT_ID = 7
CHAT = -100
PROMPT = 10
REPLY = [11, 12]


@final
class FakeMessage:
    def __init__(self, text: str, msg_id: int = PROMPT, sender_id: int = 1, reply_to: int | None = None) -> None:
        self.chat_id: int = CHAT
        self.id: int = msg_id
        self.sender_id: int = sender_id
        self.message: str = text
        self.photo: Any = None
        self.document: Any = None
        self.grouped_id: Any = None
        self.reply_to: Any = MessageReplyHeader(reply_to_msg_id=reply_to) if reply_to is not None else None

    async def get_reply_message(self) -> Any:
        return None


def _bot(live: bool = True, answered: bool = True) -> Any:
    bot = object.__new__(ChatGPTTelegramBot)
    bot.store = Store(':memory:')
    bot.models = [Model(prefix='g', name='gpt-5'), Model(prefix='c', name='claude')]
    bot.endpoints = [EndPoint(name='main', url='https://main.test')]
    bot.endpoint_by_name = {e.name: e for e in bot.endpoints}
    bot.default_endpoint = 'main'
    bot.clients = {}

    def make_client(ep_name: str, api_type: str, suffix: str) -> Any:
        return SimpleNamespace(triple=(ep_name, api_type, suffix))

    bot._make_client = make_client  # pyright: ignore[reportPrivateUsage]  # test-only stub
    bot.system_prompt = 'be helpful'
    bot.timezone = ZoneInfo('UTC')
    bot.allowed_chats = {CHAT}
    bot.bot_id = BOT_ID
    bot.username = 'the_bot'
    bot.default_image_prompt = 'Describe the image'
    bot.default_image_reply_prompt = 'Continue'
    bot.pending_reply_manager = PendingReplyManager()
    bot.generating = {}
    bot.generation_done = {}
    bot.streaming = {}
    sent: list[tuple[str, int]] = []
    runs: list[dict[str, Any]] = []
    bot.sent = sent  # pyright: ignore[reportAttributeAccessIssue]  # test-only capture
    bot.runs = runs  # pyright: ignore[reportAttributeAccessIssue]  # test-only capture

    async def send_message(_chat_id: int, body: Any, reply_to: int) -> int:
        sent.append((body, reply_to))
        return 999

    async def run_completion(
        chat_id: int, msg_id: int, adopt: list[int] | None = None, error_reply_to: int | None = None, edition: int = 1
    ) -> None:
        assert chat_id == CHAT
        runs.append({'msg_id': msg_id, 'adopt': adopt, 'error_reply_to': error_reply_to, 'edition': edition})

    bot.send_message = send_message  # pyright: ignore[reportAttributeAccessIssue]  # test-only stub
    bot._run_completion = run_completion  # pyright: ignore[reportPrivateUsage]  # test-only stub

    overrides: dict[str, str | None] | None = {'live': None} if live else None
    bot.set_msg_info(CHAT, PROMPT, MsgInfo(False, [make_text_part('first')], None, 'g', 'be helpful', overrides))
    if answered:
        bot.store.put_turn(CHAT, REPLY, MsgInfo(True, [make_text_part('answer')], PROMPT, None, None))
    return bot


def _edit(bot: Any, message: FakeMessage) -> Any:
    asyncio.run(bot.edit_handler(message))
    return bot


def _prompt_text(bot: Any, msg_id: int = PROMPT) -> str | None:
    info = bot.get_msg_info(CHAT, msg_id)
    assert info is not None
    return info.message[0].text


def test_live_is_a_flag_and_shows_in_the_header():
    model = apply_overrides(Model(prefix='g', name='gpt-5'), {'live': None})
    assert model.live
    assert not apply_overrides(model, {'live': ''}).live
    bot = _bot()
    assert bot._reply_prefix(model) == '🤖 `gpt-5` (live)\n\n'


def test_editing_a_live_prompt_answers_it_again_over_its_reply():
    bot = _edit(_bot(), FakeMessage('g,live first, edited'))
    assert _prompt_text(bot) == 'first, edited'
    assert bot.runs == [{'msg_id': PROMPT, 'adopt': REPLY, 'error_reply_to': PROMPT, 'edition': 2}]


def test_an_edit_without_the_prefix_keeps_the_model():
    bot = _edit(_bot(), FakeMessage('just the text now'))
    info = bot.get_msg_info(CHAT, PROMPT)
    assert info.message[0].text == 'just the text now'
    assert (info.prefix, info.overrides) == ('g', {'live': None})
    assert len(bot.runs) == 1


def test_an_edit_can_switch_the_model():
    bot = _edit(_bot(), FakeMessage('c,live ask claude instead'))
    info = bot.get_msg_info(CHAT, PROMPT)
    assert (info.prefix, info.message[0].text) == ('c', 'ask claude instead')
    assert len(bot.runs) == 1


def test_a_later_prompt_of_a_live_thread_is_answered_again_too():
    bot = _bot()
    bot.set_msg_info(CHAT, 20, MsgInfo(False, [make_text_part('second')], REPLY[-1], None, None))
    bot.store.put_turn(CHAT, [21], MsgInfo(True, [make_text_part('second answer')], 20, None, None))
    _ = _edit(bot, FakeMessage('second, edited', msg_id=20, reply_to=REPLY[-1]))
    assert _prompt_text(bot, 20) == 'second, edited'
    assert bot.get_msg_info(CHAT, 20).prefix is None, 'a reply naming no model still names none'
    assert bot.runs == [{'msg_id': 20, 'adopt': [21], 'error_reply_to': 20, 'edition': 2}]


def test_a_thread_that_is_not_live_is_left_alone():
    bot = _edit(_bot(live=False), FakeMessage('g first, edited'))
    assert _prompt_text(bot) == 'first'
    assert bot.runs == []


def test_an_edit_cannot_make_its_own_thread_live():
    bot = _edit(_bot(live=False), FakeMessage('g,live first'))
    assert bot.runs == []
    assert bot.get_msg_info(CHAT, PROMPT).overrides is None


def test_an_edit_that_changes_nothing_is_ignored():
    # Telegram reports a link preview being added as an edit of the same text
    bot = _edit(_bot(), FakeMessage('g,live first'))
    assert bot.runs == []


def test_a_prompt_with_no_stored_answer_is_answered_afresh():
    bot = _edit(_bot(answered=False), FakeMessage('g,live first, edited'))
    assert bot.runs == [{'msg_id': PROMPT, 'adopt': None, 'error_reply_to': None, 'edition': 1}]


def test_a_chat_command_is_reread_as_one():
    bot = _edit(_bot(), FakeMessage('/chat main/some-model,live first, edited'))
    info = bot.get_msg_info(CHAT, PROMPT)
    assert (info.prefix, info.model_spec, info.message[0].text) == (None, 'main/some-model', 'first, edited')
    assert len(bot.runs) == 1


def test_an_edit_into_another_command_is_ignored():
    bot = _edit(_bot(), FakeMessage('/help'))
    assert bot.runs == []
    assert _prompt_text(bot) == 'first'


def test_a_bad_override_is_reported_to_the_edited_message():
    bot = _edit(_bot(), FakeMessage('g,nonsense first'))
    assert bot.runs == []
    assert bot.sent == [('[!] Unknown override key: nonsense', PROMPT)]


def test_edits_the_bot_does_not_answer_are_ignored():
    bot = _bot()
    bot.allowed_chats = set()
    assert _edit(bot, FakeMessage('g,live edited')).runs == []
    assert _edit(_bot(), FakeMessage('g,live edited', sender_id=BOT_ID)).runs == []
    assert _edit(_bot(), FakeMessage('g,live edited', msg_id=555)).runs == [], 'no stored turn'
    assert _edit(_bot(), FakeMessage('g,live edited', msg_id=REPLY[0])).runs == [], 'an answer is not a prompt'


def test_an_answer_still_streaming_is_stopped_first():
    bot = _bot()
    key = (CHAT, PROMPT)
    bot.generating[key] = False
    bot.generation_done[key] = asyncio.Event()

    async def generation() -> None:
        # the streaming answer notices the abort at its next event, then stores and finishes
        while not bot.generating[key]:
            await asyncio.sleep(0)
        assert bot.runs == [], 'nothing starts before the old answer is over'
        del bot.generating[key]
        bot.generation_done.pop(key).set()

    async def both() -> None:
        # bounded, so an edit that never asks the old answer to stop fails rather than hangs
        _ = await asyncio.wait_for(
            asyncio.gather(generation(), bot.edit_handler(FakeMessage('g,live first, edited'))), timeout=5
        )

    asyncio.run(both())
    assert len(bot.runs) == 1
    assert _prompt_text(bot) == 'first, edited'


@final
class FakePress:
    """A press of one of the buttons under the message *message_id*."""

    def __init__(self, message_id: int, data: bytes) -> None:
        self.chat_id: int = CHAT
        self.message_id: int = message_id
        self.data: bytes = data
        self.answers: list[tuple[str | None, bool]] = []

    async def answer(self, message: str | None = None, alert: bool = False) -> None:
        # a round trip to Telegram, during which another press may be handled
        await asyncio.sleep(0)
        self.answers.append((message, alert))


def _press(bot: Any, message_id: int, data: bytes) -> FakePress:
    press = FakePress(message_id, data)
    asyncio.run(bot.live_reply_callback(press))
    return press


def _deleting_bot(refused: set[int] | None = None) -> Any:
    """A bot whose Telegram deletes every message asked for, except those in *refused*."""
    bot = _bot()
    deleted: list[int] = []
    bot.deleted = deleted  # pyright: ignore[reportAttributeAccessIssue]  # test-only capture

    async def delete_messages(_chat_id: int, message_ids: list[int]) -> None:
        deleted.extend(message_ids)

    async def still_there(_chat_id: int, message_ids: list[int]) -> set[int]:
        return set(message_ids) & (refused or set())

    bot.delete_messages = delete_messages  # pyright: ignore[reportAttributeAccessIssue]  # test-only stub
    bot._still_there = still_there  # pyright: ignore[reportPrivateUsage]  # test-only stub
    bot.deleting_replies = set()
    return bot


def test_regenerate_answers_the_prompt_again_over_every_part():
    bot = _bot()
    bot.deleting_replies = set()
    press = _press(bot, REPLY[1], b'live:re')
    assert bot.runs == [{'msg_id': PROMPT, 'adopt': REPLY, 'error_reply_to': REPLY[1], 'edition': 2}]
    assert press.answers == [(None, False)]


def test_a_second_regenerate_press_finds_the_prompt_claimed():
    bot = _bot()
    bot.deleting_replies = set()

    async def run_completion(chat_id: int, msg_id: int, **kwargs: Any) -> None:
        # claims the prompt before its first await, as the real one does
        bot.generating[(chat_id, msg_id)] = False
        bot.runs.append({'msg_id': msg_id, **kwargs})
        await asyncio.sleep(0.01)
        del bot.generating[(chat_id, msg_id)]

    bot._run_completion = run_completion  # pyright: ignore[reportPrivateUsage]  # test-only stub
    first, second = FakePress(REPLY[0], b'live:re'), FakePress(REPLY[0], b'live:re')

    async def both() -> None:
        _ = await asyncio.gather(bot.live_reply_callback(first), bot.live_reply_callback(second))

    asyncio.run(both())
    assert len(bot.runs) == 1
    assert second.answers == [('That reply is already being generated', True)]


def test_delete_removes_every_part_and_leaves_the_prompt():
    bot = _deleting_bot()
    press = _press(bot, REPLY[0], b'live:del')
    assert bot.deleted == REPLY
    assert all(bot.get_msg_info(CHAT, mid) is None for mid in REPLY)
    assert _prompt_text(bot) == 'first'
    assert bot.store.answers_to(CHAT, PROMPT) == []
    assert press.answers == [(None, False)]


def test_delete_is_refused_when_a_later_turn_follows():
    bot = _deleting_bot()
    bot.set_msg_info(CHAT, 20, MsgInfo(False, [make_text_part('second')], REPLY[0], None, None))
    press = _press(bot, REPLY[1], b'live:del')
    assert bot.deleted == []
    assert press.answers == [('A later turn follows this reply, so it cannot be deleted', True)]


def test_a_turn_on_a_part_no_longer_shown_still_counts_as_later():
    bot = _deleting_bot()
    # a shrunken regeneration left 12 mapped but not live, and a turn had replied to it
    bot.store.put_turn(CHAT, [REPLY[0]], MsgInfo(True, [make_text_part('short')], PROMPT, None, None))
    bot.set_msg_info(CHAT, 20, MsgInfo(False, [make_text_part('second')], REPLY[1], None, None))
    _ = _press(bot, REPLY[0], b'live:del')
    assert bot.deleted == []


def test_a_message_telegram_would_not_delete_keeps_its_row():
    bot = _deleting_bot(refused={REPLY[1]})
    press = _press(bot, REPLY[0], b'live:del')
    assert bot.get_msg_info(CHAT, REPLY[0]) is None
    assert bot.get_msg_info(CHAT, REPLY[1]) is not None, 'still on screen, so still replyable'
    assert press.answers == [('1 of the messages could not be deleted', True)]


def test_deleting_an_imported_turn_takes_its_prompt_too():
    bot = _deleting_bot()
    bot.set_msg_info(CHAT, -50, MsgInfo(False, [make_text_part('imported')], None, 'g', None, {'live': None}))
    bot.set_msg_info(CHAT, 50, MsgInfo(True, [make_text_part('imported answer')], -50, None, None))
    _ = _press(bot, 50, b'live:del')
    assert bot.get_msg_info(CHAT, 50) is None
    assert bot.get_msg_info(CHAT, -50) is None


def test_presses_are_refused_while_the_reply_is_busy():
    bot = _deleting_bot()
    bot.generating[(CHAT, PROMPT)] = False
    assert _press(bot, REPLY[0], b'live:del').answers == [('That reply is already being generated', True)]
    assert _press(bot, REPLY[0], b'live:re').answers == [('That reply is already being generated', True)]
    del bot.generating[(CHAT, PROMPT)]
    bot.deleting_replies.add((CHAT, PROMPT))
    assert _press(bot, REPLY[0], b'live:re').answers == [('That reply is being deleted', True)]
    assert bot.runs == []
    assert bot.deleted == []


def test_a_press_on_a_reply_no_longer_stored_says_so():
    bot = _deleting_bot()
    assert _press(bot, 555, b'live:re').answers == [('That reply is no longer stored', True)]
    assert _press(bot, PROMPT, b'live:del').answers == [('That reply is no longer stored', True)]
