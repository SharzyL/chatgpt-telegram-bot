"""Answering a prompt again in place."""

import asyncio
from types import SimpleNamespace
from typing import Any, final
from zoneinfo import ZoneInfo

import pytest
from telethon.tl.types import MessageReplyHeader

from chatgpt_telegram_bot.bot import LIVE_REPLY_MARKUP, ChatGPTTelegramBot
from chatgpt_telegram_bot.models import (
    EndPoint,
    Model,
    MsgInfo,
    ResponseDelta,
    make_image_part,
    make_text_part,
    regeneration_edition,
)
from chatgpt_telegram_bot.store import Store
from chatgpt_telegram_bot.utils import PendingReplyManager

BOT_ID = 7
CHAT = -100
PROMPT_ID = 10
REPLY_IDS = [11, 12, 13]
COMMAND_ID = 20


@final
class FakeMessage:
    def __init__(self, replied: Any = None, target: int | None = None, sender_id: int = 1) -> None:
        self.chat_id: int = CHAT
        self.sender_id: int = sender_id
        self.id: int = COMMAND_ID
        self.message: str = ''
        self.photo: Any = None
        self.document: Any = None
        self.reply_to: Any = MessageReplyHeader(reply_to_msg_id=target) if target is not None else None
        self._replied: Any = replied

    async def get_reply_message(self) -> Any:
        return self._replied


def _bot(prompt_parts: list[Any] | None = None, part_ids: list[int] | None = None) -> Any:
    bot = object.__new__(ChatGPTTelegramBot)
    bot.store = Store(':memory:')
    bot.models = [Model(prefix='g', name='gpt-5'), Model(prefix='c', name='claude')]
    bot.endpoints = [EndPoint(name='main', url='https://main.test'), EndPoint(name='alt', url='https://alt.test')]
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
    bot.pending_reply_manager = PendingReplyManager()
    bot.generating = {}
    bot.generation_done = {}
    sent: list[Any] = []
    runs: list[tuple[int, list[int] | None]] = []
    editions: list[int] = []
    bot.editions = editions  # pyright: ignore[reportAttributeAccessIssue]  # test-only capture
    bot.sent = sent  # pyright: ignore[reportAttributeAccessIssue]  # test-only capture
    bot.runs = runs  # pyright: ignore[reportAttributeAccessIssue]  # test-only capture

    async def send_message(_chat_id: int, body: Any, _reply_to: int | None = None) -> int:
        sent.append(body)
        return 999

    async def run_completion(
        chat_id: int, msg_id: int, adopt: list[int] | None = None, error_reply_to: int | None = None, edition: int = 1
    ) -> None:
        assert chat_id == CHAT
        assert error_reply_to == COMMAND_ID, 'failures are reported to the command'
        runs.append((msg_id, adopt))
        editions.append(edition)

    bot.send_message = send_message  # pyright: ignore[reportAttributeAccessIssue]  # test-only stub
    bot._run_completion = run_completion  # pyright: ignore[reportPrivateUsage]  # test-only stub

    bot.set_msg_info(
        CHAT,
        PROMPT_ID,
        MsgInfo(False, prompt_parts or [make_text_part('the original prompt')], None, 'g', 'rooted', {'t': 'high'}),
    )
    reply = MsgInfo(True, [make_text_part('the old answer')], PROMPT_ID, None, None, None, part_ids)
    bot.store.put_turn(CHAT, part_ids or [REPLY_IDS[0]], reply)
    return bot


def _regenerate(bot: Any, body: str = '', target: int | None = REPLY_IDS[0]) -> Any:
    replied = FakeMessage(sender_id=BOT_ID) if target is not None else None
    asyncio.run(bot.regenerate_handler(FakeMessage(replied=replied, target=target), body))
    return bot


def _prompt(bot: Any) -> MsgInfo:
    info = bot.get_msg_info(CHAT, PROMPT_ID)
    assert info is not None
    return info


def test_a_bare_regenerate_reruns_the_same_prompt_and_model():
    bot = _regenerate(_bot())
    assert bot.runs == [(PROMPT_ID, [REPLY_IDS[0]])]
    # nothing about the turn changed
    assert _prompt(bot).message[0].text == 'the original prompt'
    assert _prompt(bot).prefix == 'g'
    assert _prompt(bot).overrides == {'t': 'high'}


def test_an_endpoint_model_switches_the_model_for_the_turn():
    bot = _regenerate(_bot(), 'alt/vendor/m,suffix=/v2 a reworded prompt')
    assert bot.runs == [(PROMPT_ID, [REPLY_IDS[0]])]
    info = _prompt(bot)
    assert (info.prefix, info.model_spec, info.overrides) == (None, 'alt/vendor/m', {'suffix': '/v2'})
    assert info.message[0].text == 'a reworded prompt'
    _, model, _ = bot.construct_chat_history(CHAT, PROMPT_ID)
    assert (model.endpoint, model.name, model.suffix) == ('alt', 'vendor/m', '/v2')


def test_a_prefix_after_an_endpoint_model_clears_the_spec():
    bot = _regenerate(_bot(), 'alt/m')
    bot = _regenerate(bot, 'c')
    assert (_prompt(bot).prefix, _prompt(bot).model_spec) == ('c', None)


def test_a_slash_word_on_no_known_endpoint_is_prompt():
    bot = _regenerate(_bot(), 'and/or which one')
    info = _prompt(bot)
    assert (info.prefix, info.model_spec) == ('g', None)
    assert info.message[0].text == 'and/or which one'


def test_an_endpoint_model_with_bad_flags_is_reported():
    bot = _regenerate(_bot(), 'alt/m,api_type=grpc hi')
    assert bot.sent == ['[!] api_type must be one of openai, openai_legacy, anthropic']
    assert bot.runs == []


def test_a_prefix_switches_the_model_for_the_turn():
    bot = _regenerate(_bot(), 'c')
    assert bot.runs == [(PROMPT_ID, [REPLY_IDS[0]])]
    info = _prompt(bot)
    assert info.prefix == 'c'
    assert info.message[0].text == 'the original prompt', 'the prompt itself is untouched'
    assert info.overrides is None
    assert info.system_prompt is not None and info.system_prompt.startswith('be helpful')


def test_a_prefix_and_text_replace_the_prompt_too():
    bot = _regenerate(_bot(), 'c be brief this time')
    info = _prompt(bot)
    assert info.prefix == 'c'
    assert info.message[0].text == 'be brief this time'


def test_overrides_on_the_prefix_are_recorded():
    bot = _regenerate(_bot(), 'g,t= same prompt')
    info = _prompt(bot)
    assert info.prefix == 'g'
    assert info.overrides == {'t': ''}


def test_an_image_survives_a_new_prompt():
    bot = _bot([make_text_part('look'), make_image_part('deadbeef')])
    _ = _regenerate(bot, 'g what is this')
    parts = _prompt(bot).message
    assert [p.type_ for p in parts] == ['text', 'image']
    assert parts[0].text == 'what is this'
    assert parts[1].hash == 'deadbeef'


def test_every_part_of_a_split_reply_is_written_over():
    bot = _bot(part_ids=REPLY_IDS)
    _ = _regenerate(bot, target=REPLY_IDS[1])
    assert bot.runs == [(PROMPT_ID, REPLY_IDS)], 'replying to any part regenerates all of them'


def test_text_without_a_prefix_replaces_only_the_prompt():
    before = _prompt(_bot())
    bot = _regenerate(_bot(), 'nonsense words')
    assert bot.runs == [(PROMPT_ID, [REPLY_IDS[0]])]
    info = _prompt(bot)
    assert info.message[0].text == 'nonsense words'
    # the model, its overrides and its prompt are those the turn was already on
    assert info.prefix == 'g'
    assert info.overrides == {'t': 'high'}
    assert info.system_prompt == before.system_prompt


def test_a_bare_prompt_keeps_the_image():
    bot = _bot([make_text_part('look'), make_image_part('deadbeef')])
    _ = _regenerate(bot, '  what is this\n')
    parts = _prompt(bot).message
    assert [p.type_ for p in parts] == ['text', 'image']
    assert parts[0].text == 'what is this'


def test_a_word_that_is_a_prefix_is_read_as_one():
    bot = _regenerate(_bot(), 'c is a letter')
    info = _prompt(bot)
    assert info.prefix == 'c'
    assert info.message[0].text == 'is a letter'


def test_a_bad_override_is_reported():
    bot = _regenerate(_bot(), 'g,zz=1')
    assert bot.runs == []
    assert bot.sent[0].startswith('[!] Unknown override key')


def test_it_has_to_reply_to_something():
    bot = _regenerate(_bot(), target=None)
    assert bot.runs == []
    assert bot.sent[0].startswith('[!] `/regenerate`')


def test_it_has_to_reply_to_the_bot():
    bot = _bot()
    asyncio.run(bot.regenerate_handler(FakeMessage(replied=FakeMessage(sender_id=1), target=REPLY_IDS[0]), ''))
    assert bot.runs == []
    assert bot.sent[0].startswith('[!] `/regenerate`')


def test_a_message_with_no_stored_conversation_is_reported():
    bot = _bot()
    _ = _regenerate(bot, target=999)
    assert bot.runs == []
    assert bot.sent[0] == '[!] No stored conversation for that message'


@final
class FakeBot:
    """Just the transport BotReplyMessages drives."""

    def __init__(self) -> None:
        self.TELEGRAM_MIN_INTERVAL: float = 0.001
        self.TELEGRAM_BURST_INTERVAL: float = 0.001
        self.TELEGRAM_BURST_WINDOW: float = 0.001
        self.pending_reply_manager: PendingReplyManager = PendingReplyManager()
        self.calls: list[tuple[str, int]] = []
        self.next_id: int = 100

    async def edit_message(self, _chat_id: int, _body: Any, message_id: int) -> None:
        self.calls.append(('edit', message_id))

    async def send_message(self, _chat_id: int, _body: Any, reply_to: int) -> int:
        self.next_id += 1
        self.calls.append(('send', reply_to))
        return self.next_id

    async def delete_message(self, _chat_id: int, message_id: int) -> None:
        self.calls.append(('delete', message_id))


def _flush(adopt: list[int], text: str) -> list[tuple[str, int]]:
    from chatgpt_telegram_bot.render import ENTITY_RENDERER
    from chatgpt_telegram_bot.reply_messages import BotReplyMessages

    fake = FakeBot()

    async def go() -> None:
        async with BotReplyMessages(fake, CHAT, PROMPT_ID, '', ENTITY_RENDERER, adopt) as msgs:  # pyright: ignore[reportArgumentType]  # a stand-in for the transport
            await msgs.update(text)

    asyncio.run(go())
    return fake.calls


def test_an_adopted_message_is_edited_not_resent():
    assert _flush([REPLY_IDS[0]], 'a new answer') == [('edit', REPLY_IDS[0])]


def test_a_longer_answer_edits_what_exists_and_sends_the_rest():
    calls = _flush([REPLY_IDS[0]], 'A' * 9000)
    assert calls[0] == ('edit', REPLY_IDS[0])
    assert [kind for kind, _ in calls].count('send') >= 1
    assert ('delete', REPLY_IDS[0]) not in calls


def test_a_shorter_answer_deletes_the_parts_it_no_longer_needs():
    calls = _flush(REPLY_IDS, 'one short answer')
    assert calls[0] == ('edit', REPLY_IDS[0])
    assert ('delete', REPLY_IDS[1]) in calls
    assert ('delete', REPLY_IDS[2]) in calls


def _recording_bot() -> Any:
    """The real bot, with only the transport replaced, so _generate runs for real."""
    bot = _bot()
    bot.default_endpoint = 'openai'
    bot.OPENAI_MAX_RETRY = 3
    bot.TELEGRAM_MIN_INTERVAL = 0.001
    bot.TELEGRAM_BURST_INTERVAL = 0.001
    bot.TELEGRAM_BURST_WINDOW = 0.001
    calls: list[tuple[str, int]] = []
    bot.calls = calls
    bodies: list[str] = []
    bot.bodies = bodies
    next_id = [200]

    async def send_message(_chat_id: int, _body: Any, reply_to: int) -> int:
        next_id[0] += 1
        calls.append(('send', reply_to))
        return next_id[0]

    async def edit_message(_chat_id: int, body: Any, message_id: int) -> None:
        calls.append(('edit', message_id))
        bodies.append(body.text)

    async def delete_message(_chat_id: int, message_id: int) -> None:
        calls.append(('delete', message_id))

    bot.send_message = send_message
    bot.edit_message = edit_message
    bot.delete_message = delete_message

    def get_client(_model: Model) -> None:
        return None

    bot.get_client = get_client
    return bot


def _regenerate_for_real(
    monkeypatch: pytest.MonkeyPatch, old_parts: list[int], answer: str, old_extra: dict[str, Any] | None = None
) -> tuple[Any, dict[int, MsgInfo]]:
    bot = _recording_bot()
    old = MsgInfo(True, [make_text_part('the old answer')], PROMPT_ID, None, None, None, old_parts, None, old_extra)
    bot.store.put_turn(CHAT, old_parts, old)

    async def fake(*_args: Any, **_kwargs: Any) -> Any:
        yield ResponseDelta(text=answer)

    monkeypatch.setattr('chatgpt_telegram_bot.bot.completion', fake)
    edition = regeneration_edition(old) + 1
    asyncio.run(ChatGPTTelegramBot._run_completion(bot, CHAT, PROMPT_ID, adopt=old_parts, edition=edition))  # pyright: ignore[reportPrivateUsage]  # exercising the real path
    rows = {mid: info for mid, info in bot.store.chat_nodes(CHAT).items() if mid != PROMPT_ID}
    return bot, rows


def test_a_longer_answer_keeps_the_first_part_and_chains_the_new_ones(monkeypatch: pytest.MonkeyPatch):
    bot, rows = _regenerate_for_real(monkeypatch, [REPLY_IDS[0]], 'A' * 9000)
    sends = [reply_to for kind, reply_to in bot.calls if kind == 'send']
    assert bot.calls[0] == ('edit', REPLY_IDS[0])
    assert sends[0] == REPLY_IDS[0], 'the second part replies to the first'
    # every part records the same reply and the full id list, so any of them can be replied to
    part_ids = rows[REPLY_IDS[0]].part_ids
    assert part_ids is not None and len(part_ids) == len(rows)
    assert all(rows[mid].part_ids == part_ids for mid in part_ids)
    assert len({rows[mid].message[-1].text for mid in part_ids}) == 1


def test_a_shorter_answer_deletes_the_surplus_and_records_only_what_is_left(monkeypatch: pytest.MonkeyPatch):
    bot, rows = _regenerate_for_real(monkeypatch, REPLY_IDS, 'a short answer')
    assert ('delete', REPLY_IDS[1]) in bot.calls
    assert ('delete', REPLY_IDS[2]) in bot.calls
    assert rows[REPLY_IDS[0]].part_ids == [REPLY_IDS[0]]
    assert rows[REPLY_IDS[0]].message[-1].text == 'a short answer'


def test_a_later_turn_on_a_deleted_part_continues_from_the_new_answer(monkeypatch: pytest.MonkeyPatch):
    # the thread went on from the last part of the old answer, which the shorter one deletes
    bot, _ = _regenerate_for_real(monkeypatch, REPLY_IDS, 'a short answer')
    later = 900
    bot.set_msg_info(CHAT, later, MsgInfo(False, [make_text_part('and then?')], REPLY_IDS[-1], None, None))
    history, _, _ = bot.construct_chat_history(CHAT, later)
    assert [parts[-1].text for parts in history] == ['the original prompt', 'a short answer', 'and then?']
    # navigating from there jumps to the part that still exists
    assert bot.get_msg_info(CHAT, REPLY_IDS[-1]).part_ids == [REPLY_IDS[0]]


def test_a_second_regenerate_of_the_same_reply_is_refused():
    bot = _bot()
    bot.generating[(CHAT, PROMPT_ID)] = False
    _ = _regenerate(bot)
    assert bot.runs == []
    assert bot.sent[0] == '[!] That reply is already being generated'


def test_an_imported_turn_with_no_model_reports_to_the_command():
    # an import without a prefix records no model, and its prompt lives under a negative id
    bot = _recording_bot()
    bot.store = Store(':memory:')
    bot.set_msg_info(CHAT, -50, MsgInfo(False, [make_text_part('imported prompt')], None, None, None))
    bot.set_msg_info(CHAT, 50, MsgInfo(True, [make_text_part('imported answer')], -50, None, None, None, [50]))
    replies: list[int] = []

    async def send_message(_chat_id: int, _body: Any, reply_to: int) -> int:
        replies.append(reply_to)
        return 999

    bot.send_message = send_message
    asyncio.run(ChatGPTTelegramBot._run_completion(bot, CHAT, -50, adopt=[50], error_reply_to=COMMAND_ID))  # pyright: ignore[reportPrivateUsage]  # exercising the real path
    assert replies == [COMMAND_ID]


def test_a_regenerated_imported_turn_keeps_its_prompt_on_top(monkeypatch: pytest.MonkeyPatch):
    # the one message showing an imported turn held its prompt too, which has no message of its own
    bot = _recording_bot()
    bot.store = Store(':memory:')
    bot.set_msg_info(CHAT, -50, MsgInfo(False, [make_text_part('imported prompt')], None, 'g', None))
    bot.set_msg_info(CHAT, 50, MsgInfo(True, [make_text_part('imported answer')], -50, None, None))

    async def fake(*_args: Any, **_kwargs: Any) -> Any:
        yield ResponseDelta(text='a new answer')

    monkeypatch.setattr('chatgpt_telegram_bot.bot.completion', fake)
    asyncio.run(ChatGPTTelegramBot._run_completion(bot, CHAT, -50, adopt=[50], edition=2))  # pyright: ignore[reportPrivateUsage]  # exercising the real path
    assert bot.bodies, 'the message was written over'
    for text in bot.bodies:
        assert text.split('\n')[2:5] == ['imported prompt', '', '-----'], text
    assert bot.bodies[-1].endswith('a new answer')
    # only the answer is stored as the answer; the prompt stays its own turn
    assert bot.get_msg_info(CHAT, 50).message[-1].text == 'a new answer'
    assert bot.get_msg_info(CHAT, -50).message[0].text == 'imported prompt'


def test_an_ordinary_regenerated_reply_shows_no_prompt(monkeypatch: pytest.MonkeyPatch):
    bot, _ = _regenerate_for_real(monkeypatch, [REPLY_IDS[0]], 'again')
    assert all('-----' not in text and 'the original prompt' not in text for text in bot.bodies)


def _manipulate_imported(target: int) -> Any:
    # a split imported turn: the prompt shows on the first part only
    bot = _recording_bot()
    bot.store = Store(':memory:')
    bot.set_msg_info(CHAT, -50, MsgInfo(False, [make_text_part('imported prompt')], None, 'g', None))
    bot.store.put_turn(CHAT, [50, 51], MsgInfo(True, [make_text_part('imported answer')], -50, None, None))
    asyncio.run(bot.manipulate_handler(FakeMessage(replied=FakeMessage(sender_id=BOT_ID), target=target), 'mine'))
    assert bot.get_msg_info(CHAT, 50).message[-1].text == 'mine'
    return bot


def test_manipulating_an_imported_turn_keeps_its_prompt_on_top():
    bot = _manipulate_imported(50)
    assert bot.calls == [('edit', 50)]
    assert bot.bodies[0].split('\n')[2:5] == ['imported prompt', '', '-----']


def test_a_later_part_of_an_imported_turn_never_showed_the_prompt():
    bot = _manipulate_imported(51)
    assert bot.calls == [('edit', 51)]
    assert 'imported prompt' not in bot.bodies[0]


def test_each_regeneration_counts_one_more_edition():
    bot = _regenerate(_bot())
    assert bot.editions == [2], 'the original answer is the first edition'
    bot = _bot()
    target = bot.get_msg_info(CHAT, REPLY_IDS[0])
    bot.store.update(CHAT, REPLY_IDS[0], target._replace(extra_metadata={'regeneration_edition': 4}))
    assert _regenerate(bot).editions == [5]


def test_a_regenerated_reply_records_its_edition_and_shows_it_in_the_header(monkeypatch: pytest.MonkeyPatch):
    bot, rows = _regenerate_for_real(monkeypatch, [REPLY_IDS[0]], 'again', {'regeneration_edition': 2})
    assert rows[REPLY_IDS[0]].extra_metadata == {'regeneration_edition': 3}
    assert regeneration_edition(rows[REPLY_IDS[0]]) == 3
    # the version joins the turn's other flags in the header, from the first edit on
    assert all(text.split('\n', 1)[0] == '🤖 gpt-5 (thinking=high, version=3)' for text in bot.bodies)


def test_an_original_reply_has_no_version():
    bot = _bot()
    assert 'version' not in bot._reply_prefix(Model(prefix='g', name='gpt-5'))
    assert bot._reply_prefix(Model(prefix='g', name='gpt-5'), edition=2) == '🤖 `gpt-5` (version=2)\n\n'


def test_an_edition_that_is_not_a_positive_number_reads_as_the_original():
    for value in (None, True, 0, -3, '2', 2.5):
        info = MsgInfo(True, [make_text_part('a')], 1, None, None, extra_metadata={'regeneration_edition': value})
        assert regeneration_edition(info) == 1, value


def _generate_marked(monkeypatch: pytest.MonkeyPatch, live: bool) -> dict[int, list[Any]]:
    """Answer the prompt afresh with a three-part reply; the markup each message was given, in order."""
    bot = _recording_bot()
    if live:
        bot.store.update(CHAT, PROMPT_ID, _prompt(bot)._replace(overrides={'live': None}))
    marks: dict[int, list[Any]] = {}
    next_id = [300]

    async def send_message(_chat_id: int, body: Any, _reply_to: int) -> int:
        next_id[0] += 1
        marks[next_id[0]] = [body.markup]
        return next_id[0]

    async def edit_message(_chat_id: int, body: Any, message_id: int) -> None:
        marks[message_id].append(body.markup)

    bot.send_message = send_message
    bot.edit_message = edit_message

    async def fake(*_args: Any, **_kwargs: Any) -> Any:
        yield ResponseDelta(text='A' * 9000)

    monkeypatch.setattr('chatgpt_telegram_bot.bot.completion', fake)
    asyncio.run(ChatGPTTelegramBot._run_completion(bot, CHAT, PROMPT_ID))  # pyright: ignore[reportPrivateUsage]  # exercising the real path
    return marks


def test_every_part_of_a_live_reply_ends_with_the_buttons(monkeypatch: pytest.MonkeyPatch):
    marks = _generate_marked(monkeypatch, live=True)
    assert len(marks) == 3
    assert all(seen[-1] is LIVE_REPLY_MARKUP for seen in marks.values())
    # while it streams there is no stored turn for a press to act on
    assert next(iter(marks.values()))[0] is None


def test_a_reply_that_is_not_live_has_no_buttons(monkeypatch: pytest.MonkeyPatch):
    marks = _generate_marked(monkeypatch, live=False)
    assert all(mark is None for seen in marks.values() for mark in seen)


def test_manipulating_a_live_reply_keeps_its_buttons():
    bot = _recording_bot()
    bot.store.update(CHAT, PROMPT_ID, _prompt(bot)._replace(overrides={'live': None}))
    marks: list[Any] = []

    async def edit_message(_chat_id: int, body: Any, _message_id: int) -> None:
        marks.append(body.markup)

    bot.edit_message = edit_message
    asyncio.run(bot.manipulate_handler(FakeMessage(replied=FakeMessage(sender_id=BOT_ID), target=REPLY_IDS[0]), 'x'))
    assert marks == [LIVE_REPLY_MARKUP]
