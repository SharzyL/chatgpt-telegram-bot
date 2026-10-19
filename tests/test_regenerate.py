"""Answering a prompt again in place."""

import asyncio
from typing import Any, final
from zoneinfo import ZoneInfo

import pytest
from telethon.tl.types import MessageReplyHeader

from chatgpt_telegram_bot.bot import ChatGPTTelegramBot
from chatgpt_telegram_bot.models import Model, MsgInfo, ResponseDelta, make_image_part, make_text_part
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
    bot.db = {}  # pyright: ignore[reportAttributeAccessIssue]  # a dict is enough of a shelf here
    bot.models = [Model(prefix='g', name='gpt-5'), Model(prefix='c', name='claude')]
    bot.system_prompt = 'be helpful'
    bot.timezone = ZoneInfo('UTC')
    bot.allowed_chats = {CHAT}
    bot.bot_id = BOT_ID
    bot.pending_reply_manager = PendingReplyManager()
    bot.generating = {}
    sent: list[Any] = []
    runs: list[tuple[int, list[int] | None]] = []
    bot.sent = sent  # pyright: ignore[reportAttributeAccessIssue]  # test-only capture
    bot.runs = runs  # pyright: ignore[reportAttributeAccessIssue]  # test-only capture

    async def send_message(_chat_id: int, body: Any, _reply_to: int | None = None) -> int:
        sent.append(body)
        return 999

    async def run_completion(chat_id: int, msg_id: int, adopt: list[int] | None = None) -> None:
        assert chat_id == CHAT
        runs.append((msg_id, adopt))

    bot.send_message = send_message  # pyright: ignore[reportAttributeAccessIssue]  # test-only stub
    bot._run_completion = run_completion  # pyright: ignore[reportPrivateUsage]  # test-only stub

    bot.set_msg_info(
        CHAT,
        PROMPT_ID,
        MsgInfo(False, prompt_parts or [make_text_part('the original prompt')], None, 'g', 'rooted', {'t': 'high'}),
    )
    reply = MsgInfo(True, [make_text_part('the old answer')], PROMPT_ID, None, None, None, part_ids)
    for mid in part_ids or [REPLY_IDS[0]]:
        bot.set_msg_info(CHAT, mid, reply)
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
    from chatgpt_telegram_bot.bot import BotReplyMessages
    from chatgpt_telegram_bot.render import ENTITY_RENDERER

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
    next_id = [200]

    async def send_message(_chat_id: int, _body: Any, reply_to: int) -> int:
        next_id[0] += 1
        calls.append(('send', reply_to))
        return next_id[0]

    async def edit_message(_chat_id: int, _body: Any, message_id: int) -> None:
        calls.append(('edit', message_id))

    async def delete_message(_chat_id: int, message_id: int) -> None:
        calls.append(('delete', message_id))

    bot.send_message = send_message
    bot.edit_message = edit_message
    bot.delete_message = delete_message

    def get_client(_endpoint: str, _model: Model) -> None:
        return None

    bot.get_client = get_client
    return bot


def _regenerate_for_real(
    monkeypatch: pytest.MonkeyPatch, old_parts: list[int], answer: str
) -> tuple[Any, dict[int, MsgInfo]]:
    bot = _recording_bot()
    old = MsgInfo(True, [make_text_part('the old answer')], PROMPT_ID, None, None, None, old_parts)
    for mid in old_parts:
        bot.set_msg_info(CHAT, mid, old)

    async def fake(*_args: Any, **_kwargs: Any) -> Any:
        yield ResponseDelta(text=answer)

    monkeypatch.setattr('chatgpt_telegram_bot.bot.completion', fake)
    asyncio.run(ChatGPTTelegramBot._run_completion(bot, CHAT, PROMPT_ID, adopt=old_parts))  # pyright: ignore[reportPrivateUsage]  # exercising the real path
    rows = {int(key.rsplit('_', 1)[1]): value for key, value in bot.db.items() if key != f'msg_info_{CHAT}_{PROMPT_ID}'}
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


def test_a_second_regenerate_of_the_same_reply_is_refused():
    bot = _bot()
    bot.generating[(CHAT, PROMPT_ID)] = False
    _ = _regenerate(bot)
    assert bot.runs == []
    assert bot.sent[0] == '[!] That reply is already being generated'
