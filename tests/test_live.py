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
