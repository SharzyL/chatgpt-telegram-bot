"""Stopping a generation that is already running."""

import asyncio
from collections.abc import AsyncGenerator
from typing import Any, final
from zoneinfo import ZoneInfo

import pytest

from chatgpt_telegram_bot.bot import ChatGPTTelegramBot
from chatgpt_telegram_bot.models import Model, MsgInfo, ResponseDelta, StreamEvent, make_text_part
from chatgpt_telegram_bot.store import Store
from chatgpt_telegram_bot.utils import PendingReplyManager

CHAT = -100
PROMPT_ID = 10


@final
class FakeMessage:
    def __init__(self) -> None:
        self.chat_id: int = CHAT
        self.sender_id: int = 1
        self.id: int = 20
        self.message: str = '/abort'


def _bot() -> Any:
    bot = object.__new__(ChatGPTTelegramBot)
    bot.store = Store(':memory:')
    bot.models = [Model(prefix='g', name='gpt-5')]
    bot.system_prompt = 'be helpful'
    bot.timezone = ZoneInfo('UTC')
    bot.allowed_chats = {CHAT}
    bot.bot_id = 7
    bot.pending_reply_manager = PendingReplyManager()
    bot.generating = {}
    bot.default_endpoint = 'openai'
    bot.OPENAI_MAX_RETRY = 3
    bot.TELEGRAM_MIN_INTERVAL = 0.001
    bot.TELEGRAM_BURST_INTERVAL = 0.001
    bot.TELEGRAM_BURST_WINDOW = 0.001
    sent: list[Any] = []
    bot.sent = sent  # pyright: ignore[reportAttributeAccessIssue]  # test-only capture

    async def send_message(_chat_id: int, body: Any, _reply_to: int | None = None) -> int:
        sent.append(body)
        return 100 + len(sent)

    async def edit_message(_chat_id: int, body: Any, message_id: int) -> None:
        sent.append((message_id, body))

    async def delete_message(_chat_id: int, _message_id: int) -> None:
        return None

    bot.send_message = send_message  # pyright: ignore[reportAttributeAccessIssue]  # test-only stub
    bot.edit_message = edit_message  # pyright: ignore[reportAttributeAccessIssue]  # test-only stub
    bot.delete_message = delete_message  # pyright: ignore[reportAttributeAccessIssue]  # test-only stub
    bot.get_client = lambda _model: None  # pyright: ignore[reportAttributeAccessIssue]  # test-only stub
    bot.set_msg_info(CHAT, PROMPT_ID, MsgInfo(False, [make_text_part('go')], None, 'g', 'be helpful'))
    return bot


def test_abort_without_a_generation_says_so():
    bot = _bot()
    asyncio.run(bot.abort_handler(FakeMessage()))
    assert bot.sent == ['[!] Nothing is generating in this chat']


def test_abort_flags_every_generation_in_the_chat():
    bot = _bot()
    bot.generating = {(CHAT, 1): False, (CHAT, 2): False, (-200, 3): False}
    asyncio.run(bot.abort_handler(FakeMessage()))
    assert bot.generating == {(CHAT, 1): True, (CHAT, 2): True, (-200, 3): False}
    assert bot.sent == [], 'a successful abort is reported by the reply, not by a message'


def _run_with_stream(
    bot: Any, monkeypatch: pytest.MonkeyPatch, events: list[StreamEvent], abort_after: int | None = None
) -> list[bool]:
    """Drive _generate against a scripted stream, aborting part-way if asked."""
    closed: list[bool] = []

    async def fake_completion(*_args: Any, **_kwargs: Any) -> AsyncGenerator[StreamEvent, None]:
        try:
            for i, event in enumerate(events):
                if abort_after is not None and i == abort_after:
                    bot.generating[(CHAT, PROMPT_ID)] = True
                yield event
        finally:
            closed.append(True)

    monkeypatch.setattr('chatgpt_telegram_bot.bot.completion', fake_completion)
    asyncio.run(bot._run_completion(CHAT, PROMPT_ID))
    return closed


DELTAS: list[StreamEvent] = [ResponseDelta(text=word) for word in ('one ', 'two ', 'three ', 'four ', 'five')]


def test_an_aborted_generation_stops_early_and_is_marked(monkeypatch: pytest.MonkeyPatch):
    bot = _bot()
    closed = _run_with_stream(bot, monkeypatch, DELTAS, abort_after=2)
    body = bot.sent[-1][1] if isinstance(bot.sent[-1], tuple) else bot.sent[-1]
    text = body.text if hasattr(body, 'text') else body
    assert 'one two' in text
    assert 'four' not in text, 'the deltas after the abort should not arrive'
    assert '[!aborted]' in text
    assert closed == [True], 'the stream is closed, not left for the collector'


def test_the_partial_reply_stays_in_the_history(monkeypatch: pytest.MonkeyPatch):
    bot = _bot()
    _ = _run_with_stream(bot, monkeypatch, DELTAS, abort_after=2)
    stored = [v for k, v in bot.store.chat_nodes(CHAT).items() if k != PROMPT_ID]
    assert stored, 'the aborted reply is still a turn that can be replied to'
    info = stored[0]
    assert info.sent_by_bot
    assert info.reply_id == PROMPT_ID
    assert info.message[-1].text == 'one two '


def test_a_finished_generation_is_not_marked_aborted(monkeypatch: pytest.MonkeyPatch):
    bot = _bot()
    _ = _run_with_stream(bot, monkeypatch, DELTAS)
    body = bot.sent[-1][1] if isinstance(bot.sent[-1], tuple) else bot.sent[-1]
    text = body.text if hasattr(body, 'text') else body
    assert 'five' in text
    assert 'aborted' not in text


def test_the_registry_is_emptied_afterwards(monkeypatch: pytest.MonkeyPatch):
    bot = _bot()
    _ = _run_with_stream(bot, monkeypatch, DELTAS)
    assert bot.generating == {}, 'a finished generation must not stay abortable'
