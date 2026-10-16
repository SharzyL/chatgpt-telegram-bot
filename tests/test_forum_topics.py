"""What counts as a reply, including a message that merely sits in a forum topic."""

import asyncio
from typing import Any, final
from zoneinfo import ZoneInfo

from telethon.tl.types import MessageReplyHeader, MessageReplyStoryHeader, PeerUser

from chatgpt_telegram_bot.bot import ChatGPTTelegramBot
from chatgpt_telegram_bot.models import Model, MsgInfo
from chatgpt_telegram_bot.utils import PendingReplyManager, replied_message_id

BOT_ID = 7
HUMAN_ID = 1
TOPIC_ROOT = 42


@final
class FakeMessage:
    """Only the attributes the reply path reads."""

    def __init__(self, text: str = '', reply_to: Any = None, replied: Any = None, sender_id: int = HUMAN_ID) -> None:
        self.chat_id: int = -100
        self.sender_id: int = sender_id
        self.id: int = 500
        self.message: str = text
        self.photo: Any = None
        self.document: Any = None
        self.reply_to: Any = reply_to
        self._replied: Any = replied

    async def get_reply_message(self) -> Any:
        return self._replied


def posted_in_topic() -> MessageReplyHeader:
    """A message sent into topic 42, replying to nothing."""
    return MessageReplyHeader(forum_topic=True, reply_to_msg_id=TOPIC_ROOT)


def reply_in_topic(target: int) -> MessageReplyHeader:
    """A reply to message *target*, made inside topic 42."""
    return MessageReplyHeader(forum_topic=True, reply_to_msg_id=target, reply_to_top_id=TOPIC_ROOT)


def test_a_message_in_the_general_topic_replies_to_nothing():
    assert replied_message_id(FakeMessage()) is None


def test_a_message_posted_in_a_topic_replies_to_nothing():
    # the header points at the message that opened the topic, which is not a reply target
    assert replied_message_id(FakeMessage(reply_to=posted_in_topic())) is None


def test_a_reply_inside_a_topic_names_its_target():
    assert replied_message_id(FakeMessage(reply_to=reply_in_topic(99))) == 99


def test_a_plain_reply_outside_a_forum_names_its_target():
    assert replied_message_id(FakeMessage(reply_to=MessageReplyHeader(reply_to_msg_id=99))) == 99


def test_a_story_reply_has_no_message_to_reply_to():
    # its header has no reply_to_msg_id at all, and reading quote_text off it would raise
    story = MessageReplyStoryHeader(peer=PeerUser(HUMAN_ID), story_id=3)
    assert replied_message_id(FakeMessage(reply_to=story)) is None


def _bot() -> Any:
    bot = object.__new__(ChatGPTTelegramBot)
    bot.db = {}  # pyright: ignore[reportAttributeAccessIssue]  # a dict is enough of a shelf here
    bot.models = [Model(prefix='g', name='gpt-5')]
    bot.system_prompt = 'be helpful'
    bot.timezone = ZoneInfo('UTC')
    bot.allowed_chats = {-100}
    bot.bot_id = BOT_ID
    bot.pending_reply_manager = PendingReplyManager()
    sent: list[Any] = []
    completions: list[tuple[int, int]] = []
    bot.sent = sent  # pyright: ignore[reportAttributeAccessIssue]  # test-only capture
    bot.completions = completions  # pyright: ignore[reportAttributeAccessIssue]  # test-only capture

    async def send_message(_chat_id: int, body: Any, _reply_to: int | None = None) -> int:
        sent.append(body)
        return 999

    async def run_completion(chat_id: int, msg_id: int, adopt: list[int] | None = None) -> None:
        assert adopt is None
        completions.append((chat_id, msg_id))

    bot.send_message = send_message  # pyright: ignore[reportAttributeAccessIssue]  # test-only stub
    bot._run_completion = run_completion  # pyright: ignore[reportPrivateUsage]  # test-only stub
    return bot


def _handle(message: FakeMessage) -> Any:
    bot = _bot()
    asyncio.run(bot.reply_handler(message))
    return bot


def _stored(bot: Any) -> MsgInfo | None:
    return bot.get_msg_info(-100, 500)


def test_a_prefixed_message_in_a_topic_opens_a_thread():
    # it used to be read as a reply to the topic root, and dropped for not being the bot's
    topic_root = FakeMessage(sender_id=HUMAN_ID)
    bot = _handle(FakeMessage('g hello', reply_to=posted_in_topic(), replied=topic_root))
    info = _stored(bot)
    assert info is not None
    assert info.prefix == 'g'
    assert info.reply_id is None
    assert info.message[0].text == 'hello'
    assert bot.completions == [(-100, 500)]


def test_a_prefixed_message_in_the_general_topic_still_opens_a_thread():
    bot = _handle(FakeMessage('g hello'))
    info = _stored(bot)
    assert info is not None
    assert info.prefix == 'g'
    assert bot.completions == [(-100, 500)]


def test_a_reply_to_the_bot_inside_a_topic_continues_the_thread():
    bot = _handle(FakeMessage('more', reply_to=reply_in_topic(99), replied=FakeMessage(sender_id=BOT_ID)))
    info = _stored(bot)
    assert info is not None
    assert info.reply_id == 99
    assert info.prefix is None
    assert bot.completions == [(-100, 500)]


def test_a_reply_to_someone_else_in_a_topic_is_still_ignored():
    bot = _handle(FakeMessage('g hello', reply_to=reply_in_topic(99), replied=FakeMessage(sender_id=HUMAN_ID)))
    assert _stored(bot) is None
    assert bot.completions == []
    assert bot.sent == []


def test_an_unprefixed_message_in_a_topic_is_ignored_quietly():
    topic_root = FakeMessage(sender_id=HUMAN_ID)
    bot = _handle(FakeMessage('just chatting', reply_to=posted_in_topic(), replied=topic_root))
    assert _stored(bot) is None
    assert bot.completions == []
    assert bot.sent == [], 'a group should not be answered without a prefix'


def test_a_vanished_reply_target_is_reported_not_crashed():
    bot = _handle(FakeMessage('g hello', reply_to=reply_in_topic(99), replied=None))
    assert _stored(bot) is None
    assert bot.completions == []
