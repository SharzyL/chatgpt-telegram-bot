"""Replying to a message that carries media brings that media into the turn."""

import asyncio
from types import SimpleNamespace
from typing import Any, final
from zoneinfo import ZoneInfo

from telethon.tl.types import MessageReplyHeader

from chatgpt_telegram_bot.bot import ChatGPTTelegramBot
from chatgpt_telegram_bot.models import Model, MsgInfo
from chatgpt_telegram_bot.utils import PendingReplyManager, image_media_type

CHAT = -100
BOT_ID = 7
HUMAN_ID = 1
JPEG = b'\xff\xd8\xff\xe0 a photo'
PNG = b'\x89PNG\r\n\x1a\n an image file'


def photo() -> Any:
    return SimpleNamespace()


def document(mime: str, size: int = 10) -> Any:
    return SimpleNamespace(mime_type=mime, size=size)


@final
class FakeMessage:
    """Only the attributes the reply path reads."""

    def __init__(
        self,
        msg_id: int,
        text: str = '',
        *,
        sender_id: int = HUMAN_ID,
        photo: Any = None,
        document: Any = None,
        blob: bytes = b'',
        replied: Any = None,
        grouped_id: int | None = None,
    ) -> None:
        self.chat_id: int = CHAT
        self.sender_id: int = sender_id
        self.id: int = msg_id
        self.message: str = text
        self.photo: Any = photo
        self.document: Any = document
        self.grouped_id: int | None = grouped_id
        self._blob: bytes = blob
        self._replied: Any = replied
        self.reply_to: Any = MessageReplyHeader(reply_to_msg_id=replied.id) if replied is not None else None

    async def get_reply_message(self) -> Any:
        return self._replied

    async def download_media(self, _kind: type) -> bytes:
        return self._blob


def _bot(*neighbours: FakeMessage) -> Any:
    bot: Any = object.__new__(ChatGPTTelegramBot)
    bot.db = {}
    bot.models = [Model(prefix='g', name='gpt-5')]
    bot.system_prompt = 'be helpful'
    bot.timezone = ZoneInfo('UTC')
    bot.allowed_chats = {CHAT}
    bot.bot_id = BOT_ID
    bot.pending_reply_manager = PendingReplyManager()
    bot.default_image_prompt = 'Describe the image'
    bot.default_image_reply_prompt = 'Continue'
    bot.TEXT_FILE_SIZE_LIMIT = 900_000
    bot.IMAGE_FILE_SIZE_LIMIT = 20_000_000
    bot.cached = {}
    bot.image_cache = SimpleNamespace(set=bot.cached.__setitem__)
    bot.sent = []
    bot.completions = []

    async def get_messages(_chat_id: int, ids: list[int]) -> list[Any]:
        by_id = {m.id: m for m in neighbours}
        return [by_id.get(i) for i in ids]

    bot.bot = SimpleNamespace(get_messages=get_messages)

    async def send_message(_chat_id: int, body: Any, _reply_to: int | None = None) -> int:
        bot.sent.append(body)
        return 999

    async def run_completion(chat_id: int, msg_id: int, adopt: list[int] | None = None) -> None:
        assert adopt is None
        bot.completions.append((chat_id, msg_id))

    async def help_handler(_message: Any, error: str | None = None) -> None:
        bot.sent.append(f'help: {error}')

    bot.send_message = send_message
    bot._run_completion = run_completion
    bot.help_handler = help_handler
    return bot


def _reply(bot: Any, message: FakeMessage) -> MsgInfo | None:
    asyncio.run(bot.reply_handler(message))
    return bot.get_msg_info(CHAT, message.id)


def test_a_reply_to_a_forwarded_photo_talks_about_it():
    forwarded = FakeMessage(10, photo=photo(), blob=JPEG)
    bot = _bot()
    info = _reply(bot, FakeMessage(11, 'g what is this', replied=forwarded))
    assert info is not None
    assert [p.type_ for p in info.message] == ['text', 'image']
    assert info.message[0].text == 'what is this'
    assert info.reply_id is None, 'it opens a thread'
    assert bot.cached == {f'{CHAT}:10': JPEG}
    assert bot.completions == [(CHAT, 11)]


def test_the_replied_caption_is_left_out():
    forwarded = FakeMessage(10, 'sales by month', photo=photo(), blob=JPEG)
    info = _reply(_bot(), FakeMessage(11, 'g which month is best', replied=forwarded))
    assert info is not None
    assert info.message[0].text == 'which month is best'
    assert [p.type_ for p in info.message] == ['text', 'image']


def test_a_replied_caption_does_not_stand_in_for_the_default_prompt():
    forwarded = FakeMessage(10, 'sales by month', photo=photo(), blob=JPEG)
    info = _reply(_bot(), FakeMessage(11, 'g', replied=forwarded))
    assert info is not None
    assert info.message[0].text == 'Describe the image'


def test_a_bare_prefix_on_a_replied_photo_gets_the_default_prompt():
    info = _reply(_bot(), FakeMessage(11, 'g', replied=FakeMessage(10, photo=photo(), blob=JPEG)))
    assert info is not None
    assert info.message[0].text == 'Describe the image'


def test_a_replied_text_file_is_inlined_ahead_of_the_prompt():
    log = FakeMessage(10, document=document('text/plain'), blob=b'line one\nline two')
    info = _reply(_bot(), FakeMessage(11, 'g summarise', replied=log))
    assert info is not None
    assert [p.type_ for p in info.message] == ['text']
    assert info.message[0].text == 'line one\nline two\n\nsummarise'


def test_an_image_sent_as_a_file_is_an_image_not_text():
    image_file = FakeMessage(10, document=document('image/png'), blob=PNG)
    info = _reply(_bot(), FakeMessage(11, 'g what is this', replied=image_file))
    assert info is not None
    assert [p.type_ for p in info.message] == ['text', 'image']


def test_an_image_file_sent_directly_is_an_image_too():
    bot = _bot()
    info = _reply(bot, FakeMessage(11, 'g what is this', document=document('image/png'), blob=PNG))
    assert info is not None
    assert [p.type_ for p in info.message] == ['text', 'image']
    assert bot.sent == [], 'it used to be refused as not being UTF-8'


def test_a_replied_album_brings_every_image():
    album = [FakeMessage(i, photo=photo(), blob=JPEG, grouped_id=5) for i in (20, 21, 22)]
    stranger = FakeMessage(23, photo=photo(), blob=JPEG, grouped_id=6)
    bot = _bot(*album, stranger)
    info = _reply(bot, FakeMessage(30, 'g compare them', replied=album[2]))
    assert info is not None
    assert [p.hash for p in info.message[1:]] == [f'{CHAT}:20', f'{CHAT}:21', f'{CHAT}:22']
    assert info.message[0].text == 'compare them'


def test_a_file_the_bot_sent_is_media_not_a_thread():
    export = FakeMessage(10, sender_id=BOT_ID, document=document('application/json'), blob=b'{"title": "t"}')
    info = _reply(_bot(), FakeMessage(11, 'g explain this file', replied=export))
    assert info is not None
    assert info.reply_id is None, 'it used to be walked as a bot reply with no history'
    assert (info.message[0].text or '').startswith('{"title": "t"}')


def test_a_bot_turn_with_media_continues_its_thread():
    # no reply carries media yet, but one that did would still be a turn, not a file to read
    bot = _bot()
    bot.set_msg_info(CHAT, 10, MsgInfo(sent_by_bot=True, message=[], reply_id=9, prefix=None, system_prompt=None))
    turn = FakeMessage(10, sender_id=BOT_ID, photo=photo(), blob=JPEG)
    info = _reply(bot, FakeMessage(11, 'and then?', replied=turn))
    assert info is not None
    assert info.reply_id == 10
    assert [p.type_ for p in info.message] == ['text'], 'its media is history, not a new attachment'
    assert bot.cached == {}


def test_a_replied_photo_still_needs_a_prefix():
    bot = _bot()
    assert _reply(bot, FakeMessage(11, 'what is this', replied=FakeMessage(10, photo=photo(), blob=JPEG))) is None
    assert bot.completions == []


def test_an_unusable_replied_file_is_reported():
    binary = FakeMessage(10, document=document('application/octet-stream'), blob=b'\x00\x01')
    bot = _bot()
    assert _reply(bot, FakeMessage(11, 'g read this', replied=binary)) is None
    assert bot.sent == ['File is not text file or not valid UTF-8']


def test_a_reply_to_the_bots_text_still_continues_the_thread():
    bot = _bot()
    info = _reply(bot, FakeMessage(11, 'more', replied=FakeMessage(10, sender_id=BOT_ID)))
    assert info is not None
    assert info.reply_id == 10


def test_an_album_replying_to_a_photo_brings_both():
    bot = _bot()
    target = FakeMessage(10, photo=photo(), blob=JPEG)
    own = [FakeMessage(i, photo=photo(), blob=JPEG, replied=target, grouped_id=9) for i in (40, 41)]
    own[0].message = 'g which of these matches'
    asyncio.run(bot.album_handler(SimpleNamespace(chat_id=CHAT, sender_id=HUMAN_ID, messages=own)))
    info = bot.get_msg_info(CHAT, 41)
    assert info is not None, 'every message of the album is stored'
    assert [p.hash for p in info.message[1:]] == [f'{CHAT}:10', f'{CHAT}:40', f'{CHAT}:41']
    assert bot.completions == [(CHAT, 40)]


def test_image_media_types():
    assert image_media_type(JPEG) == 'image/jpeg'
    assert image_media_type(PNG) == 'image/png'
    assert image_media_type(b'GIF89a...') == 'image/gif'
    assert image_media_type(b'RIFF\x00\x00\x00\x00WEBPVP8 ') == 'image/webp'
    assert image_media_type(b'unknown') == 'image/jpeg'


def test_import_reads_the_file_it_replies_to():
    sent_earlier = FakeMessage(10, document=document('application/json'))
    command = FakeMessage(11, '/import', replied=sent_earlier)
    assert asyncio.run(_bot()._import_document(command)) is sent_earlier


def test_import_prefers_a_file_attached_to_the_command():
    command = FakeMessage(11, '/import', document=document('application/json'), replied=FakeMessage(10))
    assert asyncio.run(_bot()._import_document(command)) is command


def test_import_reads_a_file_the_bot_sent():
    export = FakeMessage(10, sender_id=BOT_ID, document=document('application/json'))
    assert asyncio.run(_bot()._import_document(FakeMessage(11, '/import', replied=export))) is export
