"""Replying to a message that carries media brings that media into the turn, and /chat, which opens a
thread whatever it replies to."""

import asyncio
from types import SimpleNamespace
from typing import Any, final
from zoneinfo import ZoneInfo

import pytest
from telethon.tl.types import MessageReplyHeader

from chatgpt_telegram_bot.bot import ChatGPTTelegramBot
from chatgpt_telegram_bot.commands import apply_overrides, parse_chat_spec, parse_command
from chatgpt_telegram_bot.models import EndPoint, Model, MsgInfo, make_text_part
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
    bot.endpoints = [EndPoint(name='main', url='https://main.test'), EndPoint(name='alt', url='https://alt.test')]
    bot.endpoint_by_name = {e.name: e for e in bot.endpoints}
    bot.default_endpoint = 'main'
    # every client is made on demand; a stand-in keeps the test off the network and the env
    bot.clients = {}

    def make_client(*triple: str) -> Any:
        return SimpleNamespace(triple=triple)

    bot._make_client = make_client
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

    bot.send_message = send_message
    bot._run_completion = run_completion
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


# /chat


def _chat(bot: Any, message: FakeMessage) -> MsgInfo | None:
    command = parse_command(message.message, 'the_bot')
    assert command is not None
    assert command[0] == 'chat'
    asyncio.run(bot.chat_handler(message, command[1]))
    return bot.get_msg_info(CHAT, message.id)


def test_a_spec_splits_into_model_overrides_and_prompt():
    assert parse_chat_spec('alt/some-model,suffix=/v2,t=high hello there') == (
        'alt/some-model',
        {'suffix': '/v2', 't': 'high'},
        'hello there',
    )
    assert parse_chat_spec('g,[be terse, please] hi') == ('g', {'system_prompt': 'be terse, please'}, 'hi')
    assert parse_chat_spec('g\n\nfirst line\nsecond') == ('g', {}, 'first line\nsecond')
    assert parse_chat_spec('g') == ('g', {}, '')


def test_a_configured_prefix_takes_its_flags():
    bot = _bot()
    info = _chat(bot, FakeMessage(11, '/chat g,s what is new'))
    assert info is not None
    assert (info.prefix, info.model_spec, info.overrides) == ('g', None, {'s': None})
    assert info.message[0].text == 'what is new'
    assert bot.completions == [(CHAT, 11)]


def test_an_unconfigured_model_is_stored_as_its_spec():
    bot = _bot()
    info = _chat(bot, FakeMessage(11, '/chat alt/vendor/big-model,suffix=/v2,api_type=openai_legacy hi'))
    assert info is not None
    assert info.prefix is None, 'no prefix stands for it'
    assert info.model_spec == 'alt/vendor/big-model'
    assert bot.clients == {('alt', 'openai_legacy', '/v2'): SimpleNamespace(triple=('alt', 'openai_legacy', '/v2'))}

    bot.set_msg_info(
        CHAT,
        12,
        MsgInfo(sent_by_bot=True, message=[make_text_part('hello')], reply_id=11, prefix=None, system_prompt=None),
    )
    bot.set_msg_info(
        CHAT,
        13,
        MsgInfo(sent_by_bot=False, message=[make_text_part('more')], reply_id=12, prefix=None, system_prompt=None),
    )
    _, model, _ = bot.construct_chat_history(CHAT, 13)
    assert (model.endpoint, model.name, model.suffix, model.api_type) == (
        'alt',
        'vendor/big-model',
        '/v2',
        'openai_legacy',
    )


def test_a_prefix_that_looks_like_a_spec_means_the_configured_model():
    bot = _bot()
    bot.models = [Model(prefix='alt/x', name='configured-x')]
    info = _chat(bot, FakeMessage(11, '/chat alt/x hi'))
    assert info is not None
    assert (info.prefix, info.model_spec) == ('alt/x', None)


@pytest.mark.parametrize(
    ('text', 'error'),
    [
        ('/chat nope hi', '[!] `nope` is neither a model prefix nor `<endpoint>/<model>`; /models lists both'),
        ('/chat gone/m hi', '[!] Unknown endpoint `gone`; /models lists the endpoints'),
        ('/chat alt/m,api_type=grpc hi', '[!] api_type must be one of openai, openai_legacy, anthropic'),
        ('/chat alt/m,api_type=anthropic,dt hi', '[!] dt needs api_type openai'),
    ],
)
def test_a_bad_spec_is_reported(text: str, error: str):
    bot = _bot()
    assert _chat(bot, FakeMessage(11, text)) is None
    assert bot.sent == [error]
    assert bot.completions == []


def test_a_bare_chat_shows_usage():
    bot = _bot()
    assert _chat(bot, FakeMessage(11, '/chat')) is None
    assert bot.sent[0].startswith('[!] `/chat` starts a new conversation')


def test_a_missing_api_key_is_reported_up_front():
    bot = _bot()

    def no_key(ep_name: str, *_rest: str) -> Any:
        raise ValueError(f'No API key for endpoint "{ep_name}": set OPENAI_API_KEY_{ep_name}')

    bot._make_client = no_key
    assert _chat(bot, FakeMessage(11, '/chat alt/m hi')) is None
    assert bot.sent == ['[!] No API key for endpoint "alt": set OPENAI_API_KEY_alt']


def test_replying_to_a_bot_turn_with_media_starts_afresh_with_that_media():
    bot = _bot()
    bot.set_msg_info(CHAT, 10, MsgInfo(sent_by_bot=True, message=[], reply_id=9, prefix=None, system_prompt=None))
    turn = FakeMessage(10, sender_id=BOT_ID, photo=photo(), blob=JPEG)
    info = _chat(bot, FakeMessage(11, '/chat g what is this', replied=turn))
    assert info is not None
    assert info.reply_id is None, 'a plain reply would have continued that turn'
    assert [p.type_ for p in info.message] == ['text', 'image']
    assert bot.cached == {f'{CHAT}:10': JPEG}


def test_replying_to_a_bot_text_starts_afresh_without_its_history():
    bot = _bot()
    bot.set_msg_info(CHAT, 10, MsgInfo(sent_by_bot=True, message=[], reply_id=9, prefix=None, system_prompt=None))
    info = _chat(bot, FakeMessage(11, '/chat g new topic', replied=FakeMessage(10, sender_id=BOT_ID)))
    assert info is not None
    assert info.reply_id is None
    assert [p.type_ for p in info.message] == ['text']


def test_a_photo_captioned_with_chat_is_part_of_it():
    bot = _bot()
    info = _chat(bot, FakeMessage(11, '/chat alt/m', photo=photo(), blob=JPEG))
    assert info is not None
    assert [p.type_ for p in info.message] == ['text', 'image']
    assert info.message[0].text == 'Describe the image'


def test_an_album_captioned_with_chat():
    bot = _bot()
    own = [FakeMessage(i, photo=photo(), blob=JPEG, grouped_id=9) for i in (40, 41)]
    own[0].message = '/chat alt/m compare'
    asyncio.run(bot.album_handler(SimpleNamespace(chat_id=CHAT, sender_id=HUMAN_ID, messages=own)))
    info = bot.get_msg_info(CHAT, 41)
    assert info is not None
    assert (info.model_spec, info.message[0].text) == ('alt/m', 'compare')


def test_flags_on_a_configured_model_change_its_backend():
    model = apply_overrides(Model(prefix='g', name='m'), {'api_type': 'anthropic', 't': None, 'suffix': ''})
    assert (model.api_type, model.thinking, model.suffix) == ('anthropic', 'adaptive', '')
