import asyncio
from types import SimpleNamespace
from typing import Any

from chatgpt_telegram_bot.bot import ChatGPTTelegramBot
from chatgpt_telegram_bot.utils import is_foreign_command, parse_command

ME = 'my_bot'


def test_a_bare_command_parses():
    assert parse_command('/ping', ME) == ('ping', '')
    assert parse_command('/ping@my_bot', ME) == ('ping', '')


def test_a_command_without_a_body_keeps_one_given_for_the_caller_to_refuse():
    assert parse_command('/ping now', ME) == ('ping', 'now')
    assert parse_command('/abort ', ME) == ('abort', '')
    assert parse_command(' /help', ME) == ('help', '')


def test_a_command_with_a_body_takes_the_rest():
    assert parse_command('/regenerate g be brief', ME) == ('regenerate', 'g be brief')
    assert parse_command('/export@my_bot  A title ', ME) == ('export', 'A title ')
    assert parse_command('/import', ME) == ('import', '')


def test_aliases_stand_for_their_command():
    assert parse_command('/re g be brief', ME) == ('regenerate', 'g be brief')
    assert parse_command('/re@my_bot', ME) == ('regenerate', '')
    assert parse_command('/mani\nnew text', ME) == ('manipulate', 'new text')
    assert parse_command('/rege', ME) == ('rege', ''), 'only the listed spellings, not any prefix'


def test_a_body_may_start_on_the_next_line():
    assert parse_command('/manipulate\n<think>x</think> y', ME) == ('manipulate', '<think>x</think> y')


def test_another_bots_command_is_not_ours():
    assert parse_command('/ping@other_bot', ME) is None
    assert parse_command('/import@other_bot x', ME) is None


def test_text_that_is_no_command():
    assert parse_command('', ME) is None
    assert parse_command('g hello', ME) is None
    assert parse_command('/usr/bin is a path', ME) is None
    assert parse_command('/', ME) is None


def test_an_unknown_command_still_parses():
    # the dispatcher sends a name it does not handle to the model, as before
    assert parse_command('/start', ME) == ('start', '')


def test_foreign_commands():
    assert is_foreign_command('/start@other_bot', ME)
    assert is_foreign_command('/import@other_bot file', ME)
    assert not is_foreign_command('/start@my_bot', ME)
    assert not is_foreign_command('/start@My_Bot', ME)
    assert not is_foreign_command('/start', ME)
    assert not is_foreign_command('g /start@other_bot', ME)
    assert not is_foreign_command('mail me@example.com', ME)
    assert not is_foreign_command('', ME)


def _error_bot(allowed: bool) -> Any:
    bot: Any = object.__new__(ChatGPTTelegramBot)
    bot.sent = []

    def is_allowed(_chat_id: int) -> bool:
        return allowed

    async def send_message(_chat_id: int, text: str, _reply_to: int) -> None:
        bot.sent.append(text)

    bot.is_allowed = is_allowed

    bot.send_message = send_message
    return bot


def test_an_argument_to_a_command_that_takes_none_is_refused():
    bot = _error_bot(allowed=True)
    asyncio.run(bot.command_argument_error(SimpleNamespace(chat_id=1, id=2), 'abort'))
    assert bot.sent == ['[!] `/abort` takes no argument']


def test_the_refusal_stays_silent_where_the_command_would():
    bot = _error_bot(allowed=False)
    asyncio.run(bot.command_argument_error(SimpleNamespace(chat_id=1, id=2), 'navigate'))
    assert bot.sent == []
    asyncio.run(bot.command_argument_error(SimpleNamespace(chat_id=1, id=2), 'ping'))
    assert bot.sent == ['[!] `/ping` takes no argument'], '/ping answers anywhere'
