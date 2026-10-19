from typing import Any

from telethon.extensions import html

from chatgpt_telegram_bot.bot import BOT_COMMANDS, ChatGPTTelegramBot
from chatgpt_telegram_bot.models import Model
from chatgpt_telegram_bot.utils import (
    COMMAND_ALIASES,
    COMMANDS_WITH_BODY,
    COMMANDS_WITHOUT_BODY,
    pack_html_sections,
    telegram_len,
)


def _bot() -> Any:
    bot: Any = object.__new__(ChatGPTTelegramBot)
    bot.models = [Model(prefix='g', name='gpt-test')]
    bot.default_endpoint = 'default'
    bot.system_prompt = 'be helpful'
    return bot


def _help_text() -> str:
    return '\n\n'.join(html.parse(s)[0] for s in _bot()._build_help_sections())


def test_the_palette_offers_only_commands_safe_to_send_bare():
    registered = {name for name, _ in BOT_COMMANDS}
    assert registered == COMMANDS_WITHOUT_BODY | {'export'}
    # a bare /regenerate overwrites a reply, and the other two can only fail without a body
    assert not registered & {'regenerate', 'manipulate', 'import'}


def test_help_describes_every_command_and_alias():
    text = _help_text()
    for name in COMMANDS_WITH_BODY | COMMANDS_WITHOUT_BODY | COMMAND_ALIASES.keys():
        assert f'/{name}' in text, name


def test_sections_share_a_message_while_they_fit():
    assert pack_html_sections(['<b>a</b>', 'b'], 10) == ['<b>a</b>\n\nb']


def test_tags_do_not_count_towards_the_limit():
    # 'aaaa\n\nbbbb' is 10 characters once parsed, though the source is longer
    assert pack_html_sections(['<code>aaaa</code>', '<b>bbbb</b>'], 10) == ['<code>aaaa</code>\n\n<b>bbbb</b>']


def test_a_section_that_would_overflow_starts_a_new_message():
    packed = pack_html_sections(['a' * 6, 'b' * 6, 'c' * 2], 10)
    assert packed == ['a' * 6, 'b' * 6 + '\n\n' + 'c' * 2]
    assert all(telegram_len(html.parse(m)[0]) <= 10 for m in packed)


def test_an_oversized_section_is_sent_whole():
    assert pack_html_sections(['x' * 20, 'y'], 10) == ['x' * 20, 'y']
