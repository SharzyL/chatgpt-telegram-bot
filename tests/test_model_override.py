"""Resolving which model a turn runs on, including a prefix that switches it mid-thread."""

from typing import Any
from zoneinfo import ZoneInfo

from chatgpt_telegram_bot.bot import ChatGPTTelegramBot
from chatgpt_telegram_bot.models import Model, MsgInfo, make_text_part

FAST = Model(prefix='f', name='fast-model')
SLOW = Model(prefix='s', name='slow-model', richtext=True)


def _bot() -> ChatGPTTelegramBot:
    """A bot with just the state construct_chat_history reads."""
    bot = object.__new__(ChatGPTTelegramBot)
    bot.db = {}  # pyright: ignore[reportAttributeAccessIssue]  # a dict is enough of a shelf here
    bot.models = [FAST, SLOW]
    bot.system_prompt = 'be helpful'
    bot.timezone = ZoneInfo('UTC')
    return bot


def _chain(bot: ChatGPTTelegramBot, *turns: dict[str, Any]) -> int:
    """Store a reply chain of alternating user/bot messages; returns the last message id."""
    reply_id: int | None = None
    msg_id = 0
    for msg_id, turn in enumerate(turns, start=1):
        bot.set_msg_info(
            1,
            msg_id,
            MsgInfo(
                sent_by_bot=msg_id % 2 == 0,
                message=[make_text_part(turn.get('text', 'hi'))],
                reply_id=reply_id,
                prefix=turn.get('prefix'),
                system_prompt=turn.get('system_prompt'),
                overrides=turn.get('overrides'),
            ),
        )
        reply_id = msg_id
    return msg_id


def test_model_carries_over_when_the_reply_names_none():
    bot = _bot()
    last = _chain(bot, {'prefix': 'f', 'system_prompt': 'rooted'}, {}, {})
    _, model, system_prompt = bot.construct_chat_history(1, last)
    assert model.name == 'fast-model'
    assert system_prompt == 'rooted'


def test_a_prefix_on_a_reply_switches_the_model():
    bot = _bot()
    last = _chain(bot, {'prefix': 'f', 'system_prompt': 'rooted'}, {}, {'prefix': 's', 'system_prompt': 'switched'})
    _, model, system_prompt = bot.construct_chat_history(1, last)
    assert model.name == 'slow-model'
    # the switch brings its own prompt, which is how the new renderer's guidance arrives
    assert system_prompt == 'switched'


def test_the_switch_holds_for_later_turns():
    bot = _bot()
    last = _chain(
        bot,
        {'prefix': 'f', 'system_prompt': 'rooted'},
        {},
        {'prefix': 's', 'system_prompt': 'switched'},
        {},
        {},
    )
    _, model, _ = bot.construct_chat_history(1, last)
    assert model.name == 'slow-model'


def test_the_most_recent_switch_wins():
    bot = _bot()
    last = _chain(
        bot,
        {'prefix': 's', 'system_prompt': 'rooted'},
        {},
        {'prefix': 'f', 'system_prompt': 'one'},
        {},
        {'prefix': 's', 'system_prompt': 'two'},
    )
    _, model, system_prompt = bot.construct_chat_history(1, last)
    assert model.name == 'slow-model'
    assert system_prompt == 'two'


def test_overrides_of_the_switching_reply_apply():
    bot = _bot()
    last = _chain(
        bot,
        {'prefix': 'f', 'system_prompt': 'rooted', 'overrides': {'t': 'high'}},
        {},
        {'prefix': 's', 'system_prompt': 'switched', 'overrides': {'t': 'low'}},
    )
    _, model, _ = bot.construct_chat_history(1, last)
    assert model.name == 'slow-model'
    assert model.thinking == 'low'


def test_an_unknown_stored_prefix_falls_back_to_an_earlier_one():
    # the config can lose a model between turns; the chain still has to resolve
    bot = _bot()
    last = _chain(bot, {'prefix': 'f', 'system_prompt': 'rooted'}, {}, {'prefix': 'gone'})
    _, model, _ = bot.construct_chat_history(1, last)
    assert model.name == 'fast-model'


def test_a_switch_without_a_stored_prompt_falls_back_to_the_new_model_default():
    # the reply carried a prefix but no prompt of its own, so the default is rebuilt for it
    bot = _bot()
    last = _chain(bot, {'prefix': 'f'}, {}, {'prefix': 's'})
    _, model, system_prompt = bot.construct_chat_history(1, last)
    assert model.name == 'slow-model'
    assert system_prompt.startswith('be helpful')


def test_history_is_oldest_first_and_complete():
    bot = _bot()
    last = _chain(bot, {'prefix': 'f', 'text': 'one'}, {'text': 'two'}, {'prefix': 's', 'text': 'three'})
    history, _, _ = bot.construct_chat_history(1, last)
    assert [p[0].text for p in history] == ['one', 'two', 'three']
