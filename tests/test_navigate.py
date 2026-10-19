"""Listing a thread's turns, a page at a time."""

import asyncio
from typing import Any, final

import pytest
from telethon import errors
from telethon.tl.types import MessageReplyHeader

from chatgpt_telegram_bot.bot import ChatGPTTelegramBot
from chatgpt_telegram_bot.markdown import preview_line
from chatgpt_telegram_bot.models import MsgInfo, ThreadTurn, make_image_part, make_text_part
from chatgpt_telegram_bot.reply import format_navigation_page
from chatgpt_telegram_bot.store import Store
from chatgpt_telegram_bot.utils import PendingReplyManager, message_link

GROUP = -1001234567890
PRIVATE = 629325599


def _columns(s: str) -> int:
    import unicodedata

    return sum(2 if unicodedata.east_asian_width(c) in ('W', 'F') else 1 for c in s)


# --- preview_line --------------------------------------------------------------------------


def test_a_short_line_is_kept_whole():
    assert preview_line('hello there', 36) == 'hello there'


def test_a_long_line_is_cut_with_an_ellipsis_inside_the_width():
    out = preview_line('a' * 80, 36)
    assert out.endswith('…')
    assert _columns(out) == 36


def test_wide_characters_count_two_columns():
    out = preview_line('这是一段用来测试宽字符截断的中文句子，长度明显超过一行的宽度', 36)
    assert out.endswith('…')
    assert _columns(out) <= 36
    assert len(out) < 20, 'a CJK preview holds about half as many characters'


def test_a_line_exactly_at_the_width_is_not_cut():
    assert preview_line('x' * 36, 36) == 'x' * 36


def test_newlines_and_markdown_noise_are_flattened():
    assert preview_line('### **Title**\n\n> quoted `code`\n- item', 80) == 'Title quoted code item'


# --- message_link ------------------------------------------------------------------------------


def test_a_supergroup_message_has_a_link():
    assert message_link(GROUP, 42) == 'https://t.me/c/1234567890/42'


def test_a_private_chat_has_no_links():
    assert message_link(PRIVATE, 42) is None


def test_the_stored_half_of_an_imported_turn_has_no_link():
    assert message_link(GROUP, -42) is None


# --- format_navigation_page ----------------------------------------------------------------------


def _turns(n: int) -> list[ThreadTurn]:
    return [ThreadTurn(prompt=f'prompt {i}', response=f'answer {i}', msg_id=100 + i) for i in range(1, n + 1)]


def test_a_page_numbers_turns_across_the_whole_thread():
    text, page, pages = format_navigation_page(_turns(20), 1, 8, 36, lambda _mid: None)
    assert (page, pages) == (1, 3)
    assert '<b>9.</b> 👤 prompt 9' in text
    assert '<b>16.</b>' in text
    assert '<b>17.</b>' not in text


def test_a_page_past_the_end_is_clamped():
    _, page, pages = format_navigation_page(_turns(3), 99, 8, 36, lambda _mid: None)
    assert (page, pages) == (0, 1)


def test_answers_link_to_their_message_where_the_chat_allows():
    text, _, _ = format_navigation_page(_turns(1), 0, 8, 36, lambda mid: f'https://t.me/c/1/{mid}')
    assert '<a href="https://t.me/c/1/101">answer 1</a>' in text


def test_an_unanswered_prompt_is_marked():
    turns = [ThreadTurn(prompt='waiting', response=None, msg_id=5)]
    text, _, _ = format_navigation_page(turns, 0, 8, 36, lambda _mid: None)
    assert '🤖 <i>…</i>' in text


def test_content_is_escaped():
    turns = [ThreadTurn(prompt='a <b>tag</b> & more', response='x < y', msg_id=5)]
    text, _, _ = format_navigation_page(turns, 0, 8, 36, lambda _mid: None)
    assert '&lt;b&gt;tag&lt;/b&gt; &amp; more' in text
    assert 'x &lt; y' in text


# --- walking the thread --------------------------------------------------------------------------


def _bot(chat: int = GROUP) -> Any:
    bot = object.__new__(ChatGPTTelegramBot)
    bot.store = Store(':memory:')
    bot.allowed_chats = {chat}
    bot.pending_reply_manager = PendingReplyManager()
    bot.NAVIGATE_PER_PAGE = 8
    bot.NAVIGATE_PREVIEW_WIDTH = 36
    sent: list[Any] = []
    bot.sent = sent  # pyright: ignore[reportAttributeAccessIssue]  # test-only capture

    async def send_message(_chat_id: int, body: Any, _reply_to: int) -> int:
        sent.append(('plain', body, None))
        return 999

    async def send_message_html(_chat_id: int, text: str, _reply_to: int, buttons: Any = None) -> int:
        sent.append(('html', text, buttons))
        return 999

    bot.send_message = send_message  # pyright: ignore[reportAttributeAccessIssue]  # test-only stub
    bot.send_message_html = send_message_html  # pyright: ignore[reportAttributeAccessIssue]  # test-only stub
    return bot


def _thread(bot: Any, turns: int, chat: int = GROUP) -> int:
    """Store `turns` prompt/answer pairs as a chain; returns the last answer's id."""
    reply_id: int | None = None
    mid = 0
    for i in range(1, turns + 1):
        prompt_id, answer_id = 2 * i - 1, 2 * i
        bot.set_msg_info(
            chat, prompt_id, MsgInfo(False, [make_text_part(f'prompt {i}')], reply_id, 'g' if i == 1 else None, None)
        )
        bot.set_msg_info(chat, answer_id, MsgInfo(True, [make_text_part(f'answer {i}')], prompt_id, None, None))
        reply_id = mid = answer_id
    return mid


def test_the_walk_pairs_prompts_with_their_answers_oldest_first():
    bot = _bot()
    last = _thread(bot, 3)
    turns = bot._thread_turns(GROUP, last)
    assert [(t.prompt, t.response, t.msg_id) for t in turns] == [
        ('prompt 1', 'answer 1', 2),
        ('prompt 2', 'answer 2', 4),
        ('prompt 3', 'answer 3', 6),
    ]


def test_anchored_on_a_prompt_its_answer_is_not_part_of_the_branch():
    bot = _bot()
    _ = _thread(bot, 2)
    turns = bot._thread_turns(GROUP, 3)  # prompt 2
    assert turns[-1] == ThreadTurn(prompt='prompt 2', response=None, msg_id=3)


def test_a_split_answer_links_to_its_first_part():
    bot = _bot()
    bot.set_msg_info(GROUP, 1, MsgInfo(False, [make_text_part('q')], None, 'g', None))
    answer = MsgInfo(True, [make_text_part('long')], 1, None, None, None, [10, 11, 12])
    bot.store.put_turn(GROUP, [10, 11, 12], answer)
    assert bot._thread_turns(GROUP, 12) == [ThreadTurn(prompt='q', response='long', msg_id=10)]


def test_an_image_prompt_is_marked():
    bot = _bot()
    bot.set_msg_info(GROUP, 1, MsgInfo(False, [make_text_part('look'), make_image_part('ab')], None, 'g', None))
    assert bot._thread_turns(GROUP, 1)[0].prompt == '🖼 look'


def test_an_unknown_anchor_is_none():
    assert _bot()._thread_turns(GROUP, 404) is None


def test_a_loop_in_damaged_data_terminates():
    bot = _bot()
    bot.set_msg_info(GROUP, 1, MsgInfo(False, [make_text_part('a')], 2, 'g', None))
    bot.set_msg_info(GROUP, 2, MsgInfo(True, [make_text_part('b')], 1, None, None))
    assert bot._thread_turns(GROUP, 2) is not None


# --- pages and buttons ---------------------------------------------------------------------------


def _buttons(markup: Any) -> list[tuple[str, bytes]]:
    # in this Telethon the callback payload sits on the button's type, not the button
    return [(b.text, b.type.data) for b in markup[0]] if markup else []


def test_a_short_thread_has_no_buttons():
    bot = _bot()
    text, buttons = bot._navigation(GROUP, _thread(bot, 3), None)
    assert buttons is None
    assert '3 turns' in text


def test_a_long_thread_opens_on_its_last_page():
    bot = _bot()
    last = _thread(bot, 20)
    text, buttons = bot._navigation(GROUP, last, None)
    assert '<b>20.</b>' in text
    assert _buttons(buttons) == [('‹', f'nav:{last}:1'.encode()), ('3/3', b'nav:noop')]


def test_a_middle_page_goes_both_ways():
    bot = _bot()
    last = _thread(bot, 20)
    _, buttons = bot._navigation(GROUP, last, 1)
    assert _buttons(buttons) == [
        ('‹', f'nav:{last}:0'.encode()),
        ('2/3', b'nav:noop'),
        ('›', f'nav:{last}:2'.encode()),
    ]


# --- the command and the callback --------------------------------------------------------------


@final
class FakeMessage:
    def __init__(self, target: int | None) -> None:
        self.chat_id: int = GROUP
        self.sender_id: int = 1
        self.id: int = 900
        self.message: str = '/navigate'
        self.reply_to: Any = MessageReplyHeader(reply_to_msg_id=target) if target is not None else None


def test_the_command_needs_a_reply():
    bot = _bot()
    asyncio.run(bot.navigate_handler(FakeMessage(None)))
    assert bot.sent[0][0] == 'plain'
    assert bot.sent[0][1].startswith('[!] `/navigate`')


def test_the_command_sends_the_listing():
    bot = _bot()
    asyncio.run(bot.navigate_handler(FakeMessage(_thread(bot, 2))))
    kind, text, _ = bot.sent[0]
    assert kind == 'html'
    assert 'prompt 1' in text and 'answer 2' in text


def test_an_unstored_thread_is_reported():
    bot = _bot()
    asyncio.run(bot.navigate_handler(FakeMessage(404)))
    assert bot.sent[0][1] == '[!] No stored conversation for that message'


@final
class FakeCallback:
    def __init__(self, data: bytes, chat_id: int | None = GROUP, edit_error: Exception | None = None) -> None:
        self.chat_id: int | None = chat_id
        self.data: bytes = data
        self.edits: list[tuple[str, Any]] = []
        self.answers: list[tuple[str | None, bool]] = []
        self._edit_error: Exception | None = edit_error

    async def edit(self, text: str, **kwargs: Any) -> None:
        if self._edit_error is not None:
            raise self._edit_error
        self.edits.append((text, kwargs.get('buttons')))

    async def answer(self, message: str | None = None, alert: bool = False) -> None:
        self.answers.append((message, alert))


def test_a_page_button_edits_the_listing():
    bot = _bot()
    last = _thread(bot, 20)
    press = FakeCallback(f'nav:{last}:0'.encode())
    asyncio.run(bot.navigate_callback(press))
    text, buttons = press.edits[0]
    assert '<b>1.</b>' in text and '<b>9.</b>' not in text
    assert _buttons(buttons)[0] == ('1/3', b'nav:noop')
    assert press.answers == [(None, False)], 'the spinner is always dismissed'


def test_the_page_indicator_only_dismisses_the_spinner():
    press = FakeCallback(b'nav:noop')
    asyncio.run(_bot().navigate_callback(press))
    assert press.edits == []
    assert press.answers == [(None, False)]


def test_a_chat_that_is_not_allowed_gets_nothing():
    bot = _bot()
    press = FakeCallback(f'nav:{_thread(bot, 20)}:0'.encode(), chat_id=-1009999)
    asyncio.run(bot.navigate_callback(press))
    assert press.edits == []
    assert press.answers == [(None, False)]


def test_a_thread_that_has_gone_alerts():
    press = FakeCallback(b'nav:404:0')
    asyncio.run(_bot().navigate_callback(press))
    assert press.answers == [('That thread is no longer stored', True)]


def test_malformed_data_is_ignored():
    press = FakeCallback(b'nav:what')
    asyncio.run(_bot().navigate_callback(press))
    assert press.edits == []
    assert press.answers == [(None, False)]


def test_pressing_the_page_already_shown_is_not_an_error():
    bot = _bot()
    last = _thread(bot, 20)
    press = FakeCallback(f'nav:{last}:2'.encode(), edit_error=errors.MessageNotModifiedError(request=None))
    asyncio.run(bot.navigate_callback(press))
    assert press.answers == [(None, False)]


@pytest.mark.parametrize('turns', [1, 8, 9, 64])
def test_every_page_fits_in_one_message(turns: int):
    from chatgpt_telegram_bot.markdown import telegram_len

    bot = _bot()
    long_prompt = '很长的提问' * 40
    for i in range(1, turns + 1):
        prev = 2 * i - 2 if i > 1 else None
        bot.set_msg_info(GROUP, 2 * i - 1, MsgInfo(False, [make_text_part(long_prompt)], prev, 'g', None))
        bot.set_msg_info(GROUP, 2 * i, MsgInfo(True, [make_text_part('x' * 500)], 2 * i - 1, None, None))
    for page in range(-(-turns // 8)):
        text, _ = bot._navigation(GROUP, 2 * turns, page)
        assert telegram_len(text) < 4096
