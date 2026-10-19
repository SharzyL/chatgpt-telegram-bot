"""Pruning a thread down to the line that leads to one message."""

import asyncio
from types import SimpleNamespace
from typing import Any, final

from telethon.tl.types import MessageReplyHeader

from chatgpt_telegram_bot.bot import ChatGPTTelegramBot
from chatgpt_telegram_bot.models import MsgInfo, make_text_part
from chatgpt_telegram_bot.reply import format_linearize_listing
from chatgpt_telegram_bot.store import Store
from chatgpt_telegram_bot.utils import PendingReplyManager, linearize_plan, plan_digest

CHAT = -1001234567890
ASKER = 42


def _u(text: str, parent: int | None) -> MsgInfo:
    return MsgInfo(False, [make_text_part(text)], parent, None, None)


def _b(text: str, parent: int) -> MsgInfo:
    return MsgInfo(True, [make_text_part(text)], parent, None, None)


def _tree() -> dict[int, MsgInfo]:
    """A main line 1-7 with answer 1 split over 2 and 3, a fork off each answer, and a second tree."""
    return {
        1: _u('first prompt', None),
        2: _b('first answer', 1),
        3: _b('first answer', 1),
        4: _u('second prompt', 3),
        5: _b('second answer', 4),
        6: _u('third prompt', 5),
        7: _b('third answer', 6),
        8: _u('side prompt', 2),
        9: _b('side answer', 8),
        10: _u('late fork', 5),
        11: _b('late fork answer', 10),
        20: _u('another conversation', None),
        21: _b('its answer', 20),
    }


# --- the plan -----------------------------------------------------------------------------------


def test_the_line_to_the_anchor_is_kept_and_every_branch_goes():
    keep, delete = linearize_plan(_tree(), 5) or (set(), set())
    assert keep == {1, 2, 3, 4, 5}
    assert delete == {6, 7, 8, 9, 10, 11}


def test_every_part_of_a_kept_answer_is_kept():
    # the path runs through part 3 only, but part 2 is the same answer
    keep, _ = linearize_plan(_tree(), 5) or (set(), set())
    assert {2, 3} <= keep


def test_a_prompt_anchor_keeps_its_own_answer():
    keep, _ = linearize_plan(_tree(), 4) or (set(), set())
    assert 5 in keep


def test_anchoring_on_a_fork_drops_the_main_line():
    keep, delete = linearize_plan(_tree(), 9) or (set(), set())
    assert keep == {1, 2, 3, 8, 9}
    assert delete == {4, 5, 6, 7, 10, 11}


def test_the_last_message_of_a_line_deletes_only_side_branches():
    _, delete = linearize_plan(_tree(), 7) or (set(), set())
    assert delete == {8, 9, 10, 11}


def test_another_tree_in_the_chat_is_never_touched():
    keep, delete = linearize_plan(_tree(), 5) or (set(), set())
    assert not {20, 21} & (keep | delete)


def test_an_imported_turn_goes_with_its_prompt_half():
    nodes = {
        -101: _u('imported prompt', None),
        101: _b('imported answer', -101),
        -102: _u('imported second', 101),
        102: _b('imported second answer', -102),
        103: _u('fork', 101),
        104: _b('fork answer', 103),
    }
    _, delete = linearize_plan(nodes, 104) or (set(), set())
    assert delete == {-102, 102}


def test_an_unknown_anchor_has_no_plan():
    assert linearize_plan(_tree(), 404) is None


def test_a_loop_has_no_plan():
    assert linearize_plan({1: _u('a', 2), 2: _b('b', 1)}, 2) is None


def test_the_digest_follows_the_set_not_its_order():
    assert plan_digest({3, 1, 2}) == plan_digest({1, 2, 3})
    assert plan_digest({1, 2}) != plan_digest({1, 2, 3})


# --- the listing ----------------------------------------------------------------------------------


def test_a_split_answer_is_one_line_with_its_part_count():
    nodes = {1: _u('q', None), 2: _b('long answer', 1), 3: _b('long answer', 1)}
    text, count = format_linearize_listing(nodes, {1, 2, 3}, 36, 30)
    assert count == 3
    assert text.count('long answer') == 1
    assert '(2 messages)' in text


def test_an_imported_prompt_is_not_counted_as_a_message():
    nodes = {-5: _u('hidden half', None), 5: _b('shown turn', -5)}
    text, count = format_linearize_listing(nodes, {-5, 5}, 36, 30)
    assert count == 1
    assert 'hidden half' not in text


def test_a_long_listing_is_capped():
    nodes = {i: _u(f'prompt {i}', None) for i in range(1, 51)}
    text, count = format_linearize_listing(nodes, set(nodes), 36, 10)
    assert count == 50
    assert text.count('👤') == 10
    assert '…and 40 more' in text


# --- the command and the confirmation --------------------------------------------------------------


def _bot(missing_after_delete: set[int] | None = None, delete_raises: bool = False) -> Any:
    bot = object.__new__(ChatGPTTelegramBot)
    bot.store = Store(':memory:')
    bot.allowed_chats = {CHAT}
    bot.pending_reply_manager = PendingReplyManager()
    bot.generating = {}
    bot.generation_done = {}
    bot.linearizing = set()
    bot.NAVIGATE_PREVIEW_WIDTH = 36
    bot.LINEARIZE_LIST_LIMIT = 30
    tree = _tree()
    for mid, info in tree.items():
        if mid != 3:
            bot.set_msg_info(CHAT, mid, info)
    # the split first answer is one turn shown by both of its messages, as the bot stores it
    bot.store.put_turn(CHAT, [2, 3], tree[2])
    sent: list[Any] = []
    deleted: list[list[int]] = []
    bot.sent = sent  # pyright: ignore[reportAttributeAccessIssue]  # test-only capture
    bot.deleted = deleted  # pyright: ignore[reportAttributeAccessIssue]  # test-only capture

    async def send_message(_chat_id: int, body: Any, _reply_to: int) -> int:
        sent.append(('plain', body, None))
        return 999

    async def send_message_html(_chat_id: int, text: str, _reply_to: int, buttons: Any = None) -> int:
        sent.append(('html', text, buttons))
        return 999

    async def delete_messages(_chat_id: int, ids: list[int]) -> None:
        if delete_raises:
            raise PermissionError('no delete rights')
        deleted.append(ids)

    async def get_messages(_chat_id: int, ids: list[int]) -> list[Any]:
        # what is still there: anything not deleted, plus whatever Telegram declined
        gone = {i for batch in deleted for i in batch} - (missing_after_delete or set())
        return [SimpleNamespace(id=i) if i not in gone else None for i in ids]

    bot.send_message = send_message  # pyright: ignore[reportAttributeAccessIssue]  # test-only stub
    bot.send_message_html = send_message_html  # pyright: ignore[reportAttributeAccessIssue]  # test-only stub
    bot.delete_messages = delete_messages  # pyright: ignore[reportAttributeAccessIssue]  # test-only stub
    bot.bot = SimpleNamespace(get_messages=get_messages)  # pyright: ignore[reportAttributeAccessIssue]  # test-only stub
    return bot


@final
class FakeMessage:
    def __init__(self, target: int | None) -> None:
        self.chat_id: int = CHAT
        self.sender_id: int = ASKER
        self.id: int = 900
        self.message: str = '/linearize'
        self.reply_to: Any = MessageReplyHeader(reply_to_msg_id=target) if target is not None else None


@final
class FakePress:
    def __init__(self, data: bytes, sender_id: int = ASKER, chat_id: int | None = CHAT) -> None:
        self.chat_id: int | None = chat_id
        self.sender_id: int = sender_id
        self.data: bytes = data
        self.edits: list[str] = []
        self.answers: list[tuple[str | None, bool]] = []

    async def edit(self, text: str, **_kwargs: Any) -> None:
        self.edits.append(text)

    async def answer(self, message: str | None = None, alert: bool = False) -> None:
        self.answers.append((message, alert))


def _ask(bot: Any, anchor: int) -> list[Any]:
    asyncio.run(bot.linearize_handler(FakeMessage(anchor)))
    return bot.sent


def _button_data(buttons: Any) -> list[bytes]:
    return [b.type.data for b in buttons[0]]


def test_the_command_needs_a_reply():
    bot = _bot()
    asyncio.run(bot.linearize_handler(FakeMessage(None)))
    assert bot.sent[0][1].startswith('[!] `/linearize`')


def test_the_confirmation_lists_what_will_go():
    bot = _bot()
    kind, text, buttons = _ask(bot, 5)[0]
    assert kind == 'html'
    assert 'Delete 6 messages?' in text
    for gone in ('third prompt', 'side answer', 'late fork'):
        assert gone in text
    for kept in ('first prompt', 'second answer', 'another conversation'):
        assert kept not in text
    go, no = _button_data(buttons)
    digest = plan_digest({6, 7, 8, 9, 10, 11})
    assert go == f'lin:5:{digest}:{ASKER}:go'.encode()
    assert no == f'lin:5:{digest}:{ASKER}:no'.encode()


def test_nothing_to_delete_says_so():
    bot = _bot()
    bot.store.delete(CHAT, [8, 9, 10, 11])
    assert _ask(bot, 7)[0][1] == 'Nothing to delete: that thread is already a single line.'


def test_a_thread_that_is_generating_is_refused():
    bot = _bot()
    bot.generating[(CHAT, 10)] = False  # the late fork's prompt is being answered
    assert _ask(bot, 5)[0][1].startswith('[!] Part of that thread is still generating')


def _confirm(
    bot: Any, anchor: int = 5, action: str = 'go', sender: int = ASKER, digest: str | None = None
) -> FakePress:
    if digest is None:
        plan = linearize_plan(bot._chat_nodes(CHAT), anchor)
        assert plan is not None
        digest = plan_digest(plan[1])
    press = FakePress(f'lin:{anchor}:{digest}:{ASKER}:{action}'.encode(), sender_id=sender)
    asyncio.run(bot.linearize_callback(press))
    return press


def test_confirming_deletes_the_messages_and_their_history():
    bot = _bot()
    press = _confirm(bot)
    assert sorted(i for batch in bot.deleted for i in batch) == [6, 7, 8, 9, 10, 11]
    assert press.edits[-1] == '🧹 Deleted 6 messages.'
    stored = set(bot.store.chat_nodes(CHAT))
    assert stored == {1, 2, 3, 4, 5, 20, 21}


def test_a_message_telegram_refused_keeps_its_history():
    # a survivor can still be replied to, so its row must stay
    bot = _bot(missing_after_delete={8})
    press = _confirm(bot)
    assert 8 in bot.store.chat_nodes(CHAT)
    assert 9 not in bot.store.chat_nodes(CHAT)
    assert 'Deleted 5 messages.' in press.edits[-1]
    assert '1 could not be deleted' in press.edits[-1]


def test_a_failed_delete_call_deletes_no_history():
    bot = _bot(delete_raises=True)
    press = _confirm(bot)
    assert len(bot.store.chat_nodes(CHAT)) == len(_tree())
    assert '6 could not be deleted' in press.edits[-1]


def test_someone_else_cannot_confirm():
    bot = _bot()
    press = _confirm(bot, sender=7)
    assert bot.deleted == []
    assert press.answers == [('Only whoever asked can confirm this', True)]


def test_cancel_deletes_nothing():
    bot = _bot()
    press = _confirm(bot, action='no')
    assert bot.deleted == []
    assert press.edits == ['🧹 Linearize cancelled.']


def test_a_thread_that_changed_since_is_not_deleted():
    bot = _bot()
    press = _confirm(bot, digest='00000000')
    assert bot.deleted == []
    assert press.edits == ['🧹 That thread has changed since; run /linearize again.']


def test_a_second_press_while_deleting_is_ignored():
    bot = _bot()
    bot.linearizing.add((CHAT, 5))
    press = _confirm(bot)
    assert bot.deleted == []
    assert press.answers == [('Already deleting', False)]


def test_a_chat_that_is_not_allowed_gets_nothing():
    bot = _bot()
    press = FakePress(f'lin:5:{plan_digest({1})}:{ASKER}:go'.encode(), chat_id=-1009999)
    asyncio.run(bot.linearize_callback(press))
    assert bot.deleted == []
    assert press.edits == []
