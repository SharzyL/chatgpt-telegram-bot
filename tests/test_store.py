"""The SQLite conversation store, and the migration from the shelve database before it."""

import os
import shelve
from pathlib import Path

import pytest

from chatgpt_telegram_bot.models import (
    MsgInfo,
    make_image_part,
    make_reasoning_part,
    make_text_part,
    make_thinking_text_part,
)
from chatgpt_telegram_bot.store import CODEC_JSON, CODEC_ZSTD, Store, migrate_shelf, zstd

CHAT = -100
OTHER = -200


def _user(text: str, reply_id: int | None, prefix: str | None = None) -> MsgInfo:
    return MsgInfo(False, [make_text_part(text)], reply_id, prefix, 'be helpful' if prefix else None)


def _bot(text: str, reply_id: int, part_ids: list[int] | None = None) -> MsgInfo:
    return MsgInfo(True, [make_text_part(text)], reply_id, None, None, None, part_ids)


def _count(store: Store, table: str) -> int:
    return store.conn.execute(f'SELECT count(*) FROM {table}').fetchone()[0]


def test_boolean_fields_are_declared_boolean():
    store = Store(':memory:')
    columns = {name: type_ for _, name, type_, *_ in store.conn.execute('PRAGMA table_info(turn)')}
    assert columns['sent_by_bot'] == 'BOOLEAN'


def test_a_turn_reads_back_as_written():
    store = Store(':memory:')
    info = MsgInfo(
        sent_by_bot=False,
        message=[make_text_part('look'), make_image_part('-100:7')],
        reply_id=None,
        prefix='g',
        system_prompt='be helpful',
        overrides={'t': 'high', 's': None},
        model_spec=None,
    )
    store.put_turn(CHAT, [7], info)
    got = store.get(CHAT, 7)
    assert got == info._replace(part_ids=[7])
    assert got is not None and got.sent_by_bot is False


def test_a_split_answer_is_stored_once():
    store = Store(':memory:')
    parts = [make_thinking_text_part('a long chain of thought ' * 100), make_reasoning_part('m', {'type': 'reasoning'})]
    answer = MsgInfo(True, [*parts, make_text_part('the answer')], 10, None, None)
    store.put_turn(CHAT, [11, 12, 13], answer)
    assert _count(store, 'turn') == 1
    assert _count(store, 'message') == 3
    for mid in (11, 12, 13):
        got = store.get(CHAT, mid)
        assert got is not None and got.message == answer.message and got.part_ids == [11, 12, 13]


def test_a_shorter_regeneration_keeps_dropped_parts_on_the_new_answer():
    store = Store(':memory:')
    store.put_turn(CHAT, [11, 12, 13], _bot('the old answer', 10))
    store.put_turn(CHAT, [11], _bot('a short answer', 10))
    assert _count(store, 'turn') == 1
    for mid in (11, 13):
        got = store.get(CHAT, mid)
        assert got is not None and got.message[-1].text == 'a short answer' and got.part_ids == [11]


def test_a_longer_regeneration_extends_the_same_turn():
    store = Store(':memory:')
    store.put_turn(CHAT, [11], _bot('short', 10))
    store.put_turn(CHAT, [11, 14], _bot('much longer', 10))
    assert _count(store, 'turn') == 1
    got = store.get(CHAT, 14)
    assert got is not None and got.part_ids == [11, 14]


def test_an_update_rewrites_every_part_and_keeps_the_mapping():
    store = Store(':memory:')
    store.put_turn(CHAT, [11, 12], _bot('before', 10))
    store.update(CHAT, 12, _bot('after', 10))
    for mid in (11, 12):
        got = store.get(CHAT, mid)
        assert got is not None and got.message[-1].text == 'after' and got.part_ids == [11, 12]


def test_an_update_of_an_unknown_message_stores_a_new_turn():
    store = Store(':memory:')
    store.update(CHAT, 5, _user('hi', None, 'g'))
    assert store.get(CHAT, 5) == _user('hi', None, 'g')._replace(part_ids=[5])


def test_a_turn_goes_only_with_its_last_message():
    store = Store(':memory:')
    store.put_turn(CHAT, [11, 12], _bot('answer', 10))
    store.delete(CHAT, [12])
    assert _count(store, 'turn') == 1
    got = store.get(CHAT, 11)
    assert got is not None and got.part_ids == [11]
    store.delete(CHAT, [11, 999])
    assert _count(store, 'turn') == 0
    assert store.get(CHAT, 11) is None


def test_chat_nodes_covers_one_chat_and_shares_a_turn_between_parts():
    store = Store(':memory:')
    store.put_turn(CHAT, [1], _user('q', None, 'g'))
    store.put_turn(CHAT, [2, 3], _bot('a', 1))
    store.put_turn(OTHER, [1], _user('elsewhere', None, 'g'))
    nodes = store.chat_nodes(CHAT)
    assert sorted(nodes) == [1, 2, 3]
    assert nodes[2] is nodes[3] and nodes[2].part_ids == [2, 3]
    assert store.get(OTHER, 1) == _user('elsewhere', None, 'g')._replace(part_ids=[1])


@pytest.mark.skipif(zstd is None, reason='zstd is in the standard library from Python 3.14')
def test_compression_is_on_by_default_and_rows_of_either_codec_read_back(tmp_path: Path):
    path = str(tmp_path / 'db-v2.sqlite3')
    text = 'the same sentence again. ' * 400
    store = Store(path)
    store.put_turn(CHAT, [1], _user(text, None, 'g'))
    plain = Store(path, compress=False)
    plain.put_turn(CHAT, [2], _bot(text, 1))
    codecs = dict(store.conn.execute('SELECT sent_by_bot, codec FROM turn').fetchall())
    assert codecs == {0: CODEC_ZSTD, 1: CODEC_JSON}
    sizes = dict(store.conn.execute('SELECT codec, length(parts) FROM turn').fetchall())
    assert sizes[CODEC_ZSTD] * 10 < sizes[CODEC_JSON]
    for reader in (store, plain):
        assert [reader.get(CHAT, mid).message[0].text for mid in (1, 2)] == [text, text]  # pyright: ignore[reportOptionalMemberAccess]  # both are stored


# --- migration ----------------------------------------------------------------------------------


def _shelf_rows() -> dict[tuple[int, int], MsgInfo]:
    old_cot = [make_thinking_text_part('old reasoning'), make_text_part('old split answer')]
    return {
        # a split answer recorded with part_ids
        (CHAT, 1): _user('hi', None, 'g'),
        (CHAT, 2): _bot('split answer', 1, [2, 3]),
        (CHAT, 3): _bot('split answer', 1, [2, 3]),
        # a split answer from before part_ids existed
        (CHAT, 4): _user('go on', 3),
        (CHAT, 5): MsgInfo(True, old_cot, 4, None, None),
        (CHAT, 6): MsgInfo(True, old_cot, 4, None, None),
        # a shorter regeneration after the dropped part was rewritten too
        (CHAT, 7): _user('again', 6),
        (CHAT, 8): _bot('regenerated', 7, [8]),
        (CHAT, 9): _bot('regenerated', 7, [8]),
        # and one from before: the dropped part still holds the old answer
        (CHAT, 10): _user('once more', 9),
        (CHAT, 11): _bot('new answer', 10, [11]),
        (CHAT, 12): _bot('stale answer', 10, [11, 12]),
        # two threads opening with the same words, and an album: prompts are never merged
        (CHAT, 20): _user('hi', None, 'g'),
        (CHAT, 40): _user('two photos', None, 'g'),
        (CHAT, 41): _user('two photos', None, 'g'),
        # an imported turn, its prompt under a negative id
        (CHAT, -50): _user('imported prompt', None),
        (CHAT, 50): _bot('imported answer', -50, [50]),
        # the same ids in another chat are other messages
        (OTHER, 1): _user('elsewhere', None, 'g'),
        (OTHER, 2): _bot('split answer', 1, [2, 3]),
    }


def _write_shelf(path: str, rows: dict[tuple[int, int], MsgInfo]) -> None:
    with shelve.open(path) as shelf:
        for (chat, mid), info in rows.items():
            shelf[f'msg_info_{chat}_{mid}'] = info
        shelf['unrelated'] = {'not': 'a row'}


def test_migration_keeps_every_row_and_stores_each_answer_once(tmp_path: Path):
    shelf_path, store_path = str(tmp_path / 'db'), str(tmp_path / 'db-v2.sqlite3')
    rows = _shelf_rows()
    _write_shelf(shelf_path, rows)
    migrate_shelf(shelf_path, store_path)
    assert sorted(os.listdir(tmp_path)) == ['db', 'db-v2.sqlite3']
    store = Store(store_path)

    expected_parts = {
        (CHAT, 2): [2, 3],
        (CHAT, 3): [2, 3],
        (CHAT, 5): [5, 6],
        (CHAT, 6): [5, 6],
        (CHAT, 8): [8],
        (CHAT, 9): [8],
        (CHAT, 11): [11],
        (CHAT, 12): [12],
        (CHAT, 40): [40],
        (CHAT, 41): [41],
        (CHAT, -50): [-50],
        (OTHER, 2): [2],
    }
    for (chat, mid), info in rows.items():
        got = store.get(chat, mid)
        assert got is not None, (chat, mid)
        assert got._replace(part_ids=None) == info._replace(part_ids=None), (chat, mid)
        assert got.part_ids == expected_parts.get((chat, mid), [mid]), (chat, mid)
    # 19 rows; the split answers, the regenerated one and its dropped part each become one turn
    assert _count(store, 'turn') == 16

    # rewriting one "hi" thread's prompt leaves the other alone
    store.update(CHAT, 20, _user('changed', None, 'g'))
    got = store.get(CHAT, 1)
    assert got is not None and got.message[0].text == 'hi'


def test_a_failed_check_leaves_no_store_behind(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    shelf_path, store_path = str(tmp_path / 'db'), str(tmp_path / 'db-v2.sqlite3')
    _write_shelf(shelf_path, _shelf_rows())

    def no_row(_self: Store, _chat_id: int, _msg_id: int) -> MsgInfo | None:
        return None

    monkeypatch.setattr(Store, 'get', no_row)
    with pytest.raises(RuntimeError, match='migration check failed'):
        migrate_shelf(shelf_path, store_path)
    assert not os.path.exists(store_path)
    with shelve.open(shelf_path, flag='r') as shelf:
        assert len(shelf) == len(_shelf_rows()) + 1
