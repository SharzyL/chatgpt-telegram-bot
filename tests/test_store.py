"""The SQLite conversation store, and the migration from the shelve database before it."""

import os
import shelve
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from chatgpt_telegram_bot.models import (
    MsgInfo,
    make_image_part,
    make_reasoning_part,
    make_text_part,
    make_thinking_text_part,
)
from chatgpt_telegram_bot.store import (
    CODEC_JSON,
    CODEC_ZSTD,
    MIGRATIONS,
    SCHEMA_VERSION,
    Store,
    decode_parts,
    migrate_shelf,
    zstd,
)

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


# --- extra_metadata -----------------------------------------------------------------------------


def test_extra_metadata_reads_back_and_empty_is_none():
    store = Store(':memory:')
    store.put_turn(CHAT, [1], _user('q', None, 'g')._replace(extra_metadata={'pinned': True, 'n': 2}))
    store.put_turn(CHAT, [2], _bot('a', 1)._replace(extra_metadata={}))
    one, two = store.get(CHAT, 1), store.get(CHAT, 2)
    assert one is not None and one.extra_metadata == {'pinned': True, 'n': 2}
    assert two is not None and two.extra_metadata is None


def test_a_rewrite_merges_extra_metadata_and_keeps_keys_it_does_not_name():
    store = Store(':memory:')
    store.put_turn(CHAT, [11, 12], _bot('answer', 10)._replace(extra_metadata={'from_newer': 'kept', 'a': 1}))
    # a rewrite that knows nothing of the extras, as an older version's /regenerate would be
    store.put_turn(CHAT, [11], _bot('regenerated', 10))
    got = store.get(CHAT, 12)
    assert got is not None and got.extra_metadata == {'from_newer': 'kept', 'a': 1}
    store.update(CHAT, 11, _bot('manipulated', 10)._replace(extra_metadata={'a': None, 'b': 2}))
    got = store.get(CHAT, 11)
    assert got is not None and got.extra_metadata == {'from_newer': 'kept', 'b': 2}


def test_extra_metadata_has_to_be_a_json_object():
    store = Store(':memory:')
    store.put_turn(CHAT, [1], _user('q', None, 'g'))
    with pytest.raises(sqlite3.IntegrityError):
        _ = store.conn.execute("UPDATE turn SET extra_metadata = '[1, 2]'")


def test_parts_from_another_version_still_decode():
    # one written before a field existed would be shorter; one from a newer version is longer
    parts = decode_parts(CODEC_JSON, b'[["text", null, "hi", "a field from later"]]')
    assert parts == [make_text_part('hi')]


# --- schema versions ----------------------------------------------------------------------------

# the schema version 1 shipped with, frozen: an upgrade from it must end where a new store starts
V1_SCHEMA = """
CREATE TABLE turn (
    id            INTEGER PRIMARY KEY,
    chat_id       INTEGER NOT NULL,
    sent_by_bot   BOOLEAN NOT NULL CHECK (sent_by_bot IN (0, 1)),
    -- the message this turn replies to, by message id as MsgInfo.reply_id has it
    reply_id      INTEGER,
    prefix        TEXT,
    model_spec    TEXT,
    system_prompt TEXT,
    overrides     TEXT,
    codec         INTEGER NOT NULL CHECK (codec IN (0, 1)),
    parts         BLOB NOT NULL,
    -- optional fields as a JSON object, so adding one needs no migration; see MsgInfo
    extra_metadata TEXT CHECK (extra_metadata IS NULL OR json_type(extra_metadata) = 'object')
);
CREATE TABLE message (
    chat_id INTEGER NOT NULL,
    msg_id  INTEGER NOT NULL,
    turn_id INTEGER NOT NULL REFERENCES turn (id),
    seq     INTEGER,
    PRIMARY KEY (chat_id, msg_id)
) WITHOUT ROWID;
CREATE INDEX message_turn ON message (turn_id, seq);
"""


def _shape(conn: sqlite3.Connection) -> dict[str, Any]:
    """Every table's columns and every index, which is what an upgrade has to reproduce."""
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")]
    columns = {t: [tuple(c[1:]) for c in conn.execute(f'PRAGMA table_info({t})')] for t in tables}
    indexes = sorted(
        r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'index' AND sql IS NOT NULL")
    )
    return {'columns': columns, 'indexes': indexes}


def _v1_store(path: str) -> None:
    conn = sqlite3.connect(path)
    _ = conn.executescript(V1_SCHEMA)
    _ = conn.execute('PRAGMA user_version = 1')
    conn.commit()
    conn.close()


def test_every_version_after_the_first_has_an_upgrade_step():
    assert set(MIGRATIONS) == set(range(2, SCHEMA_VERSION + 1))


def test_an_upgraded_v1_store_matches_a_new_one(tmp_path: Path):
    old, new = str(tmp_path / 'old.sqlite3'), str(tmp_path / 'new.sqlite3')
    _v1_store(old)
    assert _shape(Store(old).conn) == _shape(Store(new).conn)
    assert Store(old).conn.execute('PRAGMA user_version').fetchone()[0] == SCHEMA_VERSION


def test_an_upgrade_step_runs_once_and_keeps_the_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = str(tmp_path / 'db.sqlite3')
    Store(path).put_turn(CHAT, [1], _user('q', None, 'g'))
    monkeypatch.setattr('chatgpt_telegram_bot.store.SCHEMA_VERSION', SCHEMA_VERSION + 1)
    monkeypatch.setattr(
        'chatgpt_telegram_bot.store.MIGRATIONS', {SCHEMA_VERSION + 1: ['ALTER TABLE turn ADD COLUMN probe TEXT']}
    )
    store = Store(path)
    assert store.conn.execute('PRAGMA user_version').fetchone()[0] == SCHEMA_VERSION + 1
    assert 'probe' in [c[1] for c in store.conn.execute('PRAGMA table_info(turn)')]
    assert store.get(CHAT, 1) is not None
    _ = Store(path)  # already there: nothing to run, so the ALTER is not repeated


def test_a_failed_upgrade_step_leaves_the_store_as_it_was(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = str(tmp_path / 'db.sqlite3')
    Store(path).put_turn(CHAT, [1], _user('q', None, 'g'))
    monkeypatch.setattr('chatgpt_telegram_bot.store.SCHEMA_VERSION', SCHEMA_VERSION + 1)
    monkeypatch.setattr(
        'chatgpt_telegram_bot.store.MIGRATIONS',
        {SCHEMA_VERSION + 1: ['ALTER TABLE turn ADD COLUMN probe TEXT', 'NOT VALID SQL']},
    )
    with pytest.raises(sqlite3.OperationalError):
        _ = Store(path)
    conn = sqlite3.connect(path)
    assert conn.execute('PRAGMA user_version').fetchone()[0] == SCHEMA_VERSION
    assert 'probe' not in [c[1] for c in conn.execute('PRAGMA table_info(turn)')]


def test_a_store_from_a_newer_version_is_refused(tmp_path: Path):
    path = str(tmp_path / 'db.sqlite3')
    _ = Store(path)
    conn = sqlite3.connect(path)
    _ = conn.execute(f'PRAGMA user_version = {SCHEMA_VERSION + 1}')
    conn.commit()
    with pytest.raises(RuntimeError, match='newer than this bot knows'):
        _ = Store(path)
