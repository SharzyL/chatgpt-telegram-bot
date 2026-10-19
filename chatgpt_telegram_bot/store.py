"""The conversation store: every turn once, however many Telegram messages show it.

Two tables. A `turn` is one prompt or one answer, with its content. A `message` maps a
Telegram message id to the turn it shows, so an answer split over 42 messages is one turn
and 42 small rows rather than 42 copies of the answer. `seq` orders a turn's live parts; a
part that no longer shows the turn (a shorter regeneration deleted it) keeps its row with
`seq` NULL, since a later turn may have replied to it and its history runs through it.

The handlers still see `MsgInfo`, with `part_ids` derived from the live parts rather than
stored. Content is JSON, compressed with zstd when the interpreter has it (Python 3.14+);
`codec` says which, so compressed and plain rows can sit side by side.
"""

import json
import os
import shelve
import sqlite3
from collections.abc import Iterable
from typing import Any

from loguru import logger

from chatgpt_telegram_bot.models import MsgInfo, MsgPartInHistory

try:
    from compression import zstd
except ImportError:  # Python < 3.14 has no zstd in the standard library
    zstd = None

CODEC_JSON = 0
CODEC_ZSTD = 1
ZSTD_LEVEL = 3

SCHEMA_VERSION = 1
SCHEMA = """
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
    parts         BLOB NOT NULL
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

_TURN_COLUMNS = (
    't.id, t.sent_by_bot, t.reply_id, t.prefix, t.system_prompt, t.overrides, t.model_spec, t.codec, t.parts'
)


def encode_parts(parts: list[MsgPartInHistory], compress: bool) -> tuple[int, bytes]:
    data = json.dumps([[p.type_, p.hash, p.text] for p in parts], ensure_ascii=False, separators=(',', ':')).encode()
    if compress and zstd is not None:
        return CODEC_ZSTD, zstd.compress(data, ZSTD_LEVEL)
    return CODEC_JSON, data


def decode_parts(codec: int, blob: bytes) -> list[MsgPartInHistory]:
    if codec == CODEC_ZSTD:
        if zstd is None:
            raise RuntimeError('this row is zstd-compressed, which needs Python 3.14 or later')
        blob = zstd.decompress(blob)
    return [MsgPartInHistory(type_, hash_, text) for type_, hash_, text in json.loads(blob)]


class Store:
    def __init__(self, path: str, compress: bool = True) -> None:
        if compress and zstd is None:
            logger.warning('zstd is unavailable before Python 3.14; storing turns uncompressed')
        self.compress: bool = compress
        # the bot is one process on one event loop; each write is one `with self.conn` transaction
        self.conn: sqlite3.Connection = sqlite3.connect(path)
        _ = self.conn.execute('PRAGMA foreign_keys = ON')
        version: int = self.conn.execute('PRAGMA user_version').fetchone()[0]
        if version == 0:
            # auto_vacuum only takes effect if set before the first table exists
            _ = self.conn.execute('PRAGMA auto_vacuum = INCREMENTAL')
            _ = self.conn.executescript(SCHEMA)
            _ = self.conn.execute(f'PRAGMA user_version = {SCHEMA_VERSION}')
            self.conn.commit()
        elif version != SCHEMA_VERSION:
            raise RuntimeError(f'{path} has schema version {version}, this bot knows {SCHEMA_VERSION}')
        if path != ':memory:':
            _ = self.conn.execute('PRAGMA journal_mode = WAL')
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def _turn_of(self, chat_id: int, msg_id: int) -> int | None:
        row = self.conn.execute('SELECT turn_id FROM message WHERE chat_id = ? AND msg_id = ?', (chat_id, msg_id))
        found = row.fetchone()
        return found[0] if found else None

    def _live_parts(self, turn_id: int) -> list[int]:
        rows = self.conn.execute(
            'SELECT msg_id FROM message WHERE turn_id = ? AND seq IS NOT NULL ORDER BY seq', (turn_id,)
        )
        return [r[0] for r in rows]

    @staticmethod
    def _info(row: Any, part_ids: list[int]) -> MsgInfo:
        _, sent_by_bot, reply_id, prefix, system_prompt, overrides, model_spec, codec, parts = row
        return MsgInfo(
            sent_by_bot=bool(sent_by_bot),
            message=decode_parts(codec, parts),
            reply_id=reply_id,
            prefix=prefix,
            system_prompt=system_prompt,
            overrides=json.loads(overrides) if overrides is not None else None,
            part_ids=part_ids,
            model_spec=model_spec,
        )

    def _turn_values(self, info: MsgInfo) -> tuple[Any, ...]:
        codec, parts = encode_parts(info.message, self.compress)
        overrides = json.dumps(info.overrides, ensure_ascii=False) if info.overrides else None
        return (
            info.sent_by_bot,
            info.reply_id,
            info.prefix,
            info.model_spec,
            info.system_prompt,
            overrides,
            codec,
            parts,
        )

    def _write_turn(self, turn_id: int | None, chat_id: int, info: MsgInfo) -> int:
        values = self._turn_values(info)
        if turn_id is not None:
            _ = self.conn.execute(
                'UPDATE turn SET sent_by_bot = ?, reply_id = ?, prefix = ?, model_spec = ?, system_prompt = ?,'
                + ' overrides = ?, codec = ?, parts = ? WHERE id = ?',
                (*values, turn_id),
            )
            return turn_id
        cursor = self.conn.execute(
            'INSERT INTO turn (chat_id, sent_by_bot, reply_id, prefix, model_spec, system_prompt, overrides, codec, parts)'
            + ' VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (chat_id, *values),
        )
        assert cursor.lastrowid is not None
        return cursor.lastrowid

    def _collect_orphans(self, turn_ids: Iterable[int]) -> None:
        _ = self.conn.executemany(
            'DELETE FROM turn WHERE id = ? AND NOT EXISTS (SELECT 1 FROM message WHERE turn_id = turn.id)',
            [(t,) for t in set(turn_ids)],
        )

    def get(self, chat_id: int, msg_id: int) -> MsgInfo | None:
        row = self.conn.execute(
            f'SELECT {_TURN_COLUMNS} FROM message m JOIN turn t ON t.id = m.turn_id'
            + ' WHERE m.chat_id = ? AND m.msg_id = ?',
            (chat_id, msg_id),
        ).fetchone()
        if row is None:
            return None
        return self._info(row, self._live_parts(row[0]))

    def put_turn(self, chat_id: int, msg_ids: list[int], info: MsgInfo) -> None:
        """Store *info* as the turn shown by *msg_ids*, in order; `info.part_ids` is ignored.

        A turn already shown by ``msg_ids[0]`` is rewritten in place: a regeneration keeps its
        first part, so the turn keeps its identity, and any part it held that is not listed
        stays mapped to it but no longer live.
        """
        assert msg_ids, 'a turn is shown by at least one message'
        with self.conn:
            turn_id = self._turn_of(chat_id, msg_ids[0])
            previous = {t for mid in msg_ids if (t := self._turn_of(chat_id, mid)) is not None and t != turn_id}
            turn_id = self._write_turn(turn_id, chat_id, info)
            _ = self.conn.execute('UPDATE message SET seq = NULL WHERE turn_id = ?', (turn_id,))
            _ = self.conn.executemany(
                'INSERT INTO message (chat_id, msg_id, turn_id, seq) VALUES (?, ?, ?, ?)'
                + ' ON CONFLICT (chat_id, msg_id) DO UPDATE SET turn_id = excluded.turn_id, seq = excluded.seq',
                [(chat_id, mid, turn_id, seq) for seq, mid in enumerate(msg_ids)],
            )
            self._collect_orphans(previous)

    def update(self, chat_id: int, msg_id: int, info: MsgInfo) -> None:
        """Rewrite the turn *msg_id* shows, leaving which messages show it alone.

        A message no turn is stored for becomes a turn of its own.
        """
        turn_id = self._turn_of(chat_id, msg_id)
        if turn_id is None:
            self.put_turn(chat_id, [msg_id], info)
            return
        with self.conn:
            _ = self._write_turn(turn_id, chat_id, info)

    def delete(self, chat_id: int, msg_ids: Iterable[int]) -> None:
        """Forget these messages, and any turn left with no message showing it."""
        ids = list(msg_ids)
        with self.conn:
            turns = [t for mid in ids if (t := self._turn_of(chat_id, mid)) is not None]
            _ = self.conn.executemany(
                'DELETE FROM message WHERE chat_id = ? AND msg_id = ?', [(chat_id, mid) for mid in ids]
            )
            self._collect_orphans(turns)
        # hand the freed pages back to the filesystem; the pragma runs as its rows are read
        _ = self.conn.execute('PRAGMA incremental_vacuum').fetchall()

    def chat_nodes(self, chat_id: int) -> dict[int, MsgInfo]:
        """Every stored message of one chat, by id; the parts of a turn share one MsgInfo."""
        rows = self.conn.execute(
            f'SELECT m.msg_id, m.seq, {_TURN_COLUMNS} FROM message m JOIN turn t ON t.id = m.turn_id'
            + ' WHERE m.chat_id = ?',
            (chat_id,),
        ).fetchall()
        live: dict[int, list[tuple[int, int]]] = {}
        for msg_id, seq, turn_id, *_ in rows:
            if seq is not None:
                live.setdefault(turn_id, []).append((seq, msg_id))
        infos: dict[int, MsgInfo] = {}
        nodes: dict[int, MsgInfo] = {}
        for msg_id, _seq, *turn in rows:
            turn_id = turn[0]
            if turn_id not in infos:
                infos[turn_id] = self._info(turn, [mid for _, mid in sorted(live.get(turn_id, []))])
            nodes[msg_id] = infos[turn_id]
        return nodes


def _turn_key(chat_id: int, info: MsgInfo) -> tuple[Any, ...]:
    return (
        chat_id,
        info.reply_id,
        tuple(info.message),
        info.prefix,
        info.system_prompt,
        json.dumps(info.overrides, sort_keys=True) if info.overrides else None,
        info.model_spec,
    )


def plan_migration(rows: Iterable[tuple[int, int, MsgInfo]]) -> list[tuple[int, list[int], list[int], MsgInfo]]:
    """Group shelf rows into turns: ``(chat_id, live part ids, parts no longer live, info)`` each.

    Only answers are merged. A split answer was stored once per part, identical but for
    nothing; rows with `part_ids` group by it, and older rows, which predate it, by an
    identical answer to the same prompt, which only a split leaves behind. A part a shorter
    regeneration dropped names the live parts in its `part_ids` but not itself, and joins its
    turn as a part that is no longer live. A row left from before that was handled still holds
    the old answer, and becomes a turn of its own; of the parts it names, only those holding the
    same answer can be its live ones. Prompts are never merged: two threads opening with
    the same words are still two turns, and rewriting one must not rewrite the other.
    """
    groups: dict[tuple[Any, ...], list[tuple[int, MsgInfo]]] = {}
    plan: list[tuple[int, list[int], list[int], MsgInfo]] = []
    for chat_id, msg_id, info in rows:
        if not info.sent_by_bot:
            plan.append((chat_id, [msg_id], [], info))
            continue
        key = (*_turn_key(chat_id, info), tuple(info.part_ids) if info.part_ids else None)
        groups.setdefault(key, []).append((msg_id, info))
    for key, members in groups.items():
        chat_id = key[0]
        ids = sorted(mid for mid, _ in members)
        named = members[0][1].part_ids
        live = [mid for mid in named if mid in ids] if named else ids
        plan.append((chat_id, live, [mid for mid in ids if mid not in live], members[0][1]))
    return plan


def _same_turn(old: MsgInfo, new: MsgInfo, members: set[int]) -> bool:
    """Whether a migrated row reads back as stored.

    `part_ids` is derived now, so it is checked only where the old row had one, and against the
    parts it named that hold the same answer -- *members*, the rows of its turn.
    """
    if old.part_ids and new.part_ids != [mid for mid in old.part_ids if mid in members]:
        return False
    return old._replace(part_ids=None, overrides=old.overrides or None) == new._replace(part_ids=None)


def migrate_shelf(shelf_path: str, store_path: str, compress: bool = True) -> None:
    """Copy a `shelve` database into a new store at *store_path*, verifying every row.

    The store is built beside its final path and moved into place only once every row reads
    back as it was stored, so a failure leaves nothing behind and the shelf untouched.
    """
    building = store_path + '.migrating'
    for leftover in (building, building + '-wal', building + '-shm'):
        if os.path.exists(leftover):
            os.remove(leftover)
    with shelve.open(shelf_path, flag='r') as shelf:
        rows: list[tuple[int, int, MsgInfo]] = []
        skipped = 0
        for key in shelf:
            parts = key.split('_')
            if len(parts) != 4 or parts[:2] != ['msg', 'info']:
                skipped += 1
                continue
            rows.append((int(parts[2]), int(parts[3]), shelf[key]))
    plan = plan_migration(rows)
    store = Store(building, compress)
    try:
        with store.conn:
            for chat_id, live, dropped, info in plan:
                turn_id = store._write_turn(None, chat_id, info)  # pyright: ignore[reportPrivateUsage]  # one transaction for all rows
                _ = store.conn.executemany(
                    'INSERT INTO message (chat_id, msg_id, turn_id, seq) VALUES (?, ?, ?, ?)',
                    [(chat_id, mid, turn_id, seq) for seq, mid in enumerate(live)]
                    + [(chat_id, mid, turn_id, None) for mid in dropped],
                )
        members = {(chat_id, mid): {*live, *dropped} for chat_id, live, dropped, _ in plan for mid in (*live, *dropped)}
        for chat_id, msg_id, info in rows:
            new = store.get(chat_id, msg_id)
            if new is None or not _same_turn(info, new, members[(chat_id, msg_id)]):
                raise RuntimeError(f'migration check failed for message {msg_id} in chat {chat_id}')
        _ = store.conn.execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchall()
    finally:
        store.close()
    os.replace(building, store_path)
    for sidecar in (building + '-wal', building + '-shm'):
        if os.path.exists(sidecar):
            os.remove(sidecar)
    logger.info(f'Migrated {len(rows)} messages into {len(plan)} turns ({skipped} other keys skipped): {store_path}')
