"""Library logs that go through stdlib `logging` reach loguru."""

import logging
from collections.abc import Iterator
from typing import Any

import pytest
from loguru import logger

from chatgpt_telegram_bot.main import _route_stdlib_logging  # pyright: ignore[reportPrivateUsage]  # testing the setup


@pytest.fixture
def records() -> Iterator[list[Any]]:
    root = logging.getLogger()
    saved = (root.handlers[:], root.level, logging.getLogger('telethon').level)
    seen: list[Any] = []
    sink = logger.add(lambda message: seen.append(message.record), level='TRACE')
    _route_stdlib_logging()
    try:
        yield seen
    finally:
        logger.remove(sink)
        root.handlers[:] = saved[0]
        root.setLevel(saved[1])
        logging.getLogger('telethon').setLevel(saved[2])


def test_a_flood_wait_telethon_sleeps_off_is_logged(records: list[Any]):
    # the shape of telethon.client.users' own message
    logging.getLogger('telethon.client.users').info(
        'Sleeping%s for %ds (%s) on %s flood wait', '', 23, '0:00:23', 'EditMessageRequest'
    )
    assert [(r['level'].name, r['message']) for r in records] == [
        ('INFO', 'Sleeping for 23s (0:00:23) on EditMessageRequest flood wait')
    ]
    # attributed to the line that logged it, not to the handler
    assert records[0]['file'].name == 'test_logging.py'


def test_other_libraries_log_only_warnings(records: list[Any]):
    logging.getLogger('httpx').info('HTTP Request: POST https://example.test "HTTP/1.1 200 OK"')
    logging.getLogger('httpx').warning('something worth seeing')
    assert [r['message'] for r in records] == ['something worth seeing']
