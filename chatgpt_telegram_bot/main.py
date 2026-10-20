#!/usr/bin/env python3
import asyncio
import inspect
import logging
import os
import sys
from argparse import ArgumentParser
from typing import override

from loguru import logger

from chatgpt_telegram_bot.bot import ChatGPTTelegramBot

_LEVEL_TO_SYSLOG_PRIORITY = {
    'TRACE': 7,
    'DEBUG': 7,
    'INFO': 6,
    'SUCCESS': 5,
    'WARNING': 4,
    'ERROR': 3,
    'CRITICAL': 2,
}


def _journal_sink(message) -> None:  # pyright: ignore[reportMissingParameterType]  # loguru sink signature
    from systemd.journal import send  # already verified importable

    record = message.record
    priority = _LEVEL_TO_SYSLOG_PRIORITY.get(record['level'].name, 6)
    send(
        record['message'],
        PRIORITY=priority,
        CODE_FILE=record['file'].path,
        CODE_LINE=record['line'],
        CODE_FUNC=record['function'],
    )


class _ToLoguru(logging.Handler):
    """Hand a stdlib log record to loguru, attributed to the code that logged it."""

    @override
    def emit(self, record: logging.LogRecord) -> None:
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        # skip this frame and logging's own, so loguru reports the caller's file and line
        frame, depth = inspect.currentframe(), 0
        while frame is not None and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def _route_stdlib_logging() -> None:
    """Send what libraries log through `logging` to loguru's sinks.

    Without a handler, Python prints only warnings and up, so Telethon's INFO lines were lost --
    among them the flood waits of up to `flood_sleep_threshold` (60s) it sleeps off inside a
    call, which freeze a streaming reply with nothing in the log, and its reconnects. Telethon
    is kept at INFO for those; other libraries (httpx logs every request at INFO) at WARNING.
    """
    logging.basicConfig(handlers=[_ToLoguru()], level=logging.WARNING, force=True)
    logging.getLogger('telethon').setLevel(logging.INFO)


def _default_data_dir() -> str:
    xdg = os.environ.get('XDG_STATE_HOME', os.path.join(os.path.expanduser('~'), '.local', 'state'))
    return os.path.join(xdg, 'chatgpt-telegram-bot')


async def async_main() -> None:
    parser = ArgumentParser()
    _ = parser.add_argument('--debug', action='store_true')
    _ = parser.add_argument('--trace', action='store_true', help='DEBUG plus per-event stream logging')
    _ = parser.add_argument('-c', '--config', default='bot.toml')
    _ = parser.add_argument(
        '--data-dir', default=_default_data_dir(), help='directory for persistent data (db, image cache)'
    )

    args = parser.parse_args()

    log_level = 'TRACE' if args.trace else 'DEBUG' if args.debug else 'INFO'
    logger.remove()

    use_journal = False
    journal_stream = os.environ.get('JOURNAL_STREAM')
    if journal_stream:
        try:
            # JOURNAL_STREAM contains "dev:inode" of the journal socket assigned to this service.
            # Verify that our stderr actually points to it (not inherited from a parent session).
            expected_dev, expected_ino = (int(x) for x in journal_stream.split(':'))
            stat = os.fstat(sys.stderr.fileno())
            if stat.st_dev == expected_dev and stat.st_ino == expected_ino:
                import systemd.journal  # noqa: F401  # pyright: ignore[reportUnusedImport]

                use_journal = True
        except (ImportError, ValueError, OSError):
            pass

    if use_journal:
        _ = logger.add(_journal_sink, level=log_level)
    else:
        _ = logger.add(
            sys.stdout,
            colorize=True,
            format='<green>{time}</green> <level>{message}</level>',
            level=log_level,
        )

    _route_stdlib_logging()

    data_dir: str = args.data_dir
    logger.info(f'Data directory: {data_dir}, config path: {args.config}')
    cbot = ChatGPTTelegramBot(args.config, data_dir)
    await cbot.start()


def main():
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        logger.info('Interrupted, exiting')
