#!/usr/bin/env python3
import os
import sys
import asyncio
from argparse import ArgumentParser

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
        CODE_LINE=str(record['line']),
        CODE_FUNC=record['function'],
    )


async def async_main() -> None:
    parser = ArgumentParser()
    _ = parser.add_argument('--debug', action='store_true')
    _ = parser.add_argument('-c', '--config', default='bot.toml')

    args = parser.parse_args()

    log_level = 'DEBUG' if args.debug else 'INFO'
    logger.remove()

    use_journal = False
    if os.environ.get('JOURNAL_STREAM'):
        try:
            import systemd.journal  # noqa: F401  # pyright: ignore[reportUnusedImport]

            use_journal = True
        except ImportError:
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

    cbot = ChatGPTTelegramBot(args.config)
    await cbot.start()


def main():
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        logger.info('Interrupted, exiting')
