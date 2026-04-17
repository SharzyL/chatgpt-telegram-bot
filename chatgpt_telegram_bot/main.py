#!/usr/bin/env python3
import sys
import asyncio
from argparse import ArgumentParser

from loguru import logger

from chatgpt_telegram_bot.bot import ChatGPTTelegramBot


async def async_main() -> None:
    parser = ArgumentParser()
    _ = parser.add_argument('--debug', action='store_true')
    _ = parser.add_argument('-c', '--config', default='bot.toml')

    args = parser.parse_args()

    log_level = 'DEBUG' if args.debug else 'INFO'
    logger.remove()
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
