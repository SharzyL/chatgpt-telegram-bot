import asyncio
import hashlib
import os
from urllib.parse import urlparse

from loguru import logger
from telethon import errors

from chatgpt_telegram_bot.richtext import RichText


def parse_overrides(s: str) -> dict[str, str | None]:
    """Parse 'key=val,key2=val2,key3' into dict. Bare key (no '=') maps to None (use default)."""
    overrides: dict[str, str | None] = {}
    for part in s.split(','):
        part = part.strip()
        if not part:
            continue
        if '=' in part:
            k, v = part.split('=', 1)
            overrides[k.strip()] = v.strip()
        else:
            overrides[part] = None
    return overrides


def parse_proxy():
    proxy_env = os.getenv('ALL_PROXY')
    if proxy_env:
        proxy_url = urlparse(proxy_env)
        return {
            'proxy_type': proxy_url.scheme,
            'addr': proxy_url.hostname,
            'port': proxy_url.port,
        }
    else:
        return None


def retry(max_retry: int = 30, interval: int = 10):
    def decorator(func):  # pyright: ignore[reportMissingParameterType]  # generic decorator wrapper
        async def new_func(*args, **kwargs):  # pyright: ignore[reportMissingParameterType]  # generic decorator wrapper
            for _ in range(max_retry - 1):
                try:
                    return await func(*args, **kwargs)
                except AssertionError as e:
                    logger.exception(e)
                except ValueError as e:
                    logger.exception(e)
                except errors.FloodWaitError as e:
                    logger.exception(e)
                    await asyncio.sleep(interval)
            return await func(*args, **kwargs)

        return new_func

    return decorator


def telegram_len(s: str | RichText) -> int:
    """Length in UTF-16 code units, matching Telegram's counting."""
    if isinstance(s, RichText):
        text, _ = s.to_telegram()
        return len(text.encode('utf-16-le')) // 2
    return len(s.encode('utf-16-le')) // 2


class PendingReplyManager:
    def __init__(self) -> None:
        self.messages: dict[tuple[int, int], asyncio.Event] = {}

    def add(self, reply_id: tuple[int, int]) -> None:
        assert reply_id not in self.messages
        self.messages[reply_id] = asyncio.Event()

    def remove(self, reply_id: tuple[int, int]) -> None:
        if reply_id not in self.messages:
            return
        self.messages[reply_id].set()
        del self.messages[reply_id]

    async def wait_for(self, reply_id: tuple[int, int]) -> None:
        if reply_id not in self.messages:
            return
        logger.info('PendingReplyManager waiting for %r', reply_id)
        _ = await self.messages[reply_id].wait()
        logger.info('PendingReplyManager waiting for %r finished', reply_id)


def save_photo(photo_blob: bytes) -> str:
    h = hashlib.sha256(photo_blob).hexdigest()
    save_dir = f'photos/{h[:2]}/{h[2:4]}'
    path = f'{save_dir}/{h}'
    if not os.path.isfile(path):
        os.makedirs(save_dir, exist_ok=True)
        with open(path, 'wb') as f:
            _ = f.write(photo_blob)
    return h


def load_photo(h: str) -> bytes:
    save_dir = f'photos/{h[:2]}/{h[2:4]}'
    path = f'{save_dir}/{h}'
    with open(path, 'rb') as f:
        return f.read()
