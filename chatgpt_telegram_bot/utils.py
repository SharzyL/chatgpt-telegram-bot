import asyncio
import hashlib
import os
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import urlparse

import diskcache
from loguru import logger
from telethon import errors
from telethon.extensions import html as telethon_html

from chatgpt_telegram_bot.markdown import telegram_len
from chatgpt_telegram_bot.models import MsgInfo, MsgPartInHistory, make_text_part


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
    """Sit out Telegram's flood waits, up to *max_retry* attempts in all.

    Nothing else is retried: any other error would recur on an immediate second attempt.
    """

    def decorator(func):  # pyright: ignore[reportMissingParameterType]  # generic decorator wrapper
        async def new_func(*args, **kwargs):  # pyright: ignore[reportMissingParameterType]  # generic decorator wrapper
            for _ in range(max_retry - 1):
                try:
                    return await func(*args, **kwargs)
                except errors.FloodWaitError as e:
                    # Telegram states how long to wait; sleeping less just floods again
                    wait = getattr(e, 'seconds', None) or interval
                    logger.warning(f'Flood wait {wait}s on {func.__name__}')
                    await asyncio.sleep(wait + 1)
            return await func(*args, **kwargs)

        return new_func

    return decorator


def decode_text_file(blob: bytes) -> str | None:
    """The contents of *blob* as text, or None if it is not UTF-8 text.

    A NUL byte decodes cleanly but marks the file as binary, so it is rejected too.
    """
    try:
        text = blob.decode()
    except UnicodeDecodeError:
        return None
    if '\x00' in text:
        return None
    return text


def message_link(chat_id: int, msg_id: int) -> str | None:
    """A t.me link to a message, which exists only in a supergroup or channel.

    Their chat ids are the internal id with a -100 prefix; a private chat or a basic group has
    no linkable messages, and a negative id is a stored half of an imported turn that was never
    sent as a message.
    """
    peer = str(chat_id)
    if not peer.startswith('-100') or msg_id <= 0:
        return None
    return f'https://t.me/c/{peer[4:]}/{msg_id}'


def linearize_plan(tree: dict[int, MsgInfo], anchor_id: int) -> tuple[set[int], set[int]] | None:
    """Split one tree of branches into the node ids to keep and the ones to delete.

    *tree* is every stored node sharing the anchor's root (`MsgInfo.root_msg_id`), by id. What
    is kept is the path from the first message down to the anchor, as whole turns: a prompt on
    the path keeps every part of its answer -- a split reply is shown by several ids that all
    name the prompt -- and so does the anchor when it is a prompt. Everything else in the tree is
    deleted, including whatever came after the anchor. None when the anchor is not in the tree,
    or its chain loops.
    """
    if anchor_id not in tree:
        return None
    keep: set[int] = set()
    cur: int | None = anchor_id
    while cur is not None and cur in tree:
        if cur in keep:  # a loop in damaged data
            return None
        keep.add(cur)
        cur = tree[cur].reply_id
    prompts = {n for n in keep if not tree[n].sent_by_bot}
    keep |= {n for n, info in tree.items() if info.sent_by_bot and info.reply_id in prompts}
    return keep, set(tree) - keep


def plan_digest(ids: set[int]) -> str:
    """A short fingerprint of a set of ids, so a confirmation can tell the plan has changed."""
    return hashlib.sha1(','.join(map(str, sorted(ids))).encode()).hexdigest()[:8]


def replace_text_part(parts: list[MsgPartInHistory], text: str) -> list[MsgPartInHistory]:
    """*parts* with its text replaced, keeping an attached image where it was."""
    replaced = False
    result: list[MsgPartInHistory] = []
    for part in parts:
        if part.type_ == 'text' and not replaced:
            result.append(make_text_part(text))
            replaced = True
        else:
            result.append(part)
    if not replaced:
        result.insert(0, make_text_part(text))
    return result


def replied_message_id(message: Any) -> int | None:
    """The id of the message *message* is a reply to, or None if it replies to nothing.

    A message merely posted in a forum topic still carries a reply header, pointing at the
    message that created the topic (`forum_topic` set, no `reply_to_top_id`) — it is not a
    reply to anything. A real reply inside a topic names its target in `reply_to_msg_id` and
    the topic in `reply_to_top_id`. Only the General topic carries no header at all, so
    taking any header as a reply makes a bot answer in General and nowhere else.

    A reply to a story has a header of its own shape, with no message id to return.
    """
    header: Any = getattr(message, 'reply_to', None)
    if header is None:
        return None
    if getattr(header, 'forum_topic', False) and getattr(header, 'reply_to_top_id', None) is None:
        return None
    replied_id: Any = getattr(header, 'reply_to_msg_id', None)
    return replied_id if isinstance(replied_id, int) else None


def pack_html_sections(sections: list[str], limit: int) -> list[str]:
    """Join HTML sections, a blank line apart, into as few messages as fit under ``limit``.

    Telegram's limit applies to the text left once the tags are parsed out, so that is what is
    measured. Sections are never cut: one that is too long on its own gets a message to itself,
    and its send fails as it would have unsplit.
    """
    messages: list[str] = []
    for section in sections:
        if messages:
            joined = f'{messages[-1]}\n\n{section}'
            text, _ = telethon_html.parse(joined)
            if telegram_len(text) <= limit:
                messages[-1] = joined
                continue
        messages.append(section)
    return messages


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
        logger.info(f'PendingReplyManager waiting for {reply_id!r}')
        _ = await self.messages[reply_id].wait()
        logger.info(f'PendingReplyManager waiting for {reply_id!r} finished')


# leading bytes of the image formats a model accepts; a Telegram photo is always JPEG, but an
# image sent as a file keeps whatever format it was saved in
_IMAGE_SIGNATURES = (
    (b'\xff\xd8\xff', 'image/jpeg'),
    (b'\x89PNG\r\n\x1a\n', 'image/png'),
    (b'GIF87a', 'image/gif'),
    (b'GIF89a', 'image/gif'),
)


def image_media_type(blob: bytes) -> str:
    """The media type of an image, read from its leading bytes; JPEG when unrecognised."""
    for signature, media_type in _IMAGE_SIGNATURES:
        if blob.startswith(signature):
            return media_type
    if blob[:4] == b'RIFF' and blob[8:12] == b'WEBP':
        return 'image/webp'
    return 'image/jpeg'


def save_photo(cache: diskcache.Cache, photo_blob: bytes, chat_id: int, msg_id: int) -> str:
    key = f'{chat_id}:{msg_id}'
    _ = cache.set(key, photo_blob)
    return key


async def load_photo(
    cache: diskcache.Cache,
    key: str,
    fetcher: Callable[[int, int], Awaitable[bytes]] | None = None,
) -> bytes | None:
    blob: bytes | None = cache.get(key)  # pyright: ignore[reportAssignmentType]  # diskcache returns stored type
    if blob is not None:
        return blob
    if fetcher and ':' in key:
        chat_id_str, msg_id_str = key.split(':', 1)
        try:
            blob = await fetcher(int(chat_id_str), int(msg_id_str))
            _ = cache.set(key, blob)
            return blob
        except Exception:  # noqa: BLE001  # any re-fetch failure just means the image is unavailable
            logger.warning(f'Failed to re-fetch image for key={key}')
            return None
    return None
