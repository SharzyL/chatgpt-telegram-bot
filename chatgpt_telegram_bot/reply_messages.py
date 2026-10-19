"""`BotReplyMessages`: one streaming reply, kept in step across however many messages it needs."""

import time
from types import TracebackType
from typing import Protocol, Self

from loguru import logger
from telethon import errors

from chatgpt_telegram_bot.render import ENTITY_RENDERER, Rendered, Renderer
from chatgpt_telegram_bot.throttle import Throttle
from chatgpt_telegram_bot.utils import PendingReplyManager


class ReplyTransport(Protocol):
    """What a reply needs of the bot: its pacing, its pending replies, and the three message calls."""

    TELEGRAM_MIN_INTERVAL: float
    TELEGRAM_BURST_INTERVAL: float
    TELEGRAM_BURST_WINDOW: float
    pending_reply_manager: PendingReplyManager

    async def send_message(self, chat_id: int, body: Rendered | str, reply_to_message_id: int) -> int: ...

    async def edit_message(self, chat_id: int, body: Rendered | str, message_id: int) -> None: ...

    async def delete_message(self, chat_id: int, message_id: int) -> None: ...


class BotReplyMessages:
    def __init__(
        self,
        cbot: ReplyTransport,
        chat_id: int,
        orig_msg_id: int,
        prefix: str,
        renderer: Renderer,
        adopt: list[int] | None = None,
    ) -> None:
        self.cbot: ReplyTransport = cbot
        self.prefix: str = prefix
        self.renderer: Renderer = renderer
        self.chat_id: int = chat_id
        self.orig_msg_id: int = orig_msg_id
        # messages a regeneration writes over: seeding them as already-sent parts makes
        # `_apply` edit them, and delete the ones a shorter reply no longer needs. Their
        # recorded text is empty so the first flush always counts as a change.
        self.replied_msgs: list[tuple[int, Rendered]] = [(mid, Rendered(text='')) for mid in adopt or []]
        for mid in adopt or []:
            cbot.pending_reply_manager.add((chat_id, mid))
        self.text: str = ''
        self.collapse_first_quote: bool = False
        self.throttle: Throttle = Throttle(
            cbot.TELEGRAM_MIN_INTERVAL, cbot.TELEGRAM_BURST_INTERVAL, cbot.TELEGRAM_BURST_WINDOW
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self, type_: type[BaseException] | None, value: BaseException | None, tb: TracebackType | None
    ) -> None:
        await self.finalize()
        for msg_id, _ in self.replied_msgs:
            self.cbot.pending_reply_manager.remove((self.chat_id, msg_id))

    async def _force_update(self, text: str) -> None:
        # how the reply is cut, and how long a part may be, are the renderer's business
        try:
            await self._apply(self.renderer.render(text, self.prefix, self.collapse_first_quote))
        except errors.BadRequestError as e:
            # the server renderer rejects constructs it cannot lay out (an image with no
            # media, an unsupported block); rendering to entities instead keeps the reply
            if self.renderer is ENTITY_RENDERER or 'RICH_MESSAGE' not in str(e):
                raise
            logger.warning(f'Rich rendering rejected, falling back to entities ({self.chat_id=}): {e}')
            self.renderer = ENTITY_RENDERER
            await self._apply(self.renderer.render(text, self.prefix, self.collapse_first_quote))

    async def _apply(self, bodies: list[Rendered]) -> None:
        for i in range(min(len(bodies), len(self.replied_msgs))):
            msg_id, previous = self.replied_msgs[i]
            if bodies[i] != previous:
                await self.cbot.edit_message(self.chat_id, bodies[i], msg_id)
                self.replied_msgs[i] = (msg_id, bodies[i])
        if len(bodies) > len(self.replied_msgs):
            for i in range(len(self.replied_msgs), len(bodies)):
                if i == 0:
                    reply_to = self.orig_msg_id
                else:
                    reply_to, _ = self.replied_msgs[i - 1]
                msg_id = await self.cbot.send_message(self.chat_id, bodies[i], reply_to)
                self.replied_msgs.append((msg_id, bodies[i]))
                self.cbot.pending_reply_manager.add((self.chat_id, msg_id))
        if len(self.replied_msgs) > len(bodies):
            for i in range(len(bodies), len(self.replied_msgs)):
                msg_id, _ = self.replied_msgs[i]
                await self.cbot.delete_message(self.chat_id, msg_id)
                self.cbot.pending_reply_manager.remove((self.chat_id, msg_id))
            self.replied_msgs = self.replied_msgs[: len(bodies)]

    async def update(self, text: str, collapse_first_quote: bool = False) -> None:
        self.text = text
        self.collapse_first_quote = collapse_first_quote
        if self.throttle.take(time.time()):
            await self._force_update(self.text)
            # timed from the end of the flush: one tick can edit several messages when a
            # cut shifts, and starting the clock before that would shorten the next gap
            self.throttle.flushed(time.time())

    async def finalize(self) -> None:
        await self._force_update(self.text)
