"""Turning a markdown reply into Telegram message bodies.

Two renderers exist because Telegram offers two ways to show formatted text, and they
differ in more than the request field they fill:

- `ServerMarkdownRenderer` ships markdown source and lets Telegram render it. Headings,
  tables, horizontal rules and LaTeX math all survive, and a body may be 32768 characters,
  but the rendering is the server's and cannot be inspected or corrected.
- `EntityRenderer` renders on the client into message entities. Only what an entity can
  express survives — no tables, no math — and a body is capped at 4096 characters.

Because one splits markdown *source* and the other splits *rendered* text, a long reply is
cut differently in each mode. Both are hidden behind `Renderer.render()`, which hands back
ready message bodies, so callers never branch on the mode.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, override

from chatgpt_telegram_bot.richtext import RichText
from chatgpt_telegram_bot.utils import (
    neutralize_images,
    split_markdown,
    telegram_len,
    telegram_truncate,
    utf16_prefix,
)

# a plain message body, and the `message` field of a rich one, are capped here
PLAIN_LENGTH_LIMIT = 4096

# rich messages are rendered server-side; the server rejects a longer rendered body with
# RICH_MESSAGE_TEXT_TOO_LONG
RICH_LENGTH_LIMIT = 32768


@dataclass
class Rendered:
    """One message body, ready to hand to Telegram."""

    text: str
    entities: list[Any] = field(default_factory=list)
    rich_markdown: str | None = None


class Renderer(ABC):
    """Cuts a markdown reply into message bodies that Telegram will accept."""

    limit: int

    @abstractmethod
    def render(self, markdown: str, prefix: str = '', collapse_first_quote: bool = False) -> list[Rendered]:
        """
        Bodies for *markdown*, with *prefix* prepended to the first one.

        *collapse_first_quote* says the leading blockquote is a chain of thought and should
        be sent folded. Always returns at least one body, so an empty reply still has a
        message to occupy.
        """

    def render_one(self, markdown: str) -> Rendered:
        """The first body only, for the bot's own short messages."""
        return self.render(markdown)[0]


class ServerMarkdownRenderer(Renderer):
    """Sends markdown source for Telegram to render."""

    limit: int = RICH_LENGTH_LIMIT

    @override
    def render(self, markdown: str, prefix: str = '', collapse_first_quote: bool = False) -> list[Rendered]:
        # whether the rich markdown can express a foldable quote is untested, so the
        # request is accepted and ignored rather than guessed at
        _ = collapse_first_quote
        # an image would be read as a photo block and rejected for having no media
        markdown = neutralize_images(markdown)
        # the prefix is kept out of the split so that its length is charged to the first
        # body only, and so a growing reply re-splits the same way each time
        slices = split_markdown(markdown, self.limit - telegram_len(prefix), self.limit) or ['']
        bodies: list[Rendered] = []
        for i, piece in enumerate(slices):
            source = prefix + piece if i == 0 else piece
            # the `message` field is required by the schema even though the server discards
            # it, so it is sent as a truncated copy of the markdown
            bodies.append(Rendered(text=telegram_truncate(source, PLAIN_LENGTH_LIMIT), rich_markdown=source))
        return bodies


class EntityRenderer(Renderer):
    """Renders markdown to message entities on the client."""

    limit: int = PLAIN_LENGTH_LIMIT

    @override
    def render(self, markdown: str, prefix: str = '', collapse_first_quote: bool = False) -> list[Rendered]:
        # the prefix is part of the document here: it is formatted markdown too, and
        # slicing the rendered result places it at the head of the first body for free
        rich = RichText.from_markdown(prefix + markdown)
        if collapse_first_quote:
            # set before any slicing, so every part of a quote spanning bodies stays folded
            rich.collapse_first_blockquote()
        rich = self._trim(rich)
        bodies: list[Rendered] = []
        while len(rich) > 0:
            head, rest = self._split(rich, self.limit)
            text, entities = head.to_telegram()
            bodies.append(Rendered(text=text, entities=entities))
            rich = self._trim(rest)
        if not bodies:
            bodies.append(Rendered(text=''))
        return bodies

    @staticmethod
    def _split(rich: RichText, budget: int) -> tuple[RichText, RichText]:
        """
        Cut *rich* into a head of at most *budget* UTF-16 units and the rest.

        RichText indexes by character while Telegram counts UTF-16 units, so the cut is
        found on the rendered text, whose characters map one to one onto the RichText.
        """
        text, _ = rich.to_telegram()
        cut = max(utf16_prefix(text, budget), 1)
        # trailing whitespace is dropped from the head: Telegram trims it from the message
        # and every entity would then sit two units to the right of where it was measured
        end = cut
        while end > 0 and text[end - 1].isspace():
            end -= 1
        return rich[: end or cut], rich[cut:]

    @staticmethod
    def _trim(rich: RichText) -> RichText:
        """Drop whitespace at either end, which Telegram would strip and shift entities by."""
        if len(rich) == 0:
            return rich
        text, _ = rich.to_telegram()
        start, end = 0, len(text)
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        return rich[start:end]


SERVER_MARKDOWN_RENDERER = ServerMarkdownRenderer()
ENTITY_RENDERER = EntityRenderer()
