import json
from dataclasses import dataclass, field
from typing import Any, NamedTuple


class Model(NamedTuple):
    prefix: str
    name: str
    endpoint: str | None = None
    no_system_prompt: bool = False
    system_prompt: str | None = None
    system_prompt_append: str | None = None  # appended to default system prompt via +[...] syntax
    api_type: str = 'openai'  # 'openai', 'openai_legacy', or 'anthropic'
    suffix: str | None = None  # appended to endpoint URL, None = use endpoint's default_suffix
    thinking: int | str | None = None  # None = disabled, int = budget, str = effort level or 'adaptive'
    search: bool = False
    richtext: bool = False  # True = let Telegram render the markdown, False = client-side entities


class EndPoint(NamedTuple):
    name: str
    url: str
    default_suffix: str = ''


class MsgPartInHistory(NamedTuple):
    """
    one of "text", "image", "reasoning" or "thinking_text"
    """

    type_: str
    hash: str | None  # must present when str == "img"
    text: str | None  # must present unless str == "image"


def make_image_part(_hash: str) -> MsgPartInHistory:
    return MsgPartInHistory(type_='image', hash=_hash, text=None)


def make_text_part(text: str) -> MsgPartInHistory:
    return MsgPartInHistory(type_='text', hash=None, text=text)


def make_thinking_text_part(text: str) -> MsgPartInHistory:
    """
    Store a user-supplied chain of thought (see the /manipulate command).

    Unlike a `reasoning` part this is plain text, not an opaque provider item, so every
    backend can replay it.
    """
    return MsgPartInHistory(type_='thinking_text', hash=None, text=text)


def make_reasoning_part(model_name: str, item: dict[str, Any]) -> MsgPartInHistory:
    """
    Store a provider reasoning item verbatim so the next turn can replay it.

    The model name travels with the item so it is never replayed to a different model.
    Backends that have no notion of replayable reasoning ignore this part type.
    """
    return MsgPartInHistory(type_='reasoning', hash=None, text=json.dumps({'model': model_name, 'item': item}))


class MsgInfo(NamedTuple):
    sent_by_bot: bool
    message: list[MsgPartInHistory]
    reply_id: int | None

    """only present for head of conversation"""
    prefix: str | None

    """only present for head of conversation"""
    system_prompt: str | None

    """only present for head of conversation, stores inline parameter overrides"""
    overrides: dict[str, str | None] | None = None


@dataclass
class StreamEvent:
    pass


@dataclass
class ThinkingDelta(StreamEvent):
    text: str


@dataclass
class ResponseDelta(StreamEvent):
    text: str


@dataclass
class StatusChange(StreamEvent):
    status: str


@dataclass
class StreamMeta(StreamEvent):
    tool_calls: list[str] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)

    """provider items to persist and replay on the next turn (currently reasoning items)"""
    carry_items: list[dict[str, Any]] = field(default_factory=list)


OVERRIDE_ALIASES: dict[str, str] = {'t': 'thinking', 's': 'search', 'r': 'richtext'}
