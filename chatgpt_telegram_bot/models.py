from dataclasses import dataclass, field
from typing import Any, NamedTuple


class Model(NamedTuple):
    prefix: str
    name: str
    endpoint: str | None = None
    no_system_prompt: bool = False
    system_prompt: str | None = None
    api_type: str = 'openai'  # 'openai', 'openai_legacy', or 'anthropic'
    suffix: str | None = None  # appended to endpoint URL, None = use endpoint's default_suffix
    thinking: int | str | None = None  # None = disabled, int = budget, str = effort level or 'adaptive'
    search: bool = False


class EndPoint(NamedTuple):
    name: str
    url: str
    default_suffix: str = ''


class MsgPartInHistory(NamedTuple):
    """
    either "text" or "image"
    """

    type_: str
    hash: str | None  # must present when str == "img"
    text: str | None  # must present when str == "text"


def make_image_part(_hash: str) -> MsgPartInHistory:
    return MsgPartInHistory(type_='image', hash=_hash, text=None)


def make_text_part(text: str) -> MsgPartInHistory:
    return MsgPartInHistory(type_='text', hash=None, text=text)


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


OVERRIDE_ALIASES: dict[str, str] = {'t': 'thinking', 's': 'search'}
