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
    # True = offer a tool that does nothing, which is what makes DeepSeek keep passed-back
    # chains of thought, and pass the stored ones back as reasoning; False = drop them
    dummy_tool: bool = False


class EndPoint(NamedTuple):
    name: str
    url: str
    default_suffix: str = ''


class MsgPartInHistory(NamedTuple):
    """
    one of "text", "image", "reasoning" or "thinking_text"
    """

    type_: str
    hash: str | None  # present when type_ == "image"
    text: str | None  # present unless type_ == "image"


def make_image_part(_hash: str) -> MsgPartInHistory:
    return MsgPartInHistory(type_='image', hash=_hash, text=None)


def make_text_part(text: str) -> MsgPartInHistory:
    return MsgPartInHistory(type_='text', hash=None, text=text)


def make_thinking_text_part(text: str) -> MsgPartInHistory:
    """
    Store a chain of thought as plain text rather than as an opaque provider item.

    Used for a chain of thought the provider returns in the clear, one supplied with
    /manipulate, and one carried by an import. It is sent back only under `dt`, as a
    reasoning item; every other request drops it.
    """
    return MsgPartInHistory(type_='thinking_text', hash=None, text=text)


def text_of(parts: list[MsgPartInHistory]) -> str:
    """The text a stored message shows: its first non-empty text part."""
    return next((p.text for p in parts if p.type_ == 'text' and p.text), '')


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

    # prefix, system_prompt, overrides and model_spec are set on a user turn that names a
    # model -- the head of a thread, or a reply switching to another -- and None elsewhere

    """the configured prefix the turn named"""
    prefix: str | None

    """the system prompt that came with the model"""
    system_prompt: str | None

    """inline parameter overrides given with the prefix"""
    overrides: dict[str, str | None] | None = None

    """every message id that shows this turn, in order, so a regeneration can reuse all of them;
    the store derives it on read and ignores it on write"""
    part_ids: list[int] | None = None

    """a model named as ``<endpoint>/<name>`` (by /chat or /regenerate) rather than by a
    configured prefix; `prefix` is None then"""
    model_spec: str | None = None

    """optional fields the model's input does not depend on, as a JSON object, so adding one
    needs no schema change. Writing a turn merges this into what is stored, a None value
    removing that key, so a key a newer version wrote survives an older one rewriting the turn.
    Read each field through one helper that supplies its default and type."""
    extra_metadata: dict[str, Any] | None = None


class ThreadTurn(NamedTuple):
    """One prompt of a thread and what answered it, for /navigate."""

    prompt: str
    # None for a prompt still waiting on its answer
    response: str | None
    # the message to jump to: the answer if there is one, else the prompt itself
    msg_id: int


class ImportedTurn(NamedTuple):
    """One prompt of an imported conversation and the response to it."""

    prompt: str
    response: str
    # the response's chain of thought, from the message's optional `reasoning` field
    reasoning: str | None = None


class ImportedChat(NamedTuple):
    """A conversation exported from elsewhere, as pairs of prompt and response."""

    title: str
    turns: list[ImportedTurn]
    # a prompt the export ends on with nothing answering it; there is no turn to show for it
    trailing_prompt: str | None = None


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


OVERRIDE_ALIASES: dict[str, str] = {'t': 'thinking', 's': 'search', 'r': 'richtext', 'dt': 'dummy_tool'}
