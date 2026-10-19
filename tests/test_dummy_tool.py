"""`dt` (dummy tool): chains of thought go back as reasoning behind a tool that does nothing, or not at all."""

import asyncio
from types import SimpleNamespace
from typing import Any, Self

import pytest

from chatgpt_telegram_bot.completion import (
    DUMMY_TOOL_NAME,
    DUMMY_TOOL_RESPONSES,
    DUMMY_TOOL_RESULT,
    DUMMY_TOOL_ROUNDS,
    build_input_openai,
    completion,
    reasoning_text_item,
)
from chatgpt_telegram_bot.models import (
    Model,
    MsgPartInHistory,
    ResponseDelta,
    StreamMeta,
    ThinkingDelta,
    make_text_part,
    make_thinking_text_part,
)
from chatgpt_telegram_bot.utils import apply_overrides

HISTORY: list[list[MsgPartInHistory]] = [
    [make_text_part('pick a number')],
    [make_thinking_text_part('I pick 7.'), make_text_part('done')],
    [make_text_part('which one?')],
]


async def _no_image(_key: str) -> bytes | None:
    return None


class FakeStream:
    """An SDK stream: iterable once, closable as a context manager."""

    def __init__(self, events: list[Any]) -> None:
        self.events: list[Any] = events

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None

    def __aiter__(self) -> Any:
        return self._iterate()

    async def _iterate(self) -> Any:
        for event in self.events:
            yield event


class FakeClient:
    """Records every request and answers each with the next scripted stream."""

    def __init__(self, *rounds: list[Any]) -> None:
        self.rounds: list[list[Any]] = list(rounds)
        self.requests: list[dict[str, Any]] = []
        self.base_url: str = 'https://fake.test/v1'
        self.chat: Any = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.responses: Any = SimpleNamespace(create=self._create)

    async def _create(self, **kwargs: Any) -> FakeStream:
        # a snapshot: the backend extends its message list in place between rounds
        self.requests.append(
            {**kwargs, 'messages': list(kwargs.get('messages', [])), 'input': list(kwargs.get('input', []))}
        )
        return FakeStream(self.rounds.pop(0))


def _run(client: FakeClient, model: Model, history: list[list[MsgPartInHistory]] = HISTORY) -> list[Any]:
    async def collect() -> list[Any]:
        return [event async for event in completion(client, history, model, 'be helpful', 1, 2, _no_image)]

    return asyncio.run(collect())


# --- the override


def test_dt_is_an_override_like_the_other_flags():
    model = Model(prefix='d', name='deepseek-flash')
    assert apply_overrides(model, {'dt': None}).dummy_tool
    assert not apply_overrides(model._replace(dummy_tool=True), {'dt': ''}).dummy_tool


@pytest.mark.parametrize('api_type', ['anthropic', 'openai_legacy'])
def test_dt_is_refused_outside_the_responses_api(api_type: str):
    with pytest.raises(ValueError, match='dt needs api_type openai'):
        _ = apply_overrides(Model(prefix='c', name='m', api_type=api_type), {'dt': None})


# --- the Responses API


def test_responses_input_carries_the_cot_before_its_reply_under_dt():
    items = asyncio.run(build_input_openai(HISTORY, _no_image, 'deepseek-flash', True, dummy_tool=True))
    assert items == [
        {'role': 'user', 'content': 'pick a number'},
        reasoning_text_item('I pick 7.'),
        {'role': 'assistant', 'content': 'done'},
        {'role': 'user', 'content': 'which one?'},
    ]


def _completed(output: list[Any], text_in: int = 10) -> Any:
    usage = SimpleNamespace(
        input_tokens=text_in, output_tokens=5, input_tokens_details=None, output_tokens_details=None
    )
    return SimpleNamespace(
        type='response.completed',
        response=SimpleNamespace(output=output, status='completed', incomplete_details=None, usage=usage),
    )


def _item(dump: dict[str, Any]) -> Any:
    def model_dump(**_kw: Any) -> dict[str, Any]:
        return dict(dump)

    return SimpleNamespace(**dump, model_dump=model_dump)


def test_responses_request_under_dt_offers_the_dummy_tool():
    client = FakeClient([_completed([])])
    _ = _run(client, Model(prefix='d', name='deepseek-flash', thinking='high', dummy_tool=True))
    request = client.requests[0]
    assert request['tools'] == [DUMMY_TOOL_RESPONSES]
    assert reasoning_text_item('I pick 7.') in request['input']


def test_responses_request_without_dt_drops_the_cot():
    client = FakeClient([_completed([])])
    _ = _run(client, Model(prefix='d', name='deepseek-flash', thinking='high'))
    request = client.requests[0]
    assert 'tools' not in request
    assert not [i for i in request['input'] if i.get('type') == 'reasoning']
    assert {'role': 'assistant', 'content': 'done'} in request['input'], 'and no <think> block in the text'


def test_responses_dummy_call_is_answered_and_asked_again():
    reasoning = _item(
        {'type': 'reasoning', 'id': 'rs_1', 'summary': [], 'content': [{'type': 'reasoning_text', 'text': 'hmm'}]}
    )
    call_dump = {
        'type': 'function_call',
        'id': 'fc_1',
        'call_id': 'call_1',
        'name': DUMMY_TOOL_NAME,
        'arguments': '{}',
    }
    call = _item(call_dump)
    first = [_completed([reasoning, call], text_in=10)]
    second = [SimpleNamespace(type='response.output_text.delta', delta='seven'), _completed([], text_in=20)]
    client = FakeClient(first, second)
    events = _run(client, Model(prefix='d', name='deepseek-flash', thinking='high', dummy_tool=True))

    assert len(client.requests) == 2
    followup = client.requests[1]['input'][len(client.requests[0]['input']) :]
    assert followup == [
        reasoning_text_item('hmm'),  # the reasoning goes back in the clear, without its id
        call_dump,
        {'type': 'function_call_output', 'call_id': 'call_1', 'output': DUMMY_TOOL_RESULT},
    ]
    assert [e.text for e in events if isinstance(e, ResponseDelta)] == ['seven']
    meta = [e for e in events if isinstance(e, StreamMeta)]
    assert len(meta) == 1
    assert meta[0].usage['in'] == 30, 'usage of both requests'


def test_responses_gives_up_on_a_model_that_keeps_calling():
    call = _item(
        {'type': 'function_call', 'id': 'fc_1', 'call_id': 'call_1', 'name': DUMMY_TOOL_NAME, 'arguments': '{}'}
    )
    client = FakeClient(*[[_completed([call])] for _ in range(DUMMY_TOOL_ROUNDS)])
    events = _run(client, Model(prefix='d', name='deepseek-flash', thinking='high', dummy_tool=True))
    assert len(client.requests) == DUMMY_TOOL_ROUNDS
    assert any('kept calling the dummy tool' in e.text for e in events if isinstance(e, ResponseDelta))


# --- Chat Completions: no dt, so a chain of thought is kept but never sent


def _chunk(
    content: str | None = None, reasoning: str | None = None, finish: str | None = None, calls: Any = None
) -> Any:
    delta = SimpleNamespace(content=content, reasoning_content=reasoning, tool_calls=calls)
    return SimpleNamespace(usage=None, choices=[SimpleNamespace(finish_reason=finish, delta=delta)])


LEGACY = Model(prefix='d', name='deepseek-flash', api_type='openai_legacy', thinking='high')


def test_chat_request_never_carries_the_cot():
    client = FakeClient([_chunk('ok', finish='stop')])
    _ = _run(client, LEGACY)
    request = client.requests[0]
    assert 'tools' not in request
    assert request['messages'][2] == {'role': 'assistant', 'content': 'done'}


def test_chat_cot_is_kept_for_a_thread_that_moves_to_a_dt_model():
    client = FakeClient([_chunk(reasoning='think '), _chunk(reasoning='hard'), _chunk('ok', finish='stop')])
    events = _run(client, LEGACY)
    assert [e.text for e in events if isinstance(e, ThinkingDelta)] == ['think ', 'hard']
    meta = [e for e in events if isinstance(e, StreamMeta)]
    assert meta[-1].carry_items == [reasoning_text_item('think hard')]
