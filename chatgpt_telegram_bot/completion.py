import base64
import json
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from contextlib import aclosing
from dataclasses import dataclass, field
from typing import Any

import anthropic
import openai
from loguru import logger

from chatgpt_telegram_bot.models import (
    Model,
    MsgPartInHistory,
    ResponseDelta,
    StatusChange,
    StreamEvent,
    StreamMeta,
    ThinkingDelta,
)
from chatgpt_telegram_bot.utils import image_media_type

LoadImage = Callable[[str], Awaitable[bytes | None]]

# Request features an endpoint has already rejected, keyed by (base url, model). Endpoints
# implement the Responses API to different depths, and without this every turn would repeat
# the same rejected request before degrading.
_unsupported: dict[tuple[str, str], set[str]] = {}


def is_replayable_reasoning(item: Any) -> bool:
    """
    Whether a reasoning item carries enough of itself to be replayed later.

    OpenAI ships `encrypted_content`; DeepSeek's Responses API ships plain `reasoning_text`
    content parts instead. An item with neither is only an id, which nothing can replay
    without server-side state we do not keep.
    """
    return bool(getattr(item, 'encrypted_content', None) or getattr(item, 'content', None))


def drop_thinking_text(msg_parts: list[MsgPartInHistory]) -> list[MsgPartInHistory]:
    """The message without its stored chains of thought, for a request that does not carry them."""
    return [part for part in msg_parts if part.type_ != 'thinking_text']


def thinking_text_of(msg_parts: list[MsgPartInHistory]) -> str:
    """The stored chains of thought of a message, joined into one."""
    return '\n\n'.join(part.text for part in msg_parts if part.type_ == 'thinking_text' and part.text)


# DeepSeek keeps a passed-back chain of thought only in a request that defines a tool, and
# ignores it otherwise -- an empty tool list and tool_choice="none" both count as no tools,
# as measured by the input token count. `dt` (dummy tool) therefore sends one tool that does
# nothing, and the stored chains of thought with it. Responses API only.
DUMMY_TOOL_NAME = 'noop'
DUMMY_TOOL_RESULT = 'This tool does nothing. Answer the user directly.'
# a model that calls the dummy tool anyway is told it did nothing and asked again, this many
# requests in all before the turn gives up
DUMMY_TOOL_ROUNDS = 3
# every word of the definition lands in the model's context on every request, so it is a name
# and a few words saying to leave it alone, with no parameters; the ~225 tokens DeepSeek adds
# for having tools at all are not ours to trim
DUMMY_TOOL_RESPONSES: dict[str, Any] = {
    'type': 'function',
    'name': DUMMY_TOOL_NAME,
    'description': 'For testing, ignore.',
}
_DUMMY_TOOL_LOOP_ERROR = '\n\n[!] Error: the model kept calling the dummy tool'


def reasoning_text_item(text: str) -> dict[str, Any]:
    """A Responses API reasoning item carrying *text* in the clear.

    It has no id and no encrypted content: DeepSeek reads an item of this shape, while its
    own items echoed back with their id did not reach the input token count.
    """
    return {'type': 'reasoning', 'summary': [], 'content': [{'type': 'reasoning_text', 'text': text}]}


def _add_usage(total: dict[str, Any], usage: dict[str, Any]) -> None:
    """Sum *usage* into *total*, for a turn that took more than one request."""
    for key, value in usage.items():
        total[key] = total.get(key, 0) + value


def reasoning_text_of(item: dict[str, Any]) -> str:
    """
    The plain reasoning text an item carries, if any.

    DeepSeek's Responses API ships the chain of thought as `reasoning_text` content parts;
    OpenAI ships opaque ciphertext and leaves these empty.
    """
    parts = item.get('content') or []
    return '\n\n'.join(p['text'] for p in parts if isinstance(p, dict) and p.get('text'))


def read_reasoning_part(part: MsgPartInHistory, model_name: str) -> dict[str, Any] | None:
    """Decode a stored reasoning item, discarding it unless it belongs to model_name."""
    if not part.text:
        return None
    try:
        payload = json.loads(part.text)
    except json.JSONDecodeError:
        logger.warning('Discarding malformed reasoning part')
        return None
    if not isinstance(payload, dict):
        logger.warning('Discarding reasoning part with unexpected shape')
        return None
    stored_model = payload.get('model')
    if stored_model != model_name:
        logger.debug(f'Skipping reasoning item of another model ({stored_model=}, {model_name=})')
        return None
    item = payload.get('item')
    if not isinstance(item, dict):
        logger.warning('Discarding reasoning part without an item')
        return None
    return item


async def build_input_openai(
    chat_history: list[list[MsgPartInHistory]],
    load_image: LoadImage,
    model_name: str,
    replay_reasoning: bool,
    dummy_tool: bool = False,
) -> list[dict[str, Any]]:
    """
    Assemble Responses API input items from the neutral history.

    Stored reasoning items are replayed as top-level items immediately before the
    assistant message they produced, which is the order the API requires. With
    *dummy_tool* a stored plain chain of thought goes the same way, as a
    `reasoning_text_item()`.
    """
    items: list[dict[str, Any]] = []
    roles = ['user', 'assistant']
    for i, msg_parts in enumerate(chat_history):
        role = roles[i % len(roles)]
        text_type = 'output_text' if role == 'assistant' else 'input_text'
        content: list[dict[str, Any]] = []
        reasoning: list[dict[str, Any]] = []
        for part in msg_parts:
            if part.type_ == 'text':
                content.append({'type': text_type, 'text': part.text})
            elif part.type_ == 'image':
                assert part.hash is not None
                blob = await load_image(part.hash)
                if blob is None:
                    continue
                blob_base64 = base64.b64encode(blob).decode()
                data_url = f'data:{image_media_type(blob)};base64,{blob_base64}'
                content.append({'type': 'input_image', 'image_url': data_url})
            elif part.type_ == 'reasoning' and replay_reasoning:
                item = read_reasoning_part(part, model_name)
                if item is not None:
                    reasoning.append(item)
        if dummy_tool and role == 'assistant':
            chain_of_thought = thinking_text_of(msg_parts)
            if chain_of_thought:
                reasoning.append(reasoning_text_item(chain_of_thought))
        items.extend(reasoning)
        # a lone text part still goes out as a bare string, the shape sent before reasoning
        # replay existed, so untouched conversations keep hitting the prompt cache
        if len(content) == 1 and content[0]['type'] == text_type:
            items.append({'role': role, 'content': content[0]['text']})
        else:
            items.append({'role': role, 'content': content})
    return items


def has_reasoning_items(items: list[dict[str, Any]]) -> bool:
    return any(item.get('type') == 'reasoning' for item in items)


def strip_reasoning_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in items if item.get('type') != 'reasoning']


async def convert_history_anthropic(
    chat_history: list[list[MsgPartInHistory]], load_image: LoadImage
) -> list[list[dict[str, Any]]]:
    result: list[list[dict[str, Any]]] = []
    for msg_parts in chat_history:
        converted: list[dict[str, Any]] = []
        for part in msg_parts:
            if part.type_ == 'text':
                converted.append({'type': 'text', 'text': part.text})
            elif part.type_ == 'image':
                assert part.hash is not None
                blob = await load_image(part.hash)
                if blob is None:
                    continue
                blob_base64 = base64.b64encode(blob).decode()
                converted.append(
                    {
                        'type': 'image',
                        'source': {'type': 'base64', 'media_type': image_media_type(blob), 'data': blob_base64},
                    }
                )
        result.append(converted)
    return result


async def completion_openai(
    client: openai.AsyncOpenAI,
    chat_history: list[list[MsgPartInHistory]],
    model: Model,
    system_prompt: str,
    chat_id: int,
    msg_id: int,
    load_image: LoadImage,
) -> AsyncIterator[StreamEvent]:
    # replaying reasoning to a turn that has reasoning switched off is not meaningful,
    # and the API rejects reasoning items when no reasoning is requested
    replay_reasoning = model.thinking is not None
    input_items = await build_input_openai(chat_history, load_image, model.name, replay_reasoning, model.dummy_tool)
    n_replayed = sum(1 for item in input_items if item.get('type') == 'reasoning')
    if n_replayed:
        logger.debug(f'Replaying reasoning items ({chat_id=}, {msg_id=}): {n_replayed}')

    unsupported = _unsupported.setdefault((str(client.base_url), model.name), set())
    if unsupported:
        logger.debug(f'Endpoint features known unsupported ({chat_id=}, {msg_id=}): {sorted(unsupported)}')
    if 'reasoning_replay' in unsupported:
        input_items = strip_reasoning_items(input_items)

    kwargs: dict[str, Any] = {
        'model': model.name,
        'input': input_items,
        'stream': True,
    }
    if 'store' not in unsupported:
        # the reply chain in the shelve DB is the only conversation state we rely on, and
        # reasoning travels by value, so server-side retention buys nothing
        kwargs['store'] = False
    if system_prompt and not model.no_system_prompt:
        kwargs['instructions'] = system_prompt
    if model.thinking is not None:
        effort = model.thinking if isinstance(model.thinking, str) else 'high'
        kwargs['reasoning'] = {'effort': effort}
        if 'reasoning_summary' not in unsupported:
            kwargs['reasoning']['summary'] = 'detailed'
        if 'include' not in unsupported:
            # ask for reasoning items we can persist and replay on the following turn;
            # endpoints that return plain reasoning text instead simply ignore this
            kwargs['include'] = ['reasoning.encrypted_content']
    elif 'reasoning_none' not in unsupported:
        # a hybrid model thinks unless told not to (DeepSeek documents its Responses API as
        # thinking by default), so silence is not enough to turn it off
        kwargs['reasoning'] = {'effort': 'none'}
    tools: list[dict[str, Any]] = []
    if model.search:
        tools.append({'type': 'web_search', 'search_context_size': 'medium'})
    if model.dummy_tool:
        tools.append(DUMMY_TOOL_RESPONSES)
    if tools:
        kwargs['tools'] = tools

    meta = StreamMeta()
    completed = False
    for round_no in range(DUMMY_TOOL_ROUNDS):
        dummy_calls: list[Any] = []
        followup: list[dict[str, Any]] = []
        # aclosing, so an abort that closes this generator closes the request's stream too
        async with aclosing(_stream_openai_round(client, kwargs, unsupported, chat_id, msg_id, meta)) as events:
            async for event in events:
                if isinstance(event, _RoundDone):
                    completed = True
                    dummy_calls = event.dummy_calls
                    followup = event.followup
                    continue
                yield event
        if not dummy_calls:
            break
        if round_no + 1 == DUMMY_TOOL_ROUNDS:
            yield ResponseDelta(text=_DUMMY_TOOL_LOOP_ERROR)
            break
        logger.info(f'Model called the dummy tool, asking again ({chat_id=}, {msg_id=}, {round_no=})')
        outputs = [
            {'type': 'function_call_output', 'call_id': call.call_id, 'output': DUMMY_TOOL_RESULT}
            for call in dummy_calls
        ]
        kwargs['input'] = [*kwargs['input'], *followup, *outputs]
    if completed:
        yield meta


@dataclass
class _RoundDone(StreamEvent):
    """Internal: one Responses request finished; what it asked of the dummy tool."""

    dummy_calls: list[Any] = field(default_factory=list)
    # the output items to send back if the turn goes on, reasoning in its plain-text form
    followup: list[dict[str, Any]] = field(default_factory=list)


async def _stream_openai_round(
    client: openai.AsyncOpenAI,
    kwargs: dict[str, Any],
    unsupported: set[str],
    chat_id: int,
    msg_id: int,
    meta: StreamMeta,
) -> AsyncGenerator[StreamEvent, None]:
    """Run one Responses request, folding its reasoning, searches and usage into *meta*."""
    # an endpoint may reject the replayed items (expired or unsupported) or any of the
    # optional parameters; shed them one at a time rather than failing a turn the model can
    # still answer, and remember what was rejected so the next turn asks for less
    while True:
        try:
            stream = await client.responses.create(**kwargs)
            break
        except openai.BadRequestError as e:
            if has_reasoning_items(kwargs['input']):
                feature = 'reasoning_replay'
                kwargs['input'] = strip_reasoning_items(kwargs['input'])
            elif kwargs.get('include'):
                feature = 'include'
                del kwargs['include']
            elif isinstance(kwargs.get('reasoning'), dict) and 'summary' in kwargs['reasoning']:
                feature = 'reasoning_summary'
                del kwargs['reasoning']['summary']
            elif kwargs.get('reasoning') == {'effort': 'none'}:
                feature = 'reasoning_none'
                del kwargs['reasoning']
            elif 'store' in kwargs:
                feature = 'store'
                del kwargs['store']
            else:
                raise
            unsupported.add(feature)
            logger.warning(f'Endpoint rejected {feature} ({chat_id=}, {msg_id=}): {e}')
    search_queries: list[str] = []
    # leaving the block closes the HTTP stream, including when an abort closes this generator
    async with stream:
        async for event in stream:
            logger.trace(f'Received event ({chat_id=}, {msg_id=}): {event.type}')
            if event.type == 'response.web_search_call.searching':
                yield StatusChange(status='Searching...')
            elif event.type == 'response.output_item.done' and event.item.type == 'web_search_call':
                query = getattr(event.item.action, 'query', None) if hasattr(event.item, 'action') else None
                if query:
                    search_queries.append(query)
                yield StatusChange(status='Generating...')
            elif event.type == 'response.output_item.done' and event.item.type == 'reasoning':
                item = event.item
                encrypted = getattr(item, 'encrypted_content', None)
                replayable = 'yes' if is_replayable_reasoning(item) else 'no'
                payload = f'{len(encrypted)}B' if encrypted else 'none'
                parts = getattr(item, 'content', None) or []
                cot_chars = sum(len(getattr(c, 'text', '') or '') for c in parts)
                n_summary = len(item.summary or [])
                counts = f'content_parts={len(parts)} cot_chars={cot_chars} summary_parts={n_summary}'
                fields = sorted(item.model_dump(exclude_none=True).keys())
                desc = f'id={item.id} replayable={replayable} encrypted={payload} {counts} fields={fields}'
                logger.debug(f'Reasoning item ({chat_id=}, {msg_id=}): {desc}')
            elif event.type in ('response.reasoning_summary_text.delta', 'response.reasoning_text.delta'):
                yield ThinkingDelta(text=event.delta)
            elif event.type == 'response.output_text.delta':
                yield ResponseDelta(text=event.delta)
            elif event.type == 'response.failed':
                err = event.response.error
                detail = f'code={err.code} {err.message}' if err else 'no error object'
                logger.error(f'Response failed ({chat_id=}, {msg_id=}): {detail}')
            elif event.type == 'response.incomplete':
                details = event.response.incomplete_details
                reason = details.reason if details else None
                logger.warning(f'Response incomplete ({chat_id=}, {msg_id=}): {reason=}')
            elif event.type == 'error':
                detail = f'code={event.code} param={event.param} {event.message}'
                logger.error(f'Stream error ({chat_id=}, {msg_id=}): {detail}')
            elif event.type == 'response.completed':
                resp = event.response
                item_types = [it.type for it in resp.output]
                carry = [it for it in resp.output if it.type == 'reasoning' and is_replayable_reasoning(it)]
                dummy_calls = [it for it in resp.output if it.type == 'function_call' and it.name == DUMMY_TOOL_NAME]
                n_reasoning = sum(1 for it in resp.output if it.type == 'reasoning')
                # once per completion, after every delta has been yielded
                kept = f'{len(carry)}/{n_reasoning}'
                logger.debug(f'Completion done ({chat_id=}, {msg_id=}): output={item_types} reasoning_kept={kept}')
                if resp.status == 'incomplete' and resp.incomplete_details:
                    reason = resp.incomplete_details.reason
                    if reason == 'max_output_tokens':
                        yield ResponseDelta(text='\n\n[!] Error: Output truncated due to limit')
                    else:
                        yield ResponseDelta(text=f'\n\n[!] Error: incomplete reason="{reason}"')
                meta.carry_items.extend(it.model_dump(exclude_none=True) for it in carry)
                meta.tool_calls.extend(search_queries)
                if resp.usage:
                    usage: dict[str, Any] = {
                        'in': resp.usage.input_tokens,
                        'out': resp.usage.output_tokens,
                    }
                    if resp.usage.input_tokens_details and resp.usage.input_tokens_details.cached_tokens:
                        usage['cached'] = resp.usage.input_tokens_details.cached_tokens
                    if resp.usage.output_tokens_details and resp.usage.output_tokens_details.reasoning_tokens:
                        usage['reasoning'] = resp.usage.output_tokens_details.reasoning_tokens
                    _add_usage(meta.usage, usage)
                followup: list[dict[str, Any]] = []
                for it in resp.output:
                    dumped = it.model_dump(exclude_none=True)
                    chain_of_thought = reasoning_text_of(dumped) if it.type == 'reasoning' else ''
                    followup.append(reasoning_text_item(chain_of_thought) if chain_of_thought else dumped)
                yield _RoundDone(dummy_calls=dummy_calls, followup=followup)


async def completion_openai_legacy(
    client: openai.AsyncOpenAI,
    chat_history: list[list[MsgPartInHistory]],
    model: Model,
    system_prompt: str,
    chat_id: int,
    msg_id: int,
    load_image: LoadImage,
) -> AsyncGenerator[StreamEvent, None]:
    messages: list[dict[str, Any]] = []
    if system_prompt and not model.no_system_prompt:
        messages.append({'role': 'system', 'content': system_prompt})
    roles = ['user', 'assistant']
    for i, msg_parts in enumerate(chat_history):
        role = roles[i % len(roles)]
        parts: list[dict[str, Any]] = []
        for part in msg_parts:
            if part.type_ == 'text':
                parts.append({'type': 'text', 'text': part.text})
            elif part.type_ == 'image':
                assert part.hash is not None
                blob = await load_image(part.hash)
                if blob is None:
                    continue
                blob_base64 = base64.b64encode(blob).decode()
                data_url = f'data:{image_media_type(blob)};base64,{blob_base64}'
                parts.append({'type': 'image_url', 'image_url': {'url': data_url}})
        content: Any = parts
        if len(parts) == 1 and parts[0]['type'] == 'text':
            content = parts[0]['text']
        messages.append({'role': role, 'content': content})

    kwargs: dict[str, Any] = {
        'model': model.name,
        'messages': messages,
        'stream': True,
        'stream_options': {'include_usage': True},
    }
    if model.thinking is not None:
        effort = model.thinking if isinstance(model.thinking, str) else 'high'
        kwargs['reasoning_effort'] = effort
        kwargs['extra_body'] = {'thinking': {'type': 'enabled'}}

    meta = StreamMeta()
    reasoning = ''
    stream = await client.chat.completions.create(**kwargs)
    async with stream:  # an abort closes the request's stream with this generator
        async for chunk in stream:
            logger.trace(f'Received chunk ({chat_id=}, {msg_id=}): {chunk}')
            if chunk.usage:
                usage: dict[str, Any] = {
                    'in': chunk.usage.prompt_tokens,
                    'out': chunk.usage.completion_tokens,
                }
                if chunk.usage.prompt_tokens_details and chunk.usage.prompt_tokens_details.cached_tokens:
                    usage['cached'] = chunk.usage.prompt_tokens_details.cached_tokens
                details = chunk.usage.completion_tokens_details
                if details and details.reasoning_tokens:
                    usage['reasoning'] = details.reasoning_tokens
                meta.usage = usage
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if choice.finish_reason == 'length':
                yield ResponseDelta(text='\n\n[!] Error: Output truncated due to limit')
            if choice.delta:
                delta_reasoning = getattr(choice.delta, 'reasoning_content', None)
                if delta_reasoning:
                    reasoning += delta_reasoning
                    yield ThinkingDelta(text=delta_reasoning)
                if choice.delta.content:
                    yield ResponseDelta(text=choice.delta.content)

    # this backend never sends a chain of thought back (dt is for the Responses API only), but
    # it keeps it in the same plain form, so a thread moved to a dt model takes it along
    if reasoning:
        meta.carry_items = [reasoning_text_item(reasoning)]
    yield meta


async def completion_anthropic(
    client: anthropic.AsyncAnthropic,
    chat_history: list[list[MsgPartInHistory]],
    model: Model,
    system_prompt: str,
    chat_id: int,
    msg_id: int,
    load_image: LoadImage,
) -> AsyncIterator[StreamEvent]:
    converted = await convert_history_anthropic(chat_history, load_image)
    messages: list[Any] = []
    roles = ['user', 'assistant']
    for i, msg in enumerate(converted):
        role = roles[i % len(roles)]
        content: Any = msg
        if len(msg) == 1 and msg[0]['type'] == 'text':
            content = msg[0]['text']
        messages.append({'role': role, 'content': content})

    kwargs: dict[str, Any] = {
        'model': model.name,
        'max_tokens': 16384,
        'messages': messages,
    }
    if system_prompt and not model.no_system_prompt:
        kwargs['system'] = system_prompt
    if model.thinking == 'adaptive':
        kwargs['thinking'] = {'type': 'adaptive'}
    elif isinstance(model.thinking, int) and model.thinking > 0:
        kwargs['thinking'] = {'type': 'enabled', 'budget_tokens': model.thinking}

    if model.search:
        kwargs.setdefault('tools', []).append({'type': 'web_search_20250305', 'name': 'web_search'})

    async with client.messages.stream(**kwargs) as stream:
        async for event in stream:
            logger.trace(f'Received event ({chat_id=}, {msg_id=}): {event.type}')
            if event.type == 'content_block_start':
                logger.debug(f'Block start ({chat_id=}, {msg_id=}): type={event.content_block.type}')
                if event.content_block.type == 'server_tool_use' and event.content_block.name == 'web_search':
                    yield StatusChange(status='Searching...')
                elif event.content_block.type == 'web_search_tool_result':
                    yield StatusChange(status='Generating...')
            elif event.type == 'content_block_delta':
                if event.delta.type == 'thinking_delta':
                    yield ThinkingDelta(text=event.delta.thinking)
                elif event.delta.type == 'text_delta':
                    yield ResponseDelta(text=event.delta.text)
        final_message = await stream.get_final_message()
        if final_message.stop_reason == 'max_tokens':
            yield ResponseDelta(text='\n\n[!] Error: Output truncated due to limit')
        elif final_message.stop_reason not in ('end_turn', 'stop_sequence'):
            yield ResponseDelta(text=f'\n\n[!] Error: stop_reason="{final_message.stop_reason}"')
    # build metadata from final message
    meta = StreamMeta()
    search_queries: list[str] = []
    for block in final_message.content:
        if block.type == 'server_tool_use' and block.name == 'web_search':
            query = block.input.get('query', '')
            if query:
                search_queries.append(str(query))
    if search_queries:
        meta.tool_calls = search_queries
    usage: dict[str, Any] = {
        'in': final_message.usage.input_tokens,
        'out': final_message.usage.output_tokens,
    }
    cache_creation = getattr(final_message.usage, 'cache_creation_input_tokens', None)
    cache_read = getattr(final_message.usage, 'cache_read_input_tokens', None)
    if cache_creation:
        usage['cache_write'] = cache_creation
    if cache_read:
        usage['cache_read'] = cache_read
    server_tool_use = getattr(final_message.usage, 'server_tool_use', None)
    if server_tool_use:
        web_search_requests = getattr(server_tool_use, 'web_search_requests', None)
        if web_search_requests:
            usage['web_searches'] = web_search_requests
    meta.usage = usage
    yield meta


async def completion(
    client: Any,
    chat_history: list[list[MsgPartInHistory]],
    model: Model,
    system_prompt: str,
    chat_id: int,
    msg_id: int,
    load_image: LoadImage,
) -> AsyncGenerator[StreamEvent, None]:
    """Dispatch to the appropriate completion backend based on model.api_type.

    An AsyncGenerator rather than a plain iterator because the caller closes it to stop a
    generation early, which unwinds the backend and releases the HTTP stream.
    """
    assert len(chat_history) % 2 == 1
    # a stored chain of thought goes back only under dt, as reasoning behind the dummy tool
    # tool; anything else would have to inline it as text, which the model then imitates
    if not model.dummy_tool:
        chat_history = [drop_thinking_text(parts) for parts in chat_history]
    last_msg_parts = chat_history[-1]
    last_text = ' '.join(p.text for p in last_msg_parts if p.type_ == 'text' and p.text)
    if len(last_text) > 80:
        last_text = last_text[:80] + '…'
    n_img = sum(1 for p in last_msg_parts if p.type_ == 'image')
    img_info = f' +{n_img}img' if n_img else ''
    sys_info = f' sys="{system_prompt[:60]}…"' if len(system_prompt) > 60 else f' sys="{system_prompt}"'
    logger.info(
        f'Completion {model.api_type} {chat_id}:{msg_id}{sys_info} [{len(chat_history)}msg] "{last_text}"{img_info}'
    )

    if model.api_type == 'openai':
        backend = completion_openai(client, chat_history, model, system_prompt, chat_id, msg_id, load_image)
    elif model.api_type == 'openai_legacy':
        backend = completion_openai_legacy(client, chat_history, model, system_prompt, chat_id, msg_id, load_image)
    elif model.api_type == 'anthropic':
        backend = completion_anthropic(client, chat_history, model, system_prompt, chat_id, msg_id, load_image)
    else:
        raise ValueError(f'Unknown api_type: {model.api_type}')

    async for delta in backend:
        yield delta
