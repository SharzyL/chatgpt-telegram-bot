import base64
import json
from collections.abc import AsyncIterator, Awaitable, Callable
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
    make_text_part,
)

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


def merge_thinking_text(msg_parts: list[MsgPartInHistory]) -> list[MsgPartInHistory]:
    """
    Fold a user-supplied chain of thought into the message text as a <think> block.

    No backend takes a synthesised reasoning item — the Responses API wants encrypted
    content and Anthropic wants a signature — so the text form is the only way a
    manipulated chain of thought can reach the model.
    """
    blocks = [f'<think>\n{part.text}\n</think>' for part in msg_parts if part.type_ == 'thinking_text' and part.text]
    if not blocks:
        return msg_parts
    prelude = '\n\n'.join(blocks)
    merged: list[MsgPartInHistory] = []
    injected = False
    for part in msg_parts:
        if part.type_ == 'thinking_text':
            continue
        if part.type_ == 'text' and not injected:
            merged.append(make_text_part(prelude + '\n\n' + (part.text or '')))
            injected = True
        else:
            merged.append(part)
    if not injected:
        merged.insert(0, make_text_part(prelude))
    return merged


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
) -> list[dict[str, Any]]:
    """
    Assemble Responses API input items from the neutral history.

    Stored reasoning items are replayed as top-level items immediately before the
    assistant message they produced, which is the order the API requires.
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
                content.append({'type': 'input_image', 'image_url': 'data:image/jpeg;base64,' + blob_base64})
            elif part.type_ == 'reasoning' and replay_reasoning:
                item = read_reasoning_part(part, model_name)
                if item is not None:
                    reasoning.append(item)
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
                        'source': {'type': 'base64', 'media_type': 'image/jpeg', 'data': blob_base64},
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
    input_items = await build_input_openai(chat_history, load_image, model.name, replay_reasoning)
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
    if model.search:
        kwargs['tools'] = [{'type': 'web_search', 'search_context_size': 'medium'}]

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
            n_content = len(getattr(item, 'content', None) or [])
            n_summary = len(item.summary or [])
            counts = f'content_parts={n_content} summary_parts={n_summary}'
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
            meta = StreamMeta()
            meta.carry_items = [it.model_dump(exclude_none=True) for it in carry]
            if search_queries:
                meta.tool_calls = search_queries
            if resp.usage:
                usage: dict[str, Any] = {
                    'in': resp.usage.input_tokens,
                    'out': resp.usage.output_tokens,
                }
                if resp.usage.input_tokens_details and resp.usage.input_tokens_details.cached_tokens:
                    usage['cached'] = resp.usage.input_tokens_details.cached_tokens
                if resp.usage.output_tokens_details and resp.usage.output_tokens_details.reasoning_tokens:
                    usage['reasoning'] = resp.usage.output_tokens_details.reasoning_tokens
                meta.usage = usage
            yield meta


async def completion_openai_legacy(
    client: openai.AsyncOpenAI,
    chat_history: list[list[MsgPartInHistory]],
    model: Model,
    system_prompt: str,
    chat_id: int,
    msg_id: int,
    load_image: LoadImage,
) -> AsyncIterator[StreamEvent]:
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
                parts.append({'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + blob_base64}})
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

    stream = await client.chat.completions.create(**kwargs)
    async for chunk in stream:
        logger.trace(f'Received chunk ({chat_id=}, {msg_id=}): {chunk}')
        if chunk.usage:
            meta = StreamMeta()
            usage: dict[str, Any] = {
                'in': chunk.usage.prompt_tokens,
                'out': chunk.usage.completion_tokens,
            }
            if chunk.usage.prompt_tokens_details and chunk.usage.prompt_tokens_details.cached_tokens:
                usage['cached'] = chunk.usage.prompt_tokens_details.cached_tokens
            if chunk.usage.completion_tokens_details and chunk.usage.completion_tokens_details.reasoning_tokens:
                usage['reasoning'] = chunk.usage.completion_tokens_details.reasoning_tokens
            meta.usage = usage
            yield meta
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        if choice.finish_reason == 'length':
            yield ResponseDelta(text='\n\n[!] Error: Output truncated due to limit')
        if choice.delta:
            reasoning = getattr(choice.delta, 'reasoning_content', None)
            if reasoning:
                yield ThinkingDelta(text=reasoning)
            if choice.delta.content:
                yield ResponseDelta(text=choice.delta.content)


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
) -> AsyncIterator[StreamEvent]:
    """Dispatch to the appropriate completion backend based on model.api_type."""
    assert len(chat_history) % 2 == 1
    chat_history = [merge_thinking_text(parts) for parts in chat_history]
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
