import base64
from collections.abc import AsyncIterator
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
from chatgpt_telegram_bot.utils import load_photo


def convert_history_openai(chat_history: list[list[MsgPartInHistory]]) -> list[list[dict[str, Any]]]:
    result: list[list[dict[str, Any]]] = []
    for msg_parts in chat_history:
        converted: list[dict[str, Any]] = []
        for part in msg_parts:
            if part.type_ == 'text':
                converted.append({'type': 'input_text', 'text': part.text})
            elif part.type_ == 'image':
                assert part.hash is not None
                blob = load_photo(part.hash)
                blob_base64 = base64.b64encode(blob).decode()
                converted.append({'type': 'input_image', 'image_url': 'data:image/jpeg;base64,' + blob_base64})
        result.append(converted)
    return result


def convert_history_anthropic(chat_history: list[list[MsgPartInHistory]]) -> list[list[dict[str, Any]]]:
    result: list[list[dict[str, Any]]] = []
    for msg_parts in chat_history:
        converted: list[dict[str, Any]] = []
        for part in msg_parts:
            if part.type_ == 'text':
                converted.append({'type': 'text', 'text': part.text})
            elif part.type_ == 'image':
                assert part.hash is not None
                blob = load_photo(part.hash)
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
) -> AsyncIterator[StreamEvent]:
    converted = convert_history_openai(chat_history)
    input_messages: list[Any] = []
    roles = ['user', 'assistant']
    for i, msg in enumerate(converted):
        role = roles[i % len(roles)]
        content: Any = msg
        if len(msg) == 1 and msg[0]['type'] == 'input_text':
            content = msg[0]['text']
        input_messages.append({'role': role, 'content': content})

    kwargs: dict[str, Any] = {
        'model': model.name,
        'input': input_messages,
        'stream': True,
    }
    if system_prompt and not model.no_system_prompt:
        kwargs['instructions'] = system_prompt
    if model.thinking is not None:
        effort = model.thinking if isinstance(model.thinking, str) else 'high'
        kwargs['reasoning'] = {'effort': effort, 'summary': 'detailed'}
    if model.search:
        kwargs['tools'] = [{'type': 'web_search', 'search_context_size': 'medium'}]

    stream = await client.responses.create(**kwargs)
    search_queries: list[str] = []
    async for event in stream:
        logger.debug(f'Received event ({chat_id=}, {msg_id=}): {event.type}')
        if event.type == 'response.web_search_call.searching':
            yield StatusChange(status='Searching...')
        elif event.type == 'response.output_item.done' and event.item.type == 'web_search_call':
            query = getattr(event.item.action, 'query', None) if hasattr(event.item, 'action') else None
            if query:
                search_queries.append(query)
            yield StatusChange(status='Generating...')
        elif event.type in ('response.reasoning_summary_text.delta', 'response.reasoning_text.delta'):
            yield ThinkingDelta(text=event.delta)
        elif event.type == 'response.output_text.delta':
            yield ResponseDelta(text=event.delta)
        elif event.type == 'response.completed':
            resp = event.response
            if resp.status == 'incomplete' and resp.incomplete_details:
                reason = resp.incomplete_details.reason
                if reason == 'max_output_tokens':
                    yield ResponseDelta(text='\n\n[!] Error: Output truncated due to limit')
                else:
                    yield ResponseDelta(text=f'\n\n[!] Error: incomplete reason="{reason}"')
            meta = StreamMeta()
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
                blob = load_photo(part.hash)
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
        logger.debug(f'Received chunk ({chat_id=}, {msg_id=}): {chunk}')
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
) -> AsyncIterator[StreamEvent]:
    converted = convert_history_anthropic(chat_history)
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
            logger.debug(f'Received event ({chat_id=}, {msg_id=}): {event.type}')
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
) -> AsyncIterator[StreamEvent]:
    """Dispatch to the appropriate completion backend based on model.api_type."""
    assert len(chat_history) % 2 == 1
    log_history = [[{'type': p.type_, 'text': p.text} for p in msg] for msg in chat_history]
    logger.info(f'Starting completion ({model.api_type}) for {chat_id=}, {msg_id=}: {log_history}')

    if model.api_type == 'openai':
        backend = completion_openai(client, chat_history, model, system_prompt, chat_id, msg_id)
    elif model.api_type == 'openai_legacy':
        backend = completion_openai_legacy(client, chat_history, model, system_prompt, chat_id, msg_id)
    elif model.api_type == 'anthropic':
        backend = completion_anthropic(client, chat_history, model, system_prompt, chat_id, msg_id)
    else:
        raise ValueError(f'Unknown api_type: {model.api_type}')

    async for delta in backend:
        yield delta
