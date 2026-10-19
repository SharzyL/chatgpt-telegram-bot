"""Which generation failures are retried, and after how long."""

import anthropic
import httpx
import httpx2
import openai

from chatgpt_telegram_bot.bot import RATE_LIMIT_MAX_WAIT, RATE_LIMIT_WAIT, retry_delay

INTERVAL = 3.0
OPENAI_REQUEST = httpx.Request('POST', 'https://openai.test/v1/responses')
ANTHROPIC_REQUEST = httpx2.Request('POST', 'https://anthropic.test/v1/messages')


def _openai_status(cls: type[openai.APIStatusError], status: int, headers: dict[str, str] | None = None):
    return cls('failed', response=httpx.Response(status, headers=headers, request=OPENAI_REQUEST), body=None)


def _anthropic_status(cls: type[anthropic.APIStatusError], status: int, body: object = None):
    return cls('failed', response=httpx2.Response(status, request=ANTHROPIC_REQUEST), body=body)


def test_connection_failures_are_retried_for_both_sdks():
    assert retry_delay(openai.APIConnectionError(request=OPENAI_REQUEST), INTERVAL) == INTERVAL
    assert retry_delay(openai.APITimeoutError(request=OPENAI_REQUEST), INTERVAL) == INTERVAL
    assert retry_delay(anthropic.APIConnectionError(request=ANTHROPIC_REQUEST), INTERVAL) == INTERVAL


def test_a_stream_dropped_mid_reply_is_retried():
    # neither SDK wraps an error raised while the body is being read
    assert retry_delay(httpx.RemoteProtocolError('peer closed connection'), INTERVAL) == INTERVAL
    assert retry_delay(httpx2.ReadError('connection reset'), INTERVAL) == INTERVAL


def test_server_errors_are_retried():
    assert retry_delay(_openai_status(openai.InternalServerError, 502), INTERVAL) == INTERVAL
    assert retry_delay(_anthropic_status(anthropic.InternalServerError, 529), INTERVAL) == INTERVAL


def test_a_rate_limit_waits_as_long_as_it_is_told_within_a_cap():
    assert retry_delay(_openai_status(openai.RateLimitError, 429, {'retry-after': '7'}), INTERVAL) == 7.0
    assert retry_delay(_openai_status(openai.RateLimitError, 429, {'retry-after': '600'}), INTERVAL) == (
        RATE_LIMIT_MAX_WAIT
    )
    assert retry_delay(_anthropic_status(anthropic.RateLimitError, 429), INTERVAL) == RATE_LIMIT_WAIT


def test_overload_reported_mid_stream_is_retried():
    body = {'type': 'error', 'error': {'type': 'overloaded_error', 'message': 'Overloaded'}}
    assert retry_delay(_anthropic_status(anthropic.APIStatusError, 200, body), INTERVAL) == INTERVAL


def test_errors_a_retry_would_repeat_are_not_retried():
    assert retry_delay(_openai_status(openai.BadRequestError, 400), INTERVAL) is None
    assert retry_delay(_openai_status(openai.AuthenticationError, 401), INTERVAL) is None
    assert retry_delay(_anthropic_status(anthropic.NotFoundError, 404), INTERVAL) is None
    body = {'type': 'error', 'error': {'type': 'invalid_request_error', 'message': 'bad'}}
    assert retry_delay(_anthropic_status(anthropic.APIStatusError, 200, body), INTERVAL) is None
    assert retry_delay(ValueError('No API key for endpoint'), INTERVAL) is None
