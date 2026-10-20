from itertools import pairwise

import pytest

from chatgpt_telegram_bot.throttle import ChatLimit, Throttle, chat_throttle


def schedule(duration: float, tick: float = 0.05) -> list[float]:
    """Times at which a stream of `duration` seconds would flush, ticking every `tick`."""
    throttle = Throttle(3.0, 1.0, 15.0)
    flushes: list[float] = []
    now = 0.0
    while now <= duration:
        if throttle.take(now):
            flushes.append(now)
            throttle.flushed(now)
        now += tick
    return flushes


def gaps(flushes: list[float]) -> list[float]:
    return [round(b - a, 2) for a, b in pairwise(flushes)]


def test_burst_capacity_matches_the_window():
    # 15s of 1s edits, refilled at one per 3s, costs 10 tokens over what refill covers
    assert Throttle(3.0, 1.0, 15.0).capacity == pytest.approx(10.0)


def test_first_update_flushes_immediately():
    assert schedule(0.0)[0] == 0.0


def test_short_flow_runs_at_the_burst_interval():
    # a reply that finishes inside the window never leaves 1s cadence; gaps land on the
    # first tick at or after 1s, so allow one tick of slack
    measured = gaps(schedule(14.0))
    assert len(measured) == 13
    assert all(1.0 <= g <= 1.05 for g in measured), measured


def test_long_flow_decays_to_the_steady_interval():
    tail = gaps(schedule(40.0))[-5:]
    assert all(3.0 <= g <= 3.05 for g in tail), tail


def test_steady_rate_is_bounded_by_the_interval():
    # over a long stream the burst amortises away: never more than duration/interval + capacity
    flushes = schedule(120.0)
    assert len(flushes) <= 120.0 / 3.0 + 10 + 1


def test_burst_recovers_while_idle():
    throttle = Throttle(3.0, 1.0, 15.0)
    assert throttle.take(0.0)
    throttle.flushed(0.0)
    throttle.tokens = 0.0
    assert not throttle.take(1.0)  # bucket empty, burst spacing alone is not enough
    assert throttle.take(31.0)  # 30s idle refills the bucket to capacity
    throttle.flushed(31.0)
    assert throttle.take(32.0)  # and burst speed is available again


def test_degenerate_configuration_still_flushes():
    # burst_interval >= interval leaves no room for a burst; the bucket must not deadlock
    throttle = Throttle(3.0, 3.0, 15.0)
    assert throttle.capacity == pytest.approx(1.0)
    assert throttle.take(0.0)


def reserved(count: int, interval: float = 3.0, burst: float = 1.0, window: float = 10.0) -> list[float]:
    """The times at which `count` queued sends would go out."""
    throttle = Throttle(interval, burst, window)
    now = 0.0
    times: list[float] = []
    for _ in range(count):
        now += throttle.reserve(now)
        times.append(round(now, 2))
    return times


def test_a_short_run_goes_out_at_burst_speed():
    # six sends fit in the bucket, so they pace at 1s rather than 3s
    assert reserved(6) == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


def test_a_long_run_settles_to_the_steady_interval():
    times = reserved(20)
    assert gaps(times)[-5:] == [3.0] * 5


def test_a_queued_run_never_outpaces_the_bucket():
    # the bucket's guarantee: n sends take at least (n - capacity) * interval
    times = reserved(40)
    assert times[-1] >= (40 - Throttle(3.0, 1.0, 10.0).capacity - 1) * 3.0


def test_reserve_does_not_lose_tokens_to_a_clock_that_moved_on():
    # reserve() credits refill up to a time in the future; a later call must not see
    # negative elapsed time and dock the bucket for it
    throttle = Throttle(3.0, 1.0, 10.0)
    _ = throttle.reserve(0.0)
    before = throttle.tokens
    _ = throttle.reserve(0.5)  # earlier than the reservation it just made
    assert throttle.tokens <= before
    assert throttle.reserve(0.5) >= 0.0


def test_waiting_longer_than_the_run_needs_restores_the_burst():
    throttle = Throttle(3.0, 1.0, 10.0)
    for _ in range(10):
        _ = throttle.reserve(0.0)
    assert throttle.reserve(0.0) > 0.0, 'the bucket should be empty by now'
    assert throttle.reserve(120.0) == 0.0, 'two minutes idle should refill it'


def _flushes(throttle: Throttle, seconds: float, tick: float = 0.1) -> list[float]:
    """When a stream ticking every *tick* seconds gets to flush, each flush landing at once."""
    times: list[float] = []
    for i in range(round(seconds / tick)):
        now = i * tick
        if throttle.take(now):
            throttle.flushed(now)
            times.append(now)
    return times


def test_a_group_and_a_private_chat_each_keep_a_steady_pace():
    # to within one 0.1s tick
    assert _flushes(chat_throttle(-100, 1.0, 3.5), 15)[:4] == pytest.approx([0.0, 3.5, 7.0, 10.5], abs=0.11)
    assert _flushes(chat_throttle(42, 1.0, 3.5), 4) == pytest.approx([0.0, 1.0, 2.0, 3.0], abs=0.11)


def _limit() -> ChatLimit:
    return ChatLimit([(18, 60.0), (4, 8.0)])


def test_a_short_run_of_calls_waits_for_the_oldest_to_leave_the_window():
    limit = _limit()
    for t in (0.0, 0.1, 0.2, 0.3):
        assert limit.wait(t) == 0
        limit.record(t)
    # Telegram refused the 6th of a fast run; the 5th is held back until the first is 8s old
    assert limit.wait(0.4) == pytest.approx(7.6)
    assert limit.wait(8.0) == 0


def test_a_minute_holds_eighteen_calls_however_spaced():
    limit = _limit()
    t = 0.0
    for _ in range(18):
        t += limit.wait(t)
        limit.record(t)
        t += 2.0
    # the 21st call within a minute was refused; this stops at the 19th, until the first is 60s old
    assert limit.wait(t) == pytest.approx(60.0 - t)


def test_room_for_two_calls_is_room_for_both_within_every_limit():
    limit = _limit()
    for t in (0.0, 1.0, 2.0):
        limit.record(t)
    assert limit.wait(3.0, calls=1) == 0
    assert limit.wait(3.0, calls=2) == pytest.approx(5.0), 'the second would be the 5th within 8s'


def test_every_schedule_a_reply_can_make_fits_the_measured_group_limits():
    # whatever ticks a stream offers, calls paced through the limit never break Telegram's own
    limit = _limit()
    calls: list[float] = []
    t = 0.0
    while t < 600:
        t += limit.wait(t)
        limit.record(t)
        calls.append(t)
        t += 0.05
    assert max(sum(1 for c in calls if s <= c < s + 60) for s in calls) <= 20
    assert max(sum(1 for c in calls if s <= c < s + 7.2) for s in calls) <= 5


def test_no_limits_means_no_waiting():
    limit = ChatLimit([])
    for t in range(100):
        limit.record(float(t) / 100)
    assert limit.wait(1.0, calls=50) == 0
