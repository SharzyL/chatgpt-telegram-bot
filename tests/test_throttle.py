from itertools import pairwise

import pytest

from chatgpt_telegram_bot.utils import EditThrottle


def schedule(duration: float, tick: float = 0.05) -> list[float]:
    """Times at which a stream of `duration` seconds would flush, ticking every `tick`."""
    throttle = EditThrottle(3.0, 1.0, 15.0)
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
    assert EditThrottle(3.0, 1.0, 15.0).capacity == pytest.approx(10.0)


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
    throttle = EditThrottle(3.0, 1.0, 15.0)
    assert throttle.take(0.0)
    throttle.flushed(0.0)
    throttle.tokens = 0.0
    assert not throttle.take(1.0)  # bucket empty, burst spacing alone is not enough
    assert throttle.take(31.0)  # 30s idle refills the bucket to capacity
    throttle.flushed(31.0)
    assert throttle.take(32.0)  # and burst speed is available again


def test_degenerate_configuration_still_flushes():
    # burst_interval >= interval leaves no room for a burst; the bucket must not deadlock
    throttle = EditThrottle(3.0, 3.0, 15.0)
    assert throttle.capacity == pytest.approx(1.0)
    assert throttle.take(0.0)
