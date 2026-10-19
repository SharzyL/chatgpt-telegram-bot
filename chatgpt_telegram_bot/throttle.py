"""Pacing outgoing Telegram traffic."""


class Throttle:
    """Token bucket over outgoing Telegram traffic — streamed edits, or a run of sends.

    A flow that keeps going settles at one message per *interval*, which is what Telegram's
    own rate limit cares about. A short one may go every *burst_interval* instead, paid for
    out of a bucket that starts full and refills at ``1 / interval``. The capacity is
    derived from *burst_window*: it is exactly the number of extra messages a burst of that
    length spends over what refilling covers, so a flow shorter than the window never leaves
    burst speed and a longer one decays to *interval* on its own.

    Callers come in two shapes. A stream of edits only ever shows the latest text, so it
    uses `take()` and drops a tick that arrives too early. A run of sends has to deliver
    every message, so it uses `reserve()` and waits its turn instead.

    Time is passed in rather than read, so the schedule is testable.
    """

    def __init__(self, interval: float, burst_interval: float, burst_window: float) -> None:
        self.interval: float = interval
        self.burst_interval: float = burst_interval
        # spending one token every burst_interval while refilling one every interval
        self.capacity: float = max(1.0, burst_window * (1 / burst_interval - 1 / interval))
        self.tokens: float = self.capacity
        self.refilled: float | None = None
        self.last_flush: float | None = None

    def _refill(self, now: float) -> None:
        """Credit the tokens accrued up to *now*, which reserve() may put in the future."""
        if self.refilled is None:
            self.refilled = now
        elapsed = max(0.0, now - self.refilled)
        self.tokens = min(self.capacity, self.tokens + elapsed / self.interval)
        self.refilled = max(self.refilled, now)

    def take(self, now: float) -> bool:
        """Whether a message may go out now; consumes a token when it may."""
        self._refill(now)
        if self.tokens < 1.0 or (self.last_flush is not None and now - self.last_flush < self.burst_interval):
            return False
        self.tokens -= 1.0
        return True

    def reserve(self, now: float) -> float:
        """How long to wait before the next message may go out, taking its token now.

        Where `take()` drops a message that comes too early, this one queues it: the caller
        sleeps for the returned delay and then sends.
        """
        self._refill(now)
        wait = (1.0 - self.tokens) * self.interval if self.tokens < 1.0 else 0.0
        if self.last_flush is not None:
            wait = max(wait, self.last_flush + self.burst_interval - now)
        wait = max(wait, 0.0)
        at = now + wait
        self._refill(at)
        self.tokens -= 1.0
        self.last_flush = at
        return wait

    def flushed(self, now: float) -> None:
        """Record that a message landed; spacing is measured from here, not from take()."""
        self.last_flush = now
