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


def chat_throttle(chat_id: int, private_interval: float, group_interval: float) -> Throttle:
    """The steady pace of traffic to one chat: one message per interval, a group's or a private chat's.

    No burst: in a group it would only spend the minute's calls early (see `ChatLimit`) and
    leave the reply stalled later, and a private chat needs none. Telethon marks a group's id
    negative.
    """
    interval = group_interval if chat_id < 0 else private_interval
    return Throttle(interval, interval, 0.0)


class ChatLimit:
    """The calls Telegram lets a bot make to one chat: at most *count* in any *window* seconds, for each limit.

    Every send, edit and delete to the chat counts, whichever reply it is for, and a call is
    recorded when it is made. Measured on a group, each stopped at the first call over:
    20 in a rolling minute (the 21st was refused and told to wait until the oldest of them was
    a minute old), and 5 in a few seconds (the 6th refused within 4s, waiting until about 7s
    after the first). A private chat refused nothing at 4 calls a second.
    """

    def __init__(self, limits: list[tuple[int, float]]) -> None:
        self.limits: list[tuple[int, float]] = limits
        self.calls: list[float] = []

    def wait(self, now: float, calls: int = 1) -> float:
        """How long until *calls* more fit within every limit; 0 when they fit now."""
        wait = 0.0
        for count, window in self.limits:
            if calls > count:
                raise ValueError(f'{calls} calls can never fit in {count} per {window}s')
            # the oldest call that has to leave the window before there is room for these
            index = len(self.calls) - (count - calls) - 1
            if index >= 0:
                wait = max(wait, self.calls[index] + window - now)
        return max(wait, 0.0)

    def record(self, now: float) -> None:
        if not self.limits:
            return
        self.calls.append(now)
        # only the longest window's calls can matter again
        horizon = now - max(window for _, window in self.limits)
        while self.calls and self.calls[0] <= horizon:
            _ = self.calls.pop(0)
