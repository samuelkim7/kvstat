from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from kvstat.events import EngineCounts

# Points kept. At one scrape a second this is five minutes, wider than any terminal.
WINDOW = 300


@dataclass(frozen=True, slots=True)
class TimelinePoint:
    """One column of the pressure panel: the numbers for one scrape, one second of the run.

    The engine's gauges are copied as they were. The engine's counters and kvstat's eviction
    total are turned into the change since the previous point, so a column shows what happened
    in that second rather than since the engine started.

    Attributes:
        time: When the engine was scraped, in seconds since the epoch.
        used_fraction: The fraction of the pool that running requests hold.
        running_requests: The requests the engine is serving.
        waiting_requests: The requests queued for admission.
        preemptions: Requests preempted since the previous point. None when the engine does
            not report preemptions.
        evictions: Blocks evicted since the previous point, from the event stream.
        ttft_p50: Median seconds to first token among the requests that finished prefill
            since the previous point. None when none did, or the engine has no histogram.
        ttft_p95: The same at the 95th percentile.
        hit_rate: Prefix-cache hits over queries since the first scrape kvstat saw. None
            until a query lands, or when the engine does not report the counters.
    """

    time: float
    used_fraction: float
    running_requests: int
    waiting_requests: int
    preemptions: int | None
    evictions: int
    ttft_p50: float | None
    ttft_p95: float | None
    hit_rate: float | None


class Timeline:
    """Builds the pressure panel's points, one TimelinePoint per metrics scrape.

    Every scrape is added with add(), which computes that second's changes against the scrape
    before it and appends the point. The panel paints the points and computes nothing. The
    engine's counters reset when it restarts, so a change is clamped at zero.

    Args:
        window: Points kept. Older points fall off the front.
    """

    def __init__(self, window: int = WINDOW) -> None:
        self._points: deque[TimelinePoint] = deque(maxlen=window)
        self._previous: EngineCounts | None = None
        self._first: EngineCounts | None = None
        self._evicted_total = 0
        self.sources: set[str] = set()

    @property
    def points(self) -> tuple[TimelinePoint, ...]:
        """The points in the window, oldest first."""
        return tuple(self._points)

    @property
    def latest(self) -> TimelinePoint | None:
        """The most recent point, or None before the first scrape."""
        return self._points[-1] if self._points else None

    def add(self, counts: EngineCounts, evicted_total: int) -> TimelinePoint:
        """Add one scrape, paired with the event stream's eviction total at that moment."""
        previous = self._previous
        if self._first is None:
            self._first = counts
        self._note_sources(counts)
        p50, p95 = _ttft_percentiles(previous.ttft_buckets if previous else (), counts.ttft_buckets)
        point = TimelinePoint(
            time=counts.time,
            used_fraction=counts.used_fraction,
            running_requests=counts.running_requests,
            waiting_requests=counts.waiting_requests,
            preemptions=_delta(previous.preemptions if previous else None, counts.preemptions),
            evictions=max(0, evicted_total - self._evicted_total) if previous else 0,
            ttft_p50=p50,
            ttft_p95=p95,
            hit_rate=_hit_rate(self._first, counts),
        )
        self._points.append(point)
        self._previous = counts
        self._evicted_total = evicted_total
        return point

    def _note_sources(self, counts: EngineCounts) -> None:
        if counts.preemptions is not None:
            self.sources.add("preemptions")
        if counts.prefix_queries is not None:
            self.sources.add("prefix cache")
        if counts.ttft_buckets:
            self.sources.add("ttft")


def _delta(before: int | None, after: int | None) -> int | None:
    if after is None:
        return None
    if before is None:
        return 0
    return max(0, after - before)


def _hit_rate(first: EngineCounts, now: EngineCounts) -> float | None:
    if first.prefix_queries is None or first.prefix_hits is None:
        return None
    if now.prefix_queries is None or now.prefix_hits is None:
        return None
    queries = now.prefix_queries - first.prefix_queries
    if queries <= 0:
        return None
    return (now.prefix_hits - first.prefix_hits) / queries


def _ttft_percentiles(
    before: tuple[tuple[float, float], ...], after: tuple[tuple[float, float], ...]
) -> tuple[float | None, float | None]:
    """Percentiles of the requests counted between two cumulative histograms."""
    if not after:
        return None, None
    earlier = dict(before)
    deltas = [(bound, count - earlier.get(bound, 0.0)) for bound, count in after]
    total = deltas[-1][1]
    if total <= 0:
        return None, None
    return _bucket_at(deltas, 0.50 * total), _bucket_at(deltas, 0.95 * total)


def _bucket_at(deltas: list[tuple[float, float]], rank: float) -> float | None:
    """The upper bound of the bucket at the rank. The open bucket reports the last finite bound."""
    finite = [bound for bound, _ in deltas if math.isfinite(bound)]
    if not finite:
        return None
    for bound, cumulative in deltas:
        if cumulative >= rank:
            return bound if math.isfinite(bound) else finite[-1]
    return finite[-1]
