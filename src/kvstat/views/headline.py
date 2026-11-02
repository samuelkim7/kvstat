from __future__ import annotations

from kvstat.state.cache import CacheSnapshot
from kvstat.state.timeline import TimelinePoint


def render_headline(snap: CacheSnapshot, point: TimelinePoint | None) -> str:
    """Renders the headline: occupancy, prefix hit rate and scheduler pressure in one line."""
    if snap.capacity is None:
        return "waiting for the engine's pool size"
    if not snap.stored:
        return "waiting for the first block"
    parts = [f"{snap.cached / snap.capacity * 100:.0f}% full"]
    if point is not None:
        parts.append(f"{point.used_fraction * 100:.0f}% in use")
        if point.hit_rate is not None:
            parts.append(f"{point.hit_rate * 100:.0f}% prefix hits")
    pressure = _describe_pressure(point)
    if pressure:
        parts.append(pressure)
    return " · ".join(parts)


def _describe_pressure(point: TimelinePoint | None) -> str | None:
    """Describes preemptions or waiting requests in this scrape, or None when there are none."""
    if point is None:
        return None
    if point.preemptions:
        return f"{point.preemptions:,} preempted, {point.waiting_requests:,} waiting"
    if point.waiting_requests:
        return f"{point.waiting_requests:,} waiting for room"
    return None
