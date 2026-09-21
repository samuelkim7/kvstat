from __future__ import annotations

from typing import Any

from kvstat.state.cache import CacheSnapshot
from kvstat.state.crosscheck import EngineWindow


def format_summary(
    header: dict[str, Any], snap: CacheSnapshot, block_size: int | None, window: EngineWindow
) -> str:
    """Lay the state summary out for the console.

    The in-use line comes from the engine's gauge. The cached line comes from the events. They
    count different populations, so they print apart and neither is derived from the other.
    """
    capacity = snap.capacity
    pool = f"{capacity:,} blocks" if capacity else "? blocks"
    if block_size:
        pool += f" x {block_size} tokens"
    lines = [
        f"{header.get('model_name') or 'unknown model'}  ·  "
        f"vLLM {header.get('vllm_version') or '?'}  ·  {pool}",
        "",
        f"engine in use    {_usage(window, capacity)}",
        f"prefix cached    {_cached(snap)}",
        f"turnover         {_turnover(snap)}",
        f"lifetime         {_lifetime(snap)}",
        f"prefix re-stores {snap.restored:,}{'' if snap.restored else _NO_RESTORES}",
        f"kvstat           {snap.health.value}  ·  {snap.rebuilds} rebuild(s)  ·  "
        f"{snap.orphan_stores:,} orphan stores  ·  {snap.elapsed:.1f}s",
        f"drift            {_drift(snap)}  ({window.polls} polls)",
    ]
    return "\n".join(lines)


# Zero re-stores means the workload never asked for them, not that nothing was reused.
_NO_RESTORES = "   (needs kv_cache_report_mode: full on the workload)"


def _usage(window: EngineWindow, capacity: int | None) -> str:
    """The range of the engine's in-use gauge over the window."""
    if not window.polls:
        return "not scraped"
    span = f"{window.used_fraction_low * 100:.1f}% - {window.used_fraction_high * 100:.1f}%"
    if capacity is None:
        return span
    return f"{span}   peak {round(window.used_fraction_high * capacity):,} blocks"


def _cached(snap: CacheSnapshot) -> str:
    """The blocks in the prefix cache. Idle blocks count too."""
    if snap.capacity is None:
        return f"{snap.cached:,} blocks"
    share = snap.cached / snap.capacity * 100
    return f"{snap.cached:,} / {snap.capacity:,} ({share:.1f}%)   free {snap.free:,}"


def _turnover(snap: CacheSnapshot) -> str:
    """The time the engine needs to write one pool's worth of blocks, and the rates behind it."""
    rates = f"{snap.stored_per_s:,.0f} blocks/s stored, {snap.evicted_per_s:,.0f} evicted"
    if not snap.capacity or not snap.stored_per_s:
        return rates
    return f"every {snap.capacity / snap.stored_per_s:.1f}s   {rates}"


def _lifetime(snap: CacheSnapshot) -> str:
    """How long a block survives between its store and its eviction."""
    if snap.lifetime_p50 is None or snap.lifetime_p95 is None:
        return "no evictions seen"
    return f"p50 {snap.lifetime_p50:.1f}s   p95 {snap.lifetime_p95:.1f}s"


def _drift(snap: CacheSnapshot) -> str:
    """Whether any scrape contradicted the table."""
    if not snap.breaches:
        return "ok"
    return f"{snap.breaches} breach(es), last: {snap.last_breach}"
