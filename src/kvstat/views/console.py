from __future__ import annotations

import textwrap
from typing import Any

from kvstat.state.cache import CacheSnapshot
from kvstat.state.crosscheck import EngineWindow
from kvstat.state.hits import HitSpread, SplitPrefix


def format_summary(
    header: dict[str, Any],
    snap: CacheSnapshot,
    block_size: int | None,
    window: EngineWindow,
    hits: HitSpread | None = None,
    split: SplitPrefix | None = None,
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
        f"prefix hits      {_hits(hits)}",
        f"kvstat           {snap.health.value}  ·  {snap.rebuilds} rebuild(s)  ·  "
        f"{snap.orphan_stores:,} orphan stores  ·  {snap.elapsed:.1f}s",
        f"drift            {describe_drift(snap)}  ({window.polls} polls)",
    ]
    if split is not None:
        lines += ["", "findings", *_split_finding(split)]
    return "\n".join(lines)


def _hits(hits: HitSpread | None) -> str:
    """The prefix hits, how they spread over first blocks, and how many blocks each reused."""
    if hits is None:
        # No hits means the workload never asked vLLM to report them, not that nothing was reused.
        return "none seen   (needs kv_cache_report_mode: full on the workload)"
    if hits.first_blocks == 1:
        spread = "all from 1 first block"
    else:
        share = hits.top / hits.hits * 100
        spread = f"from {hits.first_blocks:,} first blocks, top {hits.top:,} ({share:.0f}%)"
    blocks = f"{hits.fewest_blocks}-{hits.most_blocks}"
    if hits.fewest_blocks == hits.most_blocks:
        blocks = f"{hits.most_blocks}"
    noun = "block" if blocks == "1" else "blocks"
    return f"{hits.hits:,}  ·  {spread}  ·  {blocks} {noun} per hit"


def _split_finding(split: SplitPrefix) -> list[str]:
    """Name a shared prefix that the block boundary cut off, and what to change."""
    found = (
        f"No prompt shares its first block. {split.first_blocks:,} first blocks start with the "
        f"same {split.shared_tokens:,} tokens, but a block holds {split.block_size:,}, and only "
        "whole blocks are reused."
    )
    knob = (
        "-> make the shared prefix at least one block long, or revisit what sets the block "
        "size (KV dtype, speculative decoding on hybrid models)."
    )
    return [
        *textwrap.wrap(found, 78, initial_indent="  ", subsequent_indent="  "),
        *textwrap.wrap(knob, 78, initial_indent="  ", subsequent_indent="     "),
    ]


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


def describe_drift(snap: CacheSnapshot) -> str:
    """Describes where the block table disagreed with the engine's metrics, or ok."""
    if not snap.breaches:
        return "ok"
    return f"{snap.breaches} breach(es), last: {snap.last_breach}"
