from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import assert_never

from kvstat.events import (
    AllBlocksCleared,
    BlockHash,
    BlockRemoved,
    BlockStored,
    Event,
    EventBatch,
)
from kvstat.state.crosscheck import Breach, EngineCounts, check

# Identity of one block. vLLM keys its cache map by hash and group. The tier joins the key so
# that a block offloaded to another tier does not count against the pool it left.
BlockKey = tuple[int | None, str | None, BlockHash]

# Lifetimes kept for the percentiles. An operator reads recent evictions. On the reference
# capture the whole cache turns over about every eight seconds.
LIFETIME_SAMPLES = 10_000


class Health(Enum):
    """How far kvstat's block table agrees with the engine.

    A table starts WARMING. kvstat attaches to a cache that is already full, and it learns a
    block's hash only when the engine stores that block again. The table reaches OK at the
    first scrape that agrees with it. It falls to DEGRADED when a scrape contradicts it.
    """

    WARMING = "warming"
    OK = "ok"
    DEGRADED = "degraded"


@dataclass(slots=True)
class _Counters:
    """Totals that survive a rebuild. They describe the event stream, not the table."""

    stored: int = 0
    evicted: int = 0
    restored: int = 0
    unknown_removals: int = 0
    orphan_stores: int = 0
    clears: int = 0
    rebuilds: int = 0
    breaches: int = 0


@dataclass(frozen=True, slots=True)
class CacheSnapshot:
    """kvstat's view of the KV cache at one moment, with the counters that say how far to trust it.

    Attributes:
        cached: Blocks in the prefix cache, in the counted tier. Idle blocks count too.
        capacity: Blocks the tier has room for. None when the engine did not report it.
        free: capacity minus cached. None without a capacity.
        stored: Blocks written since kvstat started, rebuilds included.
        evicted: Blocks the engine evicted since kvstat started.
        restored: Store events for a block that was already cached. vLLM emits one for a reused
            block only when the request asks for it, so this counts the prefix hits kvstat
            could see.
        unknown_removals: Evictions of a block that was not in the table. Expected after a
            resync.
        orphan_stores: Stores of a block whose parent was not in the table. Expected after a
            resync. At any other time kvstat missed the parent's store.
        clears: Times the engine reset its whole cache.
        rebuilds: Times kvstat dropped the table and started again, after a resync or a breach.
        breaches: Scrapes that contradicted the table.
        last_breach: The detail line of the most recent breach. None before the first.
        elapsed: Engine seconds since the last rebuild.
        stored_per_s: Blocks written per second over that window.
        evicted_per_s: Blocks evicted per second over that window.
        lifetime_p50: Median seconds a block survived between its store and its eviction.
        lifetime_p95: The same at the 95th percentile. Many blocks evicted young is the
            signature of thrash.
        epoch: Collector epoch the table was built under.
        health: How far the table and the engine agree right now. See Health.
    """

    cached: int
    capacity: int | None
    free: int | None
    stored: int
    evicted: int
    restored: int
    unknown_removals: int
    orphan_stores: int
    clears: int
    rebuilds: int
    breaches: int
    last_breach: str | None
    elapsed: float
    stored_per_s: float
    evicted_per_s: float
    lifetime_p50: float | None
    lifetime_p95: float | None
    epoch: int
    health: Health


def _percentile(ordered: list[float], q: float) -> float | None:
    """Nearest-rank percentile of an already sorted list."""
    if not ordered:
        return None
    return ordered[max(1, math.ceil(q * len(ordered))) - 1]


class CacheState:
    """Rebuilds the block table of the KV cache from the event stream, and cross-checks it.

    The cached count is not the engine's in-use gauge, and it never becomes one. vLLM emits no
    event when a request releases a block, so the two numbers count different populations.
    kvstat reports both and derives neither from the other.

    Three invariants hold over any valid event sequence. Cached blocks never exceed capacity.
    An evicted block leaves no entry behind. A stored block's parent is absent or already
    cached. A store with a missing parent is counted, not raised.

    Every method runs on the caller's thread.

    Args:
        capacity: Blocks the pool has room for, from the engine's cache config. Without it there
            is no free count and no capacity check.
        tier: The storage tier that capacity describes, named as the engine names it on events.
            Blocks in other tiers stay in the table but are left out of the cached count, so an
            offloaded block never counts against a pool it is not in. None counts every tier.
    """

    def __init__(self, capacity: int | None = None, tier: str | None = None) -> None:
        self.capacity = capacity
        self.tier = tier
        self._blocks: dict[BlockKey, float] = {}
        self._counters = _Counters()
        self._lifetimes: deque[float] = deque(maxlen=LIFETIME_SAMPLES)
        self._epoch = 0
        self._first_ts: float | None = None
        self._last_ts: float | None = None
        self._health = Health.WARMING
        self._recovering = True
        self._last_breach: str | None = None

    def update(self, batch: EventBatch) -> None:
        """Add one batch of events to the table.

        A batch with a newer epoch means the collector gave up on a gap that vLLM could not
        replay. The table is missing events, so it is dropped before this batch lands.
        """
        if batch.epoch != self._epoch:
            self._epoch = batch.epoch
            self._rebuild()
        if self._first_ts is None:
            self._first_ts = batch.ts
        self._last_ts = batch.ts
        for event in batch.events:
            self._add_event(event, batch.ts)

    def cross_check(self, counts: EngineCounts) -> Breach | None:
        """Compare the table with one metrics scrape, and rebuild it when the engine disagrees."""
        breach = check(self.snapshot(), counts)
        if breach is None:
            self._recovering = False
            self._health = Health.OK
            return None
        if breach.kind == "undercount" and self._recovering:
            # A table still refilling is expected to be behind. It stays WARMING, not drift.
            return None
        self._counters.breaches += 1
        self._last_breach = breach.detail
        self._rebuild()
        self._health = Health.DEGRADED
        return breach

    def snapshot(self) -> CacheSnapshot:
        """Return the current numbers as a frozen value."""
        counters = self._counters
        cached = sum(1 for key in self._blocks if self.tier in (None, key[1]))
        elapsed = self._elapsed()
        lifetimes = sorted(self._lifetimes)
        return CacheSnapshot(
            cached=cached,
            capacity=self.capacity,
            free=None if self.capacity is None else self.capacity - cached,
            stored=counters.stored,
            evicted=counters.evicted,
            restored=counters.restored,
            unknown_removals=counters.unknown_removals,
            orphan_stores=counters.orphan_stores,
            clears=counters.clears,
            rebuilds=counters.rebuilds,
            breaches=counters.breaches,
            last_breach=self._last_breach,
            elapsed=elapsed,
            stored_per_s=counters.stored / elapsed if elapsed else 0.0,
            evicted_per_s=counters.evicted / elapsed if elapsed else 0.0,
            lifetime_p50=_percentile(lifetimes, 0.50),
            lifetime_p95=_percentile(lifetimes, 0.95),
            epoch=self._epoch,
            health=self._health,
        )

    def _add_event(self, event: Event, ts: float) -> None:
        """Add one event to the block table."""
        match event:
            case BlockStored():
                self._store(event, ts)
            case BlockRemoved():
                self._remove(event, ts)
            case AllBlocksCleared():
                self._counters.clears += 1
                self._blocks.clear()
            case _:
                assert_never(event)

    def _store(self, event: BlockStored, ts: float) -> None:
        """Record a run of blocks, each one continuing the block before it."""
        group, medium = event.group_idx, event.medium
        parent = event.parent_block_hash
        for block_hash in event.block_hashes:
            key = (group, medium, block_hash)
            self._counters.stored += 1
            if key in self._blocks:
                self._counters.restored += 1
            # Only a first sighting can be an orphan. A block already in the table had its parent
            # judged on arrival.
            elif parent is not None and (group, medium, parent) not in self._blocks:
                self._counters.orphan_stores += 1
            self._blocks[key] = ts
            parent = block_hash

    def _remove(self, event: BlockRemoved, ts: float) -> None:
        """Drop evicted blocks and record how long each of them survived."""
        for block_hash in event.block_hashes:
            self._counters.evicted += 1
            stored_at = self._blocks.pop((event.group_idx, event.medium, block_hash), None)
            if stored_at is None:
                self._counters.unknown_removals += 1
            else:
                self._lifetimes.append(ts - stored_at)

    def _rebuild(self) -> None:
        """Drop the table and start again. The totals and the measured lifetimes stay.

        A rebuilt table is as far behind the engine as a cold start, so it goes back to WARMING.
        When a breach caused the rebuild, the caller sets DEGRADED afterwards.
        """
        self._blocks.clear()
        self._counters.rebuilds += 1
        self._first_ts = self._last_ts
        self._recovering = True
        self._health = Health.WARMING

    def _elapsed(self) -> float:
        """Return the engine seconds since the last rebuild."""
        if self._first_ts is None or self._last_ts is None:
            return 0.0
        return self._last_ts - self._first_ts
