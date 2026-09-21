from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from kvstat.state.cache import CacheSnapshot

# Blocks of tolerance on the undercount inequality. A scrape and the event position are never
# the same instant. On the reference capture the worst disagreement was a fifth of a block.
DEFAULT_SLACK = 4


@dataclass(frozen=True, slots=True)
class EngineCounts:
    """Two readings from one metrics scrape. The rebuilt table is compared against them.

    Attributes:
        time: When the engine was scraped, in seconds since the epoch.
        used_fraction: vllm:kv_cache_usage_perc. The fraction of the pool that running requests
            hold right now.
        running_requests: vllm:num_requests_running. The requests the engine is serving right
            now.
    """

    time: float
    used_fraction: float
    running_requests: int


@dataclass(frozen=True, slots=True)
class EngineWindow:
    """The range of the engine's readings over a run of scrapes.

    The cross-check reads one scrape at a time and forgets it. A report needs the whole window,
    and the used fraction is the one number kvstat cannot derive for itself.

    Attributes:
        polls: Scrapes compared so far.
        used_fraction_low: The lowest used fraction seen.
        used_fraction_high: The highest used fraction seen.
    """

    polls: int = 0
    used_fraction_low: float = 1.0
    used_fraction_high: float = 0.0

    def then(self, counts: EngineCounts) -> EngineWindow:
        """Return the window widened by one more scrape."""
        return EngineWindow(
            polls=self.polls + 1,
            used_fraction_low=min(self.used_fraction_low, counts.used_fraction),
            used_fraction_high=max(self.used_fraction_high, counts.used_fraction),
        )


@dataclass(frozen=True, slots=True)
class Breach:
    """One scrape that contradicts the rebuilt table.

    Attributes:
        kind: over_capacity means the table holds more blocks than the pool has. undercount
            means the table holds fewer blocks than the running requests are using.
        detail: Both numbers on one line, for the log and the status bar.
    """

    kind: Literal["over_capacity", "undercount"]
    detail: str


def check(
    snapshot: CacheSnapshot, counts: EngineCounts, *, slack: int = DEFAULT_SLACK
) -> Breach | None:
    """Compare kvstat's block table with one metrics scrape. Return the first breach, or None.

    Two inequalities hold when the table is right.

    Over capacity: cached <= capacity. The table cannot hold more blocks than the pool has.
    A failure means kvstat counted a store twice or missed an eviction.

    Undercount: blocks_in_use <= cached + running_requests + slack. blocks_in_use comes from
    the engine's gauge. Every block in use is either cached, so the table has it, or the tail
    block a request is still filling, which has no hash yet. Each request has at most one tail
    block. A failure means kvstat missed stores.

    A table still refilling after a rebuild fails the undercount check. This function reports
    it anyway. CacheState decides whether to tolerate it.

    Args:
        snapshot: kvstat's block table, as of the last event.
        counts: The engine's numbers from one scrape.
        slack: Blocks of tolerance on the undercount check, for the time between the scrape
            and the last event.
    """
    if snapshot.capacity is None:
        return None
    if snapshot.cached > snapshot.capacity:
        return Breach(
            kind="over_capacity",
            detail=f"{snapshot.cached} blocks cached, pool holds {snapshot.capacity}",
        )
    blocks_in_use = counts.used_fraction * snapshot.capacity
    if snapshot.cached + counts.running_requests + slack < blocks_in_use:
        return Breach(
            kind="undercount",
            detail=(
                f"{snapshot.cached} blocks cached and {counts.running_requests} requests "
                f"running, engine reports {blocks_in_use:.0f} blocks in use"
            ),
        )
    return None
