from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

BlockHash = bytes | int
"""Engine-assigned block identity. vLLM emits either width; kvtop never inspects the value."""


@dataclass(frozen=True, slots=True)
class BlockStored:
    """Blocks the engine wrote into its KV cache.

    Under report-mode full traffic the engine also emits this for blocks it reused, so a hash
    that is already resident may arrive again; the state layer reads that as a prefix hit.
    """

    block_hashes: tuple[BlockHash, ...]
    parent_block_hash: BlockHash | None
    token_ids: tuple[int, ...]
    block_size: int
    medium: str | None
    group_idx: int | None
    kv_cache_spec_kind: str | None
    lora_name: str | None
    session_id: str | None


@dataclass(frozen=True, slots=True)
class BlockRemoved:
    """Blocks evicted from the KV cache."""

    block_hashes: tuple[BlockHash, ...]
    medium: str | None
    group_idx: int | None


@dataclass(frozen=True, slots=True)
class AllBlocksCleared:
    """The engine reset its cache; every block is gone."""


Event = BlockStored | BlockRemoved | AllBlocksCleared


@dataclass(frozen=True, slots=True)
class EventBatch:
    """One publisher batch, events in engine order.

    ``seq`` comes from the transport frame, not the payload; a gap between consecutive batches
    means events were lost. ``skipped`` names the event types this kvtop does not know, in the
    order they appeared, so a newer engine degrades to a count instead of a lost batch.
    """

    seq: int
    ts: float
    data_parallel_rank: int | None
    events: tuple[Event, ...]
    skipped: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class IngestStats:
    """What the collector has seen so far; the status bar's trust numbers start here."""

    batches: int
    last_seq: int | None
    skipped_event_types: Mapping[str, int]


@runtime_checkable
class EventSource(Protocol):
    """Anything that yields batches in order: a live socket, a capture file, a test fake."""

    def __iter__(self) -> Iterator[EventBatch]: ...

    def close(self) -> None: ...
