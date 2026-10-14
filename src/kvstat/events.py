from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

BlockHash = bytes | int
"""Identity of one KV cache block, assigned by the engine.

vLLM emits either width; kvstat never inspects the value.
"""


@dataclass(frozen=True, slots=True)
class BlockStored:
    """Event: the engine wrote these blocks into its KV cache and gave them hashes.

    When a request asks for kv_cache_report_mode=full, the engine also emits this for blocks it
    reused. That is a prefix hit.

    Attributes:
        block_hashes: Hashes of the blocks written, in sequence order.
        parent_block_hash: Hash of the block before block_hashes[0], or None when these are the
            first blocks of a sequence. Links the blocks into a prefix chain.
        token_ids: The prompt token ids these blocks hold. Empty in a redacted capture.
        block_size: Tokens per block.
        medium: Storage tier of the blocks, such as GPU or CPU. None when the engine does not
            report it.
        group_idx: KV cache group the blocks belong to, for models with more than one.
        kv_cache_spec_kind: Attention kind of that group, such as full or sliding window.
        lora_name: LoRA adapter the request used, if any.
        session_id: Session id the request carried, if the engine echoes it.
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
    """Event: the engine evicted these blocks from its KV cache.

    Attributes:
        block_hashes: Hashes of the blocks evicted.
        medium: Storage tier the blocks left. None when the engine does not report it.
        group_idx: KV cache group they belonged to, for models with more than one.
    """

    block_hashes: tuple[BlockHash, ...]
    medium: str | None
    group_idx: int | None


@dataclass(frozen=True, slots=True)
class AllBlocksCleared:
    """Event: the engine reset its KV cache and every block is gone."""


Event = BlockStored | BlockRemoved | AllBlocksCleared


@dataclass(frozen=True, slots=True)
class EventBatch:
    """One decoded publisher message: its sequence number, timestamp and events, in engine order.

    Attributes:
        seq: Sequence number from the ZMQ frame, not from the payload. Consecutive batches have
            consecutive numbers, so a jump means batches were lost.
        ts: Engine-side timestamp of the batch, seconds since the epoch.
        data_parallel_rank: Rank of the engine that produced the batch, or None for a single
            engine.
        events: The decoded events, in the order the engine emitted them.
        skipped: Event types this kvstat version does not know, in order of appearance. A newer
            engine costs a count, not a lost batch.
        epoch: Number of unrecoverable gaps the collector had seen when this batch arrived.
            State built under an earlier epoch is out of date.
    """

    seq: int
    ts: float
    data_parallel_rank: int | None
    events: tuple[Event, ...]
    skipped: tuple[str, ...] = ()
    epoch: int = 0


# One Prometheus sample from a metrics scrape: (metric name, labels, value).
Sample = tuple[str, dict[str, str], float]


@dataclass(frozen=True, slots=True)
class EngineCounts:
    """What the engine said about itself in one metrics scrape.

    Every value is the engine's own, cumulative where the engine keeps a counter. A source the
    scrape did not carry is None or empty, never zero.

    Attributes:
        time: When the engine was scraped, in seconds since the epoch.
        used_fraction: The fraction of the pool that running requests hold right now.
        running_requests: The requests the engine is serving right now.
        waiting_requests: The requests queued for admission right now.
        preemptions: Requests the scheduler has preempted since the engine started.
        prefix_queries: Prompt tokens looked up in the prefix cache since the engine started.
        prefix_hits: Of those, the tokens found cached.
        ttft_buckets: The time-to-first-token histogram as (upper bound, cumulative count)
            pairs, in bound order, the last bound infinite.
    """

    time: float
    used_fraction: float
    running_requests: int
    waiting_requests: int = 0
    preemptions: int | None = None
    prefix_queries: int | None = None
    prefix_hits: int | None = None
    ttft_buckets: tuple[tuple[float, float], ...] = ()


@dataclass(frozen=True, slots=True)
class IngestStats:
    """Snapshot of the collector's counters: received, recovered, resynced and dropped.

    The status bar shows these, so a reader knows how much to trust the screen.

    Attributes:
        batches: Batches decoded and handed on, replayed ones included.
        last_seq: Sequence number of the most recent batch handed on.
        epoch: Current epoch; advances by one at every resync.
        gaps: Times a sequence number was skipped, whether or not replay filled it.
        replayed: Batches vLLM resent through its replay socket after they were missed.
        resyncs: Gaps that replay could not fill. Each one advanced the epoch.
        decode_errors: Frames or batches that failed to decode and were dropped.
        queue_depth: Batches waiting between the socket thread and the consumer.
        skipped_event_types: Count per unknown event type seen so far.
    """

    batches: int
    last_seq: int | None
    epoch: int
    gaps: int
    replayed: int
    resyncs: int
    decode_errors: int
    queue_depth: int
    skipped_event_types: Mapping[str, int]


@runtime_checkable
class EventSource(Protocol):
    """Protocol for anything that yields batches in order: a collector, a capture file, a fake."""

    def __iter__(self) -> Iterator[EventBatch]: ...

    def close(self) -> None: ...
