"""Builders for decoded domain events.

These make kvstat.events objects, the output side of the decoder. The producer structs in
producer.py make vLLM's wire input instead.
"""

from __future__ import annotations

from collections.abc import Iterator

from kvstat.events import BlockHash, BlockRemoved, BlockStored, Event, EventBatch, Sample


def stored(
    *hashes: BlockHash,
    parent: BlockHash | None = None,
    group: int | None = 0,
    medium: str | None = "GPU",
) -> BlockStored:
    return BlockStored(
        block_hashes=tuple(hashes),
        parent_block_hash=parent,
        token_ids=(),
        block_size=16,
        medium=medium,
        group_idx=group,
        kv_cache_spec_kind="full_attention",
        lora_name=None,
        session_id=None,
    )


def removed(*hashes: BlockHash, group: int | None = 0, medium: str | None = "GPU") -> BlockRemoved:
    return BlockRemoved(block_hashes=tuple(hashes), medium=medium, group_idx=group)


def batch(*events: Event, ts: float = 1.0, seq: int = 1, epoch: int = 0) -> EventBatch:
    return EventBatch(seq=seq, ts=ts, data_parallel_rank=None, events=tuple(events), epoch=epoch)


class Capture:
    """A scripted capture: rows are (time, batch) or (time, samples), in arrival order."""

    def __init__(self, rows: list[tuple[float, EventBatch | list[Sample]]]) -> None:
        self.rows = rows
        self.header: dict[str, object] = {
            "model_name": "test-model",
            "vllm_version": "0.28.0",
            "cache_config": {"num_gpu_blocks": "8", "block_size": "16"},
        }

    def batches(self) -> Iterator[tuple[float, EventBatch]]:
        for t, row in self.rows:
            if isinstance(row, EventBatch):
                yield t, row

    def metrics(self) -> Iterator[tuple[float, list[Sample]]]:
        for t, row in self.rows:
            if not isinstance(row, EventBatch):
                yield t, row
