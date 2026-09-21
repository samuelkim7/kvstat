"""vLLM's producer structs, copied from kv_events.py at main (f4eccdadef), to encode frames
0.28.0 never emitted: bytes block hashes, a populated session_id, an unknown event type.

These make the wire input of the decoder. The builders in factories.py make its output.
"""

from __future__ import annotations

from typing import Any

import msgspec


class ProducerBatch(msgspec.Struct, array_like=True, omit_defaults=True, gc=False):
    ts: float
    events: list[Any]
    data_parallel_rank: int | None = None


class ProducerEvent(msgspec.Struct, omit_defaults=True, gc=False, tag=True):
    pass


class BlockStored(ProducerEvent):
    block_hashes: list[bytes | int]
    parent_block_hash: bytes | int | None
    token_ids: list[int]
    block_size: int
    lora_id: int | None
    medium: str | None
    lora_name: str | None
    extra_keys: list[tuple[Any, ...] | None] | None = None
    group_idx: int | None = None
    kv_cache_spec_kind: str | None = None
    kv_cache_spec_sliding_window: int | None = None
    locality: str | None = None
    ownership: str | None = None
    session_id: str | None = None


class BlockRemoved(ProducerEvent):
    block_hashes: list[bytes | int]
    medium: str | None
    group_idx: int | None = None
    locality: str | None = None
    ownership: str | None = None


class AllBlocksCleared(ProducerEvent):
    pass


class FutureEvent(ProducerEvent):
    detail: str


_encoder = msgspec.msgpack.Encoder()


def encode_batch(*events: ProducerEvent, ts: float = 1.5, rank: int | None = None) -> bytes:
    return _encoder.encode(ProducerBatch(ts=ts, events=list(events), data_parallel_rank=rank))


def stored(
    hashes: list[bytes | int],
    parent: bytes | int | None = None,
    tokens: list[int] | None = None,
    **overrides: Any,
) -> BlockStored:
    fields: dict[str, Any] = {
        "block_hashes": hashes,
        "parent_block_hash": parent,
        "token_ids": tokens if tokens is not None else list(range(16 * len(hashes))),
        "block_size": 16,
        "lora_id": None,
        "medium": "GPU",
        "lora_name": None,
        "group_idx": 0,
        "kv_cache_spec_kind": "full_attention",
    }
    fields.update(overrides)
    return BlockStored(**fields)
