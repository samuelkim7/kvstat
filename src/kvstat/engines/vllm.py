"""vLLM's KV-event wire format and its translation into kvstat events.

The only module that knows vLLM's field names, struct options and tags. The structs mirror
``vllm/distributed/kv_events.py``; a wire change upstream is a change here and nowhere else.
Only the struct options that affect decoding are mirrored (``array_like``, ``tag``).
"""

from __future__ import annotations

from typing import Any, assert_never

import msgspec

from kvstat import events
from kvstat.errors import DecodeError


class _Batch(msgspec.Struct, array_like=True):
    # Positional on the wire: ts, events, data_parallel_rank. Reordering breaks decoding.
    ts: float
    events: list[msgspec.Raw]
    data_parallel_rank: int | None = None


class _BlockStored(msgspec.Struct, tag="BlockStored"):
    block_hashes: list[events.BlockHash]
    parent_block_hash: events.BlockHash | None
    token_ids: list[int]
    block_size: int
    lora_id: int | None = None
    medium: str | None = None
    lora_name: str | None = None
    extra_keys: list[tuple[Any, ...] | None] | None = None
    group_idx: int | None = None
    kv_cache_spec_kind: str | None = None
    kv_cache_spec_sliding_window: int | None = None
    locality: str | None = None
    ownership: str | None = None
    session_id: str | None = None


class _BlockRemoved(msgspec.Struct, tag="BlockRemoved"):
    block_hashes: list[events.BlockHash]
    medium: str | None = None
    group_idx: int | None = None
    locality: str | None = None
    ownership: str | None = None


class _AllBlocksCleared(msgspec.Struct, tag="AllBlocksCleared"):
    pass


class _Tagged(msgspec.Struct):
    type: str


_Event = _BlockStored | _BlockRemoved | _AllBlocksCleared
_KNOWN_TAGS = frozenset(
    cls.__struct_config__.tag for cls in (_BlockStored, _BlockRemoved, _AllBlocksCleared)
)

_batch_decoder = msgspec.msgpack.Decoder(_Batch)
_event_decoder = msgspec.msgpack.Decoder(_Event)
_tag_decoder = msgspec.msgpack.Decoder(_Tagged)


def sequence_number(frame: bytes) -> int:
    """Read the sequence number vLLM sends as the second ZMQ frame (8 bytes, big-endian)."""
    return int.from_bytes(frame, "big")


def decode_batch(seq: int, payload: bytes) -> events.EventBatch:
    """Turn one published payload into an ``EventBatch``.

    Event types this module does not know are skipped and named in ``skipped``. Anything else
    that fails to parse raises, because a half-read batch would put fiction on the screen.

    Raises:
        DecodeError: the envelope or a known event type did not parse.
    """
    try:
        batch = _batch_decoder.decode(payload)
    except msgspec.MsgspecError as exc:
        raise DecodeError(f"batch seq={seq}: {exc}") from exc

    decoded: list[events.Event] = []
    skipped: list[str] = []
    for raw in batch.events:
        try:
            wire = _event_decoder.decode(raw)
        except msgspec.ValidationError as exc:
            tag = _peek_tag(raw)
            if tag is not None and tag not in _KNOWN_TAGS:
                skipped.append(tag)
                continue
            raise DecodeError(f"batch seq={seq}, event {tag or '?'}: {exc}") from exc
        except msgspec.DecodeError as exc:
            raise DecodeError(f"batch seq={seq}: {exc}") from exc
        decoded.append(_translate(wire))

    return events.EventBatch(
        seq=seq,
        ts=batch.ts,
        data_parallel_rank=batch.data_parallel_rank,
        events=tuple(decoded),
        skipped=tuple(skipped),
    )


def _peek_tag(raw: msgspec.Raw) -> str | None:
    try:
        return _tag_decoder.decode(raw).type
    except msgspec.MsgspecError:
        return None


def _translate(wire: _Event) -> events.Event:
    match wire:
        case _BlockStored():
            return events.BlockStored(
                block_hashes=tuple(wire.block_hashes),
                parent_block_hash=wire.parent_block_hash,
                token_ids=tuple(wire.token_ids),
                block_size=wire.block_size,
                medium=wire.medium,
                group_idx=wire.group_idx,
                kv_cache_spec_kind=wire.kv_cache_spec_kind,
                lora_name=wire.lora_name,
                session_id=wire.session_id,
            )
        case _BlockRemoved():
            return events.BlockRemoved(
                block_hashes=tuple(wire.block_hashes),
                medium=wire.medium,
                group_idx=wire.group_idx,
            )
        case _AllBlocksCleared():
            return events.AllBlocksCleared()
        case _:
            assert_never(wire)
