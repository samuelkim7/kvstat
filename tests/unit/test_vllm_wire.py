from __future__ import annotations

import msgspec
import pytest

from kvtop import events
from kvtop.engines import vllm
from kvtop.errors import DataSourceError, DecodeError, KvtopError
from tests.unit.conftest import (
    AllBlocksCleared,
    BlockRemoved,
    FutureEvent,
    encode_batch,
    stored,
)


def test_real_frames_decode_into_domain_events(real_frames):
    batches = [vllm.decode_batch(seq, payload) for seq, payload in real_frames]

    assert [b.seq for b in batches] == [0, 223, 234]
    assert [len(b.events) for b in batches] == [11, 7, 2]
    assert all(b.skipped == () for b in batches)
    assert all(b.data_parallel_rank == 0 for b in batches)

    stored_events = [e for b in batches for e in b.events if isinstance(e, events.BlockStored)]
    removed_events = [e for b in batches for e in b.events if isinstance(e, events.BlockRemoved)]
    assert stored_events and removed_events
    assert any(e.parent_block_hash is None for e in stored_events)
    assert any(e.parent_block_hash is not None for e in stored_events)
    hashes = [h for e in stored_events + removed_events for h in e.block_hashes]
    assert hashes and all(type(h) is int for h in hashes)


def test_real_frames_carry_0_28_shape(real_frames):
    batch = vllm.decode_batch(*real_frames[0])
    first = batch.events[0]
    assert isinstance(first, events.BlockStored)
    assert first.block_size == 16
    assert first.medium == "GPU"
    assert first.kv_cache_spec_kind == "full_attention"
    assert first.group_idx == 0
    assert first.session_id is None
    assert len(first.token_ids) == first.block_size * len(first.block_hashes)
    assert isinstance(first.token_ids, tuple)


def test_synthetic_frame_keeps_bytes_hashes_and_all_event_types():
    root = b"\x01" * 32
    child = b"\x02" * 32
    payload = encode_batch(
        stored([root], parent=None, session_id="tenant-a", ownership="tier-0"),
        stored([child], parent=root, session_id="tenant-a"),
        BlockRemoved(block_hashes=[root], medium="GPU", ownership="tier-0"),
        AllBlocksCleared(),
        ts=42.0,
        rank=3,
    )

    batch = vllm.decode_batch(7, payload)

    assert (batch.seq, batch.ts, batch.data_parallel_rank) == (7, 42.0, 3)
    assert [type(e) for e in batch.events] == [
        events.BlockStored,
        events.BlockStored,
        events.BlockRemoved,
        events.AllBlocksCleared,
    ]
    first, second, removed = batch.events[:3]
    assert isinstance(first, events.BlockStored)
    assert isinstance(second, events.BlockStored)
    assert isinstance(removed, events.BlockRemoved)
    assert first.block_hashes == (root,)
    assert second.parent_block_hash == root
    assert first.session_id == "tenant-a"
    assert removed.block_hashes == (root,)
    assert not hasattr(first, "ownership")
    assert not hasattr(removed, "ownership")


def test_unknown_event_type_is_skipped_not_fatal():
    payload = encode_batch(
        stored([1]),
        FutureEvent(detail="from a newer vLLM"),
        BlockRemoved(block_hashes=[1], medium="GPU"),
    )

    batch = vllm.decode_batch(1, payload)

    assert [type(e) for e in batch.events] == [events.BlockStored, events.BlockRemoved]
    assert batch.skipped == ("FutureEvent",)


def test_two_element_envelope_defaults_rank():
    payload = msgspec.msgpack.encode([1.0, []])
    assert vllm.decode_batch(0, payload) == events.EventBatch(0, 1.0, None, ())


def test_garbage_payload_raises_decode_error():
    with pytest.raises(DecodeError) as info:
        vllm.decode_batch(5, b"\xc1not msgpack")
    assert isinstance(info.value, DataSourceError)
    assert isinstance(info.value, KvtopError)
    assert "seq=5" in str(info.value)
    assert info.value.__cause__ is not None


def test_known_event_with_wrong_field_type_raises_decode_error():
    payload = msgspec.msgpack.encode(
        [1.0, [{"type": "BlockRemoved", "block_hashes": "not-a-list", "medium": "GPU"}]]
    )
    with pytest.raises(DecodeError) as info:
        vllm.decode_batch(9, payload)
    assert "BlockRemoved" in str(info.value)


def test_sequence_number_is_big_endian_u64():
    assert vllm.sequence_number((2**40 + 5).to_bytes(8, "big")) == 2**40 + 5
    assert vllm.sequence_number(b"\x00" * 8) == 0


def test_domain_events_are_immutable():
    batch = vllm.decode_batch(0, encode_batch(stored([1])))
    event = batch.events[0]
    assert isinstance(event, events.BlockStored)
    with pytest.raises(AttributeError):
        event.block_size = 32  # type: ignore[misc]
