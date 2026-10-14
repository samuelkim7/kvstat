from __future__ import annotations

import msgspec
import pytest

from kvstat import events
from kvstat.engines import vllm
from kvstat.errors import DecodeError
from tests.unit.producer import (
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

    first = stored_events[0]
    assert (first.block_size, first.medium, first.group_idx) == (16, "GPU", 0)
    assert first.kv_cache_spec_kind == "full_attention"
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
    assert "seq=5" in str(info.value)
    assert info.value.__cause__ is not None


def test_known_event_with_wrong_field_type_raises_decode_error():
    payload = msgspec.msgpack.encode(
        [1.0, [{"type": "BlockRemoved", "block_hashes": "not-a-list", "medium": "GPU"}]]
    )
    with pytest.raises(DecodeError) as info:
        vllm.decode_batch(9, payload)
    assert "BlockRemoved" in str(info.value)


SCRAPE = [
    ("vllm:kv_cache_usage_perc", {"engine": "0"}, 0.25),
    ("vllm:num_requests_running", {"engine": "0"}, 3.0),
    ("vllm:num_requests_waiting", {"engine": "0"}, 7.0),
    ("vllm:num_preemptions_total", {"engine": "0"}, 91.0),
    ("vllm:prefix_cache_queries_total", {"engine": "0"}, 1000.0),
    ("vllm:prefix_cache_hits_total", {"engine": "0"}, 210.0),
    ("vllm:time_to_first_token_seconds_bucket", {"le": "0.1"}, 12.0),
    ("vllm:time_to_first_token_seconds_bucket", {"le": "0.5"}, 18.0),
    ("vllm:time_to_first_token_seconds_bucket", {"le": "+Inf"}, 20.0),
    ("vllm:time_to_first_token_seconds_count", {}, 20.0),
    ("vllm:time_to_first_token_seconds_sum", {}, 3.5),
]


def test_engine_counts_read_every_family_the_panels_use():
    counts = vllm.read_engine_counts(5.0, SCRAPE)
    assert counts == events.EngineCounts(
        time=5.0,
        used_fraction=0.25,
        running_requests=3,
        waiting_requests=7,
        preemptions=91,
        prefix_queries=1000,
        prefix_hits=210,
        ttft_buckets=((0.1, 12.0), (0.5, 18.0), (float("inf"), 20.0)),
    )


def test_engine_counts_accept_counters_without_the_total_suffix():
    # The spike's captures normalised counter names; a live scrape keeps _total.
    stripped = [(name.removesuffix("_total"), labels, v) for name, labels, v in SCRAPE]
    assert vllm.read_engine_counts(5.0, stripped) == vllm.read_engine_counts(5.0, SCRAPE)


def test_engine_counts_leave_an_absent_source_absent_not_zero():
    counts = vllm.read_engine_counts(5.0, SCRAPE[:2])
    assert (counts.waiting_requests, counts.preemptions) == (0, None)
    assert (counts.prefix_queries, counts.prefix_hits) == (None, None)
    assert counts.ttft_buckets == ()
