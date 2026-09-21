from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from kvstat.events import AllBlocksCleared, BlockHash, Event, EventBatch
from kvstat.state.cache import CacheState
from tests.unit.factories import removed, stored

CAPACITY = 24
Stream = tuple[list[EventBatch], int]


@st.composite
def event_streams(draw: st.DrawFn) -> Stream:
    """An event sequence a vLLM engine could emit, with the blocks left cached at the end.

    Chains grow from cached parents, removals target cached blocks, a cached block may be
    stored again when a request reuses it, and the engine never holds more blocks than it has
    room for.
    """
    cached: list[BlockHash] = []
    next_hash = 0
    batches: list[EventBatch] = []
    for seq in range(draw(st.integers(min_value=1, max_value=30))):
        events: list[Event] = []
        for _ in range(draw(st.integers(min_value=1, max_value=4))):
            action = draw(st.sampled_from(["store"] * 5 + ["remove"] * 4 + ["restore", "clear"]))
            if action == "remove" and cached:
                victim = draw(st.sampled_from(cached))
                cached.remove(victim)
                events.append(removed(victim))
            elif action == "restore" and cached:
                events.append(stored(draw(st.sampled_from(cached))))
            elif action == "clear":
                cached.clear()
                events.append(AllBlocksCleared())
            elif len(cached) < CAPACITY:
                count = draw(st.integers(min_value=1, max_value=min(4, CAPACITY - len(cached))))
                parent = draw(st.sampled_from([None, *cached])) if cached else None
                hashes = list(range(next_hash, next_hash + count))
                next_hash += count
                cached.extend(hashes)
                events.append(stored(*hashes, parent=parent))
        batches.append(
            EventBatch(seq=seq, ts=float(seq), data_parallel_rank=None, events=tuple(events))
        )
    return batches, len(cached)


@given(event_streams())
@settings(max_examples=200)
def test_an_evicted_block_leaves_no_entry_behind(stream: Stream) -> None:
    batches, expected = stream
    state = CacheState(capacity=CAPACITY)
    for b in batches:
        state.update(b)
    snap = state.snapshot()
    assert snap.cached == expected
    assert snap.unknown_removals == 0


@given(event_streams())
@settings(max_examples=200)
def test_every_stored_block_chains_to_a_cached_parent(stream: Stream) -> None:
    batches, _ = stream
    state = CacheState(capacity=CAPACITY)
    for b in batches:
        state.update(b)
    assert state.snapshot().orphan_stores == 0


@given(event_streams())
@settings(max_examples=200)
def test_totals_never_contradict_each_other(stream: Stream) -> None:
    batches, _ = stream
    state = CacheState(capacity=CAPACITY)
    for b in batches:
        state.update(b)
    snap = state.snapshot()
    assert snap.stored >= snap.evicted
    assert snap.cached <= snap.stored - snap.evicted
