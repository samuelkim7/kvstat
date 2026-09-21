from __future__ import annotations

from kvstat.events import AllBlocksCleared
from kvstat.state.cache import CacheState, Health
from kvstat.state.crosscheck import EngineCounts
from tests.unit.factories import batch, removed, stored


def test_stored_blocks_become_cached():
    state = CacheState(capacity=100)
    state.update(batch(stored(1, 2, 3)))
    snap = state.snapshot()
    assert snap.cached == 3
    assert snap.stored == 3
    assert snap.free == 97


def test_removed_blocks_leave_no_entry():
    state = CacheState(capacity=100)
    state.update(batch(stored(1, 2, 3)))
    state.update(batch(removed(2), ts=2.0))
    snap = state.snapshot()
    assert snap.cached == 2
    assert snap.evicted == 1


def test_all_blocks_cleared_empties_the_table_but_keeps_totals():
    state = CacheState(capacity=100)
    state.update(batch(stored(1, 2)))
    state.update(batch(AllBlocksCleared(), ts=2.0))
    snap = state.snapshot()
    assert snap.cached == 0
    assert snap.stored == 2
    assert snap.clears == 1
    assert snap.health is Health.WARMING


def test_the_same_hash_in_two_groups_is_two_blocks():
    state = CacheState(capacity=100)
    state.update(batch(stored(7, group=0), stored(7, group=1)))
    assert state.snapshot().cached == 2


def test_a_removal_only_evicts_its_own_group():
    state = CacheState(capacity=100)
    state.update(batch(stored(7, group=0), stored(7, group=1)))
    state.update(batch(removed(7, group=1), ts=2.0))
    assert state.snapshot().cached == 1


def test_a_block_in_another_tier_does_not_count_against_the_pool():
    state = CacheState(capacity=100, tier="GPU")
    state.update(batch(stored(1, medium="GPU"), stored(2, medium="CPU")))
    snap = state.snapshot()
    assert snap.cached == 1
    assert snap.free == 99


def test_restoring_a_cached_block_is_idempotent():
    state = CacheState(capacity=100)
    state.update(batch(stored(1)))
    state.update(batch(stored(1), ts=2.0))
    snap = state.snapshot()
    assert snap.cached == 1
    assert snap.restored == 1


def test_removing_an_unknown_block_is_counted_not_fatal():
    state = CacheState(capacity=100)
    state.update(batch(removed(42)))
    snap = state.snapshot()
    assert snap.cached == 0
    assert snap.unknown_removals == 1


def test_a_store_whose_parent_is_missing_is_counted_as_an_orphan():
    state = CacheState(capacity=100)
    state.update(batch(stored(2, parent=1)))
    assert state.snapshot().orphan_stores == 1


def test_a_chained_store_is_not_an_orphan():
    state = CacheState(capacity=100)
    state.update(batch(stored(1, 2, 3)))
    assert state.snapshot().orphan_stores == 0


def test_block_lifetime_is_measured_between_store_and_eviction():
    state = CacheState(capacity=100)
    state.update(batch(stored(1, 2), ts=10.0))
    state.update(batch(removed(1), ts=14.0))
    state.update(batch(removed(2), ts=20.0))
    snap = state.snapshot()
    assert snap.lifetime_p50 == 4.0
    assert snap.lifetime_p95 == 10.0


def test_rates_are_measured_over_the_window_since_the_rebuild():
    state = CacheState(capacity=100)
    state.update(batch(stored(1, 2, 3, 4), ts=10.0))
    state.update(batch(removed(1, 2), ts=12.0))
    snap = state.snapshot()
    assert snap.elapsed == 2.0
    assert snap.stored_per_s == 2.0
    assert snap.evicted_per_s == 1.0


def test_an_epoch_change_rebuilds_the_table():
    state = CacheState(capacity=100)
    state.update(batch(stored(1, 2), ts=10.0))
    state.update(batch(stored(3), ts=11.0, epoch=1))
    snap = state.snapshot()
    assert snap.cached == 1
    assert snap.epoch == 1
    assert snap.rebuilds == 1


def test_a_resync_stops_the_state_calling_itself_healthy():
    """Regression: _rebuild() once kept the old OK, so a resynced table reported healthy while
    holding 2,275 of the engine's 7,514 blocks."""
    state = CacheState(capacity=100)
    state.update(batch(stored(*range(60)), ts=10.0))
    state.cross_check(EngineCounts(time=10.0, used_fraction=0.50, running_requests=0))
    assert state.snapshot().health is Health.OK
    state.update(batch(stored(60), ts=11.0, epoch=1))
    assert state.snapshot().health is Health.WARMING


def test_totals_survive_a_rebuild():
    state = CacheState(capacity=100)
    state.update(batch(stored(1, 2), ts=10.0))
    state.update(batch(stored(3), ts=11.0, epoch=1))
    assert state.snapshot().stored == 3


def test_an_unknown_capacity_leaves_free_unset():
    state = CacheState(capacity=None)
    state.update(batch(stored(1)))
    snap = state.snapshot()
    assert snap.cached == 1
    assert snap.free is None
