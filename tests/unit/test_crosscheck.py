from __future__ import annotations

from kvstat.state.cache import CacheState, Health
from kvstat.state.crosscheck import EngineCounts, check
from tests.unit.factories import batch, stored


def counts(used_fraction: float, running_requests: int = 0, time: float = 1.0) -> EngineCounts:
    return EngineCounts(time=time, used_fraction=used_fraction, running_requests=running_requests)


def test_agreement_reports_no_breach():
    state = CacheState(capacity=100)
    state.update(batch(stored(*range(50))))
    assert check(state.snapshot(), counts(used_fraction=0.40)) is None


def test_more_cached_blocks_than_capacity_is_a_breach():
    state = CacheState(capacity=10)
    state.update(batch(stored(*range(12))))
    breach = check(state.snapshot(), counts(used_fraction=0.5))
    assert breach is not None
    assert breach.kind == "over_capacity"


def test_fewer_cached_blocks_than_the_engine_holds_is_a_breach():
    state = CacheState(capacity=100)
    state.update(batch(stored(1, 2)))
    breach = check(state.snapshot(), counts(used_fraction=0.90))
    assert breach is not None
    assert breach.kind == "undercount"


def test_running_requests_cover_blocks_the_engine_holds_but_never_hashed():
    state = CacheState(capacity=100)
    state.update(batch(stored(*range(40))))
    assert check(state.snapshot(), counts(used_fraction=0.45, running_requests=8)) is None


def test_the_undercount_check_allows_slack_for_poll_skew():
    state = CacheState(capacity=100)
    state.update(batch(stored(*range(49))))
    assert check(state.snapshot(), counts(used_fraction=0.50), slack=2) is None


def test_an_unknown_capacity_cannot_be_checked():
    state = CacheState(capacity=None)
    state.update(batch(stored(1)))
    assert check(state.snapshot(), counts(used_fraction=0.90)) is None


def test_reconciling_a_breach_degrades_and_rebuilds():
    state = CacheState(capacity=10)
    state.update(batch(stored(*range(12))))
    breach = state.cross_check(counts(used_fraction=0.5))
    assert breach is not None
    snap = state.snapshot()
    assert snap.health is Health.DEGRADED
    assert snap.cached == 0
    assert snap.breaches == 1
    assert snap.rebuilds == 1


def test_a_recovering_state_does_not_breach_on_what_it_knowingly_dropped():
    state = CacheState(capacity=10)
    state.update(batch(stored(*range(12))))
    state.cross_check(counts(used_fraction=0.5))
    for poll in range(5):
        assert state.cross_check(counts(used_fraction=0.9, time=2.0 + poll)) is None
    snap = state.snapshot()
    assert snap.health is Health.DEGRADED
    assert snap.breaches == 1
    assert snap.rebuilds == 1


def test_recovery_ends_at_the_first_poll_the_engine_agrees_with():
    state = CacheState(capacity=10)
    state.update(batch(stored(*range(12))))
    state.cross_check(counts(used_fraction=0.5))
    state.update(batch(stored(100, 101, 102, 103, 104), ts=2.0))
    assert state.cross_check(counts(used_fraction=0.5, time=2.0)) is None
    assert state.snapshot().health is Health.OK


def test_a_clean_reconcile_leaves_the_table_alone():
    state = CacheState(capacity=100)
    state.update(batch(stored(1, 2)))
    assert state.cross_check(counts(used_fraction=0.0)) is None
    assert state.snapshot().cached == 2
