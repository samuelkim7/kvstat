from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from kvstat.__main__ import main
from kvstat.cli.report import apply_capture
from kvstat.events import BlockStored
from kvstat.ingest.capture import CaptureReader
from kvstat.state.cache import CacheState, Health
from kvstat.state.crosscheck import EngineWindow
from tests.integration.corrupt import DoubleCounting

CAPTURE = "mixed-storm-120s.jsonl.gz"


@pytest.fixture(scope="module")
def capture(fixtures: Path) -> CaptureReader:
    return CaptureReader(fixtures / CAPTURE)


@pytest.fixture(scope="module")
def replayed(capture: CaptureReader) -> tuple[CacheState, list[object], EngineWindow]:
    capacity, _ = capture.cache_config()
    state = CacheState(capacity=capacity)
    breaches, _, window = apply_capture(state, capture)
    return state, list(breaches), window


def test_the_capture_is_redacted(capture: CaptureReader) -> None:
    assert capture.header["redacted"] is True
    tokens = sum(len(e.token_ids) for b in capture for e in b.events if isinstance(e, BlockStored))
    assert tokens == 0


def test_the_capture_carries_the_pool_size(capture: CaptureReader) -> None:
    capacity, block_size = capture.cache_config()
    assert capacity == 8285
    assert block_size == 16


def test_the_engine_never_contradicts_the_rebuilt_state(
    replayed: tuple[CacheState, list[object], EngineWindow],
) -> None:
    state, breaches, window = replayed
    assert breaches == []
    assert window.polls == 119
    assert state.snapshot().health is Health.OK


def test_the_engine_gauge_swings_while_residency_stays_pinned(
    replayed: tuple[CacheState, list[object], EngineWindow],
) -> None:
    state, _, window = replayed
    snap = state.snapshot()
    assert snap.capacity is not None
    assert window.used_fraction_low < 0.10, "the engine was nearly idle at some point"
    assert window.used_fraction_high > 0.95, "and nearly full at another"
    assert snap.cached / snap.capacity > 0.95, "while residency never moved"


def test_the_rebuilt_state_fills_the_pool_without_overflowing_it(
    replayed: tuple[CacheState, list[object], EngineWindow],
) -> None:
    snap = replayed[0].snapshot()
    assert snap.capacity is not None
    assert snap.cached <= snap.capacity
    assert snap.cached / snap.capacity > 0.95


def test_the_capture_shows_a_cache_churning_faster_than_it_grows(
    replayed: tuple[CacheState, list[object], EngineWindow],
) -> None:
    snap = replayed[0].snapshot()
    assert snap.stored_per_s > 500
    assert abs(snap.stored_per_s - snap.evicted_per_s) / snap.stored_per_s < 0.05
    assert snap.lifetime_p50 is not None and snap.lifetime_p50 < 30


def test_orphan_stores_stop_once_the_state_has_warmed_up(capture: CaptureReader) -> None:
    capacity, _ = capture.cache_config()
    state = CacheState(capacity=capacity)
    rows = list(capture.batches())
    cutoff = rows[-1][0] - 60.0
    warm = None
    for arrived, batch in rows:
        state.update(batch)
        if warm is None and arrived >= cutoff:
            warm = state.snapshot().orphan_stores
    assert warm is not None
    assert state.snapshot().orphan_stores == warm


def test_a_double_count_breaches_the_pool_size_and_is_rebuilt_away(
    capture: CaptureReader,
) -> None:
    capacity, _ = capture.cache_config()
    state = CacheState(capacity=capacity)
    breaches, _, _ = apply_capture(state, DoubleCounting(capture))
    assert breaches, "a double count should outgrow the pool"
    assert all(breach.kind == "over_capacity" for breach in breaches)
    assert "blocks cached, pool holds 8285" in breaches[0].detail
    assert state.snapshot().rebuilds >= len(breaches)


def test_dump_prints_the_summary_without_a_server(fixtures: Path) -> None:
    result = CliRunner().invoke(main, ["report", "--from", str(fixtures / CAPTURE)])
    assert result.exit_code == 0
    printed = {
        line[:16].strip(): line[16:].strip()
        for line in result.output.splitlines()
        if line[:16].strip() and line[16:].strip()
    }
    assert printed["engine in use"] == "0.0% - 100.0%   peak 8,281 blocks"
    assert printed["prefix cached"] == "8,262 / 8,285 (99.7%)   free 23"
    assert printed["turnover"] == "every 7.1s   1,166 blocks/s stored, 1,166 evicted"
    assert printed["lifetime"] == "p50 9.4s   p95 12.4s"
    assert printed["drift"] == "ok  (119 polls)"


def test_dump_says_why_it_saw_no_prefix_re_stores(fixtures: Path) -> None:
    result = CliRunner().invoke(main, ["report", "--from", str(fixtures / CAPTURE)])
    assert "kv_cache_report_mode: full" in result.output, (
        "zero re-stores means the workload never asked for them, and the report must say so"
    )


def test_the_preamble_is_unseen_until_the_burst_and_returns_with_its_recomputes(
    capture: CaptureReader,
) -> None:
    capacity, _ = capture.cache_config()
    state = CacheState(capacity=capacity)
    burst_start = capture.header["started_at"] + 41.0
    before_burst = None
    for arrived, batch in capture.batches():
        if before_burst is None and arrived >= burst_start:
            before_burst = state.prefixes()
        state.update(batch)
    assert before_burst is not None and len(before_burst.shared) == 2
    assert not any(row.in_table for row in before_burst.shared), "the attach missed the preamble"
    rows = {row.root: row for row in state.prefixes().shared}
    preamble = rows[9442039759122069047]
    assert preamble.in_table and preamble.blocks == 48
    assert preamble.recomputes > 0
    assert state.prefixes().shared_write_share < 0.01
