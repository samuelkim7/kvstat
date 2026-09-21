from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from pathlib import Path

import pytest
from click.testing import CliRunner

from kvstat.__main__ import main
from kvstat.cli.dump import apply_capture
from kvstat.events import BlockStored, Event, EventBatch
from kvstat.ingest.capture import CaptureReader
from kvstat.ingest.metrics import Sample
from kvstat.state.cache import CacheState, Health
from kvstat.state.crosscheck import EngineWindow

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


class DoubleCounting:
    """The capture with every stored block counted twice, under a hash kvstat has not seen.

    Presents the same batches()/metrics() surface as a CaptureReader, so the corrupted stream
    runs through the very same merge the dump command uses.
    """

    def __init__(self, source: CaptureReader) -> None:
        self.header = source.header
        self._source = source

    def batches(self) -> Iterator[tuple[float, EventBatch]]:
        for arrived, batch in self._source.batches():
            events: list[Event] = []
            for event in batch.events:
                events.append(event)
                if isinstance(event, BlockStored):
                    shifted = tuple(
                        h + 10**12 if isinstance(h, int) else h + b"\x00"
                        for h in event.block_hashes
                    )
                    events.append(dataclasses.replace(event, block_hashes=shifted))
            yield arrived, dataclasses.replace(batch, events=tuple(events))

    def metrics(self) -> Iterator[tuple[float, list[Sample]]]:
        return self._source.metrics()


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
    result = CliRunner().invoke(main, ["dump", "--from", str(fixtures / CAPTURE)])
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
    result = CliRunner().invoke(main, ["dump", "--from", str(fixtures / CAPTURE)])
    assert "kv_cache_report_mode: full" in result.output, (
        "zero re-stores means the workload never asked for them, and the report must say so"
    )
