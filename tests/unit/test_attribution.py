from __future__ import annotations

import dataclasses

from kvstat.events import BlockStored
from kvstat.state.cache import CacheState
from kvstat.state.prefixes import PrefixTable
from kvstat.state.timeline import TimelinePoint
from kvstat.views.attribution import render_attribution
from tests.unit.factories import batch, stored

POINT = TimelinePoint(
    time=0.0,
    used_fraction=0.0,
    running_requests=0,
    waiting_requests=0,
    preemptions=None,
    evictions=0,
    ttft_p50=None,
    ttft_p95=None,
    hit_rate=None,
    hit_span=None,
)
ROOT = 1 << 60


def table_of(*events: BlockStored) -> PrefixTable:
    state = CacheState(capacity=1000)
    state.update(batch(*events))
    return state.prefixes()


def diverging(root: int, length: int = 1, users: int = 2) -> list[BlockStored]:
    stem = [root + i for i in range(length)]
    tails = [stored(root + 100 + i, parent=stem[-1]) for i in range(users)]
    return [stored(*stem), *tails]


def lines(table: PrefixTable, point: TimelinePoint | None = None) -> list[str]:
    return render_attribution(table, point).plain.split("\n")


def test_nothing_written_yet():
    assert lines(CacheState().prefixes()) == ["waiting for the first block"]


def test_a_shared_prefix_row_and_the_unshared_row():
    out = lines(table_of(*diverging(ROOT, length=2, users=3)))
    assert out[1].startswith("10000000")
    assert out[1].split()[1:4] == ["2", "3", "40.0%"]
    assert out[2].startswith("unshared")
    assert out[2].split()[1:4] == ["3", "-", "60.0%"]


def test_the_header_lines_up_with_the_rows():
    out = lines(table_of(*diverging(ROOT)))
    assert len({len(line) for line in out[:3]}) == 1


def test_a_prefix_kvstat_never_saw_has_unknown_blocks():
    out = lines(table_of(stored(10, parent=ROOT), stored(20, parent=ROOT)))
    assert out[1].split()[:4] == ["10000000", "?", "2", "?"]


def test_a_byte_hash_shows_its_first_four_bytes():
    root = b"\xab\xcd\xef\x01\x99"
    out = lines(table_of(stored(root), stored(b"x", parent=root), stored(b"y", parent=root)))
    assert out[1].startswith("abcdef01")


def test_hits_are_a_dash_until_report_mode_traffic_arrives():
    assert lines(table_of(stored(1)))[1].endswith("-")


def test_large_counts_stay_inside_their_columns():
    table = table_of(*diverging(ROOT))
    row = dataclasses.replace(table.shared[0], users=12_345_678, recomputes=4_321, hits=123_456)
    out = lines(dataclasses.replace(table, shared=(row,)))
    assert out[1].split()[2:] == ["12.3M", "33.3%", "4,321", "123k"]
    assert len(out[1]) == len(out[0])


def test_extra_shared_prefixes_are_counted_in_the_header():
    events = [e for i in range(5) for e in diverging(10_000 * (i + 1), length=i + 1)]
    out = lines(table_of(*events))
    assert out[0].startswith("prefix +2 more")
    assert len(out) == 1 + 3 + 1 + 1


def test_the_engine_hit_rate_states_the_span_it_covers():
    table = table_of(stored(1))
    covered = dataclasses.replace(POINT, hit_rate=0.21, hit_span=20.0)
    idle = dataclasses.replace(POINT, hit_span=60.0)
    assert lines(table, covered)[-1].endswith("engine hit rate 21% over 20 s")
    assert lines(table, idle)[-1].endswith("engine hit rate: no prompts in 60 s")
    assert lines(table, POINT)[-1].endswith("engine hit rate: not reported")


def test_the_written_share_counts_shared_blocks():
    out = lines(table_of(*diverging(ROOT, length=2, users=2)))
    assert out[-1].startswith("written: 50.0% shared")


def test_every_line_fits_an_80_column_terminal():
    events = [e for i in range(5) for e in diverging(10_000 * (i + 1), length=i + 1)]
    point = dataclasses.replace(POINT, hit_rate=1.0, hit_span=60.0)
    out = lines(table_of(*events), point)
    assert max(len(line) for line in out) <= 80 - 4
