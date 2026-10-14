from __future__ import annotations

import dataclasses
from typing import Any

from rich.text import Text

from kvstat.state.timeline import TimelinePoint
from kvstat.views.pressure import render_pressure

BASE = TimelinePoint(
    time=0.0,
    used_fraction=0.0,
    running_requests=0,
    waiting_requests=0,
    preemptions=None,
    evictions=0,
    ttft_p50=None,
    ttft_p95=None,
    hit_rate=None,
)


def points(**series: list[Any]) -> list[TimelinePoint]:
    n = max(len(v) for v in series.values())
    return [
        dataclasses.replace(BASE, time=float(i), **{k: v[i] for k, v in series.items()})
        for i in range(n)
    ]


def rows(text: Text) -> dict[str, str]:
    return {line[:10].strip(): line[10:] for line in text.plain.split("\n")}


def test_no_points_yet():
    assert render_pressure([], set(), 80).plain == "waiting for the first scrape"


def test_the_fixed_rows_appear_with_the_latest_value_at_the_right():
    text = render_pressure(points(used_fraction=[0.0, 0.5, 1.0]), set(), 80)
    r = rows(text)
    assert set(r) == {"in use", "waiting", "evicted"}
    assert r["in use"].endswith("100%")
    assert "▁▅█" in r["in use"]


def test_series_the_engine_does_not_report_are_left_out():
    r = rows(render_pressure(points(used_fraction=[0.1]), {"prefix cache"}, 80))
    assert "preempted" not in r and "ttft p95" not in r


def test_preemptions_are_marks_with_the_count_in_view():
    text = render_pressure(points(preemptions=[0, 3, 0, 2]), {"preemptions"}, 80)
    r = rows(text)
    assert r["preempted"].endswith("5 in view")
    assert "·▲·▲" in r["preempted"]


def test_ttft_row_scales_to_its_own_peak_and_shows_the_latest():
    text = render_pressure(points(ttft_p95=[0.1, 2.0, None]), {"ttft"}, 80)
    r = rows(text)
    assert r["ttft p95"].endswith("-")
    assert "▁█▁" in r["ttft p95"]


def test_only_the_points_that_fit_are_drawn():
    text = render_pressure(points(waiting_requests=list(range(200))), set(), 60)
    spark = rows(text)["waiting"][: 60 - 10 - 12]
    assert len(spark) == 38 and spark[0] != " "


def test_rows_stay_aligned_before_the_window_fills():
    text = render_pressure(
        points(used_fraction=[0.5] * 3, preemptions=[1, 0, 1]), {"preemptions"}, 50
    )
    assert len({len(line) for line in text.plain.split("\n")}) == 1


def test_evictions_show_as_a_rate_per_second():
    p = points(evictions=[0, 0, 600])
    p[-1] = dataclasses.replace(p[-1], time=p[-2].time + 2.0)
    assert rows(render_pressure(p, set(), 80))["evicted"].endswith("300/s")
