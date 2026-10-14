from __future__ import annotations

from kvstat.events import EngineCounts
from kvstat.state.timeline import Timeline

INF = float("inf")


def scrape(
    t: float,
    *,
    used: float = 0.5,
    running: int = 2,
    waiting: int = 0,
    preemptions: int | None = None,
    queries: int | None = None,
    hits: int | None = None,
    ttft: tuple[tuple[float, float], ...] = (),
) -> EngineCounts:
    return EngineCounts(
        time=t,
        used_fraction=used,
        running_requests=running,
        waiting_requests=waiting,
        preemptions=preemptions,
        prefix_queries=queries,
        prefix_hits=hits,
        ttft_buckets=ttft,
    )


def test_the_first_point_has_no_deltas_to_report():
    timeline = Timeline()
    point = timeline.add(scrape(1.0, preemptions=91), evicted_total=500)
    assert (point.preemptions, point.evictions) == (0, 0)
    assert timeline.latest is point


def test_counters_become_deltas_between_points():
    timeline = Timeline()
    timeline.add(scrape(1.0, preemptions=91), evicted_total=500)
    point = timeline.add(scrape(2.0, preemptions=94), evicted_total=900)
    assert (point.preemptions, point.evictions) == (3, 400)


def test_a_counter_reset_reads_as_zero_not_negative():
    timeline = Timeline()
    timeline.add(scrape(1.0, preemptions=91), evicted_total=500)
    point = timeline.add(scrape(2.0, preemptions=0), evicted_total=500)
    assert point.preemptions == 0


def test_hit_rate_covers_the_session_not_the_engine_lifetime():
    timeline = Timeline()
    timeline.add(scrape(1.0, queries=1000, hits=900), evicted_total=0)
    point = timeline.add(scrape(2.0, queries=1200, hits=920), evicted_total=0)
    assert point.hit_rate == 0.1


def test_hit_rate_is_absent_until_a_query_lands():
    timeline = Timeline()
    first = timeline.add(scrape(1.0, queries=1000, hits=900), evicted_total=0)
    assert first.hit_rate is None
    assert timeline.add(scrape(2.0), evicted_total=0).hit_rate is None


def test_ttft_percentiles_come_from_the_histogram_delta():
    before = ((0.1, 10.0), (0.5, 10.0), (2.0, 10.0), (INF, 10.0))
    after = ((0.1, 12.0), (0.5, 18.0), (2.0, 20.0), (INF, 20.0))
    timeline = Timeline()
    timeline.add(scrape(1.0, ttft=before), evicted_total=0)
    point = timeline.add(scrape(2.0, ttft=after), evicted_total=0)
    assert (point.ttft_p50, point.ttft_p95) == (0.5, 2.0)


def test_ttft_is_absent_when_no_request_finished_prefill():
    buckets = ((0.1, 10.0), (INF, 10.0))
    timeline = Timeline()
    timeline.add(scrape(1.0, ttft=buckets), evicted_total=0)
    point = timeline.add(scrape(2.0, ttft=buckets), evicted_total=0)
    assert (point.ttft_p50, point.ttft_p95) == (None, None)


def test_ttft_in_the_open_bucket_reports_the_last_finite_bound():
    timeline = Timeline()
    timeline.add(scrape(1.0, ttft=((0.1, 0.0), (INF, 0.0))), evicted_total=0)
    point = timeline.add(scrape(2.0, ttft=((0.1, 0.0), (INF, 5.0))), evicted_total=0)
    assert point.ttft_p50 == 0.1


def test_the_window_keeps_only_the_latest_points():
    timeline = Timeline(window=3)
    for t in range(5):
        timeline.add(scrape(float(t)), evicted_total=0)
    assert [p.time for p in timeline.points] == [2.0, 3.0, 4.0]


def test_the_timeline_names_the_sources_the_stream_carries():
    timeline = Timeline()
    timeline.add(scrape(1.0, preemptions=1), evicted_total=0)
    assert timeline.sources == {"preemptions"}
    timeline.add(scrape(2.0, preemptions=1, queries=1, hits=0, ttft=((INF, 1.0),)), evicted_total=0)
    assert timeline.sources == {"preemptions", "prefix cache", "ttft"}
