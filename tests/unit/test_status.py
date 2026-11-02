from __future__ import annotations

from kvstat.events import EngineCounts, IngestStats
from kvstat.state.cache import CacheState
from kvstat.views.status import Playback, render_status
from tests.unit.factories import batch, stored


def test_health_leads_and_the_mismatch_detail_comes_last():
    state = CacheState(capacity=2)
    state.update(batch(stored(1, 2, 3)))
    state.cross_check(EngineCounts(time=1.0, used_fraction=0.5, running_requests=0))
    text = render_status(state.snapshot(), None, set(), None).plain
    first, second = text.split("\n")
    assert first.startswith(" DEGRADED ")
    assert "rebuilt 1 time" in first
    assert first.endswith("1 mismatch with engine, last: 3 blocks cached, pool holds 2")
    assert second.endswith("sources: events only")


def test_a_table_that_matches_the_engine_says_so_in_plain_words():
    first = render_status(CacheState().snapshot(), None, set(), None).plain.split("\n")[0]
    assert first.endswith("tracked 0s · rebuilt 0 times · matches engine")
    assert "epoch" not in first and "orphan" not in first


def test_a_live_stream_reports_the_collector_counters():
    stats = IngestStats(
        batches=10,
        last_seq=9,
        epoch=1,
        gaps=2,
        replayed=1,
        resyncs=1,
        decode_errors=0,
        queue_depth=4,
        skipped_event_types={},
    )
    text = render_status(CacheState().snapshot(), stats, {"preemptions", "ttft"}, None).plain
    assert "live  gaps 2  replayed 1  resyncs 1  decode errors 0  queue 4" in text
    assert "sources: preemptions, ttft" in text


def test_playback_reports_speed_position_and_pause():
    playback = Playback(speed=4.0, elapsed=61.4, paused=True, finished=False)
    text = render_status(CacheState().snapshot(), None, set(), playback).plain
    assert "playback x4  t+61s  PAUSED" in text
    ended = Playback(speed=1.0, elapsed=120.0, paused=False, finished=True)
    assert "t+120s  END" in render_status(CacheState().snapshot(), None, set(), ended).plain
