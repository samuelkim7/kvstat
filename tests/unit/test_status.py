from __future__ import annotations

from kvstat.events import EngineCounts, IngestStats
from kvstat.state.cache import CacheState
from kvstat.views.status import Playback, render_status
from tests.unit.factories import batch, stored


def test_health_and_drift_lead_the_first_line():
    state = CacheState(capacity=2)
    state.update(batch(stored(1, 2, 3)))
    state.cross_check(EngineCounts(time=1.0, used_fraction=0.5, running_requests=0))
    text = render_status(state.snapshot(), None, set(), None).plain
    first, second = text.split("\n")
    assert first.startswith(" DEGRADED ")
    assert "drift 1 breach(es), last: 3 blocks cached, pool holds 2" in first
    assert "1 rebuild(s)" in first
    assert second.endswith("sources: events only")


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
