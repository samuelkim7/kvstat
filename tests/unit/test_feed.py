from __future__ import annotations

import time
from collections.abc import Iterator

from kvstat.cli.feed import Batch, Item, PlaybackFeed, Scrape, merge
from kvstat.events import EventBatch, Sample
from tests.unit.factories import Capture, batch


def rows_at(batch_times: list[float], scrape_times: list[float]) -> Capture:
    """A capture with empty batches and empty scrapes at the given times."""
    rows: list[tuple[float, EventBatch | list[Sample]]] = [
        (t, batch(ts=t, seq=i)) for i, t in enumerate(batch_times)
    ]
    rows += [(t, []) for t in scrape_times]
    return Capture(sorted(rows, key=lambda r: (r[0], isinstance(r[1], EventBatch))))


def times(items: list[Item]) -> list[tuple[str, float]]:
    return [(type(i).__name__, i.time) for i in items]


def drain_for(feed: PlaybackFeed, seconds: float) -> list[Item]:
    out: list[Item] = []
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        out.extend(feed.drain())
        if feed.finished:
            break
        time.sleep(0.01)
    return out


def test_merge_interleaves_scrapes_and_batches_by_arrival():
    items = list(merge(rows_at([0.0, 1.0, 2.0], [0.5, 1.0, 5.0])))
    assert times(items) == [
        ("Batch", 0.0),
        ("Scrape", 0.5),
        ("Scrape", 1.0),
        ("Batch", 1.0),
        ("Batch", 2.0),
        ("Scrape", 5.0),
    ]
    assert isinstance(items[0], Batch) and isinstance(items[1], Scrape)


def test_playback_delivers_every_row_in_order_and_reports_capture_time():
    feed = PlaybackFeed(rows_at([10.0, 11.0, 12.0], [10.5]), speed=1e6)
    feed.start()
    items = drain_for(feed, 2.0)
    assert [i.time for i in items] == [10.0, 10.5, 11.0, 12.0]
    assert feed.finished
    assert feed.elapsed == 2.0
    assert feed.stats is None


def test_playback_honours_the_capture_clock_scaled_by_speed():
    feed = PlaybackFeed(rows_at([0.0, 0.4], []), speed=1.0)
    feed.start()
    try:
        assert [i.time for i in drain_for(feed, 0.15)] == [0.0]
        assert [i.time for i in drain_for(feed, 1.0)] == [0.4]
    finally:
        feed.close()


def test_a_faster_speed_shortens_the_gap():
    feed = PlaybackFeed(rows_at([0.0, 0.8], []), speed=8.0)
    feed.start()
    try:
        assert [i.time for i in drain_for(feed, 0.5)] == [0.0, 0.8]
    finally:
        feed.close()


def test_pause_holds_the_next_row_until_nudged():
    feed = PlaybackFeed(rows_at([0.0, 0.01, 0.02], []), speed=1e6)
    feed.paused = True
    feed.start()
    try:
        assert drain_for(feed, 0.1) == []
        feed.paused = False
        feed.nudge()
        assert [i.time for i in drain_for(feed, 1.0)] == [0.0, 0.01, 0.02]
    finally:
        feed.close()


def test_close_stops_a_playback_that_is_still_waiting():
    feed = PlaybackFeed(rows_at([0.0, 100.0], []), speed=1.0)
    feed.start()
    drain_for(feed, 0.05)
    started = time.monotonic()
    feed.close()
    assert time.monotonic() - started < 1.0


def test_decode_time_does_not_slow_the_capture_clock():
    class Slow(Capture):
        def batches(self) -> Iterator[tuple[float, EventBatch]]:
            for t, row in super().batches():
                time.sleep(0.03)
                yield t, row

    feed = PlaybackFeed(Slow([(i * 0.05, batch(ts=i * 0.05, seq=i)) for i in range(10)]), speed=1.0)
    started = time.monotonic()
    feed.start()
    items = drain_for(feed, 3.0)
    assert len(items) == 10 and feed.finished
    # Ten rows 50 ms apart take 450 ms on the capture clock. Adding 30 ms of decode per row
    # to each wait would take 720 ms.
    assert time.monotonic() - started < 0.62
