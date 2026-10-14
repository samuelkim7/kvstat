from __future__ import annotations

import queue
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

from kvstat.events import EventBatch, IngestStats, Sample
from kvstat.ingest.capture import CaptureSource
from kvstat.ingest.collector import Collector
from kvstat.ingest.metrics import MetricsPoller, ScrapeCallback


@dataclass(frozen=True, slots=True)
class Batch:
    """One event batch and the kvstat clock time it arrived."""

    time: float
    batch: EventBatch


@dataclass(frozen=True, slots=True)
class Scrape:
    """One metrics scrape and the kvstat clock time it was taken."""

    time: float
    samples: list[Sample]


Item = Batch | Scrape


class Feed(Protocol):
    """Where the app gets its event batches and metrics scrapes.

    The app calls drain() once per tick and gets everything that arrived since the last call.
    Between calls, the feed gathers those items on its own threads. A live server and a capture
    file are two feeds with the same methods, so the app draws either one without knowing which.
    """

    def start(self) -> None: ...

    def close(self) -> None: ...

    def drain(self) -> list[Item]: ...

    @property
    def finished(self) -> bool: ...

    @property
    def stats(self) -> IngestStats | None: ...

    @property
    def error(self) -> BaseException | None: ...


def merge(source: CaptureSource) -> Iterator[Item]:
    """Yield a capture's batches and scrapes in arrival order.

    Both rows carry kvstat's own clock, so they merge directly. One poller thread writes the
    scrapes, so they arrive in scrape order and need no sorting.
    """
    scrapes = source.metrics()
    pending = next(scrapes, None)
    for arrived, batch in source.batches():
        while pending is not None and pending[0] <= arrived:
            yield Scrape(*pending)
            pending = next(scrapes, None)
        yield Batch(arrived, batch)
    while pending is not None:
        yield Scrape(*pending)
        pending = next(scrapes, None)


class LiveFeed:
    """The feed for a running vLLM server.

    Batches come from the collector's socket thread. Scrapes come from a metrics poller this
    feed owns, on its thread. start() launches both threads and close() stops them. drain()
    returns what the two threads put aside since the last drain.

    Args:
        collector: The event stream. Not yet started.
        metrics_url: The engine's metrics endpoint, scraped once a second.
        on_scrape: Also called with every scrape. kvstat top passes CaptureWriter.write_metrics
            when it records.
    """

    def __init__(
        self, collector: Collector, metrics_url: str, *, on_scrape: ScrapeCallback | None = None
    ) -> None:
        self._collector = collector
        self._poller = MetricsPoller(metrics_url, self._on_scrape)
        self._tee = on_scrape
        self._scrapes: queue.SimpleQueue[Scrape] = queue.SimpleQueue()

    def start(self) -> None:
        self._collector.start()
        self._poller.start()

    def close(self) -> None:
        self._collector.close()
        self._poller.close()

    def drain(self) -> list[Item]:
        now = time.time()
        items: list[Item] = [Batch(now, b) for b in self._collector.drain()]
        while True:
            try:
                items.append(self._scrapes.get_nowait())
            except queue.Empty:
                return items

    @property
    def finished(self) -> bool:
        return self._collector.finished

    @property
    def stats(self) -> IngestStats | None:
        return self._collector.stats()

    @property
    def error(self) -> BaseException | None:
        return self._collector.error

    def _on_scrape(self, scraped: float, samples: list[Sample]) -> None:
        if self._tee is not None:
            self._tee(scraped, samples)
        self._scrapes.put(Scrape(scraped, samples))


class PlaybackFeed:
    """The feed for a capture file, played on the capture's own clock.

    A daemon thread walks the capture's rows and releases each one when its recorded time comes,
    divided by speed. start() launches the thread and close() stops it. drain() returns the rows
    released since the last drain.

    Args:
        source: The capture to play.
        speed: Capture seconds per wall second. Changing it takes effect at the next row.

    Attributes:
        speed: The current playback speed.
        paused: While True the thread holds the next row.
    """

    def __init__(self, source: CaptureSource, *, speed: float = 1.0) -> None:
        self._source = source
        self.speed = speed
        self.paused = False
        self._items: queue.SimpleQueue[Item | None] = queue.SimpleQueue()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="kvstat-playback", daemon=True)
        self._first: float | None = None
        self._played: float | None = None
        self._finished = False

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2)

    def drain(self) -> list[Item]:
        out: list[Item] = []
        while True:
            try:
                item = self._items.get_nowait()
            except queue.Empty:
                return out
            if item is None:
                self._finished = True
                return out
            out.append(item)
            self._played = item.time

    def nudge(self) -> None:
        """Wake the thread so a new speed or pause state applies now, not at the next row."""
        self._wake.set()

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def stats(self) -> IngestStats | None:
        return None

    @property
    def error(self) -> BaseException | None:
        return None

    @property
    def elapsed(self) -> float | None:
        """Capture seconds played so far. None before the first row."""
        if self._first is None or self._played is None:
            return None
        return self._played - self._first

    def _run(self) -> None:
        anchor: tuple[float, float] | None = None
        for item in merge(self._source):
            if self._first is None:
                self._first = item.time
            if anchor is None:
                anchor = (time.monotonic(), item.time)
            anchor = self._wait(anchor, item.time)
            if anchor is None:
                break
            self._items.put(item)
        self._items.put(None)

    def _wait(self, anchor: tuple[float, float], capture_time: float) -> tuple[float, float] | None:
        """Hold until the row is due, then return the anchor to keep. None once stopped.

        The anchor pairs a wall time with the capture time that was due then. A row is due at
        anchor plus its capture distance over the speed, so the time spent decoding a row is
        not added to the wait after it. A speed change or a pause moves the anchor to now.
        """
        wall, capture = anchor
        speed = self.speed
        while not self._stop.is_set():
            if self.paused:
                self._wake.clear()
                self._wake.wait()
                wall, capture, speed = time.monotonic(), capture_time, self.speed
                continue
            if self.speed != speed:
                wall, capture, speed = time.monotonic(), capture_time, self.speed
            remaining = wall + (capture_time - capture) / speed - time.monotonic()
            if remaining <= 0:
                return wall, capture
            self._wake.clear()
            self._wake.wait(remaining)
        return None
