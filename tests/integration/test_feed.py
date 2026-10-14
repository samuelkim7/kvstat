from __future__ import annotations

import time
from urllib.parse import urljoin

from kvstat.cli.feed import Batch, Item, LiveFeed, Scrape
from kvstat.engines import vllm
from kvstat.ingest.collector import Collector
from tests.integration.publisher import FakePublisher


def drain_until(feed: LiveFeed, want: int, timeout: float = 5.0) -> list[Item]:
    out: list[Item] = []
    deadline = time.monotonic() + timeout
    while len(out) < want and time.monotonic() < deadline:
        out.extend(feed.drain())
        time.sleep(0.01)
    return out


def test_live_feed_drains_batches_and_scrapes_without_blocking(
    publisher: FakePublisher, real_payloads: list[bytes], vllm_http: str
) -> None:
    tee: list[float] = []
    collector = Collector(publisher.endpoint, replay_endpoint=publisher.replay_endpoint)
    feed = LiveFeed(
        collector, urljoin(vllm_http, vllm.METRICS_PATH), on_scrape=lambda t, _: tee.append(t)
    )
    feed.start()
    try:
        assert collector.ready.wait(5)
        publisher.wait_for_subscriber()
        for payload in real_payloads[:3]:
            publisher.publish(payload)
        items = drain_until(feed, 4)
    finally:
        feed.close()
    batches = [i for i in items if isinstance(i, Batch)]
    scrapes = [i for i in items if isinstance(i, Scrape)]
    assert [b.batch.seq for b in batches] == [0, 1, 2]
    assert scrapes and scrapes[0].samples
    assert tee, "the poller's own callback still fires"
    assert feed.stats is not None and feed.stats.batches == 3
