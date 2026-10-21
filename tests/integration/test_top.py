from __future__ import annotations

import re
from pathlib import Path

from textual.widgets import Static

from kvstat.cli.feed import PlaybackFeed
from kvstat.cli.top import KvstatApp
from kvstat.ingest.capture import CaptureReader
from kvstat.state.cache import Health
from kvstat.views.attribution import AttributionPanel
from kvstat.views.status import StatusBar


async def test_the_storm_fixture_plays_to_the_end_with_the_state_agreeing(fixtures: Path) -> None:
    reader = CaptureReader(fixtures / "mixed-storm-120s.jsonl.gz")
    feed = PlaybackFeed(reader, speed=1e6)
    app = KvstatApp(feed, reader.header, refresh_hz=10)
    async with app.run_test(size=(120, 30)) as pilot:
        for _ in range(600):
            await pilot.pause(0.1)
            if feed.finished:
                break
        await pilot.pause(0.3)
        snap = app.state.snapshot()
        assert feed.finished and snap.health is Health.OK and snap.breaches == 0
        assert app.timeline.sources == {"preemptions", "prefix cache"}
        head = str(app.query_one("#headline", Static).render())
        assert head.startswith("100% cached")
        status = str(app.query_one(StatusBar).render())
        assert " OK " in status and "END" in status
        assert "sources: preemptions, prefix cache" in status
        panel = str(app.query_one(AttributionPanel).render())
        assert "unshared" in panel
        assert re.search(r"engine hit rate \d+% over \d+ s", panel)


async def test_the_status_bar_stays_on_screen_at_80_by_24(fixtures: Path) -> None:
    reader = CaptureReader(fixtures / "mixed-storm-120s.jsonl.gz")
    feed = PlaybackFeed(reader, speed=1e6)
    app = KvstatApp(feed, reader.header, refresh_hz=10)
    async with app.run_test(size=(80, 24)) as pilot:
        for _ in range(600):
            await pilot.pause(0.1)
            if feed.finished:
                break
        app.timeline.sources.add("ttft")
        await pilot.pause(0.3)
        assert app.query_one(StatusBar).region.bottom <= 24 - 1
