from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest
from textual.pilot import Pilot
from textual.widget import Widget
from textual.widgets import Static

from kvstat.cli.feed import PlaybackFeed
from kvstat.cli.top import KvstatApp, run_app
from kvstat.events import Sample
from kvstat.state.cache import Health
from kvstat.views.occupancy import OccupancyPanel
from kvstat.views.pressure import PressurePanel
from kvstat.views.status import StatusBar
from tests.unit.factories import Capture, batch, removed, stored


def scrape(used: float, running: int = 0, waiting: int = 0, preemptions: int = 0) -> list[Sample]:
    return [
        ("vllm:kv_cache_usage_perc", {}, used),
        ("vllm:num_requests_running", {}, float(running)),
        ("vllm:num_requests_waiting", {}, float(waiting)),
        ("vllm:num_preemptions_total", {}, float(preemptions)),
    ]


def healthy_run() -> Capture:
    return Capture(
        [
            (0.0, batch(stored(1, 2, 3, 4), ts=0.0, seq=0)),
            (0.5, scrape(0.5, running=1)),
            (1.0, batch(removed(1, 2), ts=1.0, seq=1)),
            (1.5, scrape(0.25, running=1, waiting=2, preemptions=3)),
        ]
    )


def app_over(source: Capture, speed: float = 1e6) -> KvstatApp:
    return KvstatApp(PlaybackFeed(source, speed=speed), source.header, refresh_hz=10)


async def settle(
    app: KvstatApp,
    pilot: Pilot[None],
    until: Callable[[KvstatApp], bool] = lambda app: app.feed.finished,
) -> None:
    for _ in range(100):
        await pilot.pause(0.05)
        if until(app):
            await pilot.pause(0.2)
            return
    raise AssertionError("the app never reached the expected state")


def plain(widget: Widget) -> str:
    renderable = widget.render()
    return renderable.plain if hasattr(renderable, "plain") else str(renderable)


async def test_the_panels_show_the_replayed_state():
    app = app_over(healthy_run())
    async with app.run_test() as pilot:
        await settle(app, pilot)
        assert app.state.snapshot().cached == 2
        assert app.timeline.latest is not None and app.timeline.latest.preemptions == 3
        head = plain(app.query_one("#headline", Static))
        assert head.startswith("25% cached, 25% in use: preempting, 3 just now, 2 waiting")
        assert "in use  2 (25%)" in plain(app.query_one(OccupancyPanel))
        assert "preempted" in plain(app.query_one(PressurePanel))
        status = plain(app.query_one(StatusBar))
        assert status.startswith(" OK ")
        assert "playback x1e+06" in status and "END" in status
        assert "sources: preemptions" in status


async def test_injected_drift_degrades_the_status_bar_and_a_rebuild_recovers_it():
    source = Capture(
        [
            (0.0, batch(stored(*range(12)), ts=0.0, seq=0)),
            (0.5, scrape(0.5)),
            (1.0, batch(stored(20, 21), ts=1.0, seq=1)),
            (1.5, scrape(0.1)),
        ]
    )
    seen: list[Health] = []
    app = app_over(source, speed=1.0)
    async with app.run_test() as pilot:

        def record_health(app: KvstatApp) -> bool:
            health = app.state.snapshot().health
            if not seen or seen[-1] is not health:
                seen.append(health)
            return app.feed.finished

        await settle(app, pilot, record_health)
        assert seen == [Health.WARMING, Health.DEGRADED, Health.OK]
        assert "1 breach(es)" in plain(app.query_one(StatusBar))


async def test_playback_keys_pause_and_step_the_speed():
    source = healthy_run()
    feed = PlaybackFeed(source, speed=1.0)
    app = KvstatApp(feed, source.header, refresh_hz=10)
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        await pilot.press("space")
        assert feed.paused is True
        assert "PAUSED" in plain(app.query_one(StatusBar))
        await pilot.press("plus")
        await pilot.press("plus")
        assert feed.speed == 4.0
        await pilot.press("minus")
        assert feed.speed == 2.0
        await pilot.press("space")
        assert feed.paused is False


@pytest.mark.parametrize("key", ["q", "ctrl+c"])
async def test_quit_keys_leave_the_app(key):
    app = app_over(healthy_run(), speed=1.0)
    async with app.run_test() as pilot:
        await pilot.pause(0.05)
        await pilot.press(key)
        await pilot.pause(0.05)
    assert app.return_code == 0


async def test_a_crash_in_the_tick_ends_the_app_instead_of_hanging():
    class Broken(PlaybackFeed):
        def drain(self):
            raise RuntimeError("boom")

    app = KvstatApp(Broken(healthy_run()), healthy_run().header)
    with pytest.raises(RuntimeError, match="boom"):
        async with app.run_test() as pilot:
            await pilot.pause(0.5)
    assert app.return_code == 1


async def test_sigterm_closes_the_app():
    import os
    import signal

    app = app_over(healthy_run(), speed=1.0)
    async with app.run_test() as pilot:
        await pilot.pause(0.05)
        os.kill(os.getpid(), signal.SIGTERM)
        for _ in range(20):
            await asyncio.sleep(0.05)
            if app._exit:
                break
    assert app.return_code == 0


def test_run_app_turns_a_crashed_app_into_a_failing_exit(monkeypatch):
    app = app_over(healthy_run())

    def crashed_run(**kwargs: object) -> None:
        app._return_code = 1

    monkeypatch.setattr(app, "run", crashed_run)
    with pytest.raises(SystemExit, match="1"):
        run_app(app, "WARNING")
