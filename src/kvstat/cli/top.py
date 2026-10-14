from __future__ import annotations

import asyncio
import logging
import signal
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import urljoin

import click
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.logging import TextualHandler
from textual.widgets import Footer, Static

from kvstat import __version__
from kvstat.cli.common import capture_options, configure_logging, live_options, reporting
from kvstat.cli.feed import Batch, Feed, Item, LiveFeed, PlaybackFeed, Scrape
from kvstat.engines import vllm
from kvstat.ingest.capture import CaptureReader, CaptureWriter, open_for_resume
from kvstat.ingest.collector import Collector, warn_about_replay
from kvstat.ingest.metrics import fetch_server_info
from kvstat.state.cache import CacheState
from kvstat.state.timeline import Timeline
from kvstat.views.headline import render_headline
from kvstat.views.occupancy import OccupancyPanel
from kvstat.views.pressure import PressurePanel
from kvstat.views.status import Playback, StatusBar

log = logging.getLogger("kvstat")

SPEEDS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)


@click.command(short_help="Watch the KV cache live; --out also records it.")
@live_options
@click.option(
    "--from",
    "capture_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Play a capture back instead of attaching to a server.",
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Record the live session to this capture while showing it; a .gz name compresses.",
)
@capture_options
@click.option(
    "--speed",
    type=click.FloatRange(min=0.01),
    default=1.0,
    show_default=True,
    help="Playback speed, in capture seconds per wall second.",
)
@click.option(
    "--refresh-hz",
    type=click.FloatRange(4, 10),
    default=5.0,
    show_default=True,
    help="Screen updates per second.",
)
@click.option("--log-level", default="WARNING", show_default=True)
def top(
    endpoint: str,
    replay_endpoint: str | None,
    server: str,
    topic: str,
    capture_path: Path | None,
    out_path: Path | None,
    redact_tokens: bool,
    overwrite: bool,
    speed: float,
    refresh_hz: float,
    log_level: str,
) -> None:
    """Watch the KV cache of a running vLLM server, or play a capture back.

    Add --out to record the session while you watch it. Play the capture back later with
    --from, or summarize it with kvstat report.
    """
    if capture_path is not None and (out_path or redact_tokens or overwrite):
        raise click.UsageError(
            "--out, --redact-tokens and --overwrite record a live session, not --from"
        )
    configure_logging(log_level)
    with reporting("top"), ExitStack() as stack:
        if capture_path is not None:
            reader = CaptureReader(capture_path)
            feed: Feed = PlaybackFeed(reader, speed=speed)
            header = reader.header
        else:
            feed, header = open_live(
                stack,
                endpoint,
                replay_endpoint,
                server,
                topic,
                out_path,
                redact_tokens=redact_tokens,
                overwrite=overwrite,
            )
        run_app(KvstatApp(feed, header, refresh_hz=refresh_hz), log_level)


def open_live(
    stack: ExitStack,
    endpoint: str,
    replay_endpoint: str | None,
    server: str,
    topic: str,
    out_path: Path | None,
    *,
    redact_tokens: bool,
    overwrite: bool,
) -> tuple[LiveFeed, dict[str, Any]]:
    """Describe the server, open the capture when one is wanted, and build the live feed.

    The capture writer joins the stack so it closes, and flushes, on every exit path.
    """
    header: dict[str, Any] = {
        "kvstat_version": __version__,
        "endpoint": endpoint,
        "replay_endpoint": replay_endpoint,
        "server": server,
        "topic": topic,
        "started_at": time.time(),
        **fetch_server_info(server),
    }
    warn_about_replay(replay_endpoint)
    capture = None
    resume = None
    if out_path is not None:
        resume = None if overwrite else open_for_resume(out_path)
        capture = stack.enter_context(
            CaptureWriter(out_path, header, redact_tokens=redact_tokens, resume=resume is not None)
        )
    collector = Collector(
        endpoint,
        replay_endpoint=replay_endpoint,
        topic=topic,
        on_batch=capture.write_batch if capture else None,
        resume_from=resume.seq if resume else None,
        resume_epoch=resume.epoch if resume else 0,
    )
    feed = LiveFeed(
        collector,
        urljoin(server, vllm.METRICS_PATH),
        on_scrape=capture.write_metrics if capture else None,
    )
    return feed, header


def run_app(app: KvstatApp, log_level: str) -> None:
    """Run the app with logging routed away from the terminal it is drawing on.

    Textual restores the terminal and prints the traceback when the app crashes. The exit code
    must still say it failed.
    """
    logging.basicConfig(level=log_level.upper(), handlers=[TextualHandler()], force=True)
    try:
        app.run()
    finally:
        configure_logging(log_level)
    if app.return_code:
        raise SystemExit(app.return_code)


class KvstatApp(App[None]):
    """Shows a vLLM KV cache as it changes, from a live server or a replayed capture.

    The screen has a headline sentence, the occupancy bar, the pressure panel, the status
    bar and a footer with the keys. One tick per refresh drains the feed, applies every batch
    and scrape to the state, and repaints the panels from a fresh snapshot. The feed's threads
    never touch a widget.

    Args:
        feed: Where batches and scrapes come from. Started on mount, closed on unmount.
        header: The capture header, or its live equivalent. It carries the engine's cache
            config, which gives the pool size and the block size.
        refresh_hz: Ticks per second.
    """

    CSS = """
    #title { height: 1; padding: 0 1; color: $text-muted; }
    #headline { height: 1; padding: 0 1; text-style: bold; }
    OccupancyPanel { height: 4; padding: 0 1; border: round $secondary; }
    PressurePanel { height: auto; padding: 0 1; border: round $secondary; }
    StatusBar { height: 2; padding: 0 1; }
    """
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("q", "quit", "Quit"),
        Binding("ctrl+c", "quit", "Quit", show=False, priority=True),
        Binding("space", "pause", "Pause"),
        Binding("plus", "faster", "Faster"),
        Binding("minus", "slower", "Slower"),
    ]

    def __init__(self, feed: Feed, header: dict[str, Any], *, refresh_hz: float = 5.0) -> None:
        super().__init__()
        self.feed = feed
        self.header = header
        capacity, block_size = vllm.read_cache_config(header.get("cache_config"))
        self.state = CacheState(capacity=capacity, tier=vllm.MEDIUM_GPU)
        self.timeline = Timeline()
        self._block_size = block_size
        self._refresh_hz = refresh_hz
        self._playback = feed if isinstance(feed, PlaybackFeed) else None
        self._used_fraction: float | None = None

    def compose(self) -> ComposeResult:
        """Lay the widgets out top to bottom. Textual calls this once."""
        yield Static(self._title(), id="title")
        yield Static("", id="headline")
        yield OccupancyPanel(self._block_size)
        yield PressurePanel()
        yield StatusBar()
        yield Footer()

    def on_mount(self) -> None:
        """Start the feed's threads and the tick timer once the screen is up."""
        self.title = "kvstat"
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGTERM, self.exit)
        self.feed.start()
        self.set_interval(1 / self._refresh_hz, self.tick)

    def on_unmount(self) -> None:
        """Stop the feed's threads. Every way out of the app passes through here."""
        self.feed.close()

    def tick(self) -> None:
        """Drain the feed, apply every item, then repaint once."""
        for item in self.feed.drain():
            self.apply(item)
        if self.feed.error is not None:
            raise self.feed.error
        self.repaint()

    def apply(self, item: Item) -> None:
        """Add one item to the state.

        A batch goes to the block table. A scrape goes to the timeline and then to the
        cross-check, so the timeline keeps the point even when the scrape contradicts the table.
        """
        match item:
            case Batch():
                self.state.update(item.batch)
            case Scrape():
                counts = vllm.read_engine_counts(item.time, item.samples)
                self._used_fraction = counts.used_fraction
                self.timeline.add(counts, self.state.snapshot().evicted)
                breach = self.state.cross_check(counts)
                if breach is not None:
                    log.warning("drift: %s", breach.detail)

    def repaint(self) -> None:
        """Take one snapshot and hand it to every panel."""
        snap = self.state.snapshot()
        self.query_one("#headline", Static).update(render_headline(snap, self.timeline.latest))
        self.query_one(OccupancyPanel).show(snap, self._used_fraction)
        self.query_one(PressurePanel).show(self.timeline.points, self.timeline.sources)
        self.query_one(StatusBar).show(
            snap, self.feed.stats, self.timeline.sources, self.playback()
        )

    def playback(self) -> Playback | None:
        """Where the replayed capture stands, for the status bar. None on a live stream."""
        feed = self._playback
        if feed is None:
            return None
        return Playback(feed.speed, feed.elapsed, feed.paused, feed.finished)

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Playback keys exist only when a capture is playing."""
        if action in ("pause", "faster", "slower") and self._playback is None:
            return None
        return True

    def action_pause(self) -> None:
        """Hold or resume playback. The space key."""
        if self._playback is not None:
            self._playback.paused = not self._playback.paused
            self._playback.nudge()
            self.repaint()

    def action_faster(self) -> None:
        self._step_speed(+1)

    def action_slower(self) -> None:
        self._step_speed(-1)

    def _step_speed(self, direction: int) -> None:
        """Move playback one step along SPEEDS, and stay put at either end."""
        feed = self._playback
        if feed is None:
            return
        faster = [s for s in SPEEDS if s > feed.speed]
        slower = [s for s in SPEEDS if s < feed.speed]
        if direction > 0 and faster:
            feed.speed = faster[0]
        elif direction < 0 and slower:
            feed.speed = slower[-1]
        feed.nudge()
        self.repaint()

    def _title(self) -> str:
        pool = f"{self.state.capacity:,} blocks" if self.state.capacity else "? blocks"
        if self._block_size:
            pool += f" x {self._block_size} tokens"
        model = self.header.get("model_name") or "unknown model"
        return f"kvstat  ·  {model}  ·  vLLM {self.header.get('vllm_version') or '?'}  ·  {pool}"
