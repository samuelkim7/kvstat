from __future__ import annotations

from dataclasses import dataclass

from rich.text import Text
from textual.widgets import Static

from kvstat.events import IngestStats
from kvstat.state.cache import CacheSnapshot, Health
from kvstat.views.console import describe_drift

HEALTH_STYLE = {Health.OK: "bold green", Health.WARMING: "bold yellow", Health.DEGRADED: "bold red"}


@dataclass(frozen=True, slots=True)
class Playback:
    """Playback progress of a capture file.

    Attributes:
        speed: Playback speed as a multiple of real time.
        elapsed: Seconds of the capture played so far. None before the first row.
        paused: Whether playback is paused.
        finished: Whether playback reached the end of the capture.
    """

    speed: float
    elapsed: float | None
    paused: bool
    finished: bool


class StatusBar(Static):
    """Bottom bar that shows whether the cache numbers can be trusted."""

    def show(
        self,
        snap: CacheSnapshot,
        stats: IngestStats | None,
        sources: set[str],
        playback: Playback | None,
    ) -> None:
        self.update(render_status(snap, stats, sources, playback))


def render_status(
    snap: CacheSnapshot, stats: IngestStats | None, sources: set[str], playback: Playback | None
) -> Text:
    """Renders the status bar: cache health on top, live stream or playback below."""
    first = Text()
    first.append(f" {snap.health.value.upper()} ", HEALTH_STYLE[snap.health])
    first.append(f"  state {snap.elapsed:.0f}s")
    first.append(f"  ·  drift {describe_drift(snap)}")
    first.append(f"  ·  epoch {snap.epoch}, {snap.rebuilds} rebuild(s)")
    first.append(f"  ·  {snap.orphan_stores:,} orphan stores")
    second = Text()
    if stats is not None:
        second.append(
            f"live  gaps {stats.gaps}  replayed {stats.replayed}  resyncs {stats.resyncs}  "
            f"decode errors {stats.decode_errors}  queue {stats.queue_depth}"
        )
    if playback is not None:
        second.append(f"playback x{playback.speed:g}")
        if playback.elapsed is not None:
            second.append(f"  t+{playback.elapsed:.0f}s")
        if playback.finished:
            second.append("  END", "bold")
        elif playback.paused:
            second.append("  PAUSED", "bold")
    second.append(f"  ·  sources: {', '.join(sorted(sources)) or 'events only'}")
    return Text.assemble(first, "\n", second)
