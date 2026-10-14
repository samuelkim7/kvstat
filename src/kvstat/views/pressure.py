from __future__ import annotations

from collections.abc import Sequence

from rich.text import Text
from textual.widgets import Static

from kvstat.state.timeline import TimelinePoint

BARS = "▁▂▃▄▅▆▇█"
LABEL_WIDTH = 10
VALUE_WIDTH = 12


class PressurePanel(Static):
    """Panel that charts scheduler pressure over the recent scrapes."""

    def __init__(self) -> None:
        super().__init__()
        self.border_title = "pressure"

    def show(self, points: Sequence[TimelinePoint], sources: set[str]) -> None:
        width = max(30, self.size.width - 4)
        self.update(render_pressure(points, sources, width))


def render_pressure(points: Sequence[TimelinePoint], sources: set[str], width: int) -> Text:
    """Renders one sparkline row per metric, with the newest scrape on the right.

    A metric that the engine does not report is left out, not drawn flat.
    """
    if not points:
        return Text("waiting for the first scrape")
    span = width - LABEL_WIDTH - VALUE_WIDTH
    recent = list(points)[-span:]
    latest = recent[-1]
    used = [p.used_fraction for p in recent]
    waiting = [float(p.waiting_requests) for p in recent]
    evicted = [float(p.evictions) for p in recent]
    rows = [
        _line("in use", _sparkline(used, top=1.0), f"{latest.used_fraction * 100:.0f}%", span),
        _line("waiting", _sparkline(waiting), f"{latest.waiting_requests}", span),
        _line("evicted", _sparkline(evicted), _evictions_per_second(recent), span),
    ]
    if "preemptions" in sources:
        marks = Text()
        for p in recent:
            if p.preemptions:
                marks.append("▲", "bold red")
            else:
                marks.append("·", "dim")
        total = sum(p.preemptions or 0 for p in recent)
        rows.append(_line("preempted", marks, f"{total} in view", span))
    if "ttft" in sources:
        ttft = [p.ttft_p95 or 0.0 for p in recent]
        value = "-" if latest.ttft_p95 is None else f"{latest.ttft_p95:.2f}s"
        rows.append(_line("ttft p95", _sparkline(ttft), value, span))
    return Text("\n").join(rows)


def _evictions_per_second(recent: list[TimelinePoint]) -> str:
    if len(recent) < 2:
        return "-"
    seconds = recent[-1].time - recent[-2].time
    if seconds <= 0:
        return "-"
    return f"{recent[-1].evictions / seconds:,.0f}/s"


def _sparkline(values: list[float], top: float | None = None) -> Text:
    """Draws each value as a bar against top, which defaults to the largest value."""
    top = top or max(values) or 1.0
    bars = "".join(_bar(v, top) for v in values)
    return Text(bars, "cyan")


def _bar(value: float, top: float) -> str:
    """Picks the bar character for value, with top drawn as the tallest bar."""
    level = int(value / top * len(BARS))
    return BARS[min(level, len(BARS) - 1)]


def _line(label: str, chart: Text, value: str, span: int) -> Text:
    """Lays out one row: label on the left, chart right-aligned to span, then the value."""
    chart.pad_left(span - len(chart))
    return Text.assemble(f"{label:<{LABEL_WIDTH}}", chart, f" {value:>{VALUE_WIDTH - 1}}")
