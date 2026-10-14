from __future__ import annotations

from rich.text import Text
from textual.widgets import Static

from kvstat.state.cache import CacheSnapshot

IN_USE = "black on bright_cyan"
IDLE = "white on dark_cyan"
FREE = "grey62 on grey19"


class OccupancyPanel(Static):
    """Panel that shows the KV block pool as one bar."""

    def __init__(self, block_size: int | None) -> None:
        super().__init__()
        self._block_size = block_size
        self.border_title = "occupancy"

    def show(self, snap: CacheSnapshot, used_fraction: float | None) -> None:
        width = max(20, self.size.width - 4)
        self.update(render_occupancy(snap, used_fraction, self._block_size, width))


def render_occupancy(
    snap: CacheSnapshot, used_fraction: float | None, block_size: int | None, width: int
) -> Text:
    """Renders the occupancy bar and its legend at the given width.

    In use comes from the engine's metrics, and cached comes from the events. In use can exceed
    cached because the last block of a running request has no hash yet.
    """
    if snap.capacity is None:
        return Text("pool size unknown: the engine did not report num_gpu_blocks")
    capacity = snap.capacity
    in_use = None if used_fraction is None else round(used_fraction * capacity)
    held = max(snap.cached, in_use or 0)
    idle = held - (in_use or 0)
    free = capacity - held
    segments = ((in_use, IN_USE, "in use"), (idle, IDLE, "idle cached"), (free, FREE, "free"))
    cells = _cells(tuple(count or 0 for count, _, _ in segments), width)
    bar = Text()
    for count, (_, style, _) in zip(cells, segments, strict=True):
        bar.append(" " * count, style)
    legend = Text()
    for blocks, style, label in segments:
        legend.append(f" {label} ", style)
        legend.append(f" {_share(blocks, capacity)}   ")
    legend.append(f"pool {capacity:,} blocks")
    if block_size:
        legend.append(f" x {block_size} tokens")
    return Text.assemble(bar, "\n", legend)


def _share(count: int | None, capacity: int) -> str:
    if count is None:
        return "not scraped"
    return f"{count:,} ({count / capacity * 100:.0f}%)"


def _cells(counts: tuple[int, ...], width: int) -> list[int]:
    """Splits the width across the counts so that the segments fill the width exactly."""
    total = sum(counts) or 1
    cells = [round(c / total * width) for c in counts]
    cells[-1] += width - sum(cells)
    return cells
