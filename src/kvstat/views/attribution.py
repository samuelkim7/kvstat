from __future__ import annotations

from rich.text import Text
from textual.widgets import Static

from kvstat.events import BlockHash
from kvstat.state.prefixes import PrefixRow, PrefixTable
from kvstat.state.timeline import RECENT_WINDOW, TimelinePoint

# Shared rows that fit at 80x24 with the status bar still on screen.
SHARED_ROWS = 3
LABEL_WIDTH = 18
VALUE_WIDTHS = (8, 8, 10, 12, 8)


class AttributionPanel(Static):
    """Panel that shows which shared prefixes the cache holds now."""

    def __init__(self) -> None:
        super().__init__()
        self.border_title = "attribution"

    def show(self, table: PrefixTable, point: TimelinePoint | None) -> None:
        self.update(render_attribution(table, point))


def render_attribution(table: PrefixTable, point: TimelinePoint | None) -> Text:
    """Renders the most resident shared prefixes, the unshared rest, and the hit sources."""
    if not table.written:
        return Text("waiting for the first block")
    hidden = len(table.shared) - SHARED_ROWS
    label = f"prefix +{hidden} more" if hidden > 0 else "prefix"
    recent = f"recomp/{RECENT_WINDOW:.0f}s"
    header = _columns(label, "blocks", "users", "resident", recent, "hits")
    lines = [Text(header, "dim")]
    lines += [_row(_shorten_hash(row.root), row) for row in table.shared[:SHARED_ROWS]]
    lines.append(_row("unshared", table.unshared))
    lines.append(Text(_describe_sources(table, point), "dim"))
    return Text("\n").join(lines)


def _row(label: str, row: PrefixRow) -> Text:
    blocks = _count(row.blocks) if row.in_table else "?"
    users = "-" if row.users is None else _count(row.users)
    hits = "-" if row.hits is None else _count(row.hits)
    resident = f"{row.resident_share * 100:.1f}%" if row.in_table else "?"
    return Text(_columns(label, blocks, users, resident, _count(row.recomputes), hits))


def _columns(label: str, *values: str) -> str:
    """Lays out one line: the label left-aligned, then each value right-aligned to its width."""
    cells = (f"{v:>{w}}" for v, w in zip(values, VALUE_WIDTHS, strict=True))
    return f"{label[: LABEL_WIDTH - 1]:<{LABEL_WIDTH}}" + "".join(cells)


def _count(n: int) -> str:
    """Formats a count in at most six characters, so that it never runs into the next column."""
    if n < 100_000:
        return f"{n:,}"
    if n < 10_000_000:
        return f"{n / 1_000:.0f}k"
    return f"{n / 1_000_000:.1f}M"


def _describe_sources(table: PrefixTable, point: TimelinePoint | None) -> str:
    """Says how much of what was written was shared, and the engine's own hit rate."""
    written = f"written: {table.shared_write_share * 100:.1f}% shared"
    if point is None or point.hit_span is None:
        return f"{written}   engine hit rate: not reported"
    if point.hit_rate is None:
        return f"{written}   engine hit rate: no prompts in {point.hit_span:.0f} s"
    return f"{written}   engine hit rate {point.hit_rate * 100:.0f}% over {point.hit_span:.0f} s"


def _shorten_hash(root: BlockHash | None) -> str:
    """Shortens a block hash to its first eight hex digits."""
    if root is None:
        return "?"
    if isinstance(root, bytes):
        return root[:4].hex()
    return f"{root & 0xFFFF_FFFF_FFFF_FFFF:016x}"[:8]
