from __future__ import annotations

from collections import Counter, OrderedDict, defaultdict, deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

from kvstat.events import BlockHash
from kvstat.state.timeline import RECENT_WINDOW

if TYPE_CHECKING:
    from kvstat.state.cache import BlockKey

# Evicted blocks remembered, with whether they were shared, so a recompute keeps its row.
RECOMPUTE_MEMORY = 60_000


@dataclass(slots=True)
class _Node:
    """One resident block in the prefix tree, or a placeholder for a parent that is not resident."""

    parent: BlockKey | None
    resident: bool = False
    shared: bool = False
    diverges: bool = False
    first_child: BlockKey | None = None
    resident_children: int = 0


@dataclass(slots=True)
class _Tally:
    in_table: bool = True
    blocks: int = 0
    users: int = 0
    recomputes: int = 0
    hits: int = 0


@dataclass(frozen=True, slots=True)
class PrefixRow:
    """One row of the attribution panel: a stretch of shared blocks, or every unshared block.

    Attributes:
        root: Hash of the first block in the stretch. None on the unshared row.
        in_table: False when kvstat does not hold the block where requests diverge. The row's
            blocks are then unknown.
        users: Requests that hold resident blocks below the stretch. None on the unshared row.
        recomputes: Blocks of the row stored again after an eviction, in the last RECENT_WINDOW
            seconds. The unshared row also counts blocks that have been evicted again since.
        hits: Resident blocks of the row stored again while resident, in the same window. Only
            requests with kv_cache_report_mode=full cause these. None until one is seen.
    """

    root: BlockHash | None
    in_table: bool
    blocks: int
    users: int | None
    resident_share: float
    recomputes: int
    hits: int | None


@dataclass(frozen=True, slots=True)
class PrefixTable:
    """The shared prefixes resident at one moment, and the rest of the table in one row.

    Attributes:
        shared: One row per stretch of shared blocks, the most resident first.
        unshared: Every resident block that only one request went through.
        written: Blocks written since kvstat started.
        shared_write_share: The share of those blocks that were shared when they left the
            table, or are shared now.
    """

    shared: tuple[PrefixRow, ...]
    unshared: PrefixRow
    written: int
    shared_write_share: float


class Prefixes:
    """Builds a prefix tree over the resident blocks and finds the stretches requests share.

    A block is shared once two requests went through it: a block below it got a second,
    different child, or vLLM stored it again while it was resident. A row is a stretch of
    shared blocks up to where requests diverge. A parent that is not resident stays in the tree
    as a placeholder until its last resident child leaves.
    """

    def __init__(self) -> None:
        self._nodes: dict[BlockKey, _Node] = {}
        self._shared: set[BlockKey] = set()
        self._unseen: set[BlockKey] = set()
        self._evicted: OrderedDict[BlockKey, bool] = OrderedDict()
        self._recomputes: deque[tuple[float, BlockKey, bool]] = deque()
        self._hits: deque[tuple[float, BlockKey]] = deque()
        self._resident = 0
        self._written = 0
        self._shared_gone = 0
        self._saw_hits = False

    def add(self, key: BlockKey, parent: BlockKey | None, ts: float) -> None:
        """Add one stored block below its parent. A block already resident is a hit."""
        while self._recomputes and self._recomputes[0][0] < ts - RECENT_WINDOW:
            self._recomputes.popleft()
        while self._hits and self._hits[0][0] < ts - RECENT_WINDOW:
            self._hits.popleft()
        node = self._nodes.get(key)
        if node is not None and node.resident:
            self._hits.append((ts, key))
            self._saw_hits = True
            self._mark_shared(key)
            return
        if parent is not None:
            self._attach(parent, key)
        if node is None:
            node = self._nodes[key] = _Node(parent)
        node.parent = parent
        node.resident = True
        self._unseen.discard(key)
        self._resident += 1
        self._written += 1
        was_shared = self._evicted.pop(key, None)
        if was_shared is not None:
            self._recomputes.append((ts, key, was_shared))
            node.shared = node.shared or was_shared
        if node.shared:
            self._shared.add(key)
            self._mark_shared(parent)

    def evict(self, key: BlockKey) -> None:
        """Take one evicted block out of the tree."""
        node = self._nodes.get(key)
        if node is None or not node.resident:
            return
        node.resident = False
        self._resident -= 1
        if node.shared:
            self._shared.discard(key)
            self._shared_gone += 1
        self._evicted[key] = node.shared
        if len(self._evicted) > RECOMPUTE_MEMORY:
            self._evicted.popitem(last=False)
        self._detach(node.parent)
        if not node.resident_children:
            self._drop(key)
        elif node.diverges:
            self._unseen.add(key)

    def clear(self) -> None:
        """Drop the whole tree. The since-start totals stay."""
        self._shared_gone += len(self._shared)
        self._nodes.clear()
        self._shared.clear()
        self._unseen.clear()
        self._resident = 0

    def table(self) -> PrefixTable:
        """Return the shared prefixes and the unshared rest as a frozen value."""
        return _TableBuilder(self).build()

    def _attach(self, parent: BlockKey, child: BlockKey) -> None:
        """Count a new resident child under its parent, and notice when requests diverge there."""
        node = self._nodes.get(parent)
        if node is None:
            node = self._nodes[parent] = _Node(None)
        node.resident_children += 1
        if node.first_child is None:
            node.first_child = child
        elif node.first_child != child and not node.diverges:
            node.diverges = True
            self._mark_shared(parent)
            if not node.resident:
                self._unseen.add(parent)

    def _detach(self, parent: BlockKey | None) -> None:
        """Count one resident child less under its parent, and drop a placeholder left empty."""
        if parent is None or (node := self._nodes.get(parent)) is None:
            return
        node.resident_children -= 1
        if not node.resident and not node.resident_children:
            self._drop(parent)

    def _drop(self, key: BlockKey) -> None:
        del self._nodes[key]
        self._unseen.discard(key)

    def _mark_shared(self, key: BlockKey | None) -> None:
        """Mark a block and every block above it as shared, up to the first one already marked."""
        while key is not None:
            node = self._nodes.get(key)
            if node is None or node.shared:
                return
            node.shared = True
            if node.resident:
                self._shared.add(key)
            key = node.parent


class _TableBuilder:
    """Turns the tree of a Prefixes into finished rows. Lives for one table() call.

    Args:
        prefixes: The tree to read. The builder only reads it.
    """

    def __init__(self, prefixes: Prefixes) -> None:
        self._prefixes = prefixes
        self._nodes = prefixes._nodes
        self._shared = prefixes._shared
        self._unseen = prefixes._unseen
        self._starts: dict[BlockKey, BlockKey] = {}
        self._tallies: dict[BlockKey, _Tally] = {}
        self._unshared = _Tally(blocks=prefixes._resident - len(prefixes._shared))

    def build(self) -> PrefixTable:
        """Group the shared blocks into rows, then count users, recomputes and hits per row."""
        prefixes = self._prefixes
        shared_children = Counter(self._nodes[key].parent for key in self._shared)
        for key in self._shared:
            tally = self._tallies.setdefault(self._find_start(key), _Tally())
            tally.blocks += 1
            tally.users += self._nodes[key].resident_children - shared_children[key]
        for key in self._unseen:
            tally = self._tallies.setdefault(key, _Tally(in_table=False))
            tally.users += self._nodes[key].resident_children - shared_children[key]
        for _, key, was_shared in prefixes._recomputes:
            if key in self._shared:
                self._tallies[self._starts[key]].recomputes += 1
            elif not was_shared:
                self._unshared.recomputes += 1
        for _, key in prefixes._hits:
            if key in self._shared:
                self._tallies[self._starts[key]].hits += 1
        self._roll_up_users()
        ordered = sorted(
            self._tallies.items(), key=lambda item: (item[1].blocks, item[1].users), reverse=True
        )
        written = prefixes._written
        shared_written = prefixes._shared_gone + len(self._shared)
        return PrefixTable(
            shared=tuple(self._row(start[2], tally) for start, tally in ordered),
            unshared=self._row(None, self._unshared),
            written=written,
            shared_write_share=shared_written / written if written else 0.0,
        )

    def _find_start(self, key: BlockKey) -> BlockKey:
        """Return the first block of the stretch that a shared block belongs to."""
        starts = self._starts
        path = []
        while key not in starts:
            parent = self._nodes[key].parent
            if parent not in self._shared or self._nodes[parent].diverges:
                starts[key] = key
                break
            path.append(key)
            key = parent
        for step in path:
            starts[step] = starts[key]
        return starts[key]

    def _roll_up_users(self) -> None:
        """Add the users of every row below a row to that row."""
        below: defaultdict[BlockKey, list[BlockKey]] = defaultdict(list)
        for start in self._tallies:
            above = self._nodes[start].parent
            if above in self._shared:
                below[self._starts[above]].append(start)
            elif above in self._unseen:
                below[above].append(start)

        def total(start: BlockKey) -> int:
            return self._tallies[start].users + sum(total(child) for child in below[start])

        totals = {start: total(start) for start in self._tallies}
        for start, users in totals.items():
            self._tallies[start].users = users

    def _row(self, root: BlockHash | None, tally: _Tally) -> PrefixRow:
        resident = self._prefixes._resident
        return PrefixRow(
            root=root,
            in_table=tally.in_table,
            blocks=tally.blocks,
            users=None if root is None else tally.users,
            resident_share=tally.blocks / resident if resident else 0.0,
            recomputes=tally.recomputes,
            hits=tally.hits if self._prefixes._saw_hits else None,
        )
