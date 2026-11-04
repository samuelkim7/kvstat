from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass, field

from kvstat.events import BlockHash

# Sets of first blocks remembered by their leading tokens. Bounds the token ids kept.
MAX_SETS = 64
# A chat template header stays below this, so first blocks that share only one do not count.
MIN_SHARED_TOKENS = 64


@dataclass(frozen=True, slots=True)
class HitSpread:
    """How the prefix hits of a run spread over the first blocks they started from.

    Attributes:
        hits: Prefix hits seen. vLLM reports one only for a request with kv_cache_report_mode=full.
        first_blocks: Distinct first blocks the hits started from.
        top: Hits that started from the most used first block.
        fewest_blocks: Blocks in the shortest hit.
        most_blocks: Blocks in the longest hit.
    """

    hits: int
    first_blocks: int
    top: int
    fewest_blocks: int
    most_blocks: int


@dataclass(frozen=True, slots=True)
class SplitPrefix:
    """First blocks that start with the same tokens and still differ, so no prompt reuses another's.

    Attributes:
        first_blocks: Distinct first blocks that share the leading tokens.
        shared_tokens: Leading tokens they all share.
        block_size: Tokens in one block.
    """

    first_blocks: int
    shared_tokens: int
    block_size: int


@dataclass(slots=True)
class _Set:
    tokens: tuple[int, ...]
    shared: int
    blocks: set[BlockHash] = field(default_factory=set)


class FirstBlocks:
    """Tracks the first block of every prompt: the hits that start there, and its leading tokens.

    First blocks join one set when they share at least half a block of leading tokens. A set of
    two or more is a shared prefix that the block boundary cut off.
    """

    def __init__(self) -> None:
        self._hits: Counter[BlockHash] = Counter()
        self._lengths: Counter[int] = Counter()
        self._sets: list[_Set] = []

    def add_hit(self, first: BlockHash, blocks: int) -> None:
        """Count one prefix hit that reused blocks from first on."""
        self._hits[first] += 1
        self._lengths[blocks] += 1

    def add_first(self, block: BlockHash, tokens: tuple[int, ...]) -> None:
        """Note the leading tokens of a first block that kvstat did not hold."""
        need = max(len(tokens) // 2, MIN_SHARED_TOKENS)
        for found in self._sets:
            shared = len(os.path.commonprefix([found.tokens, tokens]))
            if shared >= need:
                found.shared = min(found.shared, shared)
                found.blocks.add(block)
                return
        if len(self._sets) == MAX_SETS:
            alone = next((s for s in self._sets if len(s.blocks) == 1), None)
            if alone is None:
                return
            self._sets.remove(alone)
        self._sets.append(_Set(tokens, len(tokens), {block}))

    def spread(self) -> HitSpread | None:
        """Return how the hits spread, or None before the first hit."""
        if not self._hits:
            return None
        return HitSpread(
            hits=self._hits.total(),
            first_blocks=len(self._hits),
            top=self._hits.most_common(1)[0][1],
            fewest_blocks=min(self._lengths),
            most_blocks=max(self._lengths),
        )

    def split_prefix(self) -> SplitPrefix | None:
        """Return the largest set of first blocks that share leading tokens, or None."""
        largest = max(self._sets, key=lambda s: len(s.blocks), default=None)
        if largest is None or len(largest.blocks) < 2:
            return None
        return SplitPrefix(
            first_blocks=len(largest.blocks),
            shared_tokens=largest.shared,
            block_size=len(largest.tokens),
        )
