from __future__ import annotations

import dataclasses

from kvstat.engines import vllm
from kvstat.events import BlockStored, EventBatch
from kvstat.state.cache import CacheState
from kvstat.state.crosscheck import EngineWindow
from kvstat.views.console import format_summary
from tests.unit.factories import batch, removed, stored

BLOCK = 256
SYSTEM = tuple(range(1000, 1200))


def first(
    block: int, *rest: int, tail: int, group: int = 0, kind: str = "full_attention"
) -> BlockStored:
    """A store from a prompt's first block on, whose text is SYSTEM and then its own tokens."""
    tokens = (SYSTEM + (tail,) * BLOCK)[:BLOCK] + (0,) * BLOCK * len(rest)
    event = stored(block, *rest, group=group, kind=kind)
    return dataclasses.replace(event, token_ids=tokens, block_size=BLOCK)


def replay(*batches: EventBatch, keep_tokens: bool = True) -> CacheState:
    state = CacheState(
        capacity=100, keep_tokens=keep_tokens, contiguous_kinds=vllm.CONTIGUOUS_KINDS
    )
    for b in batches:
        state.update(b)
    return state


def test_a_store_of_held_blocks_from_the_first_block_on_is_a_hit():
    state = replay(
        batch(first(1, 2, 3, tail=7)),
        batch(first(1, tail=7), ts=2.0),
        batch(first(1, 2, 3, tail=7), ts=3.0),
    )
    hits = state.hits()
    assert hits is not None
    assert (hits.hits, hits.first_blocks, hits.top) == (2, 1, 2)
    assert (hits.fewest_blocks, hits.most_blocks) == (1, 3)


def test_a_first_block_stored_again_after_its_eviction_is_no_hit():
    state = replay(
        batch(first(1, tail=7)), batch(removed(1), ts=2.0), batch(first(1, tail=7), ts=3.0)
    )
    assert state.hits() is None


def test_hits_count_from_the_full_attention_group_alone():
    state = replay(
        batch(first(5, group=1, kind="mamba", tail=7), first(1, tail=7)),
        batch(first(5, group=1, kind="mamba", tail=7), first(1, tail=7), ts=2.0),
    )
    hits = state.hits()
    assert hits is not None
    assert hits.hits == 1


def test_first_blocks_that_share_most_of_a_block_name_the_cut_prefix():
    state = replay(batch(first(1, tail=7), first(2, tail=8), first(3, tail=9)))
    split = state.split_prefix()
    assert split is not None
    assert (split.first_blocks, split.shared_tokens, split.block_size) == (3, len(SYSTEM), BLOCK)


def test_first_blocks_that_share_only_a_template_header_name_nothing():
    header = tuple(range(30))

    def prompt(block: int, tail: int) -> BlockStored:
        event = stored(block)
        return dataclasses.replace(
            event, token_ids=(header + (tail,) * BLOCK)[:BLOCK], block_size=BLOCK
        )

    assert replay(batch(prompt(1, 7), prompt(2, 8))).split_prefix() is None


def test_a_redacted_capture_counts_hits_and_names_no_prefix():
    def bare(*hashes: int) -> BlockStored:
        return dataclasses.replace(stored(*hashes), block_size=BLOCK)

    state = replay(batch(bare(1), bare(2), bare(3)), batch(bare(1), ts=2.0))
    hits = state.hits()
    assert hits is not None
    assert hits.hits == 1
    assert state.split_prefix() is None


def test_the_report_names_the_spread_and_the_cut_prefix():
    state = replay(
        batch(first(1, tail=7), first(2, tail=8)),
        batch(first(1, tail=7), ts=2.0),
        batch(first(2, tail=8), ts=3.0),
        batch(first(2, tail=8), ts=4.0),
    )
    text = format_summary(
        {}, state.snapshot(), BLOCK, EngineWindow(), state.hits(), state.split_prefix()
    )
    assert "prefix hits      3  ·  from 2 first blocks, top 2 (67%)  ·  1 block per hit" in text
    assert "findings" in text
    finding = " ".join(text.split("findings", 1)[1].split())
    assert "2 first blocks start with the same 200 tokens, but a block holds 256" in finding


def test_the_report_says_why_it_saw_no_hits():
    state = replay(batch(first(1, tail=7)))
    text = format_summary({}, state.snapshot(), BLOCK, EngineWindow(), state.hits())
    assert "prefix hits      none seen   (needs kv_cache_report_mode: full" in text
    assert "findings" not in text
