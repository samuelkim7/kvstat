from __future__ import annotations

import dataclasses

from kvstat.events import AllBlocksCleared, BlockStored, EngineCounts, EventBatch
from kvstat.state.cache import CacheState
from kvstat.state.prefixes import PrefixRow, PrefixTable
from tests.unit.factories import batch, removed, stored

TEMPLATE = 1000
# Two sibling blocks whose text matches for the first ten tokens, as below a chat template.
COMMON = tuple(range(100, 110))
WHERE = (*COMMON, 1, 2, 3, 4, 5, 6)
RETURN = (*COMMON, 7, 8, 9, 10, 11, 12)


def replay(*batches: EventBatch, capacity: int = 100, keep_tokens: bool = False) -> CacheState:
    state = CacheState(capacity=capacity, keep_tokens=keep_tokens)
    for b in batches:
        state.update(b)
    return state


def by_root(table: PrefixTable) -> dict[object, PrefixRow]:
    return {row.root: row for row in table.shared}


def texted(event: BlockStored, *blocks: tuple[int, ...]) -> BlockStored:
    return dataclasses.replace(event, token_ids=tuple(t for block in blocks for t in block))


def two_chats() -> EventBatch:
    tail = tuple(range(16))
    return batch(
        texted(stored(TEMPLATE), tail),
        texted(stored(10, 11, parent=TEMPLATE), WHERE, tail),
        texted(stored(12, parent=10), tail),
        texted(stored(20, 21, parent=TEMPLATE), RETURN, tail),
        texted(stored(22, parent=20), tail),
    )


def test_one_request_shares_nothing():
    table = replay(batch(stored(1, 2, 3)), batch(stored(4, parent=3), ts=2.0)).prefixes()
    assert table.shared == ()
    assert table.unshared.blocks == 4
    assert table.unshared.resident_share == 1.0


def test_two_requests_share_the_stretch_above_where_they_diverge():
    table = replay(batch(stored(1, 2, 3), stored(4, 5, parent=2))).prefixes()
    [row] = table.shared
    assert (row.root, row.blocks, row.users) == (1, 2, 2)
    assert table.unshared.blocks == 3


def test_a_chat_template_is_one_small_row_and_the_tails_stay_unshared():
    tails = [stored(10 * i, 10 * i + 1, parent=TEMPLATE) for i in range(1, 6)]
    table = replay(batch(stored(TEMPLATE), *tails)).prefixes()
    [row] = table.shared
    assert (row.root, row.blocks, row.users) == (TEMPLATE, 1, 5)
    assert table.unshared.blocks == 10


def test_a_system_prompt_behind_the_template_is_its_own_row():
    table = replay(
        batch(
            stored(TEMPLATE, 1, 2, 3, 40),
            stored(41, parent=3),
            stored(50, 51, parent=TEMPLATE),
        )
    ).prefixes()
    rows = by_root(table)
    assert (rows[1].blocks, rows[1].users) == (3, 2)
    assert (rows[TEMPLATE].blocks, rows[TEMPLATE].users) == (1, 3)
    assert [row.root for row in table.shared] == [1, TEMPLATE]


def test_a_prompt_that_diverges_in_the_middle_splits_into_two_rows():
    table = replay(
        batch(
            stored(1, 2, 3, 4, 5),
            stored(20, parent=2),
            stored(30, parent=4),
        )
    ).prefixes()
    rows = by_root(table)
    assert (rows[1].blocks, rows[1].users) == (2, 3)
    assert (rows[3].blocks, rows[3].users) == (2, 2)


def test_a_recompute_counts_for_its_row_and_is_not_a_share():
    state = replay(
        batch(stored(1, 2, 3)),
        batch(removed(2, 3), ts=2.0),
        batch(stored(2, 3, parent=1), ts=3.0),
    )
    table = state.prefixes()
    assert table.unshared.recomputes == 2
    assert table.shared == ()


def test_recomputes_count_the_last_minute_only():
    state = replay(
        batch(stored(1, 2), stored(3, parent=1)),
        batch(removed(2), ts=2.0),
        batch(stored(2, parent=1), ts=3.0),
    )
    assert state.prefixes().unshared.recomputes == 1
    state.update(batch(stored(9), ts=70.0))
    assert state.prefixes().unshared.recomputes == 0


def test_hits_are_absent_until_a_resident_block_is_stored_again():
    assert replay(batch(stored(1, 2))).prefixes().unshared.hits is None


def test_a_report_mode_hit_shares_the_block_and_leaves_the_table_count_alone():
    state = replay(batch(stored(1, 2)), batch(stored(1, 2), ts=2.0))
    table = state.prefixes()
    [row] = table.shared
    assert (row.blocks, row.hits) == (2, 2)
    assert table.unshared.hits == 0
    assert state.snapshot().cached == 2
    counts = EngineCounts(time=2.0, used_fraction=0.02, running_requests=0)
    assert state.cross_check(counts) is None


def test_requests_below_a_parent_kvstat_never_saw_make_a_row_with_unknown_blocks():
    table = replay(batch(stored(10, 11, parent=7), stored(20, 21, parent=7))).prefixes()
    [row] = table.shared
    assert (row.root, row.in_table, row.blocks, row.users) == (7, False, 0, 2)
    assert table.unshared.blocks == 4


def test_the_unseen_parent_joins_the_tree_once_it_is_stored():
    state = replay(
        batch(stored(10, parent=7), stored(20, parent=7)),
        batch(stored(5, 6, 7), ts=2.0),
    )
    [row] = state.prefixes().shared
    assert (row.root, row.in_table, row.blocks, row.users) == (5, True, 3, 2)


def test_a_row_leaves_with_its_blocks():
    state = replay(
        batch(stored(1), stored(2, parent=1), stored(3, parent=1)),
        batch(removed(2, 3, 1), ts=2.0),
    )
    table = state.prefixes()
    assert table.shared == ()
    assert table.unshared.blocks == 0
    assert table.written == 3
    assert table.shared_write_share == 1 / 3


def test_the_since_start_write_share_counts_shared_blocks_once_each():
    table = replay(batch(stored(1, 2, 3), stored(4, parent=2), stored(9))).prefixes()
    assert table.written == 5
    assert table.shared_write_share == 2 / 5


def test_a_rebuild_drops_the_tree_and_keeps_the_totals():
    state = replay(batch(stored(1), stored(2, parent=1), stored(3, parent=1)))
    state.update(batch(stored(9), ts=2.0, epoch=1))
    table = state.prefixes()
    assert table.shared == ()
    assert table.unshared.blocks == 1
    assert table.written == 4
    assert table.shared_write_share == 1 / 4


def test_a_clear_empties_the_tree():
    state = replay(batch(stored(1, 2), stored(3, parent=1)), batch(AllBlocksCleared(), ts=2.0))
    table = state.prefixes()
    assert table.shared == () and table.unshared.blocks == 0


def test_blocks_in_another_tier_stay_out_of_the_tree():
    state = CacheState(capacity=100, tier="GPU")
    state.update(batch(stored(1, medium="GPU"), stored(2, medium="CPU")))
    assert state.prefixes().unshared.blocks == 1


def test_token_ids_do_not_change_the_rows():
    plain = [stored(1, 2, 3), stored(4, parent=2), stored(9)]
    with_tokens = [dataclasses.replace(e, token_ids=tuple(range(16))) for e in plain]
    assert replay(batch(*plain)).prefixes() == replay(batch(*with_tokens)).prefixes()


def test_a_row_carries_its_text_from_where_it_differs_from_the_rows_beside_it():
    rows = by_root(replay(two_chats(), keep_tokens=True).prefixes())
    assert rows[10].tokens == WHERE[len(COMMON) :]
    assert rows[20].tokens == RETURN[len(COMMON) :]
    assert rows[TEMPLATE].tokens == tuple(range(16))


def test_a_row_keeps_its_text_after_the_rows_beside_it_leave():
    state = replay(two_chats(), batch(removed(22, 21, 20), ts=2.0), keep_tokens=True)
    [row] = [row for row in state.prefixes().shared if row.root == 10]
    assert row.tokens == WHERE[len(COMMON) :]


def test_prompts_that_start_alike_are_named_from_where_they_differ():
    tail = tuple(range(16))
    state = replay(
        batch(
            texted(stored(1, 11), WHERE, tail),
            texted(stored(12, parent=1), tail),
            texted(stored(2, 21), RETURN, tail),
            texted(stored(22, parent=2), tail),
        ),
        keep_tokens=True,
    )
    rows = by_root(state.prefixes())
    assert rows[1].tokens == WHERE[len(COMMON) :]
    assert rows[2].tokens == RETURN[len(COMMON) :]


def test_rows_carry_no_token_ids_unless_asked():
    assert all(row.tokens == () for row in replay(two_chats()).prefixes().shared)


def test_an_empty_table_has_nothing_written():
    table = CacheState().prefixes()
    assert table.written == 0 and table.shared == () and table.shared_write_share == 0.0


def test_a_shared_prefix_recomputed_after_eviction_keeps_its_row():
    state = replay(
        batch(stored(1, 2), stored(3, parent=2), stored(4, parent=2)),
        batch(removed(3, 4, 2, 1), ts=2.0),
        batch(stored(1, 2, 5), ts=3.0),
    )
    [row] = state.prefixes().shared
    assert (row.root, row.blocks, row.recomputes) == (1, 2, 2)


def test_the_unshared_row_counts_recomputes_whose_blocks_left_again():
    state = replay(
        batch(stored(1, 2)),
        batch(removed(2), ts=2.0),
        batch(stored(2, parent=1), ts=3.0),
        batch(removed(2), ts=4.0),
    )
    assert state.prefixes().unshared.recomputes == 1
