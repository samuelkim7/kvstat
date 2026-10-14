from __future__ import annotations

from rich.text import Text

from kvstat.state.cache import CacheState
from kvstat.views.occupancy import FREE, IDLE, IN_USE, render_occupancy
from tests.unit.factories import batch, stored


def state_with(cached: int, capacity: int = 100) -> CacheState:
    state = CacheState(capacity=capacity)
    state.update(batch(stored(*range(cached))))
    return state


def segment_widths(text: Text) -> dict[str, int]:
    bar = text.split("\n")[0]
    widths = {style: 0 for style in (IN_USE, IDLE, FREE)}
    for span in bar.spans:
        widths[str(span.style)] += span.end - span.start
    return widths


def test_the_bar_splits_the_width_between_in_use_idle_and_free():
    text = render_occupancy(state_with(60).snapshot(), 0.25, 16, width=100)
    assert segment_widths(text) == {IN_USE: 25, IDLE: 35, FREE: 40}
    legend = text.plain.split("\n")[1]
    assert "in use  25 (25%)" in legend
    assert "idle cached  35 (35%)" in legend
    assert "free  40 (40%)" in legend
    assert legend.endswith("pool 100 blocks x 16 tokens")


def test_segments_always_add_up_to_the_width():
    text = render_occupancy(state_with(33).snapshot(), 0.1, None, width=71)
    assert sum(segment_widths(text).values()) == 71


def test_in_use_above_cached_leaves_no_idle_and_shrinks_free():
    text = render_occupancy(state_with(20).snapshot(), 0.5, None, width=100)
    assert segment_widths(text) == {IN_USE: 50, IDLE: 0, FREE: 50}


def test_without_a_scrape_in_use_is_marked_not_scraped():
    text = render_occupancy(state_with(20).snapshot(), None, None, width=100)
    assert "in use  not scraped" in text.plain
    assert segment_widths(text) == {IN_USE: 0, IDLE: 20, FREE: 80}


def test_without_a_pool_size_the_panel_says_why():
    assert "num_gpu_blocks" in render_occupancy(CacheState().snapshot(), 0.5, 16, 80).plain
