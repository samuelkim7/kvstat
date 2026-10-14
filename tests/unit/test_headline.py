from __future__ import annotations

import dataclasses

from kvstat.state.cache import CacheState
from kvstat.state.timeline import TimelinePoint
from kvstat.views.headline import render_headline
from tests.unit.factories import batch, stored

POINT = TimelinePoint(
    time=1.0,
    used_fraction=0.4,
    running_requests=3,
    waiting_requests=0,
    preemptions=0,
    evictions=0,
    ttft_p50=None,
    ttft_p95=None,
    hit_rate=None,
)


def full_cache(capacity: int = 10) -> CacheState:
    state = CacheState(capacity=capacity)
    state.update(batch(stored(*range(capacity))))
    return state


def test_without_a_pool_size_the_headline_says_what_it_waits_for():
    assert render_headline(CacheState().snapshot(), None) == "waiting for the engine's pool size"


def test_before_any_block_the_headline_says_so():
    assert (
        render_headline(CacheState(capacity=10).snapshot(), POINT) == "waiting for the first block"
    )


def test_the_hit_rate_is_stated_as_a_prefill_share_not_a_verdict():
    point = dataclasses.replace(POINT, hit_rate=0.21)
    text = render_headline(full_cache().snapshot(), point)
    assert text == "100% cached, 40% in use, 21% of prompt tokens served from cache"


def test_an_absent_hit_source_is_left_out_not_shown_as_zero():
    assert render_headline(full_cache().snapshot(), POINT) == "100% cached, 40% in use"


def test_preemptions_take_over_the_sentence():
    point = dataclasses.replace(POINT, preemptions=7, waiting_requests=12, hit_rate=0.9)
    assert render_headline(full_cache().snapshot(), point).endswith(
        "preempting, 7 just now, 12 waiting"
    )


def test_waiting_requests_are_named_when_nothing_is_preempted():
    point = dataclasses.replace(POINT, waiting_requests=4)
    assert render_headline(full_cache().snapshot(), point).endswith(": 4 requests waiting for room")


def test_without_a_scrape_the_headline_has_only_the_cached_share():
    state = CacheState(capacity=10)
    state.update(batch(stored(1, 2, 3)))
    assert render_headline(state.snapshot(), None) == "30% cached"
