from __future__ import annotations

import logging
import time
from typing import Any

import pytest
import zmq

from kvstat.engines import vllm
from kvstat.events import EventBatch
from kvstat.ingest.collector import Collector, probe_replay
from tests.integration.conftest import FakePublisher


def collect(collector: Collector, n: int, timeout: float = 5.0) -> list[EventBatch]:
    got: list[EventBatch] = []
    deadline = time.monotonic() + timeout
    it = iter(collector)
    while len(got) < n and time.monotonic() < deadline:
        try:
            got.append(next(it))
        except StopIteration:
            break
    return got


def start(publisher: FakePublisher, **kwargs: Any) -> Collector:
    collector = Collector(
        publisher.endpoint, replay_endpoint=publisher.replay_endpoint, replay_timeout=1.0, **kwargs
    ).start()
    assert collector.ready.wait(5)
    publisher.wait_for_subscriber()
    return collector


def test_gap_is_recovered_through_replay(publisher, real_payloads, caplog):
    caplog.set_level(logging.INFO, logger="kvstat.ingest.collector")
    frames = real_payloads
    collector = start(publisher)
    try:
        for i in range(10):
            publisher.publish(frames[i % 3], drop=i in (5, 6))
        batches = collect(collector, 10)
    finally:
        collector.close()

    assert [b.seq for b in batches] == list(range(10))
    assert {b.epoch for b in batches} == {0}
    stats = collector.stats()
    assert (stats.gaps, stats.replayed, stats.resyncs, stats.batches) == (1, 2, 0, 10)
    assert stats.last_seq == 9
    messages = [r.getMessage() for r in caplog.records]
    assert any(m.startswith("gap: expected seq 5, got 7") for m in messages)
    assert any(m.startswith("replay: recovered 2 batches [5, 7)") for m in messages)


def test_short_replay_forces_resync(publisher, real_payloads, caplog):
    caplog.set_level(logging.INFO, logger="kvstat.ingest.collector")
    frames = real_payloads
    collector = start(publisher)
    try:
        for i in range(10):
            # seq 5 never reaches the replay buffer, so the gap [5, 7) cannot be filled
            publisher.publish(frames[i % 3], drop=i in (5, 6), buffer=i != 5)
        batches = collect(collector, 8)
    finally:
        collector.close()

    assert [b.seq for b in batches] == [0, 1, 2, 3, 4, 7, 8, 9]
    assert [b.epoch for b in batches] == [0] * 5 + [1] * 3
    stats = collector.stats()
    assert (stats.gaps, stats.replayed, stats.resyncs, stats.epoch) == (1, 0, 1, 1)
    assert any(
        r.levelno == logging.WARNING and r.getMessage().startswith("resync: seq [5, 7)")
        for r in caplog.records
    )


def test_no_replay_endpoint_means_resync(publisher_without_replay, real_payloads):
    frames = real_payloads
    collector = start(publisher_without_replay)
    try:
        for i in range(6):
            publisher_without_replay.publish(frames[i % 3], drop=i == 2)
        batches = collect(collector, 5)
    finally:
        collector.close()

    assert [b.seq for b in batches] == [0, 1, 3, 4, 5]
    assert [b.epoch for b in batches] == [0, 0, 1, 1, 1]
    assert collector.stats().resyncs == 1


def test_on_batch_sees_every_raw_batch_including_replayed(publisher, real_payloads):
    frames = real_payloads
    seen: list[tuple[int, int, bytes]] = []
    collector = start(
        publisher,
        on_batch=lambda t, seq, epoch, payload: seen.append((seq, epoch, payload)),
    )
    try:
        for i in range(6):
            publisher.publish(frames[i % 3], drop=i == 3)
        collect(collector, 6)
    finally:
        collector.close()

    assert [s for s, _, _ in seen] == list(range(6))
    assert all(p == frames[s % 3] for s, _, p in seen)


def test_req_client_silently_gets_one_batch_where_dealer_gets_all(publisher, real_payloads):
    """vLLM's example subscriber uses REQ; against the real protocol it truncates to one batch."""
    frames = real_payloads
    for i in range(4):
        publisher.publish(frames[i % 3], drop=True)
    ctx: zmq.Context[zmq.Socket[bytes]] = zmq.Context.instance()

    req = ctx.socket(zmq.REQ)
    req.setsockopt(zmq.RCVTIMEO, 2000)
    req.setsockopt(zmq.LINGER, 0)
    req.connect(publisher.replay_endpoint)
    req.send((0).to_bytes(8, "big"))
    first = req.recv_multipart()
    assert vllm.read_sequence_number(first[1]) == 0
    with pytest.raises(zmq.ZMQError):
        req.recv_multipart()
    req.close()

    dealer = ctx.socket(zmq.DEALER)
    dealer.setsockopt(zmq.RCVTIMEO, 2000)
    dealer.setsockopt(zmq.LINGER, 0)
    dealer.connect(publisher.replay_endpoint)
    dealer.send_multipart(vllm.build_replay_request(0))
    replies = []
    while (item := vllm.parse_replay_reply(dealer.recv_multipart())) is not None:
        replies.append(item)
    dealer.close()
    assert [seq for seq, _ in replies] == [0, 1, 2, 3]
    assert [p for _, p in replies] == [frames[i % 3] for i in range(4)]


def test_bare_dealer_request_without_delimiter_gets_no_reply(publisher, real_payloads):
    publisher.publish(real_payloads[0], drop=True)
    dealer: zmq.Socket[bytes] = zmq.Context.instance().socket(zmq.DEALER)
    dealer.setsockopt(zmq.RCVTIMEO, 500)
    dealer.setsockopt(zmq.LINGER, 0)
    dealer.connect(publisher.replay_endpoint)
    dealer.send((0).to_bytes(8, "big"))
    with pytest.raises(zmq.Again):
        dealer.recv_multipart()
    dealer.close()
    assert publisher.replay_requests == 1


def test_probe_replay_finds_a_listening_replay_socket(publisher):
    assert publisher.replay_endpoint is not None
    assert probe_replay(publisher.replay_endpoint, timeout=2.0) is True


def test_probe_replay_reports_a_socket_that_is_not_there():
    assert probe_replay("tcp://127.0.0.1:1", timeout=0.3) is False


def test_probe_replay_reports_a_publisher_started_without_replay(publisher_without_replay):
    assert publisher_without_replay.replay_endpoint is None
    assert probe_replay(publisher_without_replay.endpoint, timeout=0.3) is False
