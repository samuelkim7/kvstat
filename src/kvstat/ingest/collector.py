from __future__ import annotations

import logging
import queue
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import zmq

from kvstat import events
from kvstat.engines import vllm
from kvstat.errors import DecodeError

log = logging.getLogger(__name__)

# A sequence number vLLM can never have buffered, so a replay probe gets only the end marker.
FUTURE_SEQ = 2**63

# Called with (arrival time, seq, epoch, payload) for every raw batch, before decoding.
BatchCallback = Callable[[float, int, int, bytes], None]


@dataclass(slots=True)
class _Counters:
    """Mutable twin of IngestStats, owned by the socket thread."""

    batches: int = 0
    last_seq: int | None = None
    epoch: int = 0
    gaps: int = 0
    replayed: int = 0
    resyncs: int = 0
    decode_errors: int = 0
    skipped: Counter[str] = field(default_factory=Counter)


def probe_replay(endpoint: str, timeout: float = 2.0) -> bool:
    """Check that vLLM's replay socket answers. Ask it for a sequence number it cannot have.

    Without replay, a disconnect of any length forces a resync. The operator should hear that
    before it happens. vLLM answers a request beyond its buffer with only the end marker, so
    the probe costs nothing and proves the socket is there.
    """
    ctx: zmq.Context[zmq.Socket[bytes]] = zmq.Context.instance()
    dealer = ctx.socket(zmq.DEALER)
    dealer.setsockopt(zmq.RCVTIMEO, int(timeout * 1000))
    dealer.setsockopt(zmq.LINGER, 0)
    try:
        dealer.connect(endpoint)
        request = vllm.build_replay_request(FUTURE_SEQ)
        dealer.send_multipart(request)
        return vllm.parse_replay_reply(dealer.recv_multipart()) is None
    except (zmq.ZMQError, DecodeError) as exc:
        log.debug("replay probe failed on %s: %s", endpoint, exc)
        return False
    finally:
        dealer.close()


def warn_about_replay(replay_endpoint: str | None) -> None:
    """Say up front whether a dropped connection can be recovered, instead of at the first gap."""
    if replay_endpoint is None:
        log.warning(
            "no --replay-endpoint: a dropped connection of any length loses every batch sent "
            "meanwhile, and kvstat rebuilds its state from scratch. Start vLLM with "
            "replay_endpoint in --kv-events-config and pass it here."
        )
    elif not probe_replay(replay_endpoint):
        log.warning(
            "replay socket %s did not answer: gaps will force a full rebuild. Check that vLLM "
            "was started with this replay_endpoint in --kv-events-config.",
            replay_endpoint,
        )


class Collector:
    """Receives the KV-event stream from vLLM and yields it as decoded batches, in order.

    start() launches a daemon thread. The thread reads the ZMQ sockets and puts each batch on
    an unbounded queue. Iterating the collector drains that queue. close() stops the thread.
    When a batch is missing, the thread asks vLLM to resend it. vLLM calls that a replay. When
    vLLM no longer has the batch, the collector gives up on the gap, increments the epoch and
    continues. State built before the gap is out of date.

    Args:
        endpoint: The publisher's PUB socket, for example tcp://127.0.0.1:5557.
        replay_endpoint: The publisher's replay socket. Without it a missing batch cannot be
            recovered, so every gap becomes a resync.
        topic: ZMQ subscription topic; vLLM's default is empty.
        on_batch: Called with (arrival time, seq, epoch, payload) for every raw batch, before
            it is decoded. kvstat record passes CaptureWriter.write_batch.
        replay_timeout: Seconds to wait for each replay message before giving up on the gap.
        summary_interval: Seconds between summary log lines.
        resume_from: The first sequence number kvstat still needs, when it continues a capture.
            The batches missed while kvstat was down read as an ordinary gap before the first
            live batch, so replay fills them like any other gap.
        resume_epoch: The epoch that capture ended on, so continuing it is not read as a resync.

    Attributes:
        ready: Set once the sockets are connected and subscribed.
        error: The exception that stopped the socket thread, or None.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        replay_endpoint: str | None = None,
        topic: str = "",
        on_batch: BatchCallback | None = None,
        replay_timeout: float = 2.0,
        summary_interval: float = 10.0,
        resume_from: int | None = None,
        resume_epoch: int = 0,
    ) -> None:
        self._endpoint = endpoint
        self._replay_endpoint = replay_endpoint
        self._topic = topic.encode()
        self._on_batch = on_batch
        self._replay_timeout = replay_timeout
        self._summary_interval = summary_interval
        self._queue: queue.SimpleQueue[events.EventBatch | None] = queue.SimpleQueue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="kvstat-ingest", daemon=True)
        self._dealer: zmq.Socket[bytes] | None = None
        self._expected: int | None = resume_from
        self._counters = _Counters(epoch=resume_epoch)
        self._next_summary = 0.0
        self.ready = threading.Event()
        self.error: BaseException | None = None

    def start(self) -> Collector:
        """Start the socket thread and return self, so `Collector(...).start()` reads well."""
        self._thread.start()
        return self

    def close(self) -> None:
        """Stop the socket thread and wait for it, unless called from that thread."""
        self._stop.set()
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=self._replay_timeout + 5)

    def __iter__(self) -> Iterator[events.EventBatch]:
        """Yield batches as they arrive until the socket thread stops."""
        while (item := self._queue.get()) is not None:
            yield item

    def stats(self) -> events.IngestStats:
        """Snapshot the counters; safe to call from any thread."""
        c = self._counters
        return events.IngestStats(
            batches=c.batches,
            last_seq=c.last_seq,
            epoch=c.epoch,
            gaps=c.gaps,
            replayed=c.replayed,
            resyncs=c.resyncs,
            decode_errors=c.decode_errors,
            queue_depth=self._queue.qsize(),
            skipped_event_types=dict(c.skipped),
        )

    def _run(self) -> None:
        """Thread body: connect, then poll the subscriber until stopped or failed.

        Always ends by putting the None sentinel on the queue, so __iter__ never hangs. A crash
        is kept in self.error for the owner to raise. The thread itself exits quietly.
        """
        ctx: zmq.Context[zmq.Socket[bytes]] = zmq.Context.instance()
        sub = ctx.socket(zmq.SUB)
        try:
            sub.setsockopt(zmq.RCVHWM, 0)
            sub.setsockopt(zmq.LINGER, 0)
            sub.connect(self._endpoint)
            sub.setsockopt(zmq.SUBSCRIBE, self._topic)
            if self._replay_endpoint is not None:
                self._dealer = self._open_dealer(ctx, self._replay_endpoint)
            poller = zmq.Poller()
            poller.register(sub, zmq.POLLIN)
            log.info(
                "ingest: subscribed to %s topic=%r replay=%s",
                self._endpoint,
                self._topic,
                self._replay_endpoint,
            )
            self.ready.set()
            while not self._stop.is_set():
                if poller.poll(100):
                    frames = sub.recv_multipart()
                    if len(frames) == 3:
                        _topic, seq_bytes, payload = frames
                        seq = vllm.read_sequence_number(seq_bytes)
                        self._on_frame(seq, payload)
                    else:
                        self._counters.decode_errors += 1
                self._maybe_summary()
        except Exception as exc:  # noqa: BLE001 - the owner re-raises it from self.error
            self.error = exc
        finally:
            self._queue.put(None)
            sub.close()
            if self._dealer is not None:
                self._dealer.close()
            log.info("ingest: stopped after %d batches", self._counters.batches)

    def _open_dealer(self, ctx: zmq.Context[zmq.Socket[bytes]], endpoint: str) -> zmq.Socket[bytes]:
        """Connect a fresh replay socket with the receive timeout set."""
        dealer = ctx.socket(zmq.DEALER)
        dealer.setsockopt(zmq.RCVTIMEO, int(self._replay_timeout * 1000))
        dealer.setsockopt(zmq.LINGER, 0)
        dealer.connect(endpoint)
        return dealer

    def _on_frame(self, seq: int, payload: bytes) -> None:
        """Handle one live batch: drop it if stale, recover the gap in front of it, emit it."""
        if self._expected is not None:
            if seq < self._expected:
                return
            if seq > self._expected:
                self._recover_gap(self._expected, seq)
        self._emit(seq, payload)
        self._expected = seq + 1

    def _recover_gap(self, start: int, end: int) -> None:
        """Emit the missing batches start..end-1 through replay, or advance the epoch."""
        c = self._counters
        c.gaps += 1
        log.info("gap: expected seq %d, got %d; %d missing", start, end, end - start)
        recovered = self._replay(start, end)
        if recovered is None:
            c.epoch += 1
            c.resyncs += 1
            log.warning("resync: seq [%d, %d) unrecoverable, epoch now %d", start, end, c.epoch)
            return
        for seq, payload in recovered:
            self._emit(seq, payload)
        c.replayed += len(recovered)
        log.info("replay: recovered %d batches [%d, %d)", len(recovered), start, end)

    def _replay(self, start: int, end: int) -> list[tuple[int, bytes]] | None:
        """Ask vLLM to resend seq start..end-1; return them in order, or None if any is missing."""
        dealer, endpoint = self._dealer, self._replay_endpoint
        if dealer is None or endpoint is None:
            log.info("replay: no replay endpoint configured")
            return None
        request = vllm.build_replay_request(start)
        dealer.send_multipart(request)
        replies = self._replay_replies(dealer, endpoint)
        got = {seq: payload for seq, payload in replies if start <= seq < end}
        if len(got) != end - start:
            return None
        return sorted(got.items())

    def _replay_replies(
        self, dealer: zmq.Socket[bytes], endpoint: str
    ) -> Iterator[tuple[int, bytes]]:
        """Yield (seq, payload) replies until vLLM signals the end, times out, or sends junk."""
        while True:
            try:
                frames = dealer.recv_multipart()
            except zmq.Again:
                log.warning("replay: timed out after %.1fs", self._replay_timeout)
                # A reply that arrives late would be read as the answer to the next request.
                dealer.close()
                self._dealer = self._open_dealer(zmq.Context.instance(), endpoint)
                return
            try:
                reply = vllm.parse_replay_reply(frames)
            except DecodeError as exc:
                log.warning("replay: %s", exc)
                return
            if reply is None:
                return
            yield reply

    def _emit(self, seq: int, payload: bytes) -> None:
        """Pass one batch to on_batch as raw bytes, then decode it onto the queue."""
        c = self._counters
        now = time.time()
        if self._on_batch is not None:
            self._on_batch(now, seq, c.epoch, payload)
        try:
            batch = vllm.decode_batch(seq, payload, epoch=c.epoch)
        except DecodeError as exc:
            c.decode_errors += 1
            log.debug("decode: seq %d dropped: %s", seq, exc)
            return
        c.skipped.update(batch.skipped)
        c.batches += 1
        c.last_seq = seq
        self._queue.put(batch)

    def _maybe_summary(self) -> None:
        """Log one summary line per summary_interval, once there is something to report."""
        now = time.monotonic()
        if now < self._next_summary:
            return
        self._next_summary = now + self._summary_interval
        c = self._counters
        if c.batches or c.gaps or c.decode_errors:
            log.info(
                "ingest: batches=%d last_seq=%s epoch=%d gaps=%d replayed=%d resyncs=%d "
                "decode_errors=%d queue=%d",
                c.batches,
                c.last_seq,
                c.epoch,
                c.gaps,
                c.replayed,
                c.resyncs,
                c.decode_errors,
                self._queue.qsize(),
            )
