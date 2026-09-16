from __future__ import annotations

import threading
from collections import deque

import pytest
import zmq

from kvstat.engines import vllm


class FakePublisher:
    """Stand-in for vLLM's ZmqEventPublisher: XPUB for live batches, ROUTER for replay.

    publish(drop=True) skips the live send but keeps the batch for replay, so the gap can be
    filled. publish(buffer=False) keeps nothing, so it cannot.
    """

    def __init__(
        self, *, replay: bool = True, buffer_steps: int = 10_000, topic: bytes = b""
    ) -> None:
        self._ctx: zmq.Context[zmq.Socket[bytes]] = zmq.Context.instance()
        self._topic = topic
        self._pub = self._ctx.socket(zmq.XPUB)
        self._pub.setsockopt(zmq.LINGER, 0)
        self.endpoint = f"tcp://127.0.0.1:{self._pub.bind_to_random_port('tcp://127.0.0.1')}"
        self.replay_endpoint: str | None = None
        self._router: zmq.Socket[bytes] | None = None
        if replay:
            self._router = self._ctx.socket(zmq.ROUTER)
            self._router.setsockopt(zmq.LINGER, 0)
            port = self._router.bind_to_random_port("tcp://127.0.0.1")
            self.replay_endpoint = f"tcp://127.0.0.1:{port}"
        self.buffer: deque[tuple[int, bytes]] = deque(maxlen=buffer_steps)
        self.replay_requests = 0
        self._seq = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def wait_for_subscriber(self, timeout: float = 5.0) -> None:
        """XPUB delivers the subscription as a message; publishing before it would be lost."""
        if not self._pub.poll(int(timeout * 1000)):
            raise TimeoutError("no subscriber")
        assert self._pub.recv()[0] == 1

    def publish(self, payload: bytes, *, drop: bool = False, buffer: bool = True) -> int:
        seq = self._seq
        self._seq += 1
        if buffer:
            self.buffer.append((seq, payload))
        if not drop:
            self._pub.send_multipart((self._topic, seq.to_bytes(8, "big"), payload))
        return seq

    def _serve(self) -> None:
        if self._router is None:
            return
        while not self._stop.is_set():
            if not self._router.poll(50):
                continue
            frame = self._router.recv_multipart()
            self.replay_requests += 1
            if len(frame) != 3:
                continue
            client_id, _, start_seq_bytes = frame
            start_seq = int.from_bytes(start_seq_bytes, "big")
            for seq, buf in list(self.buffer):
                if seq >= start_seq:
                    self._router.send_multipart(
                        (client_id, b"", self._topic, seq.to_bytes(8, "big"), buf)
                    )
            self._router.send_multipart((client_id, b"", b"", vllm.END_SEQ, b""))

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        self._pub.close()
        if self._router is not None:
            self._router.close()


@pytest.fixture
def publisher():
    pub = FakePublisher()
    yield pub
    pub.close()


@pytest.fixture
def publisher_without_replay():
    pub = FakePublisher(replay=False)
    yield pub
    pub.close()
