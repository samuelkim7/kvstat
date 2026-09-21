from __future__ import annotations

import base64
import gzip
import json
import logging
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Literal, Protocol, runtime_checkable

from kvstat import events
from kvstat.engines import vllm
from kvstat.errors import ConfigError
from kvstat.ingest.metrics import Sample

log = logging.getLogger(__name__)

FORMAT = 1
GZIP_MAGIC = b"\x1f\x8b"


def _open_text(path: Path, mode: Literal["r", "w", "a"]) -> IO[str]:
    """Open for reading by content (gzip magic) or for writing by name (.gz suffix).

    Appending to a .gz file starts a new gzip member. Readers walk concatenated members as one
    stream, so a resumed capture reads back exactly like an unbroken one.
    """
    if mode == "r":
        with path.open("rb") as probe:
            compressed = probe.read(2) == GZIP_MAGIC
        return (
            gzip.open(path, "rt", encoding="utf-8")
            if compressed
            else path.open("r", encoding="utf-8")
        )
    if path.suffix == ".gz":
        return (
            gzip.open(path, "at", encoding="utf-8")
            if mode == "a"
            else gzip.open(path, "wt", encoding="utf-8")
        )
    return path.open(mode, encoding="utf-8")


@runtime_checkable
class CaptureSource(Protocol):
    """Protocol for anything a command can walk as a capture: a reader, or a test's stand-in."""

    def batches(self) -> Iterator[tuple[float, events.EventBatch]]: ...

    def metrics(self) -> Iterator[tuple[float, list[Sample]]]: ...


class CaptureWriter:
    """Writes one capture file: a header line, then every batch and metrics scrape as it arrives.

    JSON lines, one object per line, told apart by kind:

    - header: format number, redacted flag, and the caller's header fields.
    - batch: time (when kvstat received it, Unix seconds), seq, epoch and the raw payload as
      base64. Storing the raw bytes lets a later kvstat decode the file with its own decoder.
    - metrics: time of the scrape and samples as [name, labels, value] triples.

    Args:
        path: Output file. A name ending in .gz turns on gzip compression.
        header: Extra fields for the header line.
        redact_tokens: Remove prompt token ids from each payload before writing it. A capture
            that already dropped them stays redacted whatever a resumed run asks for.
        resume: Append to an existing capture and keep its header, instead of starting a file.
    """

    def __init__(
        self,
        path: Path,
        header: dict[str, Any],
        *,
        redact_tokens: bool = False,
        resume: bool = False,
    ) -> None:
        self.path = path
        if resume:
            redact_tokens = redact_tokens or bool(CaptureReader(path).header.get("redacted"))
        self._redact = redact_tokens
        self._lock = threading.Lock()
        self._fh = _open_text(path, "a" if resume else "w")
        if not resume:
            self._write({"kind": "header", "format": FORMAT, "redacted": redact_tokens, **header})

    def write_batch(self, time: float, seq: int, epoch: int, payload: bytes) -> None:
        """Append one batch row; fits Collector's on_batch."""
        if self._redact:
            payload = vllm.redact_token_ids(payload)
        encoded = base64.b64encode(payload).decode("ascii")
        self._write({"kind": "batch", "time": time, "seq": seq, "epoch": epoch, "payload": encoded})

    def write_metrics(self, time: float, samples: list[Sample]) -> None:
        """Append one metrics row; fits MetricsPoller's on_scrape."""
        self._write({"kind": "metrics", "time": time, "samples": [list(s) for s in samples]})

    def close(self) -> None:
        """Flush and close the file; further writes fail."""
        with self._lock:
            self._fh.flush()
            self._fh.close()

    def __enter__(self) -> CaptureWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _write(self, row: dict[str, Any]) -> None:
        """Serialize one row and write it under the lock."""
        line = json.dumps(row, separators=(",", ":")) + "\n"
        with self._lock:
            self._fh.write(line)


class CaptureReader:
    """Reads a capture file back as an EventSource plus a series of metrics scrapes.

    Iterating over the reader yields the batch rows, decoded, in file order. metrics() yields
    the metrics rows. Each pass reopens the file, so the two can be read in any order.

    Attributes:
        path: The capture file.
        header: The header row, as written.

    Raises:
        ConfigError: the file does not start with a header this kvstat version understands.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        with _open_text(path, "r") as fh:
            first = fh.readline()
        try:
            header = json.loads(first) if first else {}
        except ValueError as exc:
            raise ConfigError(f"{path}: not a kvstat capture") from exc
        if header.get("kind") != "header" or header.get("format") != FORMAT:
            raise ConfigError(f"{path}: not a kvstat capture (format {FORMAT})")
        self.header: dict[str, Any] = header

    def __iter__(self) -> Iterator[events.EventBatch]:
        """Yield every batch row decoded, in file order."""
        for _, batch in self.batches():
            yield batch

    def batches(self) -> Iterator[tuple[float, events.EventBatch]]:
        """Yield every batch row as (arrival time, decoded batch), in file order.

        The arrival time is kvstat's own clock. metrics() reports the same clock, so the two
        streams merge. The batch itself carries the engine's clock, which may run on another
        host.
        """
        for row in self._rows("batch"):
            payload = base64.b64decode(row["payload"])
            batch = vllm.decode_batch(row["seq"], payload, epoch=row["epoch"])
            yield row["time"], batch

    def metrics(self) -> Iterator[tuple[float, list[Sample]]]:
        """Yield every metrics scrape as (scrape time, samples), in file order."""
        for row in self._rows("metrics"):
            yield row["time"], [tuple(s) for s in row["samples"]]

    def cache_config(self) -> tuple[int | None, int | None]:
        """Return the engine's GPU block count and block size, as the header recorded them."""
        return vllm.read_cache_config(self.header.get("cache_config"))

    def last_batch(self) -> tuple[int, int] | None:
        """Return (sequence number, epoch) of the final batch row. None when there is no batch row.

        A resumed capture continues from here. The next sequence number is the first one still
        missing. The epoch carries forward, so a restart does not read as a resync.
        """
        last = None
        for row in self._rows("batch"):
            last = (row["seq"], row["epoch"])
        return last

    def close(self) -> None:
        """No file stays open between passes. Present so the reader satisfies EventSource."""

    def _rows(self, kind: str) -> Iterator[dict[str, Any]]:
        """Reopen the file, skip the header, and yield the rows of one kind."""
        with _open_text(self.path, "r") as fh:
            next(fh)
            for line in fh:
                row = json.loads(line)
                if row["kind"] == kind:
                    yield row


@dataclass(frozen=True, slots=True)
class Resume:
    """Where a capture left off, for a run that continues it.

    Attributes:
        seq: The first sequence number still missing, or None when nothing was recorded yet.
        epoch: The epoch that capture ended on.
    """

    seq: int | None
    epoch: int


def open_for_resume(path: Path) -> Resume | None:
    """Read where the capture at path left off. Return None when there is no capture yet.

    Raises:
        ConfigError: a file is there and kvstat did not write it. Overwriting it would destroy
            a file kvstat cannot replace.
    """
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        reader = CaptureReader(path)
    except ConfigError as exc:
        raise ConfigError(f"{exc}; pass --overwrite to replace it") from exc
    last = reader.last_batch()
    if last is None:
        log.info("continuing %s, which holds no batches yet", path)
        return Resume(seq=None, epoch=0)
    seq, epoch = last
    log.info(
        "continuing %s from seq %d; the batches missed since are requested from replay, which "
        "holds vLLM's last %d batches by default",
        path,
        seq + 1,
        vllm.REPLAY_BUFFER_BATCHES,
    )
    return Resume(seq=seq + 1, epoch=epoch)
