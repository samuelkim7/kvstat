from __future__ import annotations

import base64
import gzip
import json
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import IO, Any, Literal

from kvstat import events
from kvstat.engines import vllm
from kvstat.errors import ConfigError
from kvstat.ingest.metrics import Sample

FORMAT = 1
GZIP_MAGIC = b"\x1f\x8b"


def _open_text(path: Path, mode: Literal["r", "w"]) -> IO[str]:
    """Open for reading by content (gzip magic) or for writing by name (.gz suffix)."""
    if mode == "r":
        with path.open("rb") as probe:
            compressed = probe.read(2) == GZIP_MAGIC
        return (
            gzip.open(path, "rt", encoding="utf-8")
            if compressed
            else path.open("r", encoding="utf-8")
        )
    if path.suffix == ".gz":
        return gzip.open(path, "wt", encoding="utf-8")
    return path.open("w", encoding="utf-8")


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
        redact_tokens: Remove prompt token ids from each payload before writing it.
    """

    def __init__(self, path: Path, header: dict[str, Any], *, redact_tokens: bool = False) -> None:
        self.path = path
        self._redact = redact_tokens
        self._lock = threading.Lock()
        self._fh = _open_text(path, "w")
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
        for row in self._rows("batch"):
            payload = base64.b64decode(row["payload"])
            yield vllm.decode_batch(row["seq"], payload, epoch=row["epoch"])

    def metrics(self) -> Iterator[tuple[float, list[Sample]]]:
        """Yield every metrics scrape as (scrape time, samples), in file order."""
        for row in self._rows("metrics"):
            yield row["time"], [(name, labels, value) for name, labels, value in row["samples"]]

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
