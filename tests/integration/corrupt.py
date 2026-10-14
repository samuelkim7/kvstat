from __future__ import annotations

import dataclasses
from collections.abc import Iterator

from kvstat.events import BlockStored, Event, EventBatch, Sample
from kvstat.ingest.capture import CaptureReader


class DoubleCounting:
    """The capture with every stored block counted twice, under a hash kvstat has not seen.

    Presents the same batches()/metrics() surface as a CaptureReader, so the corrupted stream
    runs through the very same merge the report command uses.
    """

    def __init__(self, source: CaptureReader) -> None:
        self.header = source.header
        self._source = source

    def batches(self) -> Iterator[tuple[float, EventBatch]]:
        for arrived, batch in self._source.batches():
            events: list[Event] = []
            for event in batch.events:
                events.append(event)
                if isinstance(event, BlockStored):
                    shifted = tuple(
                        h + 10**12 if isinstance(h, int) else h + b"\x00"
                        for h in event.block_hashes
                    )
                    events.append(dataclasses.replace(event, block_hashes=shifted))
            yield arrived, dataclasses.replace(batch, events=tuple(events))

    def metrics(self) -> Iterator[tuple[float, list[Sample]]]:
        return self._source.metrics()
