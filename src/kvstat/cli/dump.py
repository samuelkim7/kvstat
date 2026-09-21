from __future__ import annotations

import logging
from pathlib import Path

import click

from kvstat.cli import configure_logging, reporting
from kvstat.engines import vllm
from kvstat.ingest.capture import CaptureReader, CaptureSource
from kvstat.ingest.metrics import Sample
from kvstat.state.cache import CacheState
from kvstat.state.crosscheck import Breach, EngineCounts, EngineWindow
from kvstat.views.console import format_summary

log = logging.getLogger("kvstat")


@click.command()
@click.option(
    "--from",
    "capture_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Capture file to read, as written by kvstat record.",
)
@click.option("--log-level", default="WARNING", show_default=True)
def dump(capture_path: Path, log_level: str) -> None:
    """Print what the KV cache held over a capture, and whether the numbers hold up."""
    configure_logging(log_level)
    with reporting("dump"):
        reader = CaptureReader(capture_path)
        capacity, block_size = reader.cache_config()
        state = CacheState(capacity=capacity, tier=vllm.MEDIUM_GPU)
        breaches, batches, window = apply_capture(state, reader)
    snap = state.snapshot()
    click.echo(f"{capture_path.name}  ·  {batches:,} batches")
    click.echo(format_summary(reader.header, snap, block_size, window))
    if breaches:
        raise SystemExit(1)


def read_counts(time: float, samples: list[Sample]) -> EngineCounts:
    """Turn one scrape into the engine counts for the cross-check."""
    used_fraction, running = vllm.read_kv_usage(samples)
    return EngineCounts(time=time, used_fraction=used_fraction, running_requests=int(running))


def apply_capture(
    state: CacheState, reader: CaptureSource
) -> tuple[list[Breach], int, EngineWindow]:
    """Apply a capture's batches and scrapes to the state in arrival order.

    Both rows carry kvstat's own clock, so they merge directly. One poller thread writes the
    scrapes, so they arrive in scrape order and need no sorting.

    Returns:
        The breaches, the batches applied, and what the engine said about itself meanwhile.
    """
    scrapes = reader.metrics()
    pending = next(scrapes, None)
    breaches: list[Breach] = []
    window = EngineWindow()
    batches = 0

    def cross_check(time: float, samples: list[Sample]) -> None:
        nonlocal window
        counts = read_counts(time, samples)
        window = window.then(counts)
        breach = state.cross_check(counts)
        if breach is not None:
            breaches.append(breach)
            log.warning("drift: %s", breach.detail)

    for arrived, batch in reader.batches():
        while pending is not None and pending[0] <= arrived:
            cross_check(*pending)
            pending = next(scrapes, None)
        state.update(batch)
        batches += 1
    while pending is not None:
        cross_check(*pending)
        pending = next(scrapes, None)
    return breaches, batches, window
