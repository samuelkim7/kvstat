from __future__ import annotations

import logging
from pathlib import Path

import click

from kvstat.cli.common import configure_logging, reporting
from kvstat.cli.feed import Batch, merge
from kvstat.engines import vllm
from kvstat.ingest.capture import CaptureReader, CaptureSource
from kvstat.state.cache import CacheState
from kvstat.state.crosscheck import Breach, EngineWindow
from kvstat.views.console import format_summary

log = logging.getLogger("kvstat")


@click.command(short_help="Summarize a capture.")
@click.option(
    "--from",
    "capture_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Capture file to read, as written by kvstat record.",
)
@click.option("--log-level", default="WARNING", show_default=True)
def report(capture_path: Path, log_level: str) -> None:
    """Print what the KV cache held over a capture, and whether the numbers hold up."""
    configure_logging(log_level)
    with reporting("report"):
        reader = CaptureReader(capture_path)
        capacity, block_size = reader.cache_config()
        state = CacheState(capacity=capacity, tier=vllm.MEDIUM_GPU)
        breaches, batches, window = apply_capture(state, reader)
    snap = state.snapshot()
    click.echo(f"{capture_path.name}  ·  {batches:,} batches")
    click.echo(format_summary(reader.header, snap, block_size, window))
    if breaches:
        raise SystemExit(1)


def apply_capture(
    state: CacheState, reader: CaptureSource
) -> tuple[list[Breach], int, EngineWindow]:
    """Apply a capture's batches and scrapes to the state in arrival order.

    Returns:
        The breaches, the batches applied, and what the engine said about itself meanwhile.
    """
    breaches: list[Breach] = []
    window = EngineWindow()
    batches = 0
    for item in merge(reader):
        if isinstance(item, Batch):
            state.update(item.batch)
            batches += 1
            continue
        counts = vllm.read_engine_counts(item.time, item.samples)
        window = window.then(counts)
        breach = state.cross_check(counts)
        if breach is not None:
            breaches.append(breach)
            log.warning("drift: %s", breach.detail)
    return breaches, batches, window
