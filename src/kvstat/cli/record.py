from __future__ import annotations

import signal
import threading
import time
from pathlib import Path
from urllib.parse import urljoin

import click

from kvstat import __version__
from kvstat.cli.common import capture_options, configure_logging, live_options, reporting
from kvstat.engines import vllm
from kvstat.ingest.capture import CaptureWriter, open_for_resume
from kvstat.ingest.collector import Collector, warn_about_replay
from kvstat.ingest.metrics import MetricsPoller, fetch_server_info


@click.command(short_help="Record a capture without the screen.")
@live_options
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Capture file; a .gz name compresses. Default: kvstat-<time>.jsonl.gz",
)
@click.option(
    "--duration",
    type=float,
    default=None,
    help="Stop after this many seconds; default runs until Ctrl-C.",
)
@capture_options
@click.option("--log-level", default="INFO", show_default=True)
def record(
    endpoint: str,
    replay_endpoint: str | None,
    server: str,
    topic: str,
    out_path: Path | None,
    duration: float | None,
    redact_tokens: bool,
    overwrite: bool,
    log_level: str,
) -> None:
    """Record the KV-event stream and metrics of a running vLLM server to a capture file.

    Runs until Ctrl-C, SIGTERM or --duration elapses. Pointed at a capture it already wrote,
    kvstat continues it. The batches missed in between come from vLLM's replay socket, so a
    stop and restart costs nothing while they are still in its buffer.
    """
    configure_logging(log_level)
    if out_path is None:
        out_path = Path(time.strftime("kvstat-%Y%m%d-%H%M%S.jsonl.gz"))
    header = {
        "kvstat_version": __version__,
        "endpoint": endpoint,
        "replay_endpoint": replay_endpoint,
        "server": server,
        "topic": topic,
        "started_at": time.time(),
        **fetch_server_info(server),
    }
    warn_about_replay(replay_endpoint)
    with reporting("record"):
        resume = None if overwrite else open_for_resume(out_path)
        with CaptureWriter(
            out_path, header, redact_tokens=redact_tokens, resume=resume is not None
        ) as capture:
            collector = Collector(
                endpoint,
                replay_endpoint=replay_endpoint,
                topic=topic,
                on_batch=capture.write_batch,
                resume_from=resume.seq if resume else None,
                resume_epoch=resume.epoch if resume else 0,
            )
            poller = MetricsPoller(urljoin(server, vllm.METRICS_PATH), capture.write_metrics)

            # Signal handlers pass (signum, frame), the timer passes nothing; both are ignored.
            def stop(*_: object) -> None:
                collector.close()

            previous = [(sig, signal.signal(sig, stop)) for sig in (signal.SIGINT, signal.SIGTERM)]
            timer = threading.Timer(duration, stop) if duration else None
            collector.start()
            poller.start()
            if timer:
                timer.start()
            try:
                # The capture already got every batch raw through on_batch; this loop only
                # drains the decoded queue and blocks until the collector stops.
                for _ in collector:
                    pass
            finally:
                if timer:
                    timer.cancel()
                poller.close()
                for sig, handler in previous:
                    signal.signal(sig, handler)
            if collector.error is not None:
                raise collector.error
    stats = collector.stats()
    click.echo(
        f"recorded {stats.batches} batches (gaps {stats.gaps}, replayed {stats.replayed}, "
        f"resyncs {stats.resyncs}, decode errors {stats.decode_errors}), "
        f"{poller.scrapes} metric scrapes -> {out_path}"
    )
