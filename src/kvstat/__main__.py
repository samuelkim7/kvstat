"""Command-line entry point."""

from __future__ import annotations

import logging
import signal
import threading
import time
from pathlib import Path
from urllib.parse import urljoin

import click

from kvstat import __version__
from kvstat.engines import vllm
from kvstat.ingest.capture import CaptureWriter
from kvstat.ingest.collector import Collector
from kvstat.ingest.metrics import MetricsPoller, fetch_server_info

log = logging.getLogger("kvstat")


@click.group()
@click.version_option(__version__, prog_name="kvstat")
def main() -> None:
    """See inside vLLM's KV cache, request by request."""


@main.command()
@click.option(
    "--endpoint",
    default="tcp://127.0.0.1:5557",
    show_default=True,
    help="KV-event publisher (ZMQ PUB).",
)
@click.option(
    "--replay-endpoint",
    default=None,
    help="vLLM's replay socket, which resends missed batches; without it a gap forces a resync.",
)
@click.option(
    "--server",
    default="http://127.0.0.1:8000",
    show_default=True,
    help="vLLM HTTP server, for /metrics and /version.",
)
@click.option("--topic", default="", help="ZMQ topic to subscribe to; vLLM's default is empty.")
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Capture file; a .gz name compresses. Default: kvstat-<time>.jsonl.gz",
)
@click.option(
    "--redact-tokens", is_flag=True, help="Strip prompt token ids before they reach disk."
)
@click.option(
    "--duration",
    type=float,
    default=None,
    help="Stop after this many seconds; default runs until Ctrl-C.",
)
@click.option("--log-level", default="INFO", show_default=True)
def record(
    endpoint: str,
    replay_endpoint: str | None,
    server: str,
    topic: str,
    out_path: Path | None,
    redact_tokens: bool,
    duration: float | None,
    log_level: str,
) -> None:
    """Record the KV-event stream and metrics of a running vLLM server to a capture file.

    Runs until Ctrl-C, SIGTERM or --duration elapses.
    """
    logging.basicConfig(
        level=log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
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
    try:
        with CaptureWriter(out_path, header, redact_tokens=redact_tokens) as capture:
            collector = Collector(
                endpoint, replay_endpoint=replay_endpoint, topic=topic, on_batch=capture.write_batch
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
    except Exception as exc:
        log.error("record failed: %s", exc)
        raise click.ClickException(str(exc)) from exc
    stats = collector.stats()
    click.echo(
        f"recorded {stats.batches} batches (gaps {stats.gaps}, replayed {stats.replayed}, "
        f"resyncs {stats.resyncs}, decode errors {stats.decode_errors}), "
        f"{poller.scrapes} metric scrapes -> {out_path}"
    )


if __name__ == "__main__":
    main()
