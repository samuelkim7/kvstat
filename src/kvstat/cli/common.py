from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TypeVar

import click

log = logging.getLogger("kvstat")

Command = TypeVar("Command", bound=Callable[..., None])


def live_options(command: Command) -> Command:
    """The options that name a running vLLM server. record and top share them."""
    options = [
        click.option(
            "--endpoint",
            default="tcp://127.0.0.1:5557",
            show_default=True,
            help="KV-event publisher (ZMQ PUB).",
        ),
        click.option(
            "--replay-endpoint",
            default=None,
            help=(
                "vLLM's replay socket, which resends missed batches. Without it a gap forces "
                "a resync."
            ),
        ),
        click.option(
            "--server",
            default="http://127.0.0.1:8000",
            show_default=True,
            help="vLLM HTTP server, for /metrics and /version.",
        ),
        click.option(
            "--topic", default="", help="ZMQ topic to subscribe to; vLLM's default is empty."
        ),
    ]
    for option in reversed(options):
        command = option(command)
    return command


def capture_options(command: Command) -> Command:
    """The options that shape a capture file. record and top --out share them."""
    options = [
        click.option(
            "--redact-tokens", is_flag=True, help="Strip prompt token ids before they reach disk."
        ),
        click.option(
            "--overwrite",
            is_flag=True,
            help="Start the output file again instead of continuing the capture already there.",
        ),
    ]
    for option in reversed(options):
        command = option(command)
    return command


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )


@contextmanager
def reporting(action: str) -> Iterator[None]:
    """Turn any failure inside into a clean CLI error. The one broad catch kvstat makes."""
    try:
        yield
    except Exception as exc:
        log.error("%s failed: %s", action, exc)
        raise click.ClickException(str(exc)) from exc
