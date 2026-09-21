from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

import click

log = logging.getLogger("kvstat")


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
