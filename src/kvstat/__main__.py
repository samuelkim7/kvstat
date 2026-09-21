"""Command-line entry point."""

from __future__ import annotations

import click

from kvstat import __version__
from kvstat.cli.record import record


@click.group()
@click.version_option(__version__, prog_name="kvstat")
def main() -> None:
    """See inside vLLM's KV cache, request by request."""


main.add_command(record)

if __name__ == "__main__":
    main()
