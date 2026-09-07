"""Command-line entry point."""

from __future__ import annotations

import click

from kvstat import __version__


@click.group()
@click.version_option(__version__, prog_name="kvstat")
def main() -> None:
    """See inside vLLM's KV cache, request by request."""


if __name__ == "__main__":
    main()
