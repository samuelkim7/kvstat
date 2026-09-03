"""Command-line entry point."""

from __future__ import annotations

import click

from kvtop import __version__


@click.group()
@click.version_option(__version__, prog_name="kvtop")
def main() -> None:
    """Monitor vLLM's KV cache, htop-style."""


if __name__ == "__main__":
    main()
