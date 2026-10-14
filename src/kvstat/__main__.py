"""Command-line entry point."""

from __future__ import annotations

import click

from kvstat import __version__
from kvstat.cli.record import record
from kvstat.cli.report import report
from kvstat.cli.top import top


class _InOrder(click.Group):
    def list_commands(self, ctx: click.Context) -> list[str]:
        return list(self.commands)


@click.group(cls=_InOrder)
@click.version_option(__version__, prog_name="kvstat")
def main() -> None:
    """See inside vLLM's KV cache, request by request."""


main.add_command(top)
main.add_command(record)
main.add_command(report)

if __name__ == "__main__":
    main()
