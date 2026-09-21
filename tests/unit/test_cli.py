from __future__ import annotations

from importlib.metadata import version

from click.testing import CliRunner

from kvstat import __version__
from kvstat.__main__ import main


def test_help_lists_every_command() -> None:
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "  record  " in result.output
    assert "  dump  " in result.output


def test_version_flag_reports_package_version() -> None:
    result = CliRunner().invoke(main, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_version_matches_distribution_metadata() -> None:
    assert __version__ == version("kvstat")
