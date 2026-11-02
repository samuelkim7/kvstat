from __future__ import annotations

from importlib.metadata import version

from click.testing import CliRunner

from kvstat import __version__
from kvstat.__main__ import main
from kvstat.ingest.capture import CaptureWriter


def test_help_lists_every_command() -> None:
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "  record  " in result.output
    assert "  report  " in result.output
    assert "  top  " in result.output


def test_top_refuses_recording_flags_with_a_capture(tmp_path):
    capture = tmp_path / "run.jsonl"
    capture.write_text("")
    result = CliRunner().invoke(main, ["top", "--from", str(capture), "--out", "x.jsonl"])
    assert result.exit_code == 2
    assert "record a live session" in result.output


def test_top_stops_before_the_screen_when_the_tokenizer_cannot_load(tmp_path):
    capture = tmp_path / "run.jsonl"
    with CaptureWriter(capture, {}):
        pass
    broken = tmp_path / "tokenizer.json"
    broken.write_text("{}")
    result = CliRunner().invoke(main, ["top", "--from", str(capture), "--tokenizer", str(broken)])
    assert result.exit_code == 1
    assert f"cannot load tokenizer {broken}" in result.output


def test_top_refuses_a_tokenizer_on_a_redacted_capture(tmp_path):
    capture = tmp_path / "run.jsonl"
    with CaptureWriter(capture, {}, redact_tokens=True):
        pass
    result = CliRunner().invoke(main, ["top", "--from", str(capture), "--tokenizer", "any"])
    assert result.exit_code == 1
    assert "holds no token ids" in result.output


def test_version_flag_reports_package_version() -> None:
    result = CliRunner().invoke(main, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_version_matches_distribution_metadata() -> None:
    assert __version__ == version("kvstat")
