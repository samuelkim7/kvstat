from __future__ import annotations

import gzip
import json

import pytest

from kvstat.engines import vllm
from kvstat.errors import ConfigError
from kvstat.events import BlockStored
from kvstat.ingest.capture import CaptureReader, CaptureWriter, Resume, open_for_resume


@pytest.mark.parametrize("name", ["cap.jsonl", "cap.jsonl.gz"])
def test_round_trip_plain_and_gzip(tmp_path, real_frames, name):
    path = tmp_path / name
    with CaptureWriter(path, {"vllm_version": "0.28.0"}) as writer:
        for seq, payload in real_frames:
            writer.write_batch(1.0 + seq, seq, 0, payload)
        writer.write_metrics(2.0, [("vllm:x", {"engine": "0"}, 1.5)])
        writer.write_batch(3.0, 999, 2, real_frames[0][1])

    reader = CaptureReader(path)
    assert reader.header["vllm_version"] == "0.28.0"
    assert reader.header["redacted"] is False
    batches = list(reader)
    assert batches[:3] == [vllm.decode_batch(seq, payload) for seq, payload in real_frames]
    assert (batches[3].seq, batches[3].epoch) == (999, 2)
    assert list(reader.metrics()) == [(2.0, [("vllm:x", {"engine": "0"}, 1.5)])]
    assert list(reader) == batches, "a second pass re-reads the file"


def test_gzip_is_detected_by_content_not_name(tmp_path, real_frames):
    path = tmp_path / "capture.gz"
    with CaptureWriter(path, {}) as writer:
        writer.write_batch(1.0, 0, 0, real_frames[0][1])
    assert path.read_bytes()[:2] == b"\x1f\x8b"
    assert len(list(CaptureReader(path))) == 1


def test_reader_rejects_files_that_are_not_captures(tmp_path):
    bad = tmp_path / "x.jsonl"
    bad.write_text("not json\n")
    with pytest.raises(ConfigError):
        CaptureReader(bad)
    wrong = tmp_path / "y.jsonl"
    wrong.write_text('{"kind":"header","format":99}\n')
    with pytest.raises(ConfigError):
        CaptureReader(wrong)


def test_resuming_appends_without_a_second_header(tmp_path):
    path = tmp_path / "run.jsonl.gz"
    with CaptureWriter(path, {"server": "a"}) as writer:
        writer.write_batch(1.0, 7, 0, b"first")
    with CaptureWriter(path, {"server": "b"}, resume=True) as writer:
        writer.write_batch(2.0, 8, 0, b"second")
    rows = [json.loads(line) for line in gzip.open(path, "rt")]
    assert [row["kind"] for row in rows] == ["header", "batch", "batch"]
    assert rows[0]["server"] == "a"


def test_last_batch_reports_where_to_continue(tmp_path):
    path = tmp_path / "run.jsonl.gz"
    with CaptureWriter(path, {}) as writer:
        writer.write_batch(1.0, 7, 0, b"a")
        writer.write_batch(2.0, 8, 3, b"b")
    assert CaptureReader(path).last_batch() == (8, 3)


def test_last_batch_is_none_when_nothing_was_recorded(tmp_path):
    path = tmp_path / "run.jsonl.gz"
    CaptureWriter(path, {}).close()
    assert CaptureReader(path).last_batch() is None


def test_resuming_a_redacted_capture_keeps_redacting(tmp_path, real_frames):
    path = tmp_path / "run.jsonl.gz"
    CaptureWriter(path, {}, redact_tokens=True).close()
    with CaptureWriter(path, {}, resume=True) as writer:
        writer.write_batch(1.0, 0, 0, real_frames[0][1])
    (batch,) = CaptureReader(path)
    stored = [e for e in batch.events if isinstance(e, BlockStored)]
    assert stored
    assert all(e.token_ids == () for e in stored)


def test_open_for_resume_finds_nothing_to_continue(tmp_path):
    assert open_for_resume(tmp_path / "missing.jsonl") is None
    empty = tmp_path / "empty.jsonl"
    empty.touch()
    assert open_for_resume(empty) is None


def test_open_for_resume_continues_after_the_last_batch(tmp_path):
    path = tmp_path / "run.jsonl.gz"
    with CaptureWriter(path, {}) as writer:
        writer.write_batch(1.0, 7, 2, b"a")
    assert open_for_resume(path) == Resume(seq=8, epoch=2)


def test_open_for_resume_refuses_a_file_kvstat_did_not_write(tmp_path):
    foreign = tmp_path / "notes.jsonl"
    foreign.write_text("hello\n")
    with pytest.raises(ConfigError, match="--overwrite"):
        open_for_resume(foreign)
