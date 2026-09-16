from __future__ import annotations

import pytest

from kvstat.engines import vllm
from kvstat.errors import ConfigError
from kvstat.ingest.capture import CaptureReader, CaptureWriter
from kvstat.ingest.metrics import parse


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


def test_parse_keeps_only_engine_families():
    text = (
        "# TYPE vllm:num_requests_running gauge\n"
        'vllm:num_requests_running{engine="0"} 3.0\n'
        "# TYPE vllm:num_preemptions_total counter\n"
        'vllm:num_preemptions_total{engine="0"} 91.0\n'
        "# TYPE vllm:e2e_request_latency_seconds histogram\n"
        'vllm:e2e_request_latency_seconds_bucket{le="1.0"} 4.0\n'
        'vllm:e2e_request_latency_seconds_bucket{le="+Inf"} 5.0\n'
        "vllm:e2e_request_latency_seconds_count 5.0\n"
        "vllm:e2e_request_latency_seconds_sum 2.5\n"
        "# TYPE process_resident_memory_bytes gauge\n"
        "process_resident_memory_bytes 1000.0\n"
    )
    samples = parse(text)
    assert ("vllm:num_requests_running", {"engine": "0"}, 3.0) in samples
    assert ("vllm:num_preemptions_total", {"engine": "0"}, 91.0) in samples
    assert ("vllm:e2e_request_latency_seconds_bucket", {"le": "+Inf"}, 5.0) in samples
    assert ("vllm:e2e_request_latency_seconds_sum", {}, 2.5) in samples
    assert not [s for s in samples if s[0].startswith("process_")]
