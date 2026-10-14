from __future__ import annotations

import threading
from pathlib import Path

from click.testing import CliRunner, Result

from kvstat.__main__ import main
from kvstat.engines import vllm
from kvstat.events import BlockStored
from kvstat.ingest.capture import CaptureReader
from kvstat.ingest.metrics import fetch_server_info
from tests.integration.publisher import FakePublisher


def test_server_info_reads_version_and_cache_config(vllm_http):
    assert fetch_server_info(vllm_http) == {
        "vllm_version": "0.28.0",
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "cache_config": {"block_size": "16", "num_gpu_blocks": "8285"},
    }


def test_server_info_degrades_to_none_when_unreachable():
    assert fetch_server_info("http://127.0.0.1:9", timeout=0.5) == {
        "vllm_version": None,
        "model_name": None,
        "cache_config": None,
    }


def run_record(
    publisher: FakePublisher,
    real_payloads: list[bytes],
    vllm_http: str,
    out: Path,
    *extra: str,
) -> Result:
    frames = real_payloads
    assert publisher.replay_endpoint is not None

    def feed() -> None:
        publisher.wait_for_subscriber()
        for i in range(9):
            publisher.publish(frames[i % 3], drop=i == 4)

    feeder = threading.Thread(target=feed, daemon=True)
    feeder.start()
    result = CliRunner().invoke(
        main,
        [
            "record",
            "--endpoint",
            publisher.endpoint,
            "--replay-endpoint",
            publisher.replay_endpoint,
            "--server",
            vllm_http,
            "--out",
            str(out),
            "--duration",
            "2",
            *extra,
        ],
    )
    feeder.join(timeout=5)
    assert result.exit_code == 0, result.output
    return result


def test_record_writes_a_capture_that_reads_back_identically(
    publisher, real_payloads, vllm_http, tmp_path
):
    out = tmp_path / "cap.jsonl.gz"
    result = run_record(publisher, real_payloads, vllm_http, out)
    assert "recorded 9 batches (gaps 1, replayed 1, resyncs 0" in result.output

    reader = CaptureReader(out)
    assert reader.header["vllm_version"] == "0.28.0"
    assert reader.header["model_name"] == "Qwen/Qwen2.5-7B-Instruct"
    assert reader.header["cache_config"]["num_gpu_blocks"] == "8285"
    assert reader.header["redacted"] is False
    frames = real_payloads
    expected = [vllm.decode_batch(i, frames[i % 3]) for i in range(9)]
    assert list(reader) == expected

    scrapes = list(reader.metrics())
    assert len(scrapes) >= 1
    names = {name for _, samples in scrapes for name, _, _ in samples}
    assert "vllm:time_to_first_token_seconds_bucket" in names
    assert "vllm:num_preemptions_total" in names
    assert not any(n.startswith(("process_", "python_")) for n in names)


def test_record_with_redaction_keeps_hashes_and_drops_tokens(
    publisher, real_payloads, vllm_http, tmp_path
):
    out = tmp_path / "redacted.jsonl.gz"
    run_record(publisher, real_payloads, vllm_http, out, "--redact-tokens")

    reader = CaptureReader(out)
    assert reader.header["redacted"] is True
    frames = real_payloads
    for batch, original in zip(
        reader, [vllm.decode_batch(i, frames[i % 3]) for i in range(9)], strict=True
    ):
        assert (batch.seq, batch.ts, len(batch.events)) == (
            original.seq,
            original.ts,
            len(original.events),
        )
        for got, want in zip(batch.events, original.events, strict=True):
            assert type(got) is type(want)
            if isinstance(got, BlockStored):
                assert isinstance(want, BlockStored)
                assert got.token_ids == ()
                assert (got.block_hashes, got.parent_block_hash, got.block_size) == (
                    want.block_hashes,
                    want.parent_block_hash,
                    want.block_size,
                )
            else:
                assert got == want


def test_recording_again_continues_the_capture_and_replays_the_hole(
    publisher, real_payloads, vllm_http, tmp_path
):
    out = tmp_path / "cap.jsonl.gz"
    run_record(publisher, real_payloads, vllm_http, out)
    first = [batch.seq for batch in CaptureReader(out)]

    # kvstat is down for these, so only vLLM's replay buffer still has them.
    for payload in real_payloads:
        publisher.publish(payload)

    run_record(publisher, real_payloads, vllm_http, out)
    second = [batch.seq for batch in CaptureReader(out)]

    assert second[: len(first)] == first, "the earlier recording survived"
    assert second == sorted(set(second)), "no duplicates and no reordering across the restart"
    assert second[len(first)] == first[-1] + 1, "the batches missed while down were recovered"


def test_recording_again_refuses_to_destroy_a_file_it_did_not_write(
    publisher, real_payloads, vllm_http, tmp_path
):
    out = tmp_path / "notes.jsonl.gz"
    out.write_bytes(b"something the operator cares about")
    result = CliRunner().invoke(
        main,
        ["record", "--endpoint", publisher.endpoint, "--server", vllm_http, "--out", str(out)],
    )
    assert result.exit_code != 0
    assert "--overwrite" in result.output
    assert out.read_bytes() == b"something the operator cares about"


def test_overwrite_starts_the_capture_again(publisher, real_payloads, vllm_http, tmp_path):
    out = tmp_path / "cap.jsonl.gz"
    run_record(publisher, real_payloads, vllm_http, out)
    first = [batch.seq for batch in CaptureReader(out)]
    run_record(publisher, real_payloads, vllm_http, out, "--overwrite")
    assert len([batch.seq for batch in CaptureReader(out)]) <= len(first) + 1


def test_resuming_keeps_a_redacted_capture_redacted(publisher, real_payloads, vllm_http, tmp_path):
    out = tmp_path / "cap.jsonl.gz"
    run_record(publisher, real_payloads, vllm_http, out, "--redact-tokens")
    run_record(publisher, real_payloads, vllm_http, out)
    tokens = sum(
        len(event.token_ids)
        for batch in CaptureReader(out)
        for event in batch.events
        if isinstance(event, BlockStored)
    )
    assert tokens == 0
