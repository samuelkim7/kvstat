from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from click.testing import CliRunner, Result

from kvstat.__main__ import main
from kvstat.engines import vllm
from kvstat.events import BlockStored
from kvstat.ingest.capture import CaptureReader
from kvstat.ingest.metrics import fetch_server_info
from tests.integration.conftest import FakePublisher

EXPOSITION = """\
# HELP vllm:cache_config_info Information of the LLMEngine CacheConfig
# TYPE vllm:cache_config_info gauge
vllm:cache_config_info{block_size="16",num_gpu_blocks="8285",engine=""} 1.0
# HELP vllm:kv_cache_usage_perc GPU KV-cache usage. 1 means 100 percent usage.
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{engine="0",model_name="Qwen/Qwen2.5-7B-Instruct"} 0.241
# HELP vllm:num_preemptions_total Cumulative number of preemption from the engine.
# TYPE vllm:num_preemptions_total counter
vllm:num_preemptions_total{engine="0",model_name="Qwen/Qwen2.5-7B-Instruct"} 91.0
# HELP vllm:time_to_first_token_seconds Histogram of time to first token in seconds.
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{le="0.1",model_name="Qwen/Qwen2.5-7B-Instruct"} 12.0
vllm:time_to_first_token_seconds_bucket{le="+Inf",model_name="Qwen/Qwen2.5-7B-Instruct"} 20.0
vllm:time_to_first_token_seconds_count{model_name="Qwen/Qwen2.5-7B-Instruct"} 20.0
vllm:time_to_first_token_seconds_sum{model_name="Qwen/Qwen2.5-7B-Instruct"} 3.5
# HELP process_cpu_seconds_total Total user and system CPU time spent in seconds.
# TYPE process_cpu_seconds_total counter
process_cpu_seconds_total 12.3
# HELP python_gc_objects_collected_total Objects collected during gc
# TYPE python_gc_objects_collected_total counter
python_gc_objects_collected_total{generation="0"} 100.0
"""


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        body = {"/version": json.dumps({"version": "0.28.0"}), "/metrics": EXPOSITION}.get(
            self.path
        )
        self.send_response(200 if body is not None else 404)
        self.end_headers()
        if body is not None:
            self.wfile.write(body.encode())

    def log_message(self, *args: object) -> None:
        return


@pytest.fixture
def vllm_http():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


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
