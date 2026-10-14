from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from tests.integration.publisher import FakePublisher


@pytest.fixture
def publisher():
    pub = FakePublisher()
    yield pub
    pub.close()


@pytest.fixture
def publisher_without_replay():
    pub = FakePublisher(replay=False)
    yield pub
    pub.close()


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
