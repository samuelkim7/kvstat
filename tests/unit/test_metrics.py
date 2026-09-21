from __future__ import annotations

from kvstat.ingest.metrics import parse


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
