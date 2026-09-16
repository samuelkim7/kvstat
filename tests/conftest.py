from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="session")
def real_frames() -> list[tuple[int, bytes]]:
    """Raw payloads recorded from vLLM 0.28.0 on an H100 (KVTOP-8 smoke capture)."""
    lines = (FIXTURES / "vllm-0.28.0-frames.jsonl").read_text().splitlines()
    rows = [json.loads(line) for line in lines]
    return [(row["seq"], base64.b64decode(row["payload_b64"])) for row in rows]


@pytest.fixture(scope="session")
def real_payloads(real_frames: list[tuple[int, bytes]]) -> list[bytes]:
    return [payload for _, payload in real_frames]
