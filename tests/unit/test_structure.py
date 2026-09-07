from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

import kvstat

SRC = Path(kvstat.__file__).resolve().parent

# Rank by the first path segment under kvstat/. Data flows ingest -> state -> views; imports
# point the other way, never across. A new top-level module must be registered here.
LAYER = {
    "errors": 0,
    "events": 0,
    "engines": 1,
    "ingest": 1,
    "state": 2,
    "views": 3,
    "__init__": 9,
    "__main__": 9,
}
ALLOWED_LAYERS = {0: {0}, 1: {0, 1}, 2: {0, 2}, 3: {0, 2, 3}, 9: {0, 1, 2, 3, 9}}
THIRD_PARTY_LAYERS = {
    "zmq": {1, 9},
    "urllib": {1, 9},
    "http": {1, 9},
    "textual": {3, 9},
    "rich": {3, 9},
}


def _modules() -> list[tuple[str, int, set[str]]]:
    found = []
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).with_suffix("")
        segment = rel.parts[0]
        if segment not in LAYER:
            pytest.fail(f"{rel}: register its layer in LAYER before adding modules there")
        imports: set[str] = set()
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)
        found.append((".".join(("kvstat", *rel.parts)), LAYER[segment], imports))
    return found


@pytest.mark.parametrize(
    ("module", "layer", "imports"), _modules(), ids=lambda m: m if isinstance(m, str) else None
)
def test_imports_point_down_the_layers(module, layer, imports):
    for name in imports:
        parts = name.split(".")
        if parts[0] == "kvstat" and len(parts) > 1:
            target = LAYER.get(parts[1])
            assert target is not None, f"{module} imports unregistered {name}"
            assert target in ALLOWED_LAYERS[layer], f"{module} imports upward: {name}"
        elif parts[0] in THIRD_PARTY_LAYERS:
            assert layer in THIRD_PARTY_LAYERS[parts[0]], f"{module} may not import {name}"


def test_public_surface_is_exactly_all():
    expected = {
        "AllBlocksCleared",
        "BlockHash",
        "BlockRemoved",
        "BlockStored",
        "ConfigError",
        "DataSourceError",
        "DecodeError",
        "Event",
        "EventBatch",
        "EventSource",
        "IngestStats",
        "KvstatError",
        "ReconciliationError",
        "StateError",
        "__version__",
    }
    assert set(kvstat.__all__) == expected
    assert len(kvstat.__all__) == len(expected)
    for name in kvstat.__all__:
        assert hasattr(kvstat, name), name


def test_every_module_imports():
    for module, _, _ in _modules():
        importlib.import_module(module)
