"""kvtop's exception hierarchy.

KvtopError
├── DataSourceError       the engine, a socket, a capture or the metrics endpoint failed us
│   └── DecodeError       bytes that are not a KV event batch kvtop understands
├── StateError            the block table or prefix tree contradicted itself: a kvtop bug
├── ReconciliationError   the drift check could not run or could not recover
└── ConfigError           a flag, endpoint or capture header kvtop cannot act on
"""

from __future__ import annotations


class KvtopError(Exception):
    """Root of every exception kvtop raises on purpose."""


class DataSourceError(KvtopError):
    """Fires when something outside kvtop stops delivering data it promised."""


class DecodeError(DataSourceError):
    """Fires when a payload is not a KV event batch in a form kvtop knows how to read."""


class StateError(KvtopError):
    """Fires when the rebuilt KV state violates one of its own invariants. Never swallowed."""


class ReconciliationError(KvtopError):
    """Fires when derived state and engine metrics cannot be compared or brought back together."""


class ConfigError(KvtopError):
    """Fires when a command-line option, endpoint or capture header cannot be used as given."""
