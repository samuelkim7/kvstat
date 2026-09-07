"""kvstat's exception hierarchy.

KvstatError
├── DataSourceError       the engine, a socket, a capture or the metrics endpoint failed us
│   └── DecodeError       bytes that are not a KV event batch kvstat understands
├── StateError            the block table or prefix tree contradicted itself: a kvstat bug
├── ReconciliationError   the drift check could not run or could not recover
└── ConfigError           a flag, endpoint or capture header kvstat cannot act on
"""

from __future__ import annotations


class KvstatError(Exception):
    """Root of every exception kvstat raises on purpose."""


class DataSourceError(KvstatError):
    """Fires when something outside kvstat stops delivering data it promised."""


class DecodeError(DataSourceError):
    """Fires when a payload is not a KV event batch in a form kvstat knows how to read."""


class StateError(KvstatError):
    """Fires when the rebuilt KV state violates one of its own invariants. Never swallowed."""


class ReconciliationError(KvstatError):
    """Fires when derived state and engine metrics cannot be compared or brought back together."""


class ConfigError(KvstatError):
    """Fires when a command-line option, endpoint or capture header cannot be used as given."""
