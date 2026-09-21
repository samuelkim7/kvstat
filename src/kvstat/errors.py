"""kvstat's exception hierarchy.

KvstatError
├── DataSourceError       the engine, a socket, a capture or the metrics endpoint failed us
│   └── DecodeError       a payload kvstat cannot read as a KV event batch
├── StateError            the block table or prefix tree broke its own invariant: a kvstat bug
├── CrossCheckError       the cross-check could not run or could not recover
└── ConfigError           a flag, endpoint or capture header kvstat cannot act on
"""

from __future__ import annotations


class KvstatError(Exception):
    """Root of every exception kvstat raises on purpose."""


class DataSourceError(KvstatError):
    """Raised when something outside kvstat stops delivering data it promised."""


class DecodeError(DataSourceError):
    """Raised when kvstat cannot read a payload as a KV event batch."""


class StateError(KvstatError):
    """Raised when the block table breaks one of its own invariants. Never swallowed."""


class CrossCheckError(KvstatError):
    """Raised when the block table and the engine's metrics cannot be compared."""


class ConfigError(KvstatError):
    """Raised when a command-line option, endpoint or capture header cannot be used as given."""
