"""Small deterministic helpers shared by the teaching path."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> str:
    """Return a stable JSON representation suitable for hashing."""

    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    """Human provenance only; never used by a simulator's logical clock."""

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def percentile(values: list[int | float], percentile_value: float) -> float:
    """Nearest-rank percentile with a deterministic empty-value convention."""

    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = max(
        0,
        min(
            len(ordered) - 1,
            int((percentile_value / 100.0) * len(ordered) + 0.999999) - 1,
        ),
    )
    return ordered[index]


def mean(values: list[int | float]) -> float:
    return (sum(float(value) for value in values) / len(values)) if values else 0.0
