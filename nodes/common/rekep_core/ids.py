from __future__ import annotations

import hashlib
import json
import time
from typing import Any


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


def digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value)).hexdigest()


def timestamp_ns() -> str:
    """Return Unix time in nanoseconds without exceeding JSON safe integers."""
    return str(time.time_ns())
