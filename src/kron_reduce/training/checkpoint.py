from __future__ import annotations

import hashlib
import json
from pathlib import Path


def fingerprint_config(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def assert_cache_matches(path: Path, expected_fingerprint: str) -> None:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("fingerprint") != expected_fingerprint:
        raise ValueError("checkpoint/cache fingerprint mismatch")
