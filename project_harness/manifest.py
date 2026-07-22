from __future__ import annotations

import json
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_manifest(root: Path | None = None) -> dict:
    root = root or project_root()
    path = root / "configs" / "project_manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("project") != "kron-reduce":
        raise ValueError("unexpected project manifest")
    return data
