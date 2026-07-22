from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py

from .manifest import load_manifest


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_data(root: Path) -> dict:
    manifest = load_manifest(root)
    entry = manifest["datasets"][manifest["default_dataset"]]
    dataset_dir = root / entry["path"]
    checksum_path = dataset_dir / entry["checksum_manifest"]
    findings = []
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(None, 1)
        path = dataset_dir / relative.strip()
        actual = sha256_file(path) if path.is_file() else "MISSING"
        findings.append({"path": relative.strip(), "expected": expected, "actual": actual, "pass": actual == expected})
    with h5py.File(dataset_dir / entry["h5"], "r") as handle:
        event_count = len(handle["events"])
        revision = str(handle["metadata"].attrs["dataset_revision"])
        build_id = str(handle["metadata"].attrs["build_id"])
    schema = json.loads((dataset_dir / "schema.json").read_text(encoding="utf-8"))
    passed = all(row["pass"] for row in findings) and event_count == entry["expected_events"] and revision == entry["revision"]
    return {"status": "PASS" if passed else "FAIL", "files": len(findings), "event_count": event_count, "dataset_revision": revision, "build_id": build_id, "schema_event_count": schema["n_active_events"], "findings": findings}
