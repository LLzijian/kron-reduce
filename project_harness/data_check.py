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


def _verify_checksum_manifest(
    dataset_dir: Path,
    checksum_name: str,
    *,
    dataset_name: str,
) -> list[dict[str, str | bool]]:
    checksum_path = dataset_dir / checksum_name
    if not checksum_path.is_file():
        return [
            {
                "dataset": dataset_name,
                "path": checksum_name,
                "expected": "PRESENT",
                "actual": "MISSING",
                "pass": False,
            }
        ]
    findings: list[dict[str, str | bool]] = []
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(None, 1)
        relative = relative.strip()
        path = dataset_dir / relative
        actual = sha256_file(path) if path.is_file() else "MISSING"
        findings.append(
            {
                "dataset": dataset_name,
                "path": relative,
                "expected": expected,
                "actual": actual,
                "pass": actual == expected,
            }
        )
    return findings


def verify_data(root: Path) -> dict:
    manifest = load_manifest(root)
    default_name = manifest["default_dataset"]
    entry = manifest["datasets"][default_name]
    dataset_dir = root / entry["path"]
    findings = _verify_checksum_manifest(
        dataset_dir,
        entry["checksum_manifest"],
        dataset_name=default_name,
    )
    auxiliary: dict[str, dict[str, object]] = {}
    for name, candidate in manifest["datasets"].items():
        if name == default_name or candidate.get("status") != "ready":
            continue
        checksum_name = candidate.get("checksum_manifest")
        if not checksum_name:
            auxiliary[name] = {"status": "FAIL", "reason": "checksum manifest undeclared"}
            continue
        candidate_findings = _verify_checksum_manifest(
            root / candidate["path"],
            checksum_name,
            dataset_name=name,
        )
        findings.extend(candidate_findings)
        auxiliary[name] = {
            "status": "PASS" if all(row["pass"] for row in candidate_findings) else "FAIL",
            "files": len(candidate_findings),
            "revision": candidate.get("revision"),
        }
    with h5py.File(dataset_dir / entry["h5"], "r") as handle:
        event_count = len(handle["events"])
        revision = str(handle["metadata"].attrs["dataset_revision"])
        build_id = str(handle["metadata"].attrs["build_id"])
    schema = json.loads((dataset_dir / "schema.json").read_text(encoding="utf-8"))
    passed = (
        all(row["pass"] for row in findings)
        and all(item["status"] == "PASS" for item in auxiliary.values())
        and event_count == entry["expected_events"]
        and revision == entry["revision"]
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "files": len(findings),
        "event_count": event_count,
        "dataset_revision": revision,
        "build_id": build_id,
        "schema_event_count": schema["n_active_events"],
        "auxiliary_datasets": auxiliary,
        "findings": findings,
    }
