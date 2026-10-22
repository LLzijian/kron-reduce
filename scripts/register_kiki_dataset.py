"""Freeze metadata and hashes without reading any holdout trajectory arrays."""
from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from pathlib import Path

import h5py

from project_harness.evidence import sha256_file


def register(root: Path, source_root: Path) -> dict:
    destination = root / "data/kiki_ieee39_v1"
    destination.mkdir(parents=True, exist_ok=False)
    shards = {}
    all_groups = {role: {"fault_line": set(), "fault_bus": set()}
                  for role in ("train", "val", "test")}
    all_keys = set()
    signatures = set()
    for name in ("ieee39_postfault", "ieee39_expansion"):
        source = source_root / name
        local = destination / name
        local.mkdir()
        for filename in ("dataset_index.csv", "splits.json", "qa_report.md"):
            shutil.copy2(source / filename, local / filename)
        with (source / "dataset_index.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        index = {row["event_id"]: row for row in rows}
        if len(index) != len(rows):
            raise ValueError("duplicate event IDs")
        split = json.loads((source / "splits.json").read_text(encoding="utf-8"))
        roles = {"train": split["train"], "val": split["validation"], "test": split["test"]}
        seen = set()
        counts = {}
        for role, ids in roles.items():
            if len(ids) != len(set(ids)) or seen & set(ids):
                raise ValueError("duplicate or overlapping split IDs")
            seen.update(ids)
            counts[role] = dict(Counter(index[e]["label"] for e in ids))
            counts[role]["events"] = len(ids)
            for field in ("fault_line", "fault_bus"):
                all_groups[role][field].update(index[e][field] for e in ids)
        if seen != set(index):
            raise ValueError("splits do not partition the index")
        for row in rows:
            key = (row["fault_bus"], row["fault_duration_s"], row["fault_xf"])
            if key in all_keys:
                raise ValueError("duplicate physical disturbance metadata")
            all_keys.add(key)
            signatures.add(row["postfault_signature"])
        # Group names/shapes/attributes only: no event dataset[...] operations.
        with h5py.File(source / "dataset.h5", "r") as handle:
            if set(handle["events"]) != set(index):
                raise ValueError("HDF5/index IDs differ")
            for event_id, row in index.items():
                event = handle["events"][event_id]
                length = int(row["valid_length"])
                if any(event[key].shape != (length, 10)
                       for key in ("delta", "omega", "Pm", "Pe")):
                    raise ValueError("invalid channel shape")
            attributes = {key: str(value) for key, value in handle.attrs.items()}
        shards[name] = {
            "events": len(rows), "counts": counts,
            "sha256": {filename: sha256_file(source / filename)
                       for filename in ("dataset.h5", "dataset_index.csv", "splits.json")},
            "attributes": attributes,
        }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        for field in ("fault_line", "fault_bus"):
            if all_groups[left][field] & all_groups[right][field]:
                raise ValueError(f"cross-shard {field} crosses {left}/{right}")
    if len(signatures) != 1:
        raise ValueError("restored physical systems differ")
    provenance = destination / "provenance"
    provenance.mkdir()
    kiki_root = source_root.parent
    for relative in ("configs/private_dataset.json", "configs/private_dataset_expansion.json",
                     "configs/test_release_v1.json",
                     "experiments/generate_private_dataset.py", "private_data/expansion_0039_cross_audit.json"):
        shutil.copy2(kiki_root / relative, provenance / Path(relative).name)
    exposure = root.parent / "task3_baselines/paper/campaign2_final/test_exposure_notice.md"
    shutil.copy2(exposure, provenance / "test_exposure_notice.md")
    manifest = {
        "dataset_id": "kiki_ieee39_v1", "revision": "kiki_ieee39_v1_20261023",
        "source_root_env": "KIKI_DATA_ROOT", "source_root_relative": "../kiki/private_data",
        "event_count": sum(s["events"] for s in shards.values()),
        "shards": shards, "group_isolation": "fault_bus_and_fault_line_across_all_shards",
        "unique_disturbance_metadata": len(all_keys),
        "restored_system_signature": next(iter(signatures)),
        "holdout_payload_arrays_read_by_registration": 0,
        "test_status": "prior_exposure_detected_not_fresh",
        "prospective_test_allowed": False,
        "test_exposure_evidence": "provenance/test_exposure_notice.md",
        "claim_boundary": "One operating point with restored topology; not multi-operating-point or Level 6 boundary evidence.",
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    files = sorted(p for p in destination.rglob("*") if p.is_file())
    (destination / "checksums.sha256").write_text(
        "".join(f"{sha256_file(p)}  {p.relative_to(destination).as_posix()}\n" for p in files),
        encoding="utf-8",
    )
    return manifest


if __name__ == "__main__":
    import os

    project = Path(__file__).resolve().parents[1]
    source = Path(os.environ.get("KIKI_DATA_ROOT", project.parent / "kiki/private_data"))
    result = register(project, source)
    print(json.dumps({"events": result["event_count"], "test_status": result["test_status"]}))
