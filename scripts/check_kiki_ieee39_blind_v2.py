"""Check the frozen IEEE39 reuse bundle without decoding trajectories."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parents[1] / "data" / "kiki_ieee39_blind_v2"


def _sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _check_layout(path, manifest, schema, rows):
    import h5py

    layout = schema["hdf5"]
    with h5py.File(path, "r") as handle:
        expected = {"dataset_id": schema["dataset_id"], "case_sha256": manifest["case_sha256"],
                    "postfault_signature": manifest["restored_system_signature"],
                    "sample_interval_s": schema["clock"]["sample_interval_s"]}
        if any(handle.attrs.get(key) != value for key, value in expected.items()):
            raise ValueError("HDF5 identity differs from the sealed manifest")
        if set(handle["events"]) != {row["event_id"] for row in rows}:
            raise ValueError("HDF5 event keys differ from the sealed roster")

        def check_array(dataset, shape):
            if dataset.shape != tuple(shape) or dataset.dtype.name != layout["dtype"]:
                raise ValueError(f"HDF5 array layout differs for {dataset.name}")

        for name, shape in layout["static_shapes"].items():
            check_array(handle[name], shape)
        for row in rows:
            group = handle[layout["event_group"].format(event_id=row["event_id"])]
            length = int(row["valid_length"])
            if (group.attrs.get("label") != row["label"] or group.attrs.get("valid_length") != length
                    or set(group) != set(layout["event_arrays"])):
                raise ValueError(f"HDF5 event metadata differs for {group.name}")
            for name, spec in layout["event_arrays"].items():
                check_array(group[name], [length if dim == "T" else dim for dim in spec["shape"]])


def check_bundle(directory, *, verify_hdf5=False):
    directory = Path(directory).resolve()
    schema = json.loads((DATASET_DIR / "schema.json").read_bytes())
    seal = directory / schema["seal"]["file"]
    if _sha256(seal) != schema["seal"]["sha256"]:
        raise ValueError("checksums.sha256 differs from the published seal")
    checks = {name: digest for digest, name in
              (line.split("  ", 1) for line in seal.read_text(encoding="utf-8").splitlines())}
    hdf5_name = schema["hdf5"]["file"]
    verified = [seal.name]
    for name, digest in checks.items():
        if name == hdf5_name and not verify_hdf5:
            continue
        if _sha256(directory / name) != digest:
            raise ValueError(f"SHA256 mismatch for {name}")
        verified.append(name)

    manifest = json.loads((directory / "manifest.json").read_bytes())
    cohort = schema["cohort"]
    splits = json.loads((directory / "splits.json").read_bytes())
    with (directory / schema["index"]["file"]).open(encoding="utf-8", newline="") as handle:
        index = csv.DictReader(handle)
        if index.fieldnames != list(schema["index"]["columns"]):
            raise ValueError("CSV columns differ from the data dictionary")
        rows = list(index)
    ids = [row["event_id"] for row in rows]
    labels = Counter(row["label"] for row in rows)
    groups = {tuple(row[key] for key in cohort["group_columns"]) for row in rows}
    if (manifest["dataset_id"] != schema["dataset_id"] or len(ids) != cohort["event_count"]
            or len(set(ids)) != len(ids) or any(not re.fullmatch(cohort["event_id_pattern"], value) for value in ids)
            or splits != {cohort["role"]: ids} or labels != cohort["labels"] or len(groups) != cohort["group_count"]
            or manifest["cases"] != len(ids) or manifest["labels"] != labels
            or manifest["sample_interval_s"] != schema["clock"]["sample_interval_s"]):
        raise ValueError("metadata differs from the registered cohort, clock or roster")
    for row in rows:
        low, high = cohort["event_length_by_label"][row["label"]]
        if not low <= int(row["valid_length"]) <= high:
            raise ValueError(f"invalid record length for {row['event_id']}")
    hdf5_path = directory / hdf5_name
    if verify_hdf5:
        _check_layout(hdf5_path, manifest, schema, rows)
    return {"status": "PASS", "dataset_id": schema["dataset_id"], "directory": str(directory),
            "events": len(ids), "labels": dict(labels), "fault_groups": len(groups), "verified_files": verified,
            "hdf5_present": hdf5_path.is_file(), "hdf5_sha256_and_layout_verified": verify_hdf5,
            "trajectory_arrays_decoded": 0, "seal_sha256": schema["seal"]["sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR, help="directory containing the sealed files")
    parser.add_argument("--verify-hdf5", action="store_true", help="check HDF5 file SHA256 and dataset shapes/dtypes")
    options = parser.parse_args()
    try:
        checked = check_bundle(options.dataset_dir, verify_hdf5=options.verify_hdf5)
    except (ImportError, KeyError, OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(checked, indent=2))


if __name__ == "__main__":
    main()
