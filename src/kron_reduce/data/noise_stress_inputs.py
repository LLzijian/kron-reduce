"""Hash-bound input access for the measurement-noise study."""

from __future__ import annotations

import csv
import io
import json
import re

import h5py
import numpy as np

from kron_reduce.data.task3_full import REFERENCE, Task3FullReader
from kron_reduce.physics.coordinates import coi_project, coi_project_power
from kron_reduce.workflows.blind_confirmation_m5 import _sealed_paths
from project_harness.evidence import sha256_file


class DevelopmentCache(Task3FullReader):
    """Only decode selected development members of the sealed TASK3 cache."""

    def __init__(self, root, source, journal):
        # The study verifies the entire historical source contract against M5.
        # The legacy constructor also reads holdout index metadata.
        self.root = root
        self.plan = {"files": {key: source["files"][key] for key in ("static", "development")}}
        self.journal = journal
        self.static = None
        self._rosters = {key: source["rosters"][key] for key in ("train", "val")}

    def load(self, role, freeze_path=None):
        if role not in self._rosters or self.static is not None or freeze_path is not None:
            raise PermissionError("development reader accepts one train or val cohort")
        expected = {"train": 346, "val": 107}[role]
        roster = self._rosters[role]
        if len(roster) != expected or len({item["id"] for item in roster}) != expected:
            raise ValueError("incomplete frozen development roster")
        self._static()
        self.journal.append({"stage": "cohort_reserved", "role": role, "count": expected})
        with np.load(io.BytesIO(self._bytes("development")), allow_pickle=False) as arrays:
            events = [self._event(item, arrays, item["raw_id"] + "_", role) for item in roster]
        self.journal.append({"stage": "cohort_complete", "role": role, "count": len(events)})
        return events


def fixed_test_events(root, plan, source, journal, freeze_path):
    if sha256_file(root / REFERENCE) != plan["split_reference_sha256"]:
        raise ValueError("the official TASK3 roster changed")
    reader = Task3FullReader(root, source | {"allowed_roles": ["test"]}, journal)
    return reader.load("test", freeze_path=freeze_path)


def blind_events(root, plan, journal):
    """Decode the sealed M5 rows, retaining preprojection mechanical power."""
    basedir, hashes = _sealed_paths(root, {"dataset_checksums_sha256": plan["blind_checksums_sha256"]})
    if hashes["manifest.json"] != plan["blind_manifest_sha256"]:
        raise ValueError("the sealed blind population differs from M5")

    def read_json(name):
        return json.loads((basedir / name).read_bytes())

    manifest = read_json("manifest.json")
    roster = read_json("splits.json")["blind"]
    with (basedir / "dataset_index.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    dataset = "kiki_ieee39_blind_v2"
    if (manifest["dataset_id"] != dataset or manifest["cases"] != 94
            or manifest["labels"] != {"unstable": 32, "stable": 62}
            or len(roster) != len(set(roster)) or len(roster) != 94
            or any(re.fullmatch(r"blind_[0-9]{4}", item) is None for item in roster)
            or [row["event_id"] for row in rows] != roster
            or read_json("qa.json")["status"] != "PASS"
            or read_json("disjointness_audit.json")["accepted_overlap_count"] != 0):
        raise ValueError("invalid M5 roster metadata")

    events = []
    journal.append({"stage": "decode_container_reserved", "role": "blind_v2", "events": 94,
                    "sha256": hashes["dataset.h5"]})
    with h5py.File(basedir / "dataset.h5", "r") as store:
        bindings = {"dataset_id": dataset, "case_sha256": manifest["case_sha256"],
                    "postfault_signature": manifest["restored_system_signature"],
                    "sample_interval_s": .01, "omega_s": 2 * np.pi * 60}
        if set(store["events"]) != set(roster) or any(store.attrs[k] != v for k, v in bindings.items()):
            raise ValueError("the sealed HDF5 header differs from the M5 manifest")
        omega_s = float(store.attrs["omega_s"])
        mass = store["inertia"][...] / omega_s
        delta_star = store["equilibrium/delta"][...]
        pm_star = store["equilibrium/Pm"][...]
        if any(x.shape != (10,) or not np.isfinite(x).all() for x in (mass, delta_star, pm_star)):
            raise ValueError("invalid sealed ten-machine equilibrium")
        equilibrium = coi_project(delta_star, mass)
        for row in rows:
            name, label = row["event_id"], row["label"]
            canonical_id = f"{dataset}/{name}"
            event = store["events"][name]
            length = min(int(row["valid_length"]), 811)
            if (label not in ("stable", "unstable") or length < 12
                    or (label == "stable" and length != 811)
                    or any(str(event.attrs[k]) != row[k]
                           for k in ("label", "valid_length", "event_time_s", "fault_bus", "fault_line"))):
                raise ValueError(f"sealed index mismatch: {canonical_id}")
            journal.append({"stage": "event_decode_reserved", "role": "blind_v2", "id": canonical_id,
                            "selection": {"t": length, "delta": length, "omega": length, "Pm_row": 10}})
            t = event["t"][:length]
            raw_q = coi_project(event["delta"][:length] - delta_star, mass)
            raw_v = (event["omega"][:length] - 1.) * omega_s
            raw_p0 = event["Pm"][10] - pm_star
            raw_x = np.column_stack((raw_q, raw_v))
            x = np.column_stack((coi_project(raw_q, mass), coi_project(raw_v, mass)))
            if (not np.array_equal(t, np.arange(length) * .01) or x.shape != (length, 20)
                    or not np.isfinite(x).all() or not np.isfinite(raw_p0).all()):
                raise ValueError(f"invalid sealed samples: {canonical_id}")
            events.append({"id": canonical_id, "role": "blind_v2", "label": label,
                           "group": f"bus_{row['fault_bus']}/{row['fault_line']}",
                           "raw_x": raw_x, "raw_p0": raw_p0, "x": x, "mass": mass,
                           "p0": coi_project_power(raw_p0, mass), "equilibrium": equilibrium,
                           "omega_s": omega_s,
                           "event_time_s": float(row["event_time_s"]) if label == "unstable" else None})
            journal.append({"stage": "event_decoded", "role": "blind_v2", "id": canonical_id})
    if len({event["group"] for event in events}) != 21 or sum(e["label"] == "unstable" for e in events) != 32:
        raise ValueError("the M5 fault groups or truth labels changed")
    return events
