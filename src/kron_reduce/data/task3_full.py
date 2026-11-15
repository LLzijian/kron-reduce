"""Sealed 100 ms TASK3 caches with source-qualified cohorts and delayed holdouts."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from pathlib import Path

import numpy as np

from kron_reduce.data.history_contract import CausalStateHistory
from kron_reduce.physics.coordinates import coi_project, coi_project_power
from project_harness.evidence import sha256_file

REFERENCE = "configs/proposals/kiki_task3_split_reference_v1.json"
COUNTS = {"train": 346, "val": 107, "test": 76}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _binding(path, digest):
    _require(re.fullmatch(r"[0-9a-f]{64}", digest) is not None, "invalid source SHA-256")
    return {"path": str(Path(path).resolve()), "sha256": digest}


def _verified(binding):
    value = Path(binding["path"]).read_bytes()
    _require(hashlib.sha256(value).hexdigest() == binding["sha256"],
             f"source hash mismatch: {binding['path']}")
    return value


def source_plan(root):
    """Read metadata only; inherit cache hashes from the frozen source manifests."""
    reference_path = Path(root).resolve() / REFERENCE
    reference_bytes = reference_path.read_bytes()
    reference = json.loads(reference_bytes)
    files = {"reference": _binding(reference_path, hashlib.sha256(reference_bytes).hexdigest())}

    def metadata(name, path, digest):
        files[name] = _binding(path, digest)
        return _verified(files[name])

    final = reference["task3_campaign2_final"]
    freeze_path = Path(final["freeze_manifest_path"])
    freeze = json.loads(metadata("source_freeze", freeze_path, final["freeze_manifest_sha256"]))
    access = json.loads(metadata("source_access", final["source_path"], final["source_sha256"]))
    _require(access["status"] == "ALL_ELIGIBLE_EVENTS_CACHED"
             and access["frozen_manifest_sha256"] == final["freeze_manifest_sha256"]
             and freeze["selection_closed"] and freeze["no_post_test_training"],
             "source test cache is not frozen and complete")
    task3_root = freeze_path.parents[2]
    sealed = {name.replace("\\", "/"): digest for name, digest in freeze["files"].items()}
    for name in ("static", "development", "events"):
        relative = f"runs/campaign2/preflight/{name}.{'json' if name == 'events' else 'npz'}"
        files[name] = _binding(task3_root / relative, sealed[relative])
    cached = json.loads(_verified(files["events"]))
    rosters = {role: [] for role in COUNTS}
    indexes = {}

    def index(source, path, digest):
        name = f"index/{source}"
        if source not in indexes:
            rows = list(csv.DictReader(io.StringIO(metadata(name, path, digest).decode("utf-8-sig"))))
            indexes[source] = {row["event_id"]: row for row in rows}
            _require(len(indexes[source]) == len(rows), "duplicate source index IDs")
        else:
            _require(files[name] == _binding(path, digest), "source index bindings disagree")
        return indexes[source]

    def row(source, raw_id, indexed):
        _require(re.fullmatch(r"[a-z][a-z0-9_]+", raw_id) is not None, "unsafe event ID")
        value = indexed[raw_id]
        _require(value["label"] in {"stable", "unstable"}, "unknown frozen truth label")
        _require(float(value["trajectory_end_s"]) > 0.1 + 1e-9, "ineligible frozen event")
        event_time = value.get("event_time_s")
        return {"id": f"{source}/{raw_id}", "raw_id": raw_id,
                "source_dataset": source, "label": value["label"], "group": value["fault_bus"],
                "event_time_s": float(event_time) if event_time else None}

    for source in reference["official_kiki"]["sources"]:
        indexed = index(source["name"], source["index_path"], source["index_sha256"])
        split = json.loads(metadata(f"split/{source['name']}", source["splits_path"],
                                    source["splits_sha256"]))
        groups = [set(split[name]) for name in ("train", "validation", "test")]
        _require(not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]),
                 "source splits overlap")
        for role, original in (("train", "train"), ("val", "validation")):
            expected = source["roles"][original]["eligible_100ms"]
            actual = {f"{source['name']}/{eid}" for eid in split[original]
                      if float(indexed[eid]["trajectory_end_s"]) > 0.1 + 1e-9}
            _require(set(expected) == actual and len(expected) == len(actual),
                     "official eligible roster mismatch")
            for qualified in expected:
                name, raw_id = qualified.split("/")
                _require(name == source["name"], "source-qualified identity mismatch")
                rosters[role].append(row(name, raw_id, indexed))
    for role, original in (("train", "train"), ("val", "dev")):
        recorded = {item["id"]: item for item in cached[original]}
        _require(len(recorded) == len(cached[original]) == len(rosters[role])
                 and set(recorded) == {item["raw_id"] for item in rosters[role]},
                 "development cache roster mismatch")
        for item in rosters[role]:
            old = recorded[item["raw_id"]]
            _require(old["label"] == item["label"] and old["group"] == item["group"],
                     "cached truth metadata differs from frozen source")

    qualified_test = {}
    for source in freeze["test_data"]["roots"]:
        directory = Path(source["path"])
        indexed = index(directory.name, directory / "dataset_index.csv", source["index_sha256"])
        for raw_id in source["eligible"]:
            _require(raw_id not in qualified_test, "ambiguous raw test identity")
            qualified_test[raw_id] = row(directory.name, raw_id, indexed)
    ids = final["eligible_ids"]
    _require(len(ids) == len(set(ids)) and set(ids) == set(qualified_test) == set(access["events"])
             and ids == freeze["test_data"]["eligible_ids"], "final test roster mismatch")
    for raw_id in ids:
        entry = access["events"][raw_id]
        item = qualified_test[raw_id]
        _require(entry["status"] == "CACHED"
                 and Path(entry["root"]).name == item["source_dataset"],
                 "final test provenance mismatch")
        files[f"test/{raw_id}"] = _binding(freeze_path.parent / "test_cache" / f"{raw_id}.npz",
                                                   entry["cache_sha256"])
        rosters["test"].append(item)
    all_ids = [item["id"] for rows in rosters.values() for item in rows]
    _require(len(all_ids) == len(set(all_ids))
             and all(len(rosters[role]) == count for role, count in COUNTS.items()),
             "frozen cohorts must be disjoint train=346, val=107, test=76")
    for rows in rosters.values():
        rows.sort(key=lambda item: item["id"])
    return {"files": files, "rosters": rosters}


class Task3FullReader:
    """Consume each cohort once; failed reads cannot be silently retried."""

    def __init__(self, root, plan, journal):
        self.root = Path(root).resolve()
        self._roles = tuple(plan.get("allowed_roles", ("train", "val")))
        _require(self._roles in (("train", "val"), ("train",), ("test",)), "unregistered reader stage roles")
        _require({key: value for key, value in plan.items() if key != "allowed_roles"}
                 == source_plan(self.root), "source plan differs from frozen metadata")
        self.plan = json.loads(json.dumps(plan))
        self.journal = journal
        self._loaded = set()
        self._failed = False
        self._freeze_digest = None
        self.static = None

    def _bytes(self, name):
        binding = self.plan["files"][name]
        self.journal.append({"stage": "read_reserved", "source": name, **binding})
        value = _verified(binding)
        self.journal.append({"stage": "bytes_verified", "source": name, "bytes": len(value)})
        return value

    def _freeze(self, path):
        if path is None:
            raise PermissionError("val/test require a model and preprocessing freeze")
        path = Path(path)
        path = path.resolve() if path.is_absolute() else (self.root / path).resolve()
        _require(path.is_relative_to(self.root), "freeze is outside the registered checkout")
        value = path.read_bytes()
        digest = hashlib.sha256(value).hexdigest()
        frozen = json.loads(value)
        for name in ("checkpoint", "preprocessing"):
            relative = Path(frozen[f"{name}_path"])
            bound = (self.root / relative).resolve()
            _require(not relative.is_absolute() and bound.is_relative_to(self.root),
                     "freeze assets must be checkout-relative")
            _require(sha256_file(bound) == frozen[f"{name}_sha256"], "frozen asset hash mismatch")
        _require(self._freeze_digest in (None, digest), "freeze changed between val and test")
        self._freeze_digest = digest
        self.journal.append({"stage": "freeze_verified", "sha256": digest, "path": str(path)})

    def _static(self):
        self.journal.append({"stage": "static_decode_reserved",
                             "members": ["mass", "delta_star", "pm_star", "omega_s"]})
        with np.load(io.BytesIO(self._bytes("static")), allow_pickle=False) as arrays:
            static = {key: np.asarray(arrays[key], dtype=float)
                      for key in ("mass", "delta_star", "pm_star", "omega_s")}
        for key in ("mass", "delta_star", "pm_star"):
            _require(static[key].shape == (10,) and np.isfinite(static[key]).all(),
                     "invalid ten-generator static prior")
        _require(np.all(static["mass"] > 0) and static["omega_s"].shape == ()
                 and np.isclose(static["omega_s"], 2 * np.pi * 60, rtol=0, atol=1e-12),
                 "invalid mass or synchronous frequency")
        for value in static.values():
            value.flags.writeable = False
        self.static = static
        self.journal.append({"stage": "static_decoded"})

    def _event(self, item, arrays, prefix, role):
        self.journal.append({"stage": "decode_reserved", "role": role, "event_id": item["id"],
                             "members": [prefix + "x", prefix + "p0"]})
        raw_x = np.array(arrays[prefix + "x"], dtype=float, copy=True)
        raw_p0 = np.array(arrays[prefix + "p0"], dtype=float, copy=True)
        _require(raw_x.ndim == 2 and raw_x.shape[1] == 20 and 12 <= len(raw_x) <= 811
                 and raw_p0.shape == (10,) and np.isfinite(raw_x).all()
                 and np.isfinite(raw_p0).all(), f"invalid cached trajectory: {item['id']}")
        mass = self.static["mass"]
        x = np.c_[coi_project(raw_x[:, :10], mass), coi_project(raw_x[:, 10:], mass)]
        p0 = coi_project_power(raw_p0, mass)
        equilibrium = coi_project(self.static["delta_star"], mass)
        history = CausalStateHistory(np.arange(-10, 1) * 0.01, x[:11, :10], x[:11, 10:], p0)
        clock = np.arange(len(x) - 10) * 0.01
        for value in (raw_x, raw_p0, x, p0, equilibrium, clock):
            value.flags.writeable = False
        self.journal.append({"stage": "decoded", "role": role, "event_id": item["id"],
                             "points": len(x), "forecast_points": len(clock)})
        return {**item, "role": role, "x": x, "raw_x": raw_x, "p0": p0, "raw_p0": raw_p0,
                "mass": mass, "equilibrium": equilibrium, "omega_s": float(self.static["omega_s"]),
                "history": history, "forecast_time": clock, "target_delta": x[10:, :10],
                "target_omega": x[10:, 10:]}

    def load(self, role, freeze_path=None):
        if self._failed or role not in self._roles or role in self._loaded:
            raise PermissionError("reader failed, cohort is unknown, or cohort was already consumed")
        if role != self._roles[len(self._loaded)]:
            raise PermissionError("cohorts must be consumed in registered stage order")
        if role != "train":
            self._freeze(freeze_path)
        self.journal.append({"stage": "cohort_reserved", "role": role,
                             "count": len(self.plan["rosters"][role])})
        self._loaded.add(role)
        try:
            if self.static is None:
                self._static()
            if role in {"train", "val"}:
                with np.load(io.BytesIO(self._bytes("development")), allow_pickle=False) as arrays:
                    events = [self._event(item, arrays, item["raw_id"] + "_", role)
                              for item in self.plan["rosters"][role]]
            else:
                events = []
                for item in self.plan["rosters"][role]:
                    with np.load(io.BytesIO(self._bytes(f"test/{item['raw_id']}")),
                                 allow_pickle=False) as arrays:
                        events.append(self._event(item, arrays, "", role))
        except BaseException as error:
            self._failed = True
            self.journal.append({"stage": "cohort_failed", "role": role,
                                 "exception_type": type(error).__name__})
            raise
        self.journal.append({"stage": "cohort_complete", "role": role, "count": len(events)})
        return events
