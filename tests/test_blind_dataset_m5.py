import csv
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

from kron_reduce.data import blind_dataset_m5 as blind
from kron_reduce.workflows.blind_confirmation_m5 import _load_blind_events
from scripts.check_kiki_ieee39_blind_v2 import DATASET_DIR, check_bundle


def test_seal_retains_failures_excludes_first_frame_collisions_and_reserves_quota(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mapping = [[i, f"synthetic_{i}", "unit", "G1"] for i in range(blind.FRAME_WIDTH)]
    history_file = tmp_path / "history.h5"
    state = np.ones(blind.FRAME_WIDTH)
    condition = {"fault_bus": 5, "fault_line": "L1", "fault_location": 0., "fault_duration_s": .15,
                 "fault_xf": .001, "label": "stable"}
    with h5py.File(history_file, "x") as f:
        f.attrs.update(andes_version="2.0.0", case_sha256="synthetic_case", postfault_signature="synthetic_system",
                       state_mapping=json.dumps(mapping))
        item = f.create_group("events/old")
        item.attrs.update(condition)
        item.attrs["record_start_delay_s"] = .0001
        item.create_dataset("full_state", data=np.stack((state, np.full_like(state, np.nan))))
    history = {"roster": [{**condition, "source": "synthetic", "id": "synthetic/old", "event_id": "old", "role": "train", "valid_length": "2",
                           "record_start_delay_s": ".0001"}], "configurations": [{}],
               "hdf_paths": {"synthetic": history_file}}
    monkeypatch.setattr(blind, "load_history_metadata", lambda _: history)
    monkeypatch.setattr(blind, "HISTORY_SOURCES", ("synthetic",))

    m = {"case_sha256": "synthetic_case", "postfault_signature": "synthetic_system",
         "state_mapping": json.dumps(mapping), "raw_units": "synthetic"}
    (tmp_path / "registry.json").write_text(json.dumps({"shards": {"ieee39_postfault": {"attributes": m}}}))
    for name in ("source.json", "generator.py", "config.json",
                 "kiki/src/kiki/data/private_dataset.py", "kiki/src/kiki/physics/coordinates.py"):
        path = Path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")

    outcomes = [("failure", 0., 1), ("exact", 1., 1), ("rounded", 1. + 4e-11, 1)]
    outcomes += [("stable", float(i + 3), i + 7 if i < 4 else 5) for i in range(62)]
    outcomes += [("early", float(i + 70), 5) for i in range(29)]
    outcomes += [("late", float(i + 100), bus) for i, bus in enumerate((10, 11, 12, 12))]
    outcomes += [("never_attempted", 0., 1)]
    events = [{"id": f"blind_{i:04d}", "bus": bus, "line": "L1", "location": 0., "duration_s": .1,
               "xf": .001, "unseen_train_val_bus": kind == "stable" and 7 <= bus <= 10,
               "near_historical_transition": kind == "late"} for i, (kind, _, bus) in enumerate(outcomes, start=1)]
    monkeypatch.setattr(blind, "prepare_candidates", lambda _: events)
    monkeypatch.setattr(blind, "MAX_ATTEMPTS", len(events))

    def simulate(event, _config, _workdir, *, diagnostics):
        i = int(event["id"].split("_")[1]) - 1
        kind, initial, bus = outcomes[i]
        assert kind != "never_attempted"
        if kind == "failure":
            diagnostics.update(failed_steps=1, converged_steps=20)
            raise ValueError("numerical_TDS_failure")
        stable = kind == "stable"
        n = 1501 if stable else 41 if kind == "late" else 16
        t = np.arange(n) * .01
        arrays = {"t": t, "full_state": np.broadcast_to([initial] * blind.FRAME_WIDTH, (n, blind.FRAME_WIDTH))}
        arrays |= {key: np.zeros((n, 10)) for key in ("delta", "omega", "Pm")}
        meta = {"fault_bus": bus, "fault_line": event["line"], "fault_location": event["location"],
                "fault_duration_s": event["duration_s"], "fault_xf": event["xf"],
                "record_start_delay_s": .0001, "event_time_s": "" if stable else t[-1],
                "label": "stable" if stable else "unstable", "solver_success": True, "failed_steps": 0,
                "postfault_signature": "synthetic_system", "valid_length": n,
                "termination_reason": "stable_horizon_reached" if stable else "physical_loss_of_synchronism"}
        return arrays, meta, {"delta": np.zeros(10), "Pm": np.zeros(10)}, np.ones(10), mapping, []

    monkeypatch.setattr(blind, "_simulate_call", lambda *_: (simulate, {
        "andes_version": "2.0.0", "sample_interval_s": .01, "postfault_horizon_s": 15.}))
    keys = {"source_contract": "source.json", "registry_manifest": "registry.json",
            "generator": "generator.py", "generator_config": "config.json"}
    bound = [*keys.values(), "kiki/src/kiki/data/private_dataset.py", "kiki/src/kiki/physics/coordinates.py"]
    plan = keys | {"candidates": events, "kiki_src": "kiki/src", "file_sha256": {p: blind._sha(p) for p in bound},
                   "history_hdf": {"synthetic": {"path": str(history_file), "bytes": history_file.stat().st_size}}}
    evidence, dataset = tmp_path / "receipt", tmp_path / "sealed"
    evidence.mkdir()
    result = blind.generate_and_seal(plan, dataset, solver_workdir="scratch/one_off/solver",
                                     evidence_directory=evidence)
    assert result["labels"] == {"stable": 62, "unstable": 32}
    assert result["late_instability_count"] == 4
    assert result["attempts"] == len(events) - 1
    assert result["dataset_manifest_sha256"] == blind._sha(dataset / "manifest.json")
    assert result["dataset_checksums_sha256"] == blind._sha(dataset / "checksums.sha256")
    rows = [json.loads(line) for line in (evidence / "attempts.jsonl").read_text().splitlines()]
    returned = [item for item in rows if item["phase"] == "returned"]
    assert [row["status"] for row in returned[:3]] == ["simulator_failure", "clearing_frame_overlap", "clearing_frame_overlap"]
    assert returned[0]["failed_steps"] == 1
    assert returned[1]["collision_ids"] == returned[2]["collision_ids"] == ["synthetic/old"]
    assert [item["status"] for item in returned].count("allocated_quota_full") == 1
    ids = json.loads((dataset / "splits.json").read_bytes())["blind"]
    assert len(ids) == len(set(ids)) == 94
    with (dataset / "dataset_index.csv").open(newline="") as stream:
        assert ids == [row["event_id"] for row in csv.DictReader(stream)]
    with h5py.File(dataset / "dataset.h5", "r") as f:
        assert list(f["events"]) == ids
        assert f["events"][ids[0]]["t"].shape == (1501,)
    assert {p.name for p in dataset.iterdir()} == {
        "dataset.h5", "dataset_index.csv", "splits.json", "qa.json", "qa_report.md",
        "disjointness_audit.json", "manifest.json", "checksums.sha256",
    }
    for line in (dataset / "checksums.sha256").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert blind._sha(dataset / name) == digest
    access = []
    with h5py.File(dataset / "dataset.h5", "r") as f:
        hash_binding = {"dataset.h5": blind._sha(dataset / "dataset.h5")}
        assert f.attrs["dataset_id"] == blind.DATASET
    events = _load_blind_events(dataset, hash_binding, access)
    assert len(events) == 94 and len(events[0]["x"]) == 811
    assert all(len(event["x"]) < 811 for event in events if event["label"] == "unstable")
    assert events[0]["group"] == "bus_7/L1"
    assert sum(row["stage"] == "event_decoded" for row in access) == 94


def test_reuse_check_supports_metadata_only_and_rejects_changed_payload(tmp_path):
    manifest = (DATASET_DIR / "checksums.sha256").read_text(encoding="utf-8")
    names = [line.split("  ", 1)[1] for line in manifest.splitlines()]
    for name in ("checksums.sha256", *names):
        if name != "dataset.h5":
            shutil.copyfile(DATASET_DIR / name, tmp_path / name)

    result = check_bundle(tmp_path)
    assert result["events"] == 94 and result["fault_groups"] == 21
    assert result["labels"] == {"stable": 62, "unstable": 32}
    assert not result["hdf5_present"] and result["trajectory_arrays_decoded"] == 0

    (tmp_path / "dataset.h5").write_bytes(b"corrupt copy")
    with pytest.raises(ValueError, match="SHA256 mismatch for dataset.h5"):
        check_bundle(tmp_path, verify_hdf5=True)
