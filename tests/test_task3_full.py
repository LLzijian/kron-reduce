"""Synthetic caches exercise cohort identity, causal slices and access gates."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from kron_reduce.data.task3_full import Task3FullReader, source_plan
from project_harness.evidence import sha256_file


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def fixture(tmp_path):
    root = tmp_path / "project"
    task3 = tmp_path / "task3"
    preflight = task3 / "runs/campaign2/preflight"
    final = task3 / "runs/campaign2_final"
    preflight.mkdir(parents=True)
    (final / "test_cache").mkdir(parents=True)
    mass = np.arange(1, 11, dtype=float)
    q = np.arange(10, dtype=float)
    q -= q @ mass / mass.sum()
    x = np.tile(np.r_[q, q + 2.0], (12, 1))
    p0 = np.arange(10, dtype=float)
    np.savez(preflight / "static.npz", mass=mass, delta_star=q,
             pm_star=np.ones(10), omega_s=2 * np.pi * 60)
    sources, test_roots, events, access, arrays = [], [], {"train": [], "dev": []}, {}, {}
    for name, prefix, counts in (
        ("ieee39_postfault", "pilot", (217, 68, 0)),
        ("ieee39_expansion", "expansion", (129, 39, 43)),
        ("ieee39_holdout_supplement", "holdout", (0, 0, 33)),
    ):
        directory = tmp_path / "kiki/private_data" / name
        directory.mkdir(parents=True)
        split, rows, roles = {}, [], {}
        offset = 0
        for role, count in zip(("train", "validation", "test"), counts):
            ids = [f"{prefix}_{i:04}" for i in range(offset, offset + count)]
            offset += count
            split[role] = ids
            roles[role] = {"eligible_100ms": [f"{name}/{eid}" for eid in ids]}
            for eid in ids:
                label = "unstable" if eid.endswith("0000") else "stable"
                rows.append({"event_id": eid, "label": label, "fault_bus": "1",
                             "trajectory_end_s": "0.11", "event_time_s": ""})
                if role != "test":
                    events["dev" if role == "validation" else role].append(
                        {"id": eid, "label": label, "group": "1"})
                    arrays[eid + "_x"] = np.tile(x[:1], (811, 1)) if eid == "pilot_0000" else x
                    arrays[eid + "_p0"] = p0
                else:
                    cache = final / "test_cache" / f"{eid}.npz"
                    np.savez(cache, x=x, p0=p0)
                    access[eid] = {"status": "CACHED", "root": str(directory),
                                   "cache_sha256": sha256_file(cache)}
        index = directory / "dataset_index.csv"
        with index.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        write_json(directory / "splits.json", split)
        if name != "ieee39_holdout_supplement":
            sources.append({"name": name, "roles": roles, "index_path": str(index),
                            "index_sha256": sha256_file(index),
                            "splits_path": str(directory / "splits.json"),
                            "splits_sha256": sha256_file(directory / "splits.json")})
        if counts[2]:
            test_roots.append({"path": str(directory), "index_sha256": sha256_file(index),
                               "eligible": split["test"]})
    np.savez(preflight / "development.npz", **arrays)
    write_json(preflight / "events.json", events)
    freeze = {"selection_closed": True, "no_post_test_training": True,
              "files": {str(p.relative_to(task3)): sha256_file(p) for p in preflight.iterdir()},
              "test_data": {"roots": test_roots, "eligible_ids": list(access)}}
    write_json(final / "freeze_manifest.json", freeze)
    write_json(final / "holdout_access.json", {
        "status": "ALL_ELIGIBLE_EVENTS_CACHED", "events": access,
        "frozen_manifest_sha256": sha256_file(final / "freeze_manifest.json")})
    reference = {"official_kiki": {"sources": sources}, "task3_campaign2_final": {
        "freeze_manifest_path": str(final / "freeze_manifest.json"),
        "freeze_manifest_sha256": sha256_file(final / "freeze_manifest.json"),
        "source_path": str(final / "holdout_access.json"),
        "source_sha256": sha256_file(final / "holdout_access.json"), "eligible_ids": list(access)}}
    write_json(root / "configs/proposals/kiki_task3_split_reference_v1.json", reference)
    (root / "checkpoint.bin").write_bytes(b"frozen model")
    (root / "preprocessing.json").write_bytes(b"frozen train-only scales")
    write_json(root / "freeze.json", {
        "checkpoint_path": "checkpoint.bin", "checkpoint_sha256": sha256_file(root / "checkpoint.bin"),
        "preprocessing_path": "preprocessing.json",
        "preprocessing_sha256": sha256_file(root / "preprocessing.json")})
    return root


def test_metadata_planner_never_decodes_payload_and_binds_exact_roster(fixture, monkeypatch):
    monkeypatch.setattr(np, "load", lambda *a, **kw: pytest.fail("metadata must not decode arrays"))
    plan = source_plan(fixture)
    assert {role: len(rows) for role, rows in plan["rosters"].items()} == {
        "train": 346, "val": 107, "test": 76}
    assert all("/" in row["id"] for rows in plan["rosters"].values() for row in rows)
    assert len({row["id"] for rows in plan["rosters"].values() for row in rows}) == 529
    plan["rosters"]["train"][0]["label"] = "stable"
    with pytest.raises(ValueError, match="source plan differs"):
        Task3FullReader(fixture, plan, [])


def test_train_only_reader_rejects_other_cohorts_before_access(fixture):
    journal = []
    reader = Task3FullReader(fixture, source_plan(fixture) | {"allowed_roles": ["train"]}, journal)
    for role in ("val", "test"):
        with pytest.raises(PermissionError):
            reader.load(role, "freeze.json")
    assert not journal
    assert len(reader.load("train")) == 346
    assert [row["source"] for row in journal if row["stage"] == "read_reserved"] == ["static", "development"]
    with pytest.raises(PermissionError):
        reader.load("val", "freeze.json")


def test_full_reader_preserves_labels_projects_inputs_and_retains_common_frequency(fixture):
    journal = []
    reader = Task3FullReader(fixture, source_plan(fixture), journal)
    train = reader.load("train")
    event = next(item for item in train if item["raw_id"] == "pilot_0000")
    assert event["label"] == "unstable"
    assert event["history"].delta.shape == (11, 10)
    np.testing.assert_array_equal(event["history"].time, np.arange(-10, 1) * 0.01)
    np.testing.assert_allclose(event["x"][:, 10:] @ event["mass"], 0, atol=1e-12)
    np.testing.assert_allclose(event["raw_x"][:, 10:] @ event["mass"] / event["mass"].sum(), 2)
    np.testing.assert_allclose(event["p0"].sum(), 0, atol=1e-12)
    np.testing.assert_array_equal(event["target_delta"][0], event["history"].delta[-1])
    assert event["forecast_time"].shape == (801,)
    assert event["forecast_time"][-1] == 8.0
    assert train[1]["forecast_time"].tolist() == [0, 0.01]
    assert event["event_time_s"] is None
    assert not event["x"].flags.writeable
    with pytest.raises(PermissionError, match="already consumed"):
        reader.load("train")
    assert len(reader.load("val", "freeze.json")) == 107
    test_reader = Task3FullReader(fixture, {**source_plan(fixture), "allowed_roles": ["test"]}, journal)
    assert len(test_reader.load("test", "freeze.json")) == 76
    assert [row["role"] for row in journal if row["stage"] == "cohort_complete"] == [
        "train", "val", "test"]
    for i, row in enumerate(journal):
        if row["stage"] == "decoded":
            assert journal[i - 1]["stage"] == "decode_reserved"


def test_holdout_gate_precedes_payload_and_detects_changed_checkpoint(fixture, monkeypatch):
    reader = Task3FullReader(fixture, source_plan(fixture), [])
    with pytest.raises(PermissionError):
        reader.load("test", "freeze.json")
    with pytest.raises(PermissionError, match="order"):
        reader.load("val", "freeze.json")
    reader.load("train")
    monkeypatch.setattr(np, "load", lambda *a, **kw: pytest.fail("holdout must remain closed"))
    with pytest.raises(PermissionError, match="require a model"):
        reader.load("val")
    (fixture / "checkpoint.bin").write_bytes(b"changed after freeze")
    with pytest.raises(ValueError, match="frozen asset hash mismatch"):
        reader.load("val", "freeze.json")


def test_changed_cache_hash_fails_before_decode_and_blocks_retry(fixture):
    journal = []
    plan = source_plan(fixture)
    reader = Task3FullReader(fixture, plan, journal)
    Path(plan["files"]["development"]["path"]).write_bytes(b"unsealed changed cache")
    with pytest.raises(ValueError, match="source hash mismatch"):
        reader.load("train")
    assert not any(row["stage"] == "decode_reserved" for row in journal)
    assert journal[-1]["stage"] == "cohort_failed"
    with pytest.raises(PermissionError, match="reader failed"):
        reader.load("train")


def test_standalone_test_stage_requires_freeze_and_cannot_open_development(fixture, monkeypatch):
    journal = []
    plan = {**source_plan(fixture), "allowed_roles": ["test"]}
    reader = Task3FullReader(fixture, plan, journal)
    with pytest.raises(PermissionError):
        reader.load("train")
    with pytest.raises(PermissionError, match="require a model"):
        reader.load("test")
    monkeypatch.setattr(np, "load", lambda *a, **kw: pytest.fail("holdout must remain closed"))
    (fixture / "preprocessing.json").write_bytes(b"changed train scales")
    with pytest.raises(ValueError, match="frozen asset hash mismatch"):
        reader.load("test", "freeze.json")
    assert not any(row["stage"] == "read_reserved" for row in journal)


@pytest.mark.parametrize("invalid", ["nonfinite", "too_short", "too_long", "wrong_nodes"])
def test_bad_trajectory_is_explicit_failure(fixture, invalid):
    plan = source_plan(fixture)
    reader = Task3FullReader(fixture, plan, [])
    reader._static()
    x = np.zeros((12, 20))
    if invalid == "nonfinite":
        x[0, 0] = np.nan
    elif invalid == "too_short":
        x = x[:11]
    elif invalid == "too_long":
        x = np.zeros((812, 20))
    else:
        x = x[:, :19]
    with pytest.raises(ValueError, match="invalid cached trajectory"):
        reader._event(plan["rosters"]["train"][0], {"x": x, "p0": np.zeros(10)}, "", "train")
