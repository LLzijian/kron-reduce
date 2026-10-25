from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from kron_reduce.data import pm_impulse_reader as reader_module
from kron_reduce.data.limited_hdf_audit import ROOT_ATTRS
from kron_reduce.data.pm_impulse_reader import FAMILIES, RUN, SEEDS, PmImpulseReader, validate_plan


def _plan() -> dict:
    npz_files = []
    for path, family in [(f"results/{RUN}/permitted_inputs.npz", "inputs")]:
        npz_files.append(_npz_row(path, family))
    for seed in SEEDS:
        npz_files.append(_npz_row(f"results/{RUN}/static-{seed}.npz", "static"))
        for index in range(258):
            npz_files.append(_npz_row(
                f"results/{RUN}/seed-{seed}-event-{index:03d}/integral.npz", "integral"
            ))
    events = []
    for index in range(258):
        source = "ieee39_expansion" if index < 129 else "ieee39_postfault"
        raw = f"event_{index:03d}"
        events.append({
            "event_id": f"{source}/{raw}", "source_dataset": source,
            "role": "train", "source_role": "train", "cache_index": index,
            "raw_id": raw, "declared_length": 12,
        })
    sources = {
        source: {"path": f"{source}/dataset.h5", "size_bytes": 1, "sha256": "b" * 64}
        for source in ("ieee39_expansion", "ieee39_postfault")
    }
    return {
        "npz_files": npz_files, "events": events, "source_inventory": sources,
        "metadata_files": [],
    }


def _npz_row(path: str, family: str) -> dict:
    return {
        "path": path, "size_bytes": 1, "sha256": "a" * 64,
        "members": [
            {"name": name, "shape": list(shape), "dtype": "<f8"}
            for name, shape in FAMILIES[family].items()
        ],
    }


def _reader(tmp_path: Path, plan: dict | None = None) -> PmImpulseReader:
    root, source_root = tmp_path / "results", tmp_path / "source"
    root.mkdir(parents=True, exist_ok=True)
    source_root.mkdir(parents=True, exist_ok=True)
    return PmImpulseReader(root, source_root, _plan() if plan is None else plan, tmp_path / "audit")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda plan: plan["npz_files"].__setitem__(1, plan["npz_files"][0]), "path whitelist"),
        (lambda plan: plan["npz_files"][0]["members"][0].update(shape=[258, 10]), "descriptor differs"),
        (lambda plan: plan["events"][0].update(role="val"), "event whitelist"),
        (lambda plan: plan["events"][0].update(raw_id="../escape"), "event whitelist"),
    ],
)
def test_plan_rejects_duplicate_shape_nontrain_and_unsafe_raw_id(mutate, message):
    plan = _plan()
    mutate(plan)
    with pytest.raises(ValueError, match=message):
        validate_plan(plan)


def test_payload_requires_initial_integrity_freeze_and_unknown_or_repeat_is_denied(tmp_path):
    blocked_reader = _reader(tmp_path / "blocked")
    row = blocked_reader.plan["npz_files"][0]
    with pytest.raises(PermissionError, match="initial integrity freeze"):
        blocked_reader.read_npz(row)
    assert blocked_reader.failed
    assert row["path"] not in blocked_reader.opened

    unknown_reader = _reader(tmp_path / "unknown")
    unknown_reader.hash_passes = {"before"}
    row = unknown_reader.plan["npz_files"][0]
    unknown = dict(row)
    unknown["path"] = "results/unregistered.npz"
    with pytest.raises(PermissionError, match="unregistered"):
        unknown_reader.read_npz(unknown)
    assert unknown_reader.failed

    reader = _reader(tmp_path / "success")
    reader.hash_passes = {"before"}
    row = reader.plan["npz_files"][0]
    npz_path = tmp_path / "success" / "results" / row["path"]
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, history_time=np.zeros((258, 11)), constant_power=np.zeros((258, 10)), mass=np.ones(10))
    result = reader.read_npz(row)
    assert set(result) == {"history_time", "constant_power", "mass"}
    with pytest.raises(RuntimeError, match="already spent"):
        reader.read_npz(row)


@pytest.mark.parametrize("bad_array", [
    lambda: np.zeros((258, 11), dtype=object),
    lambda: np.zeros((258, 10), dtype=np.float32),
    lambda: np.zeros((258, 10), dtype=np.float64),
])
def test_npz_bad_header_fails_before_payload_and_unlisted_member_is_not_decoded(tmp_path, bad_array):
    reader = _reader(tmp_path)
    reader.hash_passes = {"before"}  # Synthetic reader-unit isolation; no integrity pass is performed.
    row = reader.plan["npz_files"][0]
    path = tmp_path / "results" / row["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, history_time=bad_array(), constant_power=np.zeros((258, 10)), mass=np.ones(10), extra=np.ones(3))
    with pytest.raises((ValueError, TypeError)):
        reader.read_npz(row)
    snap = reader.ledger.snapshot()
    assert snap["failed"]
    history_key = row["path"] + "::history_time"
    slots = {item["key"]: item for item in snap["selectors"]}
    assert slots[history_key + "::descriptor"]["status"] == "failed"
    assert slots[history_key + "::array"]["status"] == "not_attempted"
    assert not any("extra" in item["key"] for item in snap["selectors"])


def test_npz_journal_records_descriptor_before_payload(tmp_path):
    reader = _reader(tmp_path)
    reader.hash_passes = {"before"}  # Synthetic reader-unit isolation; payload is a tiny temporary NPZ.
    row = reader.plan["npz_files"][0]
    path = tmp_path / "results" / row["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, history_time=np.zeros((258, 11)), constant_power=np.zeros((258, 10)), mass=np.ones(10), extra=np.ones(1))
    reader.read_npz(row)
    transitions = [json.loads(line) for line in (tmp_path / "audit" / "array_access.jsonl").read_text(encoding="utf-8").splitlines()]
    keys = [entry["key"] for entry in transitions]
    assert keys.index(row["path"] + "::history_time::descriptor") < keys.index(row["path"] + "::history_time::array")
    assert not any("extra" in key for key in keys)


class _Attrs:
    def __init__(self, values):
        self.values = values
        self.read = []

    def __getitem__(self, name):
        self.read.append(name)
        return self.values[name]

    def keys(self):
        raise AssertionError("HDF attribute enumeration is forbidden")


class _Dataset:
    def __init__(self, value, log, label):
        self.value = np.asarray(value, dtype="<f8")
        self.log, self.label = log, label

    @property
    def shape(self):
        self.log.append(("shape", self.label))
        return self.value.shape

    @property
    def dtype(self):
        self.log.append(("dtype", self.label))
        return self.value.dtype

    def __getitem__(self, selection):
        self.log.append(("payload", self.label, selection))
        return self.value[selection]


class _Group:
    def __init__(self, entries):
        self.entries = entries

    def __getitem__(self, key):
        if "/" in key:
            current = self.entries
            for part in key.split("/"):
                current = current[part]
            return current
        return self.entries[key]


class _HdfContext:
    def __init__(self, entries, attrs):
        self.entries = entries
        self.attrs = attrs

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def __getitem__(self, key):
        return _Group(self.entries).__getitem__(key)


def _fake_hdf(plan):
    log = []
    event_rows = [row for row in plan["events"] if row["source_dataset"] == "ieee39_expansion"]
    events = {}
    for row in event_rows:
        raw = row["raw_id"]
        events[raw] = {
            "Pm": _Dataset(np.ones((12, 10)), log, f"events/{raw}/Pm"),
            "t": _Dataset(np.arange(12.0), log, f"events/{raw}/t"),
            "Pe": _Dataset(np.ones((12, 10)), log, f"events/{raw}/Pe"),
        }
    entries = {
        "inertia": _Dataset(np.ones(10), log, "inertia"),
        "equilibrium": {"Pm": _Dataset(np.ones(10), log, "equilibrium/Pm")},
        "events": events,
    }
    attrs = _Attrs({key: f"synthetic:{key}" for key in ROOT_ATTRS})
    return _HdfContext(entries, attrs), log, attrs


def test_hdf_reads_only_root_attrs_static_priors_and_first_11_pm_time(tmp_path, monkeypatch):
    reader = _reader(tmp_path)
    reader.hash_passes = {"before"}  # Synthetic reader-unit isolation; no HDF integrity pass is performed.
    fake, log, attrs = _fake_hdf(reader.plan)
    monkeypatch.setattr(reader_module.h5py, "File", lambda *_args, **_kwargs: fake)
    root = reader.source_root / "ieee39_expansion" / "dataset.h5"
    root.parent.mkdir(parents=True)
    root.touch()
    _, static, events = reader.read_source("ieee39_expansion", lambda *_args: None)
    assert set(attrs.read) == set(ROOT_ATTRS)
    assert set(static) == {"inertia", "equilibrium/Pm"}
    assert len(events) == 129
    assert all(row["Pm"].shape == (11, 10) and row["t"].shape == (11,) for row in events.values())
    assert not any("Pe" in item[1] for item in log)
    event_pm_reads = [item for item in log if item[0] == "payload" and item[1].startswith("events/") and item[1].endswith("/Pm")]
    assert event_pm_reads
    assert all(item[2] == (slice(0, 11), slice(None)) for item in event_pm_reads)


def test_hdf_descriptor_failure_does_not_read_payload_or_retry(tmp_path, monkeypatch):
    reader = _reader(tmp_path)
    reader.hash_passes = {"before"}
    fake, log, _ = _fake_hdf(reader.plan)
    first = reader.plan["events"][0]
    fake.entries["events"][first["raw_id"]]["Pm"] = _Dataset(np.ones((12, 9)), log, "bad-Pm")
    monkeypatch.setattr(reader_module.h5py, "File", lambda *_args, **_kwargs: fake)
    hdf_path = reader.source_root / first["source_dataset"] / "dataset.h5"
    hdf_path.parent.mkdir(parents=True)
    hdf_path.touch()
    with pytest.raises(ValueError, match="descriptor"):
        reader.read_source(first["source_dataset"], lambda *_args: None)
    assert not any(item[0] == "payload" and item[1] == "bad-Pm" for item in log)
    with pytest.raises(RuntimeError, match="poisoned"):
        reader.read_source(first["source_dataset"], lambda *_args: None)
