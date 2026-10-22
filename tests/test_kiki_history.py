import csv
import json

import h5py
import numpy as np
import pytest

from kron_reduce.data.kiki_history import KikiHistorySamples, allocate_roles

RAW_UNITS = (
    "delta rad; omega absolute pu; Pm=GENROU.tm, Pe=GENROU.Pe, te=GENROU.te on system MVA base"
)


def make_source(root, name, lengths=None):
    directory = root / name
    directory.mkdir(parents=True)
    ids = [f"e{i}" for i in range(6)]
    lengths = lengths or {}
    rows = [
        {
            "event_id": eid,
            "fault_line": f"{name}_L{i}",
            "fault_bus": f"{name}_B{i}",
            "valid_length": str(lengths.get(eid, 25)),
        }
        for i, eid in enumerate(ids)
    ]
    (directory / "splits.json").write_text(
        json.dumps({"train": ids, "validation": ["val_only"], "test": ["test_only"]})
    )
    with (directory / "dataset_index.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    inertia = np.arange(1, 11, dtype=float)
    mapping = [[i, "delta", "GENROU", f"GENROU_{i + 1}"] for i in range(10)]
    with h5py.File(directory / "dataset.h5", "w") as h:
        h.attrs["raw_units"] = RAW_UNITS
        h.attrs["omega_s"] = 2 * np.pi * 60
        h.attrs["mva_base"] = 100.0
        h.attrs["case_sha256"] = "fixture-case"
        h.attrs["state_mapping"] = json.dumps(mapping)
        h["inertia"] = inertia
        eq = np.linspace(-0.2, 0.3, 10)
        h["equilibrium/delta"] = eq
        h["equilibrium/Pm"] = np.zeros(10)
        events = h.create_group("events")
        for i, row in enumerate(rows):
            n = int(row["valid_length"])
            g = events.create_group(row["event_id"])
            g["t"] = np.arange(n) * 0.01
            g["delta"] = np.tile(eq, (n, 1)) + i * 1e-4
            g["omega"] = 1 + np.tile(np.linspace(0, 0.09, 10), (n, 1))
            pm = np.full((n, 10), np.nan)
            pm[0] = np.linspace(0, 0.09, 10)
            g["Pm"] = pm
            g["Pe"] = np.full((n, 10), np.nan)
            g["te"] = np.full((n, 10), np.nan)
    return directory


def build_sources(root, lengths=None):
    for source in ("A", "B"):
        make_source(root, source, lengths)


def test_allocate_roles_deterministic_and_line_bus_disjoint(tmp_path):
    build_sources(tmp_path)
    rows, _ = allocate_roles(tmp_path, ["A", "B"], seed=324)
    rows_again, _ = allocate_roles(tmp_path, ["A", "B"], seed=324)
    assert rows == rows_again
    assert len(rows) == 12
    assert all(r["source_role"] == "train" for r in rows)
    assert not (
        {r["fault_line"] for r in rows if r["role"] == "train"}
        & {r["fault_line"] for r in rows if r["role"] == "val"}
    )
    for source in ("A", "B"):
        with (tmp_path / source / "dataset_index.csv").open(newline="") as stream:
            bus_by_id = {r["event_id"]: r["fault_bus"] for r in csv.DictReader(stream)}
        train_buses = {
            bus_by_id[r["raw_id"]]
            for r in rows
            if r["source_dataset"] == source and r["role"] == "train"
        }
        val_buses = {
            bus_by_id[r["raw_id"]]
            for r in rows
            if r["source_dataset"] == source and r["role"] == "val"
        }
        assert not train_buses & val_buses


def test_reader_units_static_priors_access_allowlist_and_eligibility(tmp_path):
    build_sources(tmp_path, {"e0": 21, "e1": 22})
    rows, bindings = allocate_roles(tmp_path, ["A"])
    rows = [dict(r, role="train") for r in rows]
    reader = KikiHistorySamples(tmp_path, rows, bindings)
    mass, samples = reader.load("train")
    expected = np.arange(1, 11, dtype=float) / (2 * np.pi * 60)
    assert np.allclose(mass, expected)
    audit = reader.access_audit()
    assert audit["static_priors"]["A"]["raw_units"] == RAW_UNITS
    assert audit["static_priors"]["A"]["generator_order"] == [f"GENROU_{i}" for i in range(1, 11)]
    assert audit["static_priors"]["A"]["mva_base"] == 100.0
    assert audit["static_priors"]["A"]["case_sha256"] == "fixture-case"
    assert {x["real_count"] for x in audit["coverage"]} >= {21, 22}
    assert [x["history_eligible"] for x in audit["coverage"] if x["real_count"] in (21, 22)] == [
        False,
        True,
    ]
    assert any(s.event_id.endswith("/e1") for s in samples)
    assert not any(s.event_id.endswith("/e0") for s in samples)
    assert all(np.isfinite(s.history.omega).all() for s in samples)
    pattern = np.linspace(0, 0.09, 10)
    expected_speed = (pattern - pattern @ mass / mass.sum()) * (2 * np.pi * 60)
    expected_power = pattern - pattern.sum() * mass / mass.sum()
    assert all(np.allclose(s.history.omega, expected_speed, rtol=0, atol=1e-12) for s in samples)
    assert all(
        np.allclose(s.history.constant_power, expected_power, rtol=0, atol=1e-12) for s in samples
    )
    assert all(np.max(np.abs(s.history.omega @ mass)) < 1e-12 for s in samples)
    assert all(x["dataset"] in {"t", "delta", "omega", "Pm"} for x in audit["accesses"])
    assert all(
        x["selection"] != "slice(None, None, None)"
        for x in audit["accesses"]
        if x["dataset"] == "Pm"
    )
    assert audit["source_val_ids_read"] == audit["test_ids_read"] == []
    with h5py.File(tmp_path / "A" / "dataset.h5") as h:
        assert "val_only" not in h["events"] and "test_only" not in h["events"]


def test_actual_short_target_is_retained_and_locked_source_holdout_rejected(tmp_path):
    build_sources(tmp_path, {"e0": 22})
    rows, bindings = allocate_roles(tmp_path, ["A"])
    rows = [dict(r, role="train") for r in rows]
    reader = KikiHistorySamples(tmp_path, rows, bindings)
    _, samples = reader.load("train")
    sample = next(s for s in samples if s.event_id.endswith("/e0"))
    assert len(sample.forecast_time) == 2
    with pytest.raises(ValueError, match="locked"):
        KikiHistorySamples(tmp_path, [dict(rows[0], source_role="validation")], bindings)
