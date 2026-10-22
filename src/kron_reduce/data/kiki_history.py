"""Read-only kiki TRAIN adapter with physical units and locked source holdouts."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

import h5py
import numpy as np

from kron_reduce.data.history_contract import CausalStateHistory
from kron_reduce.models.nonlinear_latent import EquilibriumLearningSample
from kron_reduce.physics.coordinates import coi_project, coi_project_power
from project_harness.evidence import sha256_file


def resolve_source_root(project_root: Path, configured: str) -> Path:
    path = Path(os.environ.get("KIKI_DATA_ROOT", configured))
    return path if path.is_absolute() else (project_root / path).resolve()


def allocate_roles(source_root: Path, names: list[str], seed: int = 324):
    rows, bindings = [], {}
    rng = np.random.default_rng(seed)
    for name in names:
        directory = source_root / name
        split = json.loads((directory / "splits.json").read_text(encoding="utf-8"))
        train, val, test = [set(split[k]) for k in ("train", "validation", "test")]
        if train & (val | test) or val & test:
            raise ValueError("source split IDs overlap")
        with (directory / "dataset_index.csv").open(encoding="utf-8", newline="") as stream:
            index_rows = list(csv.DictReader(stream))
        index = {r["event_id"]: r for r in index_rows}
        if len(index) != len(index_rows) or not train <= index.keys():
            raise ValueError("duplicate or missing source event IDs")
        groups = sorted({index[e]["fault_line"] for e in train})
        if len(groups) < 6:
            raise ValueError("too few original train line groups for registered split")
        val_groups = set(rng.permutation(groups)[: max(3, math.ceil(len(groups) * 0.25))])
        train_buses, val_buses = set(), set()
        for event_id in sorted(train):
            row = index[event_id]
            role = "val" if row["fault_line"] in val_groups else "train"
            (val_buses if role == "val" else train_buses).add(row["fault_bus"])
            rows.append(
                {
                    "event_id": f"{name}/{event_id}",
                    "raw_id": event_id,
                    "source_dataset": name,
                    "fault_line": row["fault_line"],
                    "role": role,
                    "source_role": "train",
                    "declared_length": int(row["valid_length"]),
                }
            )
        if train_buses & val_buses:
            raise ValueError("fault buses cross new train/val groups")
        bindings[name] = {
            key: sha256_file(directory / key)
            for key in ("splits.json", "dataset_index.csv", "dataset.h5")
        }
    train_lines = {r["fault_line"] for r in rows if r["role"] == "train"}
    val_lines = {r["fault_line"] for r in rows if r["role"] == "val"}
    if train_lines & val_lines:
        raise ValueError("physical fault lines cross source train/val groups")
    return rows, bindings


class KikiHistorySamples:
    def __init__(self, source_root: Path, rows, bindings, *, history_count=21, horizon_s=3.0):
        if history_count != 21 or horizon_s != 3.0:
            raise ValueError("kiki rerun locks 21 observations and 3 s targets")
        if any(r["source_role"] != "train" for r in rows):
            raise ValueError("original source holdouts remain locked")
        identities = set()
        for name in {row["source_dataset"] for row in rows}:
            split = json.loads((source_root / name / "splits.json").read_text(encoding="utf-8"))
            allowed = set(split["train"])
            forbidden = set(split["validation"]) | set(split["test"])
            if allowed & forbidden:
                raise ValueError("source split IDs overlap")
            for row in rows:
                if row["source_dataset"] != name:
                    continue
                identity = (name, row["raw_id"])
                if row["raw_id"] not in allowed or row["raw_id"] in forbidden:
                    raise ValueError("raw event ID is outside the source train allowlist")
                if identity in identities or row["role"] not in ("train", "val"):
                    raise ValueError("duplicate event or forbidden target role")
                identities.add(identity)
        self.source_root, self.rows, self.bindings = source_root, tuple(rows), bindings
        self.accesses, self.coverage, self.static = [], [], {}

    def load(self, role):
        if role not in ("train", "val"):
            raise ValueError("source val/test/audit may not be opened")
        samples, mass_reference = [], None
        for name in sorted({r["source_dataset"] for r in self.rows}):
            with h5py.File(self.source_root / name / "dataset.h5", "r") as handle:
                inertia = np.asarray(handle["inertia"][...], dtype=float)
                angular_rate = float(handle.attrs["omega_s"])
                mva_base = float(handle.attrs["mva_base"])
                if (
                    inertia.shape != (10,)
                    or not np.isfinite(inertia).all()
                    or np.any(inertia <= 0)
                    or not np.isclose(angular_rate, 2 * np.pi * 60, rtol=0, atol=1e-12)
                    or mva_base != 100.0
                ):
                    raise ValueError("mass, frequency or system power base differs")
                units = str(handle.attrs["raw_units"])
                if (
                    units
                    != "delta rad; omega absolute pu; Pm=GENROU.tm, Pe=GENROU.Pe, te=GENROU.te on system MVA base"
                ):
                    raise ValueError("raw units differ from the registered source contract")
                mapping = json.loads(handle.attrs["state_mapping"])
                delta_order = [r[3] for r in mapping if r[1] == "delta" and r[2] == "GENROU"]
                if delta_order != [f"GENROU_{i}" for i in range(1, 11)]:
                    raise ValueError("generator channel order differs from registered order")
                mass = inertia / angular_rate
                if mass_reference is not None and not np.allclose(
                    mass, mass_reference, rtol=0, atol=1e-12
                ):
                    raise ValueError("source mass priors differ")
                mass_reference = mass
                eq = coi_project(np.asarray(handle["equilibrium/delta"][...]), mass)
                pm_eq = np.asarray(handle["equilibrium/Pm"][...])
                self.static[name] = {
                    "mass": mass.tolist(),
                    "omega_s": angular_rate,
                    "mva_base": mva_base,
                    "case_sha256": str(handle.attrs["case_sha256"]),
                    "raw_units": units,
                    "generator_order": delta_order,
                }
                for row in self.rows:
                    if row["role"] != role or row["source_dataset"] != name:
                        continue
                    group = handle["events"][row["raw_id"]]
                    count = group["t"].shape[0]
                    if count != row["declared_length"]:
                        raise ValueError("index and raw target lengths disagree")
                    entry = {
                        **row,
                        "real_count": count,
                        "history_eligible": count >= 22,
                        "future_count": max(0, min(count - 21, 300)),
                    }
                    self.coverage.append(entry)
                    if count < 22:
                        continue
                    end = min(count, 321)

                    def read(key, selection, row=row, group=group):
                        self.accesses.append(
                            {
                                "event_id": row["event_id"],
                                "role": role,
                                "dataset": key,
                                "selection": str(selection),
                            }
                        )
                        return np.asarray(group[key][selection], dtype=float)

                    clock = read("t", slice(0, end))
                    if not np.allclose(clock, np.arange(end) * 0.01, rtol=0, atol=1e-10):
                        raise ValueError("source clock is not the registered real .01 s clock")
                    delta = coi_project(read("delta", slice(0, end)), mass) - eq
                    # Absolute pu synchronous frequency is removed by COI.
                    omega = coi_project((read("omega", slice(0, end)) - 1.0) * angular_rate, mass)
                    power = coi_project_power(read("Pm", 0) - pm_eq, mass)
                    history = CausalStateHistory(
                        clock[:21] - clock[20], delta[:21], omega[:21], power
                    )
                    samples.append(
                        EquilibriumLearningSample(
                            row["event_id"],
                            name,
                            row["fault_line"],
                            role,
                            history,
                            np.arange(end - 20) * 0.01,
                            delta[20:],
                            omega[20:],
                            eq,
                        )
                    )
        return mass_reference, tuple(samples)

    def access_audit(self):
        return {
            "source_policy": "original_train_only",
            "source_val_ids_read": [],
            "test_ids_read": [],
            "future_pm_reads": [],
            "hidden_state_reads": [],
            "pe_voltage_reads": [],
            "source_routing": False,
            "accesses": self.accesses,
            "coverage": self.coverage,
            "static_priors": self.static,
            "data_bindings": self.bindings,
        }
