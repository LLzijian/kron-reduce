"""F1 train-only pilot of learned history initialization with causal reads."""

from __future__ import annotations

import json
import tomllib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from kron_reduce.data.causal_history import read_measured_history
from kron_reduce.data.p2_roles import P2RoleRow, build_p2_role_rows
from kron_reduce.models.learned_latent import LatentLearningSample, LearnedLatentModel
from kron_reduce.physics.coordinates import coi_project, coi_project_power
from kron_reduce.physics.units import effective_mass
from kron_reduce.workflows.learned_causal_latent import execute_learning_study
from project_harness.data_check import verify_data
from project_harness.evidence import sha256_file


class AuthorizedHistorySamples:
    def __init__(
        self, dataset_dir: Path, rows: Sequence[P2RoleRow], mass: np.ndarray, result_dir: Path
    ) -> None:
        if not rows or any(row.canonical_role != "train" for row in rows):
            raise ValueError("history pilot requires canonical train roles only")
        if len({row.event_id for row in rows}) != len(rows):
            raise ValueError("history roles must have unique IDs")
        self.dataset_dir, self.rows, self.mass, self.result_dir = (
            dataset_dir,
            tuple(rows),
            mass,
            result_dir,
        )
        self.accesses: list[dict[str, Any]] = []
        self.origins: list[dict[str, Any]] = []
        self.audit_open_count = 0

    def _authorize_audit(self) -> None:
        if self.audit_open_count:
            raise RuntimeError("independent audit trajectory data may only be opened once")
        path = self.result_dir / "selection_freeze.json"
        freeze = json.loads(path.read_text(encoding="utf-8"))
        fit_ids = {row.event_id for row in self.rows if row.role in {"fit", "selection"}}
        for name, key in (
            ("selected-final.npz", "final_checkpoint_sha256"),
            ("memoryless-final.npz", "memoryless_checkpoint_sha256"),
        ):
            checkpoint = self.result_dir / name
            if sha256_file(checkpoint) != freeze[key]:
                raise RuntimeError("audit checkpoint differs from the frozen selection")
            fitted = set(LearnedLatentModel.load(checkpoint).fit_record["fitted_event_ids"])
            if fitted != fit_ids:
                raise RuntimeError(
                    "frozen checkpoint fitted IDs differ from registered train and val"
                )
        self.audit_open_count += 1

    def _make_sample(self, *args, equilibrium_delta: np.ndarray) -> LatentLearningSample:
        return LatentLearningSample(*args)

    def __call__(
        self, config: dict[str, Any], role: str, duration: float
    ) -> tuple[np.ndarray, tuple[LatentLearningSample, ...]]:
        mapping = {"train": "fit", "val": "selection", "audit": "internal_audit"}
        if role not in mapping or duration not in config["history_durations_s"]:
            raise ValueError("locked role or unregistered history window")
        if role == "audit":
            self._authorize_audit()
        count = config["history_sample_counts"][config["history_durations_s"].index(duration)]
        horizon = (
            config.get("audit_forecast_duration_s", config["forecast_duration_s"])
            if role == "audit"
            else config["forecast_duration_s"]
        )
        steps = round(float(horizon) / 0.01)
        samples = []
        with h5py.File(self.dataset_dir / "ieee39_postclear_canonical_v1.h5", "r") as handle:
            for row in self.rows:
                if row.role != mapping[role]:
                    continue
                group = handle["events"][row.event_id]

                def read(name, selection, *, row=row, group=group):
                    self.accesses.append(
                        {
                            "event_id": row.event_id,
                            "role": role,
                            "dataset": name,
                            "selection": str(selection),
                            "purpose": "static_or_target",
                        }
                    )
                    return np.asarray(group[name][selection], dtype=np.float64)

                equilibrium = read("equilibrium/delta0", (...,))
                pm_name = "Pm_equilibrium" if "Pm_equilibrium" in group["equilibrium"] else "Pm0"
                pm_equilibrium = read(f"equilibrium/{pm_name}", (...,))
                pm_clear = read("postclear/Pm", 0)
                power = coi_project_power(pm_clear - pm_equilibrium, self.mass)
                window = read_measured_history(
                    group,
                    sample_count=count,
                    mass=self.mass,
                    constant_power=power,
                    equilibrium_delta=equilibrium,
                )
                self.accesses.extend(
                    {**access, "event_id": row.event_id, "role": role, "purpose": "history_input"}
                    for access in window.access_log
                )
                selection = slice(count - 1, count + steps)
                source_time = read("postclear/t", selection)
                expected = window.forecast_origin_after_clearing_s + np.arange(steps + 1) * 0.01
                if source_time.shape != expected.shape or not np.allclose(
                    source_time, expected, rtol=0.0, atol=1e-10
                ):
                    raise ValueError("real future target samples do not cover the registered clock")
                delta = (
                    coi_project(read("postclear/delta", selection), self.mass)
                    - window.equilibrium_delta
                )
                omega = coi_project(read("postclear/omega", selection), self.mass)
                samples.append(
                    self._make_sample(
                        row.event_id,
                        row.source_dataset,
                        row.fault_line,
                        role,
                        window.history,
                        np.arange(steps + 1) * 0.01,
                        delta,
                        omega,
                        equilibrium_delta=window.equilibrium_delta,
                    )
                )
                self.origins.append(
                    {
                        "event_id": row.event_id,
                        "role": role,
                        "history_sample_count": count,
                        "nominal_history_duration_s": duration,
                        "actual_forecast_origin_after_clearing_s": window.forecast_origin_after_clearing_s,
                        "forecast_duration_s": horizon,
                    }
                )
        return self.mass, tuple(samples)

    def access_audit(self) -> dict[str, Any]:
        forbidden = [
            row
            for row in self.accesses
            if row["dataset"] == "postclear/Pm" and row["selection"] != "0"
        ]
        if forbidden:
            raise RuntimeError("future observed power entered history pilot")
        return {
            "schema": "causal_history_pilot_access.v1",
            "accesses": self.accesses,
            "forecast_origins": self.origins,
            "ieee39_trajectory_reads": sorted({row["event_id"] for row in self.accesses}),
            "canonical_val_ids_read": [],
            "test_ids_read": [],
            "other_folds_read": [],
            "future_pm_reads": [],
            "labels_read": False,
            "pe_or_voltage_read": False,
            "source_routing": False,
            "audit_open_count": self.audit_open_count,
        }


def run(
    *, root: Path, experiment: dict[str, Any], stage: str, result_dir: Path, run_id: str
) -> None:
    if (
        stage != "history_learning"
        or experiment.get("data_policy") != "canonical_F1_train_roles_only"
    ):
        raise ValueError("unregistered IEEE39 history learning execution")
    config_path = root / "configs" / "experiments" / experiment["config"]
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    if config["fold"] != "F1" or config["seed"] != 324:
        raise ValueError("history pilot locks F1 and seed 324")
    data_check = verify_data(root)
    if data_check["status"] != "PASS":
        raise RuntimeError("history pilot dataset verification failed")
    dataset = root / "data" / config["dataset"]
    rows, audit = build_p2_role_rows(dataset)
    if audit["role_manifest_sha256"] != config["role_manifest_sha256"]:
        raise ValueError("history pilot role manifest differs from the frozen allocation")
    schema = json.loads((dataset / "schema.json").read_text(encoding="utf-8"))
    mass = effective_mass(np.asarray(schema["M_prior"]))
    (result_dir / "role_allocation.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (result_dir / "protocol_binding.json").write_text(
        json.dumps(
            {
                "protocol_path": config["protocol_path"],
                "protocol_sha256": sha256_file(root / config["protocol_path"]),
                "config_sha256": sha256_file(config_path),
                "role_manifest_sha256": audit["role_manifest_sha256"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    provider = AuthorizedHistorySamples(dataset, rows, mass, result_dir)
    execute_learning_study(
        root=root,
        config_path=config_path,
        config=config,
        result_dir=result_dir,
        run_id=run_id,
        sample_provider=provider,
        data_policy="canonical_F1_train_roles_only",
        extra_source_paths=(
            "src/kron_reduce/workflows/ieee39_history_learning.py",
            "src/kron_reduce/data/causal_history.py",
            "src/kron_reduce/data/p2_roles.py",
            "src/kron_reduce/physics/units.py",
        ),
    )
