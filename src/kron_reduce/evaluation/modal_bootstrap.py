"""Fault-line cluster bootstrap for modes of a fitted weak physical equation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from kron_reduce.models.m0 import M0Model

from .modal import DmdMode, match_dmd_modes
from .model_modes import physical_model_modes


@dataclass(frozen=True)
class BootstrapPhysicalMode:
    mode: DmdMode
    support_fraction: float
    supporting_draws: int
    total_draws: int
    frequency_ci95_hz: tuple[float, float]
    damping_ci95: tuple[float, float]
    median_mac: float

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.mode.as_dict(),
            "support_fraction": self.support_fraction,
            "supporting_draws": self.supporting_draws,
            "total_draws": self.total_draws,
            "frequency_ci95_hz": self.frequency_ci95_hz,
            "damping_ci95": self.damping_ci95,
            "median_mac": self.median_mac,
        }


def bootstrap_m0_physical_modes(
    model: M0Model,
    samples: Sequence[dict[str, Any]],
    *,
    draws: int,
    seed: int,
    windows: tuple[int, ...],
    horizon_steps: int,
    max_fit_steps: int = 0,
    max_windows_per_case_horizon: int = 0,
    maximum_relative_frequency_error: float = 0.1,
    minimum_mac: float = 0.8,
) -> tuple[list[BootstrapPhysicalMode], dict[str, Any]]:
    """Refit M0 by resampling whole fault-line clusters and audit its poles.

    Hyperparameters and the base mode list are frozen before resampling. This
    estimates parameter/model-mode stability; it does not use event labels as
    features and it does not require every disturbance to excite every pole.
    """
    if draws < 1:
        raise ValueError("bootstrap draws must be positive")
    if model.selected_ridge is None:
        raise ValueError("bootstrap requires a frozen selected ridge")
    if not samples:
        raise ValueError("bootstrap requires at least one sample")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        metadata = sample.get("training_context", {}).get("metadata", {})
        fault_line = metadata.get("fault_line")
        if fault_line is None:
            raise KeyError("fault_line is required as the cluster resampling unit")
        grouped[str(fault_line)].append(sample)
    lines = tuple(sorted(grouped))
    if len(lines) < 2:
        raise ValueError("at least two fault-line clusters are required")

    systems: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for line in lines:
        matrix, vector, _ = M0Model.build_weak_system(
            grouped[line],
            windows=windows,
            max_cases=None,
            horizon_steps=horizon_steps,
            max_fit_steps=max_fit_steps,
            max_windows_per_case_horizon=max_windows_per_case_horizon,
        )
        systems[line] = matrix, vector
    static = samples[0]["model_inputs"]["static"]
    reference, reference_diagnostics = physical_model_modes(model, static)
    observations: list[list[dict[str, float]]] = [[] for _ in reference]
    conditions: list[float] = []
    full_rank_draws = 0
    successful_draws = 0
    rng = np.random.default_rng(int(seed))
    n_nodes = len(np.asarray(static["mass"]))
    for _ in range(int(draws)):
        selected = rng.choice(lines, size=len(lines), replace=True)
        matrix = np.concatenate([systems[str(line)][0] for line in selected])
        vector = np.concatenate([systems[str(line)][1] for line in selected])
        theta, diagnostics = M0Model._solve_nonnegative_ridge(
            matrix,
            vector,
            float(model.selected_ridge),
        )
        conditions.append(float(diagnostics["matrix_condition_number"]))
        full_rank_draws += int(
            int(diagnostics["effective_rank"]) == int(diagnostics["n_parameters"])
            if "n_parameters" in diagnostics
            else int(diagnostics["effective_rank"]) == matrix.shape[1]
        )
        if not bool(diagnostics["solver_success"]):
            continue
        successful_draws += 1
        fitted = M0Model(
            damping=theta[:n_nodes],
            coupling=theta[n_nodes:],
            selected_ridge=float(model.selected_ridge),
        )
        candidate, _ = physical_model_modes(fitted, static)
        matches = match_dmd_modes(
            reference,
            candidate,
            max_relative_frequency_error=maximum_relative_frequency_error,
        )
        for match in matches:
            if float(match["mac"]) < float(minimum_mac):
                continue
            reference_index = next(
                index
                for index, mode in enumerate(reference)
                if mode.frequency_hz == match["reference_frequency_hz"]
            )
            observations[reference_index].append(match)

    supported: list[BootstrapPhysicalMode] = []
    for mode, rows in zip(reference, observations, strict=True):
        frequencies = np.asarray([row["candidate_frequency_hz"] for row in rows])
        dampings = np.asarray([row["candidate_damping_ratio"] for row in rows])
        macs = np.asarray([row["mac"] for row in rows])
        frequency_ci = (
            (float(np.quantile(frequencies, 0.025)), float(np.quantile(frequencies, 0.975)))
            if len(frequencies)
            else (float("nan"), float("nan"))
        )
        damping_ci = (
            (float(np.quantile(dampings, 0.025)), float(np.quantile(dampings, 0.975)))
            if len(dampings)
            else (float("nan"), float("nan"))
        )
        supported.append(
            BootstrapPhysicalMode(
                mode=mode,
                support_fraction=len(rows) / float(draws),
                supporting_draws=len(rows),
                total_draws=int(draws),
                frequency_ci95_hz=frequency_ci,
                damping_ci95=damping_ci,
                median_mac=float(np.median(macs)) if len(macs) else 0.0,
            )
        )
    finite_conditions = np.asarray([value for value in conditions if np.isfinite(value)])
    return supported, {
        "cluster_unit": "fault_line",
        "fault_line_clusters": len(lines),
        "draws": int(draws),
        "successful_draw_fraction": successful_draws / float(draws),
        "full_rank_draw_fraction": full_rank_draws / float(draws),
        "median_design_condition_number": (
            float(np.median(finite_conditions)) if len(finite_conditions) else float("inf")
        ),
        "maximum_design_condition_number": (
            float(np.max(finite_conditions)) if len(finite_conditions) else float("inf")
        ),
        "frozen_ridge": float(model.selected_ridge),
        "reference_model_diagnostics": reference_diagnostics,
        "reference_modes": [asdict(mode) for mode in reference],
    }
