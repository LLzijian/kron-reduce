from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RunContext:
    seed: int = 324
    epochs: int = 2
    horizon_steps: int = 20
    max_cases: int = 2
    device: str = "cpu"
    learning_rate: float = 1e-3
    ridge_values: tuple[float, ...] = (1e-8, 1e-6, 1e-4, 1e-2)
    integral_windows: tuple[int, ...] = (1, 5, 20)
    sindy_thresholds: tuple[float, ...] = (0.0, 1e-6, 1e-4, 1e-3, 1e-2)
    rollout_refine: bool = False
    refinement_min_improvement: float = 0.02
    max_internal_step: float = 0.01
    m2a_rollout_shortlist: int = 4
    m2a_residual_scales: tuple[float, ...] = (0.01, 0.02, 0.05, 0.1)
    m2a_validation_cvar_fraction: float = 0.1
    m2a_max_cvar_degradation: float = 0.0
    m2a_enforce_dissipative_linear_terms: bool = False
    graph_prune_thresholds: tuple[float, ...] = (0.0, 1e-3, 1e-2)
    graph_nonoverlap_windows: bool = True
    fit_max_steps: int = 0
    weak_max_windows_per_case_horizon: int = 0


@dataclass
class FitResult:
    method: str
    status: str
    train_case_ids: tuple[str, ...]
    val_case_ids: tuple[str, ...]
    best_validation_loss: float
    details: dict[str, Any] = field(default_factory=dict)


class DynamicsModel(ABC):
    method_name = "abstract"

    @abstractmethod
    def fit(
        self, train: list[dict[str, Any]], val: list[dict[str, Any]], run_context: RunContext
    ) -> FitResult: ...

    @abstractmethod
    def rollout(
        self,
        initial_state: dict[str, np.ndarray],
        time_grid: np.ndarray,
        exogenous: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]: ...

    @abstractmethod
    def save(self, path: str | Path) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> DynamicsModel: ...

    @abstractmethod
    def physics_audit(self) -> dict[str, float | str | bool]: ...


def sample_initial_state(sample: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        key: np.asarray(value)
        for key, value in sample["model_inputs"]["static"].items()
    }


def sample_rollout_inputs(
    sample: dict[str, Any]
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray]]:
    return (
        sample_initial_state(sample),
        np.asarray(sample["model_inputs"]["exogenous"]["t"], dtype=np.float64),
        {
            "Pm": np.asarray(sample["model_inputs"]["exogenous"]["Pm"], dtype=np.float64),
            "interpolation": "zoh" if sample["training_context"]["claim_type"] == "causal_zoh_rollout" else "linear",
        },
    )
