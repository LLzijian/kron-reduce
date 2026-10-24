"""Passive sine-network closure with a causal, dynamics-consistent initializer.

Coordinates are relative to the supplied event equilibrium, not absolute
angles. The shifted sine potential is not globally positive or convex.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear

from kron_reduce.data.history_contract import CausalStateHistory
from kron_reduce.models.causal_latent import PassiveLatentClosure
from kron_reduce.models.learned_latent import (
    LatentLearningSample,
    event_weights,
    history_features,
    visible_scales,
)
from kron_reduce.physics.coordinates import coi_basis, coi_project, coi_project_power


@lru_cache(maxsize=32)
def complete_incidence(n: int) -> np.ndarray:
    columns = []
    for i in range(n):
        for j in range(i + 1, n):
            column = np.zeros(n)
            column[i], column[j] = 1.0, -1.0
            columns.append(column)
    result = np.stack(columns, axis=1)
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class EquilibriumLearningSample(LatentLearningSample):
    equilibrium_delta: np.ndarray

    def __post_init__(self) -> None:
        super().__post_init__()
        equilibrium = np.array(self.equilibrium_delta, dtype=float, copy=True)
        if (
            equilibrium.shape != self.history.constant_power.shape
            or not np.isfinite(equilibrium).all()
        ):
            raise ValueError("equilibrium must be a finite, node-aligned static prior")
        equilibrium.flags.writeable = False
        object.__setattr__(self, "equilibrium_delta", equilibrium)


@dataclass(frozen=True)
class NonlinearPassiveClosure:
    mass: np.ndarray
    edge_stiffness: np.ndarray
    damping: np.ndarray
    coupling: np.ndarray
    latent_skew: np.ndarray
    latent_dissipation: np.ndarray
    backbone: str = "sine"

    def __post_init__(self) -> None:
        for name in (
            "mass",
            "edge_stiffness",
            "damping",
            "coupling",
            "latent_skew",
            "latent_dissipation",
        ):
            value = np.array(getattr(self, name), dtype=float, copy=True)
            if not np.isfinite(value).all():
                raise ValueError("closure parameters must be finite")
            value.flags.writeable = False
            object.__setattr__(self, name, value)
        n = len(self.mass)
        if (
            self.backbone not in {"sine", "linear"}
            or self.edge_stiffness.shape != (n * (n - 1) // 2,)
            or np.any(self.edge_stiffness < 0)
        ):
            raise ValueError("invalid nonnegative edge coupling or backbone")
        # Reuse strict dimension, COI-port, PSD, and skew checks.
        PassiveLatentClosure(
            self.mass,
            np.zeros((n, n)),
            self.damping,
            self.coupling,
            self.latent_skew,
            self.latent_dissipation,
        )

    @property
    def n_nodes(self) -> int:
        return len(self.mass)

    @property
    def latent_dimension(self) -> int:
        return self.coupling.shape[1]

    def rhs(self, state: np.ndarray, power: np.ndarray, equilibrium: np.ndarray) -> np.ndarray:
        n = self.n_nodes
        q, v, z = state[:n], state[n : 2 * n], state[2 * n :]
        b = complete_incidence(n)
        edge, reference = b.T @ q, b.T @ equilibrium
        force = edge if self.backbone == "linear" else np.sin(edge + reference) - np.sin(reference)
        acceleration = (
            coi_project_power(
                power - b @ (self.edge_stiffness * force) - self.damping @ v - self.coupling @ z,
                self.mass,
            )
            / self.mass
        )
        return np.r_[
            v, acceleration, self.coupling.T @ v + (self.latent_skew - self.latent_dissipation) @ z
        ]

    def energy(self, state: np.ndarray, equilibrium: np.ndarray) -> float:
        n = self.n_nodes
        q, v, z = state[:n], state[n : 2 * n], state[2 * n :]
        b = complete_incidence(n)
        edge, reference = b.T @ q, b.T @ equilibrium
        potential = (
            edge**2 / 2
            if self.backbone == "linear"
            else np.cos(reference) - np.cos(reference + edge) - edge * np.sin(reference)
        )
        return float(self.edge_stiffness @ potential + (v * self.mass) @ v / 2 + z @ z / 2)

    def energy_identity(
        self, state: np.ndarray, power: np.ndarray, equilibrium: np.ndarray
    ) -> dict:
        n = self.n_nodes
        q, v, z = state[:n], state[n : 2 * n], state[2 * n :]
        b = complete_incidence(n)
        gradient = (
            b.T @ q
            if self.backbone == "linear"
            else np.sin(b.T @ (q + equilibrium)) - np.sin(b.T @ equilibrium)
        )
        dh = float(
            np.r_[b @ (self.edge_stiffness * gradient), self.mass * v, z]
            @ self.rhs(state, power, equilibrium)
        )
        supply = float(v @ power - v @ self.damping @ v - z @ self.latent_dissipation @ z)
        return {
            "energy_rate": dh,
            "supply_minus_dissipation": supply,
            "identity_error": abs(dh - supply),
        }

    def rollout(self, q, v, z, time, power, equilibrium, *, max_step=0.01) -> np.ndarray:
        # Validate using the same input contract as the linear prototype.
        checker = PassiveLatentClosure(
            self.mass,
            np.zeros((self.n_nodes, self.n_nodes)),
            self.damping,
            self.coupling,
            self.latent_skew,
            self.latent_dissipation,
        )
        state = checker._state(q, v, z)
        checker._power(power)
        clock = np.asarray(time, dtype=float)
        equilibrium = np.asarray(equilibrium, dtype=float)
        if equilibrium.shape != self.mass.shape or not np.isfinite(equilibrium).all():
            raise ValueError("invalid equilibrium prior")
        if (
            clock.ndim != 1
            or not len(clock)
            or not np.isfinite(clock).all()
            or clock[0] != 0
            or np.any(np.diff(clock) <= 0)
            or not 0 < max_step <= 0.01
        ):
            raise ValueError("invalid causal rollout clock or integration step")
        output = [state.copy()]
        for interval in np.diff(clock):
            count = max(1, int(np.ceil(interval / max_step - 1e-12)))
            h = interval / count
            for _ in range(count):
                k1 = self.rhs(state, power, equilibrium)
                k2 = self.rhs(state + h * k1 / 2, power, equilibrium)
                k3 = self.rhs(state + h * k2 / 2, power, equilibrium)
                k4 = self.rhs(state + h * k3, power, equilibrium)
                state = state + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
                state[: self.n_nodes] = coi_project(state[: self.n_nodes], self.mass)
                state[self.n_nodes : 2 * self.n_nodes] = coi_project(
                    state[self.n_nodes : 2 * self.n_nodes], self.mass
                )
                if not np.isfinite(state).all() or np.max(np.abs(state)) > 1e12:
                    raise FloatingPointError("divergent nonlinear augmented rollout")
            output.append(state.copy())
        return np.stack(output)


@dataclass(frozen=True)
class NonlinearFitSettings:
    latent_dimension: int = 2
    backbone: str = "sine"
    frames: int = 3
    epochs: int = 100
    learning_rate: float = 0.01
    encoder_penalty: float = 0.0001
    history_weight: float = 1.0
    horizons_s: tuple[float, ...] = (1.0, 3.0)
    seed: int = 324
    encoder_mode: str = "frames"

    def __post_init__(self):
        if self.encoder_mode not in {"frames", "fixed_quadratic"}:
            raise ValueError("unregistered history representation")
        if self.encoder_mode == "fixed_quadratic" and self.frames != 3:
            raise ValueError("fixed quadratic representation has exactly three coefficients")
        if (
            type(self.latent_dimension) is not int
            or self.latent_dimension < 0
            or self.backbone not in {"sine", "linear"}
        ):
            raise ValueError("invalid closure family")
        if (
            type(self.frames) is not int
            or self.frames < 2
            or type(self.epochs) is not int
            or self.epochs < 1
        ):
            raise ValueError("invalid optimization budget")
        if (
            not np.isfinite(
                [self.learning_rate, self.encoder_penalty, self.history_weight, *self.horizons_s]
            ).all()
            or self.learning_rate <= 0
            or min(self.encoder_penalty, self.history_weight) < 0
        ):
            raise ValueError("invalid loss or learning rate")
        if (
            not self.horizons_s
            or min(self.horizons_s) <= 0
            or sorted(set(self.horizons_s)) != list(self.horizons_s)
        ):
            raise ValueError("horizons must be positive and strictly increasing")


@lru_cache(maxsize=1)
def fixed_quadratic_basis() -> np.ndarray:
    """Data-independent, signed equal-weight Gram-Schmidt basis on 11 samples."""
    s = (np.arange(11, dtype=float) - 5) / 5
    columns = []
    for vector in (np.ones(11), s, s**2):
        value = vector.copy()
        for previous in columns:
            value -= (previous @ value) * previous
        value /= np.linalg.norm(value)
        if value[np.flatnonzero(np.abs(value) > 1e-14)[0]] < 0:
            value *= -1
        columns.append(value)
    result = np.stack(columns, axis=1)
    result.flags.writeable = False
    return result


def _features(history: CausalStateHistory, equilibrium: np.ndarray, frames: int,
              encoder_mode: str = "frames") -> np.ndarray:
    # Static equilibrium is allowed; targets, source, line, and identity are not.
    if encoder_mode == "fixed_quadratic":
        if frames != 3 or len(history.time) != 11 or not np.allclose(
            history.time, np.arange(11) * 0.01 - 0.1, rtol=0, atol=1e-12
        ):
            raise ValueError("fixed summary requires the registered 11-point clock")
        basis = fixed_quadratic_basis()
        return np.concatenate(((basis.T @ history.delta).reshape(-1),
                               (basis.T @ history.omega).reshape(-1),
                               history.constant_power, equilibrium))
    if encoder_mode != "frames":
        raise ValueError("unregistered history representation")
    return np.r_[history_features(history, frames), equilibrium]


@dataclass
class LearnedNonlinearModel:
    closure: NonlinearPassiveClosure
    encoder: np.ndarray
    feature_scale: np.ndarray
    target_scale: np.ndarray
    settings: NonlinearFitSettings
    fit_record: dict

    def initialize(self, history, equilibrium, *, max_step=0.01) -> tuple[np.ndarray, np.ndarray]:
        seed = self.encoder @ (
            _features(history, equilibrium, self.settings.frames, self.settings.encoder_mode)
            / self.feature_scale
        )
        path = self.closure.rollout(
            history.delta[0],
            history.omega[0],
            seed,
            history.time - history.time[0],
            history.constant_power,
            equilibrium,
            max_step=max_step,
        )
        return path[-1, 2 * self.closure.n_nodes :], path

    def predict(self, history, time, equilibrium, *, zero_latent=False, max_step=0.01) -> np.ndarray:
        z = (
            np.zeros(self.closure.latent_dimension)
            if zero_latent
            else self.initialize(history, equilibrium, max_step=max_step)[0]
        )
        return self.closure.rollout(
            history.delta[-1], history.omega[-1], z, time, history.constant_power, equilibrium,
            max_step=max_step,
        )

    def score(self, samples, *, zero_latent=False) -> dict:
        weights, rows = event_weights(samples), []
        for sample in samples:
            try:
                prediction = self.predict(
                    sample.history,
                    sample.forecast_time,
                    sample.equilibrium_delta,
                    zero_latent=zero_latent,
                )
                n = self.closure.n_nodes
                error = (
                    prediction[1:, : 2 * n]
                    - np.c_[sample.target_delta[1:], sample.target_omega[1:]]
                )
                rows.append(
                    {
                        "event_id": sample.event_id,
                        "source_dataset": sample.source_dataset,
                        "fault_line": sample.fault_line,
                        "role": sample.role,
                        "status": "ok",
                        "normalized_mse": float(np.mean((error / self.target_scale) ** 2)),
                        "delta_mse_rad2": float(np.mean(error[:, :n] ** 2)),
                        "omega_mse_rad2_per_s2": float(np.mean(error[:, n:] ** 2)),
                        "forecast_duration_s": float(sample.forecast_time[-1]),
                    }
                )
            except (ValueError, FloatingPointError, OverflowError) as error:
                rows.append(
                    {"event_id": sample.event_id, "status": "failed", "failure_reason": str(error)}
                )
        failed = sum(row["status"] != "ok" for row in rows)
        result = {
            "rows": rows,
            "failed_event_count": failed,
            "failures_imputed_as_zero": False,
            "normalized_rmse": None,
            "weighting": "source_line_event_time_equal",
            "score_includes_forecast_origin": False,
        }
        if not failed:
            result.update(
                normalized_rmse=float(np.sqrt(weights @ [row["normalized_mse"] for row in rows])),
                median_event_normalized_rmse=float(
                    np.median([np.sqrt(row["normalized_mse"]) for row in rows])
                ),
                pooled_delta_rmse_rad=float(
                    np.sqrt(weights @ [row["delta_mse_rad2"] for row in rows])
                ),
                pooled_omega_rmse_rad_per_s=float(
                    np.sqrt(weights @ [row["omega_mse_rad2_per_s2"] for row in rows])
                ),
            )
        return result

    def save(self, path: Path):
        with path.open("xb") as handle:
            np.savez_compressed(
                handle,
                metadata=np.asarray(
                    json.dumps(
                        {
                            "schema": "nonlinear_passive_history.v1",
                            "settings": asdict(self.settings),
                            "fit_record": self.fit_record,
                            "backbone": self.closure.backbone,
                        },
                        allow_nan=False,
                    )
                ),
                encoder=self.encoder,
                feature_scale=self.feature_scale,
                target_scale=self.target_scale,
                **{
                    name: getattr(self.closure, name)
                    for name in self.closure.__dataclass_fields__
                    if name != "backbone"
                },
            )

    @classmethod
    def load(cls, path: Path):
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"]))
            if meta["schema"] != "nonlinear_passive_history.v1":
                raise ValueError("unknown nonlinear checkpoint schema")
            meta["settings"]["horizons_s"] = tuple(meta["settings"]["horizons_s"])
            closure = NonlinearPassiveClosure(
                **{
                    name: data[name]
                    for name in NonlinearPassiveClosure.__dataclass_fields__
                    if name != "backbone"
                },
                backbone=meta["backbone"],
            )
            return cls(
                closure,
                data["encoder"].copy(),
                data["feature_scale"].copy(),
                data["target_scale"].copy(),
                NonlinearFitSettings(**meta["settings"]),
                meta["fit_record"],
            )


def _integral_start(samples, mass, backbone, scale, *, include_last=False):
    """Train-only nonnegative momentum-integral warm start; no derivatives."""
    n = len(mass)
    b = complete_incidence(n)
    p = np.eye(n) - mass[:, None] / mass.sum()
    basis = coi_basis(np.ones(n))
    design, target = [], []
    for weight, sample in zip(event_weights(samples), samples, strict=True):
        q, v, t = sample.target_delta, sample.target_omega, sample.forecast_time
        edge = q @ b
        if backbone == "sine":
            ref = sample.equilibrium_delta @ b
            edge = np.sin(edge + ref) - np.sin(ref)
        # Balanced PSD damping parameterized in the same quotient basis as fit.
        velocity = v @ basis
        terms = np.concatenate(
            (edge[:, None, :] * b[None, :, :], velocity[:, None, :] * basis[None, :, :]), axis=2
        )
        integral = np.cumsum((terms[1:] + terms[:-1]) * np.diff(t)[:, None, None] / 2, axis=0)
        rhs = t[1:, None] * sample.history.constant_power - (v[1:] - v[0]) * mass
        select = np.arange(9, len(t) - 1, 10)
        if include_last:
            select = np.unique(np.r_[select, len(t) - 2])
        factor = np.sqrt(weight / max(1, len(select)))
        design.append(
            ((p @ integral[select]) / (mass * scale[n:])[None, :, None]).reshape(-1, terms.shape[2])
            * factor
        )
        target.append((rhs[select] / (mass * scale[n:])).reshape(-1) * factor)
    a, y = np.concatenate(design), np.concatenate(target)
    augmented = np.r_[a, 1e-3 * np.eye(a.shape[1])]
    solution = lsq_linear(
        augmented, np.r_[y, np.zeros(a.shape[1])], bounds=(0, np.inf), tol=1e-10, max_iter=300
    )
    if not solution.success:
        raise RuntimeError("nonnegative integral warm start did not converge")
    return np.maximum(solution.x, 1e-5)


def _array_sha256(value: np.ndarray) -> str:
    """Hash a numeric array independently of its original storage layout."""
    array = np.ascontiguousarray(value, dtype="<f8")
    header = json.dumps({"dtype": "<f8", "shape": list(array.shape)}, sort_keys=True).encode()
    return hashlib.sha256(header + b"\0" + array.tobytes(order="C")).hexdigest()


def _validated_paired_start(paired_start, mass, feature_scale, target_scale):
    """Validate a shared visible backbone and frozen train normalization."""
    names = {"mass", "edge_stiffness", "damping", "feature_scale", "target_scale"}
    if not isinstance(paired_start, dict) or set(paired_start) != names:
        raise ValueError("paired_start must contain exactly the five registered arrays")
    arrays = {name: np.array(paired_start[name], dtype=float, copy=True) for name in names}
    n = len(mass)
    expected = {"mass": (n,), "edge_stiffness": (n * (n - 1) // 2,),
                "damping": (n, n), "feature_scale": feature_scale.shape,
                "target_scale": target_scale.shape}
    if any(arrays[name].shape != expected[name] or not np.isfinite(arrays[name]).all()
           for name in names):
        raise ValueError("paired_start arrays must be finite and dimensionally aligned")
    if not np.array_equal(arrays["mass"], mass):
        raise ValueError("paired_start mass differs from the registered priors")
    for name, computed in (("feature_scale", feature_scale), ("target_scale", target_scale)):
        if np.any(arrays[name] <= 0) or not np.allclose(
            arrays[name], computed, rtol=0, atol=1e-12
        ):
            raise ValueError(f"paired_start {name} differs from recomputed train normalization")
    basis = coi_basis(np.ones(n))
    damping = arrays["damping"]
    diagonal = np.diag(basis.T @ damping @ basis)
    if (np.any(arrays["edge_stiffness"] <= 0) or np.any(diagonal <= 0)
            or np.linalg.norm((basis * diagonal) @ basis.T - damping, ord="fro") > 1e-12):
        raise ValueError("paired_start backbone is not representable by positive edge and damping parameters")
    positive = np.r_[arrays["edge_stiffness"], diagonal]
    # Match torch softplus(beta=1, threshold=20), including its linear branch.
    raw = np.where(positive > 20.0, positive, positive + np.log(-np.expm1(-positive)))
    array_hashes = {name: _array_sha256(arrays[name]) for name in sorted(names)}
    binding = {"paired_start_array_sha256": array_hashes,
               "paired_start_sha256": hashlib.sha256(
                   json.dumps(array_hashes, sort_keys=True, separators=(",", ":")).encode()
               ).hexdigest()}
    return raw, arrays, binding


def _validated_full_start(full_start, mass, feature_scale, target_scale, latent_dimension):
    """Recover a physical checkpoint in a canonical raw parameterization."""
    backbone_names = {"mass", "edge_stiffness", "damping", "feature_scale", "target_scale"}
    names = backbone_names | {"coupling", "latent_skew", "latent_dissipation", "encoder"}
    if not isinstance(full_start, dict) or set(full_start) != names:
        raise ValueError("full_start must contain exactly the nine registered arrays")
    arrays = {name: np.array(full_start[name], dtype=float, copy=True) for name in names}
    raw, _, _ = _validated_paired_start(
        {name: arrays[name] for name in backbone_names}, mass, feature_scale, target_scale
    )
    n, r = len(mass), latent_dimension
    expected = {"coupling": (n, r), "latent_skew": (r, r),
                "latent_dissipation": (r, r), "encoder": (r, len(feature_scale))}
    if any(arrays[name].shape != shape or not np.isfinite(arrays[name]).all()
           for name, shape in expected.items()):
        raise ValueError("full_start memory arrays must be finite and dimensionally aligned")
    basis = coi_basis(np.ones(n))
    port = basis.T @ arrays["coupling"]
    skew = (arrays["latent_skew"] - arrays["latent_skew"].T) / 2
    diagonal = np.diag(arrays["latent_dissipation"])
    errors = {
        "coupling": float(np.linalg.norm(basis @ port - arrays["coupling"], ord="fro")),
        "latent_skew": float(np.linalg.norm(skew - arrays["latent_skew"], ord="fro")),
        "latent_dissipation": float(np.linalg.norm(
            np.diag(diagonal) - arrays["latent_dissipation"], ord="fro")),
    }
    positive = diagonal - 1e-5
    if np.any(positive <= 0) or max(errors.values()) > 1e-12:
        raise ValueError("full_start memory is not representable by the registered parameterization")
    decay = np.where(positive > 20.0, positive, positive + np.log(-np.expm1(-positive)))
    parameters = {"raw": raw, "port": port, "skew": skew, "decay": decay,
                  "encoder": arrays["encoder"].copy()}
    hashes = {name: _array_sha256(arrays[name]) for name in sorted(names)}
    binding = {
        "full_start_array_sha256": hashes,
        "full_start_sha256": hashlib.sha256(
            json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "full_start_representation_errors": errors,
        "raw_skew_reconstruction": "canonical_antisymmetric_representative",
        "optimizer_initialization": "fresh_adam_zero_moments_step_zero",
        "exact_optimizer_resume": False,
    }
    return parameters, arrays, binding


def _stored_array_sha256(value):
    """Hash optimizer arrays without normalizing away dtype or scalar shape."""
    array = np.array(value, copy=True, order="C")
    header = json.dumps({"dtype": array.dtype.str, "shape": list(array.shape)},
                        sort_keys=True).encode()
    return hashlib.sha256(header + b"\0" + array.tobytes(order="C")).hexdigest()


def _validated_optimizer_start(optimizer_start, parameter_templates, learning_rate):
    """Validate a saved Adam state, preserving raw parameters and all moments."""
    names = tuple(parameter_templates)
    metadata = {"schema", "completed_optimizer_steps", "planned_optimizer_steps",
                "adam_learning_rate", "adam_betas", "adam_eps", "adam_weight_decay", "adam_amsgrad"}
    required = metadata | {f"{prefix}__{name}" for name in names
                           for prefix in ("parameter", "adam_exp_avg", "adam_exp_avg_sq", "adam_step")}
    optional = {"initial_optimizer_step", "total_optimizer_steps", "planned_total_optimizer_steps"}
    if (not isinstance(optimizer_start, dict) or not required <= set(optimizer_start)
            or set(optimizer_start) - required - optional):
        raise ValueError("optimizer_start must contain a complete registered Adam state")
    state = {key: np.array(value, copy=True) for key, value in optimizer_start.items()}
    if state["schema"].shape != () or str(state["schema"].item()) != "nonlinear_adam_training_state.v1":
        raise ValueError("unknown optimizer_start schema")

    def integer(key, default=None):
        if key not in state:
            return default
        value = state[key]
        if value.shape != () or value.dtype.kind not in "iu" or int(value) < 0:
            raise ValueError(f"optimizer_start {key} must be a nonnegative integer scalar")
        return int(value)

    completed, planned = integer("completed_optimizer_steps"), integer("planned_optimizer_steps")
    initial = integer("initial_optimizer_step", 0)
    total = integer("total_optimizer_steps", initial + completed)
    planned_total = integer("planned_total_optimizer_steps", initial + planned)
    if not 0 < completed <= planned or total != initial + completed or planned_total != initial + planned:
        raise ValueError("optimizer_start local and total step counts are inconsistent")
    hyperparameters = {"adam_learning_rate": learning_rate, "adam_betas": np.array([.9, .999]),
                       "adam_eps": 1e-8, "adam_weight_decay": 0, "adam_amsgrad": False}
    for key, expected in hyperparameters.items():
        if not np.array_equal(state[key], np.asarray(expected)):
            raise ValueError(f"optimizer_start {key} differs from the registered Adam hyperparameter")
    parameters = {}
    for name, template in parameter_templates.items():
        shape = template.shape
        for prefix in ("parameter", "adam_exp_avg", "adam_exp_avg_sq"):
            value = state[f"{prefix}__{name}"]
            if value.shape != shape or value.dtype != np.dtype("float64") or not np.isfinite(value).all():
                raise ValueError(f"optimizer_start {prefix}__{name} must be a finite float64 array")
            if prefix == "adam_exp_avg_sq" and np.any(value < 0):
                raise ValueError("optimizer_start second moments must be nonnegative")
        step = state[f"adam_step__{name}"]
        if (step.shape != () or step.dtype.kind != "f" or not np.isfinite(step)
                or float(step) != total):
            raise ValueError("optimizer_start Adam steps must all match total_optimizer_steps")
        parameters[name] = state[f"parameter__{name}"].copy()
    hashes = {key: _stored_array_sha256(value) for key, value in sorted(state.items())}
    binding = {
        "optimizer_start_array_sha256": hashes,
        "optimizer_start_sha256": hashlib.sha256(
            json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "initial_optimizer_step": total,
        "initial_optimizer_moments_zero": all(
            not np.any(state[f"adam_{moment}__{name}"])
            for name in names for moment in ("exp_avg", "exp_avg_sq")),
        "raw_skew_reconstruction": "saved_raw_array_exact",
        "optimizer_initialization": "restored_adam_parameters_moments_and_step",
        "exact_optimizer_resume": True,
    }
    return parameters, state, binding


def _tilted_risk(event_losses, event_weights, temperature):
    """Return stable weighted exponential risk and its event-gradient weights."""
    import torch

    logits = torch.log(event_weights) + event_losses / temperature
    objective = temperature * torch.logsumexp(logits, dim=0)
    return objective, torch.softmax(logits.detach(), dim=0)


def fit_nonlinear_model(
    samples,
    mass,
    settings: NonlinearFitSettings,
    *,
    target_scale_samples=None,
    allow_partial_horizons=False,
    progress_callback=None,
    paired_start=None,
    freeze_backbone=False,
    full_start=None,
    checkpoint_steps=(),
    checkpoint_callback=None,
    optimizer_start=None,
    risk_temperature=None,
    training_max_step=0.01,
    temporal_temperature=None,
    rhs_audit=None,
) -> LearnedNonlinearModel:
    """Fit train only, with visible history consistency and free multi-horizon loss."""
    import torch

    if type(freeze_backbone) is not bool or (freeze_backbone and paired_start is None):
        raise ValueError("freezing requires a verified paired_start and a boolean update mask")
    if full_start is not None and (paired_start is not None or freeze_backbone):
        raise ValueError("full_start is mutually exclusive with paired_start and freeze_backbone")
    if optimizer_start is not None and full_start is None:
        raise ValueError("optimizer_start requires the matching full_start physical checkpoint")
    if risk_temperature is not None:
        if (isinstance(risk_temperature, (bool, np.bool_))
                or not isinstance(risk_temperature, (int, float, np.integer, np.floating))
                or not np.isfinite(risk_temperature) or risk_temperature <= 0):
            raise ValueError("risk_temperature must be a finite positive scalar or None")
        risk_temperature = float(risk_temperature)
    if (isinstance(training_max_step, (bool, np.bool_))
            or not isinstance(training_max_step, (int, float, np.integer, np.floating))
            or not np.isfinite(training_max_step) or training_max_step <= 0
            or training_max_step > 0.01):
        raise ValueError("training_max_step must be finite, positive, and no greater than 0.01 s")
    training_max_step = float(training_max_step)
    if temporal_temperature is not None:
        if (type(temporal_temperature) not in (int, float)
                or not np.isfinite(temporal_temperature) or temporal_temperature <= 0):
            raise ValueError("temporal_temperature must be a fixed finite positive Python scalar or None")
        temporal_temperature = float(temporal_temperature)
        if risk_temperature is not None:
            raise ValueError("risk_temperature and temporal_temperature are mutually exclusive")
        if optimizer_start is not None:
            raise ValueError("temporal objective does not support optimizer_start restoration")
    if optimizer_start is not None and training_max_step != 0.01:
        raise ValueError("refined training requires a fresh Adam state")
    if rhs_audit is not None and not callable(rhs_audit):
        raise TypeError("rhs_audit must be callable or None")
    checkpoint_steps = tuple(checkpoint_steps)
    if (any(type(step) is not int or not 0 < step <= settings.epochs for step in checkpoint_steps)
            or tuple(sorted(set(checkpoint_steps))) != checkpoint_steps
            or (bool(checkpoint_steps) != (checkpoint_callback is not None))
            or (checkpoint_callback is not None and not callable(checkpoint_callback))):
        raise ValueError("checkpoint steps require an ordered unique positive schedule and callback")
    if not samples or {sample.role for sample in samples} != {"train"}:
        raise ValueError("nonlinear optimizer accepts train only")
    if any(not isinstance(sample, EquilibriumLearningSample) for sample in samples):
        raise ValueError("nonlinear learning requires explicit equilibrium priors")
    mass = np.array(mass, dtype=float, copy=True)
    n, r = len(mass), settings.latent_dimension
    clock = (
        np.arange(round(max(settings.horizons_s) / 0.01) + 1) * 0.01
        if allow_partial_horizons
        else samples[0].forecast_time
    )
    history_count = len(samples[0].history.time)
    if mass.ndim != 1 or not np.isfinite(mass).all() or np.any(mass <= 0) or n < 2:
        raise ValueError("invalid mass priors")
    if any(
        (not allow_partial_horizons and not np.array_equal(sample.forecast_time, clock))
        or len(sample.history.time) != history_count
        or sample.history.delta.shape[1] != n
        for sample in samples
    ):
        raise ValueError("learning requires one clock, history count, and node count")
    if allow_partial_horizons and any(
        len(s.forecast_time) > len(clock)
        or not np.allclose(s.forecast_time, clock[: len(s.forecast_time)], rtol=0, atol=1e-12)
        for s in samples
    ):
        raise ValueError("partial targets must be real prefixes of the registered clock")
    if (
        not np.allclose(np.diff(clock), 0.01, rtol=0, atol=1e-12)
        or abs(clock[-1] - max(settings.horizons_s)) > 1e-10
    ):
        raise ValueError("train clock must cover registered horizons at 0.01 s")
    for sample in samples:
        if (
            np.max(np.diff(sample.history.time)) > 0.01 + 1e-12
            or abs(sample.history.constant_power.sum()) > 1e-10
            or any(
                np.max(np.abs(value @ mass)) > 1e-10
                for value in (
                    sample.history.delta,
                    sample.history.omega,
                    sample.target_delta,
                    sample.target_omega,
                )
            )
        ):
            raise ValueError("learning violates history-step, power, or COI contract")
    training_options_active = (training_max_step != 0.01 or temporal_temperature is not None
                               or rhs_audit is not None)
    if training_options_active:
        history_intervals = np.stack([np.diff(sample.history.time) for sample in samples])
        history_substeps = np.maximum(
            1, np.ceil(history_intervals / training_max_step - 1e-12).astype(np.int64))
        if training_max_step == 0.005 and np.any(history_substeps != 2):
            raise ValueError("0.005 s refinement requires exactly two RK4 substeps per history interval")
        if np.any(history_substeps > 2):
            raise ValueError("refined training currently supports at most two RK4 substeps per history interval")
        if np.any(history_substeps != history_substeps[0:1]):
            raise ValueError("refined training requires a shared history substep schedule across events")
        history_substeps_per_interval = history_substeps[0]
        future_substeps = max(1, int(np.ceil(0.01 / training_max_step - 1e-12)))
        if future_substeps > 2:
            raise ValueError("refined training currently supports at most two RK4 substeps per future interval")
        training_event_steps_per_update = int(history_substeps.sum()) + future_substeps * sum(
            len(sample.forecast_time) - 1 for sample in samples)
    else:
        history_substeps_per_interval = None
        future_substeps = 1
        training_event_steps_per_update = None
    scale_population = samples if target_scale_samples is None else target_scale_samples
    if {s.event_id for s in scale_population} != {s.event_id for s in samples}:
        raise ValueError("normalization population differs from train fitting IDs")
    scales, weights = visible_scales(scale_population), event_weights(samples)
    features = np.stack(
        [_features(s.history, s.equilibrium_delta, settings.frames, settings.encoder_mode)
         for s in samples]
    )
    feature_scale = np.maximum(np.sqrt(weights @ features**2), 1e-5)
    paired_binding = None
    full_parameters = None
    optimizer_arrays, optimizer_binding = None, None
    initial_optimizer_step = 0
    if full_start is not None:
        full_parameters, paired_arrays, paired_binding = _validated_full_start(
            full_start, mass, feature_scale, scales, r
        )
        if optimizer_start is not None:
            full_parameters, optimizer_arrays, optimizer_binding = _validated_optimizer_start(
                optimizer_start, full_parameters, settings.learning_rate
            )
            paired_binding.update(optimizer_binding)
            initial_optimizer_step = optimizer_binding["initial_optimizer_step"]
        paired_raw = full_parameters["raw"]
        feature_scale = paired_arrays["feature_scale"].copy()
        scales = paired_arrays["target_scale"].copy()
    elif paired_start is None:
        start = _integral_start(
            scale_population, mass, settings.backbone, scales, include_last=allow_partial_horizons
        )
    else:
        paired_raw, paired_arrays, paired_binding = _validated_paired_start(
            paired_start, mass, feature_scale, scales
        )
        # The recomputation above checks provenance; use the stored train arrays.
        feature_scale = paired_arrays["feature_scale"].copy()
        scales = paired_arrays["target_scale"].copy()
    old_threads, old_determinism = (
        torch.get_num_threads(),
        torch.are_deterministic_algorithms_enabled(),
    )
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(settings.seed)
            tensor = lambda value: torch.tensor(value, dtype=torch.float64)
            b, basis, mt, wt, scale = (
                tensor(complete_incidence(n)),
                tensor(coi_basis(np.ones(n))),
                tensor(mass),
                tensor(weights),
                tensor(scales),
            )
            x = tensor(features / feature_scale)
            power = tensor(np.stack([s.history.constant_power for s in samples]))
            eq = tensor(np.stack([s.equilibrium_delta for s in samples])) @ b
            history = tensor(np.stack([np.c_[s.history.delta, s.history.omega] for s in samples]))
            hist_dt = tensor(np.stack([np.diff(s.history.time) for s in samples]))[:, :, None]
            if allow_partial_horizons:
                # Neutral tensor storage only, never an observation or feature.
                # Missing targets have exactly zero loss weight below.
                target_buffer = np.zeros((len(samples), len(clock), 2 * n))
                for index, sample in enumerate(samples):
                    target_buffer[index, : len(sample.forecast_time)] = np.c_[
                        sample.target_delta, sample.target_omega
                    ]
                truth = tensor(target_buffer)
            else:
                truth = tensor(np.stack([np.c_[s.target_delta, s.target_omega] for s in samples]))
            raw = torch.nn.Parameter(tensor(
                np.log(np.expm1(np.minimum(start, 40))) + np.maximum(start - 40, 0)
                if paired_binding is None else paired_raw
            ))
            if full_parameters is None:
                port = torch.nn.Parameter(0.1 * torch.randn((n - 1, r), dtype=torch.float64))
                skew = torch.nn.Parameter(0.2 * torch.randn((r, r), dtype=torch.float64))
                decay = torch.nn.Parameter(torch.full((r,), -0.5, dtype=torch.float64))
                encoder = torch.nn.Parameter(0.01 * torch.randn((r, x.shape[1]), dtype=torch.float64))
            else:
                port, skew, decay, encoder = (
                    torch.nn.Parameter(tensor(full_parameters[name]))
                    for name in ("port", "skew", "decay", "encoder")
                )
            parameters = [raw, port, skew, decay, encoder]
            parameter_names = ("raw", "port", "skew", "decay", "encoder")
            updated_scalars = sum(p.numel() for p in parameters) - (raw.numel() if freeze_backbone else 0)
            paired_record = {}
            if paired_binding is not None:
                initial_positive = torch.nn.functional.softplus(raw).detach().numpy()
                reconstructed_damping = (basis.numpy() * initial_positive[b.shape[1]:]) @ basis.numpy().T
                edge_error = float(np.linalg.norm(initial_positive[:b.shape[1]] - paired_arrays["edge_stiffness"]))
                damping_error = float(np.linalg.norm(reconstructed_damping - paired_arrays["damping"], ord="fro"))
                if max(edge_error, damping_error) > 1e-12:
                    raise ValueError("paired_start inverse-softplus round trip exceeds 1e-12")
                initial_raw = raw.detach().clone()
                initial_arrays = {name: p.detach().numpy().copy()
                                  for name, p in zip(parameter_names, parameters, strict=True)}
                paired_record = {
                    **paired_binding,
                    "initial_parameter_sha256": {name: _array_sha256(value)
                                                 for name, value in initial_arrays.items()},
                    "initial_parameters": {name: value.tolist() for name, value in initial_arrays.items()},
                    "backbone_frozen": freeze_backbone,
                    "optimizer_updated_scalars": updated_scalars,
                    "paired_start_roundtrip_edge_error": edge_error,
                    "paired_start_roundtrip_damping_error": damping_error,
                    "gradient_clip_population": "complete_parameter_vector_before_F_backbone_update_mask",
                    "new_nnls_performed": False,
                    "loss_parameter_timing": "before_update",
                    "checkpoint_rule": "last_update",
                }
                if full_parameters is not None:
                    reconstructed = {
                        "edge_stiffness": initial_positive[:b.shape[1]],
                        "damping": reconstructed_damping,
                        "coupling": basis.numpy() @ port.detach().numpy(),
                        "latent_skew": ((skew - skew.T) / 2).detach().numpy(),
                        "latent_dissipation": np.diag(
                            torch.nn.functional.softplus(decay).detach().numpy() + 1e-5),
                        "encoder": encoder.detach().numpy(),
                    }
                    errors = {name: float(np.linalg.norm(value - paired_arrays[name]))
                              for name, value in reconstructed.items()}
                    if max(errors.values()) > 1e-12:
                        raise ValueError("full_start physical round trip exceeds 1e-12")
                    paired_record.update(
                        full_start_roundtrip_errors=errors,
                        gradient_clip_population="complete_parameter_vector",
                        planned_optimizer_steps=settings.epochs,
                        registered_checkpoint_steps=list(checkpoint_steps),
                        initial_optimizer_step=initial_optimizer_step,
                        initial_optimizer_moments_zero=(True if optimizer_binding is None
                                                       else optimizer_binding["initial_optimizer_moments_zero"]),
                    )
            optimizer = torch.optim.Adam(parameters, lr=settings.learning_rate)
            if optimizer_arrays is not None:
                for name, parameter in zip(parameter_names, parameters, strict=True):
                    if parameter.detach().numpy().tobytes() != optimizer_arrays[f"parameter__{name}"].tobytes():
                        raise ValueError("optimizer_start raw parameter bytes changed during restoration")
                    optimizer.state[parameter] = {
                        key: torch.from_numpy(optimizer_arrays[f"adam_{key}__{name}"].copy())
                        for key in ("exp_avg", "exp_avg_sq", "step")
                    }
            time_weight = np.zeros(len(clock) - 1)
            for horizon in settings.horizons_s:
                count = round(horizon / 0.01)
                if abs(count * 0.01 - horizon) > 1e-10:
                    raise ValueError("horizon must lie on registered clock")
                time_weight[:count] += 1 / count / len(settings.horizons_s)
            tw = tensor(time_weight)
            if allow_partial_horizons:
                time_weight = np.zeros((len(samples), len(clock) - 1))
                for index, sample in enumerate(samples):
                    for horizon in settings.horizons_s:
                        count = min(round(horizon / 0.01), len(sample.forecast_time) - 1)
                        time_weight[index, :count] += 1 / count / len(settings.horizons_s)
                tw = tensor(time_weight)
            trace = []

            def model_snapshot(completed_steps):
                positive_values = torch.nn.functional.softplus(raw).detach().numpy()
                bs = basis.numpy()
                closure = NonlinearPassiveClosure(
                    mass,
                    positive_values[: b.shape[1]],
                    (bs * positive_values[b.shape[1] :]) @ bs.T,
                    bs @ port.detach().numpy(),
                    ((skew - skew.T) / 2).detach().numpy(),
                    np.diag(torch.nn.functional.softplus(decay).detach().numpy() + 1e-5),
                    settings.backbone,
                )
                event_steps = completed_steps * (
                    len(samples) * (history_count - 1)
                    + sum(len(s.forecast_time) - 1 for s in samples)
                ) if not training_options_active else completed_steps * training_event_steps_per_update
                record = {
                    "fitted_event_ids": sorted(s.event_id for s in samples),
                    "fitted_roles": ["train"],
                    "hidden_targets_read": False,
                    "future_observations_in_encoder": False,
                    "source_routing": False,
                    "normalization": "train_only",
                    "initializer": "shared_encoder_at_history_start_then_free_history_dynamics",
                    "warm_start": "nonnegative_visible_momentum_integral",
                    "loss_trace": deepcopy(trace),
                    "actual_optimizer_steps": completed_steps,
                    "parameter_count": sum(p.numel() for p in parameters),
                    "device": str(raw.device),
                    "dtype": str(raw.dtype),
                    "training_rk4_event_steps": event_steps,
                    "training_rhs_event_evaluations": 4 * event_steps,
                }
                if training_options_active:
                    record.update(
                        training_max_step=training_max_step,
                        temporal_aggregation=("within_event_logsumexp" if temporal_temperature is not None
                                              else "original_weighted_time_mean"),
                        temporal_temperature=temporal_temperature,
                        training_event_rk4_steps_per_optimizer_update=training_event_steps_per_update,
                        rhs_audit_enabled=rhs_audit is not None,
                    )
                if paired_binding is not None:
                    record.update(**deepcopy(paired_record),
                                  warm_start="verified_paired_visible_backbone_no_nnls",
                                  final_checkpoint_optimizer_step=completed_steps,
                                  gradient_clipped_steps=sum(row["gradient_clipped"] for row in trace),
                                  frozen_backbone_verified_each_update=freeze_backbone)
                if full_parameters is not None:
                    record.update(warm_start="verified_full_physical_checkpoint_fresh_adam",
                                  checkpoint_rule="registered_update_count",
                                  total_optimizer_steps=initial_optimizer_step + completed_steps,
                                  planned_total_optimizer_steps=initial_optimizer_step + settings.epochs)
                if optimizer_arrays is not None:
                    record.update(warm_start="verified_full_raw_checkpoint_exact_adam_resume")
                if optimizer_arrays is not None or risk_temperature is not None:
                    record.update(risk_temperature=risk_temperature,
                                  objective_aggregation=(
                                      "within_event_logsumexp" if temporal_temperature is not None
                                      else "weighted_mean" if risk_temperature is None
                                      else "weighted_logsumexp"))
                if allow_partial_horizons:
                    record.update(
                        partial_horizons=True,
                        loss_masks_are_features=False,
                        real_target_counts={s.event_id: len(s.forecast_time) - 1 for s in samples},
                        missing_target_weight=0.0,
                    )
                return LearnedNonlinearModel(
                    closure, encoder.detach().numpy().copy(), feature_scale.copy(), scales.copy(),
                    deepcopy(settings), record
                )

            def optimizer_snapshot(completed_steps):
                state = {
                    "schema": np.asarray("nonlinear_adam_training_state.v1"),
                    "completed_optimizer_steps": np.asarray(completed_steps, dtype=np.int64),
                    "planned_optimizer_steps": np.asarray(settings.epochs, dtype=np.int64),
                    "initial_optimizer_step": np.asarray(initial_optimizer_step, dtype=np.int64),
                    "total_optimizer_steps": np.asarray(initial_optimizer_step + completed_steps, dtype=np.int64),
                    "planned_total_optimizer_steps": np.asarray(initial_optimizer_step + settings.epochs, dtype=np.int64),
                    "adam_learning_rate": np.asarray(optimizer.param_groups[0]["lr"]),
                    "adam_betas": np.asarray(optimizer.param_groups[0]["betas"]),
                    "adam_eps": np.asarray(optimizer.param_groups[0]["eps"]),
                    "adam_weight_decay": np.asarray(optimizer.param_groups[0]["weight_decay"]),
                    "adam_amsgrad": np.asarray(optimizer.param_groups[0]["amsgrad"]),
                }
                for name, parameter in zip(parameter_names, parameters, strict=True):
                    state[f"parameter__{name}"] = parameter.detach().numpy().copy()
                    moment = optimizer.state.get(parameter, {})
                    for key in ("exp_avg", "exp_avg_sq", "step"):
                        value = moment.get(key)
                        if value is None:
                            value = torch.zeros_like(parameter) if key != "step" else torch.tensor(0.)
                        state[f"adam_{key}__{name}"] = value.detach().numpy().copy()
                return state

            for epoch in range(settings.epochs):
                optimizer.zero_grad()
                positive = torch.nn.functional.softplus(raw)
                k, d = positive[: b.shape[1]], positive[b.shape[1] :]
                damping, coupling = (basis * d) @ basis.T, basis @ port
                j, decay_matrix = (
                    (skew - skew.T) / 2,
                    torch.diag(torch.nn.functional.softplus(decay) + 1e-5),
                )

                def rhs(
                    state,
                    k=k,
                    damping=damping,
                    coupling=coupling,
                    j=j,
                    decay_matrix=decay_matrix,
                    power=power,
                    eq=eq,
                ):
                    q, v, z = state[:, :n], state[:, n : 2 * n], state[:, 2 * n :]
                    edge = q @ b
                    force = (
                        edge
                        if settings.backbone == "linear"
                        else torch.sin(edge + eq) - torch.sin(eq)
                    )
                    f = power - (force * k) @ b.T - v @ damping.T - z @ coupling.T
                    f = f - f.sum(dim=1, keepdim=True) * mt / mt.sum()
                    return torch.cat((v, f / mt, v @ coupling + z @ (j - decay_matrix).T), dim=1)

                def step(state, h, indices=None, rhs=rhs):
                    def field(value):
                        if rhs_audit is None:
                            return (rhs(value) if indices is None else
                                    rhs(value, power=power[indices], eq=eq[indices]))
                        kwargs = {} if indices is None else {
                            "power": power[indices], "eq": eq[indices]}
                        return rhs_audit(rhs, value, **kwargs)

                    a = field(state)
                    bb = field(state + h * a / 2)
                    c = field(state + h * bb / 2)
                    dd = field(state + h * c)
                    return state + h * (a + 2 * bb + 2 * c + dd) / 6

                def advance(state, h, indices=None, substeps=1):
                    if not training_options_active:
                        return step(state, h, indices)
                    increment = h / substeps
                    for _ in range(substeps):
                        state = step(state, increment, indices)
                    return state

                state = torch.cat((history[:, 0], x @ encoder.T), dim=1)
                history_error = torch.zeros(len(samples), dtype=torch.float64)
                for index in range(history_count - 1):
                    if training_options_active:
                        state = advance(state, hist_dt[:, index],
                                        substeps=int(history_substeps_per_interval[index]))
                    else:
                        state = step(state, hist_dt[:, index])
                    history_error = history_error + torch.mean(
                        ((state[:, : 2 * n] - history[:, index + 1]) / scale) ** 2, dim=1
                    ) / (history_count - 1)
                # At the forecast origin only measured visible coordinates reset.
                # The hidden state follows the shared model's history rollout.
                state = torch.cat((history[:, -1], state[:, 2 * n :]), dim=1)
                future_error = torch.zeros(len(samples), dtype=torch.float64)
                future_point_losses = []
                for index in range(1, len(clock)):
                    if allow_partial_horizons:
                        indices = torch.as_tensor(
                            [i for i, s in enumerate(samples) if len(s.forecast_time) > index],
                            dtype=torch.long,
                        )
                        if len(indices):
                            state = state.index_copy(
                                0, indices, advance(state[indices], 0.01, indices, future_substeps)
                            )
                    else:
                        state = advance(state, 0.01, substeps=future_substeps)
                    weight = tw[:, index - 1] if allow_partial_horizons else tw[index - 1]
                    point_loss = torch.mean(
                        ((state[:, : 2 * n] - truth[:, index]) / scale) ** 2, dim=1
                    )
                    if temporal_temperature is not None:
                        future_point_losses.append(point_loss)
                    future_error = future_error + weight * point_loss
                future_loss, history_loss = (
                    torch.sum(wt * future_error),
                    torch.sum(wt * history_error),
                )
                risk_weights = wt
                future_objective = future_loss
                temporal_risk_weights = None
                if risk_temperature is not None:
                    future_objective, risk_weights = _tilted_risk(future_error, wt, risk_temperature)
                elif temporal_temperature is not None:
                    from kron_reduce.models.temporal_risk import temporal_logsumexp_risk

                    point_mse = torch.stack(future_point_losses, dim=1)
                    event_time_weights = (tw if allow_partial_horizons
                                          else tw.unsqueeze(0).expand(len(samples), -1))
                    event_temporal_risk, temporal_risk_weights = temporal_logsumexp_risk(
                        point_mse, event_time_weights, temporal_temperature,
                        return_time_risk_weights=True,
                    )
                    future_objective = torch.sum(wt * event_temporal_risk)
                loss = future_objective + settings.history_weight * history_loss
                encoder_penalty = settings.encoder_penalty * torch.mean(encoder**2) if r else loss.new_zeros(())
                if r:
                    loss = loss + encoder_penalty
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite nonlinear training objective")
                loss.backward()
                gradient_record = {}
                if paired_binding is None:
                    torch.nn.utils.clip_grad_norm_(parameters, 10)
                else:
                    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
                        raise FloatingPointError("nonfinite or missing paired training gradient")
                    backbone_norm = torch.linalg.vector_norm(raw.grad)
                    memory_norm = torch.linalg.vector_norm(torch.stack(
                        [torch.linalg.vector_norm(p.grad) for p in parameters[1:]]
                    ))
                    # F still contributes its backbone gradient to this same clipping operation.
                    full_norm = torch.nn.utils.clip_grad_norm_(parameters, 10, error_if_nonfinite=True)
                    gradient_record = {
                        "backbone_gradient_norm": float(backbone_norm.detach()),
                        "memory_gradient_norm": float(memory_norm.detach()),
                        "full_gradient_norm": float(full_norm.detach()),
                        "gradient_clipped": bool(full_norm > 10),
                        "optimizer_updated_scalars": updated_scalars,
                        "updated_optimizer_steps": epoch + 1,
                    }
                    if freeze_backbone:
                        raw.grad = None
                optimizer.step()
                if paired_binding is not None:
                    if any(not torch.isfinite(p).all() for p in parameters):
                        raise FloatingPointError("nonfinite paired training parameter after update")
                    if freeze_backbone and not torch.equal(raw.detach(), initial_raw):
                        raise RuntimeError("frozen backbone changed during optimizer update")
                if progress_callback is not None:
                    progress_callback(epoch + 1)
                log_step = epoch == 0 or (epoch + 1) % 25 == 0 or epoch + 1 == settings.epochs
                if paired_binding is not None or training_options_active or log_step:
                    row = {
                        "epoch": epoch + 1,
                        "objective": float(loss.detach()),
                        "future_loss": float(future_loss.detach()),
                        "history_loss": float(history_loss.detach()),
                    }
                    if optimizer_arrays is not None or risk_temperature is not None:
                        row.update(
                            future_objective=float(future_objective.detach()),
                            tilted_future_risk=None if risk_temperature is None else float(future_objective.detach()),
                            risk_temperature=risk_temperature,
                            risk_effective_sample_size=float(1 / torch.sum(risk_weights**2)),
                            risk_max_event_weight=float(torch.max(risk_weights)),
                            total_optimizer_steps=initial_optimizer_step + epoch + 1,
                        )
                    if training_options_active:
                        row.update(
                            future_objective=float(future_objective.detach()),
                            temporal_temperature=temporal_temperature,
                            training_max_step=training_max_step,
                            temporal_aggregation=("within_event_logsumexp" if temporal_temperature is not None
                                                  else "original_weighted_time_mean"),
                        )
                    if temporal_temperature is not None:
                        row.update(
                            temporal_aggregation="within_event_logsumexp",
                            temporal_effective_time_points=(
                                1.0 / torch.sum(temporal_risk_weights**2, dim=1)
                            ).detach().tolist(),
                        )
                    if paired_binding is not None:
                        row.update(encoder_penalty=float(encoder_penalty.detach()), **gradient_record)
                    trace.append(row)
                if log_step:
                    print(f"epoch {epoch + 1}: objective={row['objective']:.6g}", flush=True)
                if epoch + 1 in checkpoint_steps:
                    checkpoint_callback(epoch + 1, model_snapshot(epoch + 1),
                                        optimizer_snapshot(epoch + 1))
            return model_snapshot(settings.epochs)
    finally:
        torch.set_num_threads(old_threads)
        torch.use_deterministic_algorithms(old_determinism)
