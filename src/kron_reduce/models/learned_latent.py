"""Learn a passive linear closure and one shared causal history encoder.

Only visible trajectories enter training. Hidden coordinates are internal
optimization variables and have no assumed match to physical device states.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from kron_reduce.models.causal_latent import CausalStateHistory, PassiveLatentClosure
from kron_reduce.physics.coordinates import coi_basis


@dataclass(frozen=True)
class LatentLearningSample:
    event_id: str
    source_dataset: str
    fault_line: str
    role: str
    history: CausalStateHistory
    forecast_time: np.ndarray
    target_delta: np.ndarray
    target_omega: np.ndarray

    def __post_init__(self) -> None:
        if not all((self.event_id, self.source_dataset, self.fault_line)) or self.role not in {
            "train",
            "val",
            "audit",
        }:
            raise ValueError("learning sample needs registered identity and role")
        for name in ("forecast_time", "target_delta", "target_omega"):
            value = np.array(getattr(self, name), dtype=np.float64, copy=True)
            if not np.isfinite(value).all():
                raise ValueError("learning targets and clock must be finite")
            value.flags.writeable = False
            object.__setattr__(self, name, value)
        time, n = self.forecast_time, self.history.delta.shape[1]
        if time.ndim != 1 or len(time) < 2 or time[0] != 0.0 or np.any(np.diff(time) <= 0.0):
            raise ValueError("forecast clock must start at zero and increase")
        if (
            self.target_delta.shape != (len(time), n)
            or self.target_omega.shape != self.target_delta.shape
        ):
            raise ValueError("visible target dimensions differ")
        if not np.allclose(
            self.target_delta[0], self.history.delta[-1], atol=1e-12, rtol=0.0
        ) or not np.allclose(self.target_omega[0], self.history.omega[-1], atol=1e-12, rtol=0.0):
            raise ValueError("forecast must begin at the last measured state")


def event_weights(samples: Sequence[LatentLearningSample]) -> np.ndarray:
    if not samples or len({sample.event_id for sample in samples}) != len(samples):
        raise ValueError("learning populations must be nonempty and unique")
    sources = sorted({sample.source_dataset for sample in samples})
    output = []
    for sample in samples:
        lines = {row.fault_line for row in samples if row.source_dataset == sample.source_dataset}
        count = sum(
            row.source_dataset == sample.source_dataset and row.fault_line == sample.fault_line
            for row in samples
        )
        output.append(1.0 / len(sources) / len(lines) / count)
    return np.asarray(output)


def visible_scales(samples: Sequence[LatentLearningSample]) -> np.ndarray:
    if {sample.role for sample in samples} != {"train"}:
        raise ValueError("visible normalization uses train only")
    weights = event_weights(samples)
    moments = []
    for sample in samples:
        states = np.concatenate((sample.target_delta, sample.target_omega), axis=1)
        moments.append(np.mean(states[1:] ** 2, axis=0))
    return np.maximum(np.sqrt(weights @ np.stack(moments)), 1e-5)


def history_features(history: CausalStateHistory, frames: int) -> np.ndarray:
    if type(frames) is not int or frames < 2 or len(history.time) < frames:
        raise ValueError("history lacks the registered number of real observation frames")
    indices = np.linspace(0, len(history.time) - 1, frames).round().astype(int)
    return np.concatenate(
        (
            history.delta[indices].reshape(-1),
            history.omega[indices].reshape(-1),
            history.constant_power,
        )
    )


@dataclass(frozen=True)
class LatentFitSettings:
    latent_dimension: int = 2
    frames: int = 5
    epochs: int = 300
    learning_rate: float = 0.02
    encoder_penalty: float = 1e-6
    seed: int = 324
    maximum_step_s: float = 0.01

    def __post_init__(self) -> None:
        if type(self.latent_dimension) is not int or self.latent_dimension < 0:
            raise ValueError("latent dimension must be a nonnegative integer")
        if (
            type(self.frames) is not int
            or self.frames < 2
            or type(self.epochs) is not int
            or self.epochs < 1
        ):
            raise ValueError("invalid frame or epoch budget")
        if (
            not np.isfinite([self.learning_rate, self.encoder_penalty, self.maximum_step_s]).all()
            or self.learning_rate <= 0
            or self.encoder_penalty < 0
            or not 0 < self.maximum_step_s <= 0.01
        ):
            raise ValueError("invalid optimization or integration budget")


@dataclass
class LearnedLatentModel:
    closure: PassiveLatentClosure
    encoder: np.ndarray
    feature_scale: np.ndarray
    target_scale: np.ndarray
    settings: LatentFitSettings
    fit_record: dict[str, Any]

    def predict(
        self, history: CausalStateHistory, time: np.ndarray, *, zero_latent: bool = False
    ) -> np.ndarray:
        feature = history_features(history, self.settings.frames)
        latent = (
            np.zeros(self.closure.latent_dimension)
            if zero_latent
            else self.encoder @ (feature / self.feature_scale)
        )
        return self.closure.rollout(
            history.delta[-1],
            history.omega[-1],
            latent,
            time,
            history.constant_power,
            max_step=self.settings.maximum_step_s,
        )

    def score(
        self, samples: Sequence[LatentLearningSample], *, zero_latent: bool = False
    ) -> dict[str, Any]:
        rows = []
        weights = event_weights(samples)
        for sample in samples:
            prediction = self.predict(sample.history, sample.forecast_time, zero_latent=zero_latent)
            n = self.closure.n_nodes
            truth = np.concatenate((sample.target_delta, sample.target_omega), axis=1)
            error = prediction[1:, : 2 * n] - truth[1:]
            rows.append(
                {
                    "event_id": sample.event_id,
                    "source_dataset": sample.source_dataset,
                    "fault_line": sample.fault_line,
                    "role": sample.role,
                    "normalized_mse": float(np.mean((error / self.target_scale) ** 2)),
                    "delta_rmse_rad": float(np.sqrt(np.mean(error[:, :n] ** 2))),
                    "omega_rmse_rad_per_s": float(np.sqrt(np.mean(error[:, n:] ** 2))),
                    "forecast_duration_s": float(sample.forecast_time[-1]),
                    "observation_duration_s": float(-sample.history.time[0]),
                }
            )
        return {
            "normalized_rmse": float(np.sqrt(weights @ [row["normalized_mse"] for row in rows])),
            "mean_delta_rmse_rad": float(weights @ [row["delta_rmse_rad"] for row in rows]),
            "mean_omega_rmse_rad_per_s": float(
                weights @ [row["omega_rmse_rad_per_s"] for row in rows]
            ),
            "weighting": "source_line_event_time_equal",
            "score_includes_forecast_origin": False,
            "rows": rows,
        }

    def save(self, path: Path) -> None:
        with path.open("xb") as handle:
            np.savez_compressed(
                handle,
                metadata=np.asarray(
                    json.dumps(
                        {
                            "schema": "learned_passive_latent.v1",
                            "settings": asdict(self.settings),
                            "fit_record": self.fit_record,
                        },
                        allow_nan=False,
                    )
                ),
                encoder=self.encoder,
                feature_scale=self.feature_scale,
                target_scale=self.target_scale,
                **{name: getattr(self.closure, name) for name in self.closure.__dataclass_fields__},
            )

    @classmethod
    def load(cls, path: Path) -> LearnedLatentModel:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            if metadata["schema"] != "learned_passive_latent.v1":
                raise ValueError("learned checkpoint schema mismatch")
            closure = PassiveLatentClosure(
                **{name: data[name] for name in PassiveLatentClosure.__dataclass_fields__}
            )
            return cls(
                closure,
                data["encoder"],
                data["feature_scale"],
                data["target_scale"],
                LatentFitSettings(**metadata["settings"]),
                metadata["fit_record"],
            )


def fit_latent_model(
    samples: Sequence[LatentLearningSample],
    mass: np.ndarray,
    settings: LatentFitSettings,
    *,
    target_scale: np.ndarray | None = None,
) -> LearnedLatentModel:
    if {sample.role for sample in samples} != {"train"}:
        raise ValueError("optimizer accepts train samples only")
    scales = (
        visible_scales(samples) if target_scale is None else np.asarray(target_scale, dtype=float)
    )
    return _fit(samples, mass, settings, scales)


def refit_latent_model(
    train: Sequence[LatentLearningSample],
    val: Sequence[LatentLearningSample],
    mass: np.ndarray,
    settings: LatentFitSettings,
    target_scale: np.ndarray,
) -> LearnedLatentModel:
    if {sample.role for sample in train} != {"train"} or {sample.role for sample in val} != {"val"}:
        raise ValueError("final refit requires the registered train and val populations")
    if {sample.event_id for sample in train} & {sample.event_id for sample in val}:
        raise ValueError("train and val IDs overlap")
    if {(s.source_dataset, s.fault_line) for s in train} & {
        (s.source_dataset, s.fault_line) for s in val
    }:
        raise ValueError("train and val fault lines overlap")
    features = np.stack([history_features(sample.history, settings.frames) for sample in train])
    feature_scale = np.maximum(np.sqrt(event_weights(train) @ (features**2)), 1e-5)
    return _fit(
        tuple(train) + tuple(val), mass, settings, target_scale, feature_scale=feature_scale
    )


def _fit(
    samples: Sequence[LatentLearningSample],
    mass: np.ndarray,
    settings: LatentFitSettings,
    scales: np.ndarray,
    *,
    feature_scale: np.ndarray | None = None,
) -> LearnedLatentModel:
    import torch

    weights = event_weights(samples)
    mass = np.asarray(mass, dtype=np.float64)
    n, r = len(mass), settings.latent_dimension
    if mass.ndim != 1 or n < 2 or not np.isfinite(mass).all() or np.any(mass <= 0.0):
        raise ValueError("learning requires positive finite mass priors")
    if scales.shape != (2 * n,) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("train target scales must be finite and positive")
    clock = samples[0].forecast_time
    if any(
        not np.array_equal(sample.forecast_time, clock) or sample.history.delta.shape[1] != n
        for sample in samples
    ):
        raise ValueError("batched learning needs one registered clock and node count")
    for sample in samples:
        if abs(sample.history.constant_power.sum()) > 1e-10 or any(
            np.max(np.abs(value @ mass)) > 1e-10
            for value in (
                sample.history.delta,
                sample.history.omega,
                sample.target_delta,
                sample.target_omega,
            )
        ):
            raise ValueError(
                "learning inputs and targets must obey balanced power and COI constraints"
            )
    if not np.allclose(np.diff(clock), np.diff(clock)[0], rtol=0.0, atol=1e-12):
        raise ValueError("batched learning requires a uniform output clock")
    features = np.stack([history_features(sample.history, settings.frames) for sample in samples])
    if feature_scale is None:
        feature_scale = np.maximum(np.sqrt(weights @ (features**2)), 1e-5)
    old_threads = torch.get_num_threads()
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(settings.seed)
            dtype = torch.float64
            tensor = lambda value: torch.tensor(value, dtype=dtype)
            basis = tensor(coi_basis(np.ones(n)))
            mt, wt = tensor(mass), tensor(weights)
            x = tensor(features / feature_scale)
            q0 = tensor(np.stack([sample.history.delta[-1] for sample in samples]))
            v0 = tensor(np.stack([sample.history.omega[-1] for sample in samples]))
            power = tensor(np.stack([sample.history.constant_power for sample in samples]))
            truth = tensor(
                np.stack(
                    [
                        np.concatenate((sample.target_delta, sample.target_omega), axis=1)
                        for sample in samples
                    ]
                )
            )
            scale = tensor(scales)
            kraw = torch.nn.Parameter(torch.full((n - 1,), 0.0, dtype=dtype))
            draw = torch.nn.Parameter(torch.full((n - 1,), -1.5, dtype=dtype))
            port = torch.nn.Parameter(0.1 * torch.randn((n - 1, r), dtype=dtype))
            skew = torch.nn.Parameter(0.2 * torch.randn((r, r), dtype=dtype))
            decay = torch.nn.Parameter(torch.full((r,), -0.5, dtype=dtype))
            encoder = torch.nn.Parameter(0.01 * torch.randn((r, x.shape[1]), dtype=dtype))
            parameters = [kraw, draw, port, skew, decay, encoder]
            optimizer = torch.optim.Adam(parameters, lr=settings.learning_rate)

            def operators():
                stiffness = (basis * torch.nn.functional.softplus(kraw)) @ basis.T
                damping = (basis * torch.nn.functional.softplus(draw)) @ basis.T
                coupling = basis @ port
                j = (skew - skew.T) / 2.0
                dissipation = torch.diag(torch.nn.functional.softplus(decay) + 1e-5)
                generator = torch.cat(
                    (
                        torch.cat(
                            (
                                torch.zeros((n, n), dtype=dtype),
                                torch.eye(n, dtype=dtype),
                                torch.zeros((n, r), dtype=dtype),
                            ),
                            dim=1,
                        ),
                        torch.cat(
                            (
                                -stiffness / mt[:, None],
                                -damping / mt[:, None],
                                -coupling / mt[:, None],
                            ),
                            dim=1,
                        ),
                        torch.cat(
                            (torch.zeros((r, n), dtype=dtype), coupling.T, j - dissipation), dim=1
                        ),
                    ),
                    dim=0,
                )
                return stiffness, damping, coupling, j, dissipation, generator

            interval = float(clock[1] - clock[0])
            substeps = max(1, int(np.ceil(interval / settings.maximum_step_s - 1e-12)))
            step = interval / substeps
            losses = []
            for epoch in range(settings.epochs):
                optimizer.zero_grad()
                *_, matrix = operators()
                identity = torch.eye(2 * n + r, dtype=dtype)
                a2 = matrix @ matrix
                a3 = a2 @ matrix
                propagation = (
                    identity
                    + step * matrix
                    + step**2 * a2 / 2
                    + step**3 * a3 / 6
                    + step**4 * (a3 @ matrix) / 24
                )
                forcing_op = (
                    step * identity + step**2 * matrix / 2 + step**3 * a2 / 6 + step**4 * a3 / 24
                )
                forcing = (
                    torch.cat(
                        (
                            torch.zeros_like(q0),
                            power / mt,
                            torch.zeros((len(samples), r), dtype=dtype),
                        ),
                        dim=1,
                    )
                    @ forcing_op.T
                )
                state = torch.cat((q0, v0, x @ encoder.T), dim=1)
                error_sum = torch.zeros(len(samples), dtype=dtype)
                for index in range(1, len(clock)):
                    for _ in range(substeps):
                        state = state @ propagation.T + forcing
                    error_sum = error_sum + torch.mean(
                        ((state[:, : 2 * n] - truth[:, index]) / scale) ** 2, dim=1
                    )
                loss = torch.sum(wt * error_sum) / (len(clock) - 1)
                if r:
                    loss = loss + settings.encoder_penalty * torch.mean(encoder**2)
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite latent training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 10.0)
                optimizer.step()
                if epoch == 0 or epoch == settings.epochs - 1 or (epoch + 1) % 50 == 0:
                    losses.append({"epoch": epoch + 1, "train_objective": float(loss.detach())})
            matrices = [value.detach().numpy().copy() for value in operators()[:5]]
            closure = PassiveLatentClosure(mass, *matrices)
            record = {
                "settings": asdict(settings),
                "actual_optimizer_steps": settings.epochs,
                "fitted_event_ids": sorted(sample.event_id for sample in samples),
                "fitted_roles": sorted({sample.role for sample in samples}),
                "targets": ["delta", "omega"],
                "hidden_targets_read": False,
                "source_routing": False,
                "future_observations_in_encoder": False,
                "deterministic_algorithms": True,
                "torch_threads_during_fit": 1,
                "loss_trace": losses,
                "normalization": "train_only_fixed_before_selection",
                "rhs_parameter_count": sum(value.numel() for value in parameters[:5]),
                "encoder_parameter_count": encoder.numel(),
            }
            return LearnedLatentModel(
                closure,
                encoder.detach().numpy().copy(),
                feature_scale,
                np.array(scales, copy=True),
                settings,
                record,
            )
    finally:
        torch.set_num_threads(old_threads)
        torch.use_deterministic_algorithms(old_deterministic)
