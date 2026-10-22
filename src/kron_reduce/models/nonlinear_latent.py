"""Passive sine-network closure with a causal, dynamics-consistent initializer.

Coordinates are relative to the supplied event equilibrium, not absolute
angles. The shifted sine potential is not globally positive or convex.
"""

from __future__ import annotations

import json
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

    def __post_init__(self):
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


def _features(history: CausalStateHistory, equilibrium: np.ndarray, frames: int) -> np.ndarray:
    # Static equilibrium is allowed; targets, source, line, and identity are not.
    return np.r_[history_features(history, frames), equilibrium]


@dataclass
class LearnedNonlinearModel:
    closure: NonlinearPassiveClosure
    encoder: np.ndarray
    feature_scale: np.ndarray
    target_scale: np.ndarray
    settings: NonlinearFitSettings
    fit_record: dict

    def initialize(self, history, equilibrium) -> tuple[np.ndarray, np.ndarray]:
        seed = self.encoder @ (
            _features(history, equilibrium, self.settings.frames) / self.feature_scale
        )
        path = self.closure.rollout(
            history.delta[0],
            history.omega[0],
            seed,
            history.time - history.time[0],
            history.constant_power,
            equilibrium,
        )
        return path[-1, 2 * self.closure.n_nodes :], path

    def predict(self, history, time, equilibrium, *, zero_latent=False) -> np.ndarray:
        z = (
            np.zeros(self.closure.latent_dimension)
            if zero_latent
            else self.initialize(history, equilibrium)[0]
        )
        return self.closure.rollout(
            history.delta[-1], history.omega[-1], z, time, history.constant_power, equilibrium
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


def fit_nonlinear_model(
    samples,
    mass,
    settings: NonlinearFitSettings,
    *,
    target_scale_samples=None,
    allow_partial_horizons=False,
) -> LearnedNonlinearModel:
    """Fit train only, with visible history consistency and free multi-horizon loss."""
    import torch

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
    scale_population = samples if target_scale_samples is None else target_scale_samples
    if {s.event_id for s in scale_population} != {s.event_id for s in samples}:
        raise ValueError("normalization population differs from train fitting IDs")
    scales, weights = visible_scales(scale_population), event_weights(samples)
    features = np.stack(
        [_features(s.history, s.equilibrium_delta, settings.frames) for s in samples]
    )
    feature_scale = np.maximum(np.sqrt(weights @ features**2), 1e-5)
    start = _integral_start(
        scale_population, mass, settings.backbone, scales, include_last=allow_partial_horizons
    )
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
            raw = torch.nn.Parameter(
                tensor(np.log(np.expm1(np.minimum(start, 40))) + np.maximum(start - 40, 0))
            )
            port = torch.nn.Parameter(0.1 * torch.randn((n - 1, r), dtype=torch.float64))
            skew = torch.nn.Parameter(0.2 * torch.randn((r, r), dtype=torch.float64))
            decay = torch.nn.Parameter(torch.full((r,), -0.5, dtype=torch.float64))
            encoder = torch.nn.Parameter(0.01 * torch.randn((r, x.shape[1]), dtype=torch.float64))
            parameters = [raw, port, skew, decay, encoder]
            optimizer = torch.optim.Adam(parameters, lr=settings.learning_rate)
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
                        return (
                            rhs(value)
                            if indices is None
                            else rhs(value, power=power[indices], eq=eq[indices])
                        )

                    a = field(state)
                    bb = field(state + h * a / 2)
                    c = field(state + h * bb / 2)
                    dd = field(state + h * c)
                    return state + h * (a + 2 * bb + 2 * c + dd) / 6

                state = torch.cat((history[:, 0], x @ encoder.T), dim=1)
                history_error = torch.zeros(len(samples), dtype=torch.float64)
                for index in range(history_count - 1):
                    state = step(state, hist_dt[:, index])
                    history_error = history_error + torch.mean(
                        ((state[:, : 2 * n] - history[:, index + 1]) / scale) ** 2, dim=1
                    ) / (history_count - 1)
                # At the forecast origin only measured visible coordinates reset.
                # The hidden state follows the shared model's history rollout.
                state = torch.cat((history[:, -1], state[:, 2 * n :]), dim=1)
                future_error = torch.zeros(len(samples), dtype=torch.float64)
                for index in range(1, len(clock)):
                    if allow_partial_horizons:
                        indices = torch.as_tensor(
                            [i for i, s in enumerate(samples) if len(s.forecast_time) > index],
                            dtype=torch.long,
                        )
                        if len(indices):
                            state = state.index_copy(
                                0, indices, step(state[indices], 0.01, indices)
                            )
                    else:
                        state = step(state, 0.01)
                    weight = tw[:, index - 1] if allow_partial_horizons else tw[index - 1]
                    future_error = future_error + weight * torch.mean(
                        ((state[:, : 2 * n] - truth[:, index]) / scale) ** 2, dim=1
                    )
                future_loss, history_loss = (
                    torch.sum(wt * future_error),
                    torch.sum(wt * history_error),
                )
                loss = future_loss + settings.history_weight * history_loss
                if r:
                    loss = loss + settings.encoder_penalty * torch.mean(encoder**2)
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite nonlinear training objective")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 10)
                optimizer.step()
                if epoch == 0 or (epoch + 1) % 25 == 0 or epoch + 1 == settings.epochs:
                    row = {
                        "epoch": epoch + 1,
                        "objective": float(loss.detach()),
                        "future_loss": float(future_loss.detach()),
                        "history_loss": float(history_loss.detach()),
                    }
                    trace.append(row)
                    print(f"epoch {epoch + 1}: objective={row['objective']:.6g}", flush=True)
            positive = torch.nn.functional.softplus(raw).detach().numpy()
            bs = basis.numpy()
            closure = NonlinearPassiveClosure(
                mass,
                positive[: b.shape[1]],
                (bs * positive[b.shape[1] :]) @ bs.T,
                bs @ port.detach().numpy(),
                ((skew - skew.T) / 2).detach().numpy(),
                np.diag(torch.nn.functional.softplus(decay).detach().numpy() + 1e-5),
                settings.backbone,
            )
            record = {
                "fitted_event_ids": sorted(s.event_id for s in samples),
                "fitted_roles": ["train"],
                "hidden_targets_read": False,
                "future_observations_in_encoder": False,
                "source_routing": False,
                "normalization": "train_only",
                "initializer": "shared_encoder_at_history_start_then_free_history_dynamics",
                "warm_start": "nonnegative_visible_momentum_integral",
                "loss_trace": trace,
                "actual_optimizer_steps": settings.epochs,
                "parameter_count": sum(p.numel() for p in parameters),
            }
            if allow_partial_horizons:
                record.update(
                    partial_horizons=True,
                    loss_masks_are_features=False,
                    real_target_counts={s.event_id: len(s.forecast_time) - 1 for s in samples},
                    missing_target_weight=0.0,
                )
            return LearnedNonlinearModel(
                closure, encoder.detach().numpy().copy(), feature_scale, scales, settings, record
            )
    finally:
        torch.set_num_threads(old_threads)
        torch.use_deterministic_algorithms(old_determinism)
