from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

from kron_reduce.physics.coordinates import coi_project, coi_project_power

from .base import DynamicsModel, FitResult, RunContext, sample_rollout_inputs
from .port_structured_discovery import PortStructuredDiscoveryM0Model


class LowRankModalPortM0Model(DynamicsModel):
    """Reciprocal low-rank passive modal port around an M0-PSD backbone.

    A rank-``r`` port uses only ``2r`` dynamic states, irrespective of the
    number of graph edges::

        q_dot = p
        p_dot = L.T @ nu - omega0**2*q - 2*zeta*omega0*p
        edge_force = L @ p

    Here ``nu=B.T@omega`` and the same signed loading matrix ``L`` is used at
    the input and output port.  The storage ``1/2*(p.T@p+omega0**2*q.T@q)``
    therefore has the exact supply identity

        H_dot = (L@p).T@nu - 2*zeta*omega0*p.T@p.

    This reciprocity is the key distinction from independent per-edge
    resonators: the learned column space is an explicit coherent edge-mode
    subspace and the induced transfer matrix is symmetric positive
    semidefinite.
    """

    method_name = "M0-LRMRP"

    def __init__(
        self,
        backbone: PortStructuredDiscoveryM0Model | None = None,
        *,
        frequency_hz: float = 1.0,
        damping_ratio: float = 0.1,
        rank: int = 0,
        loadings: np.ndarray | None = None,
        ridge: float = 1e-6,
        max_internal_step: float = 0.01,
        diagnostics: dict[str, Any] | None = None,
    ) -> None:
        self.backbone = backbone
        self.frequency_hz = float(frequency_hz)
        self.damping_ratio = float(damping_ratio)
        self.rank = int(rank)
        self.loadings = None if loadings is None else np.asarray(loadings, dtype=float)
        self.ridge = float(ridge)
        self.max_internal_step = float(max_internal_step)
        self.diagnostics = diagnostics or {}
        if not np.isfinite(self.frequency_hz) or self.frequency_hz <= 0:
            raise ValueError("frequency_hz must be finite and positive")
        if not np.isfinite(self.damping_ratio) or not 0 < self.damping_ratio < 1:
            raise ValueError("damping_ratio must lie in (0, 1)")
        if self.rank < 0:
            raise ValueError("rank must be nonnegative")
        if not np.isfinite(self.ridge) or self.ridge < 0:
            raise ValueError("ridge must be finite and nonnegative")
        if not np.isfinite(self.max_internal_step) or self.max_internal_step <= 0:
            raise ValueError("max_internal_step must be finite and positive")

    @property
    def angular_frequency(self) -> float:
        return 2.0*np.pi*self.frequency_hz

    def _require_fitted(self) -> tuple[PortStructuredDiscoveryM0Model, np.ndarray]:
        if self.backbone is None:
            raise RuntimeError("M0-LRMRP requires a fitted port backbone")
        self.backbone._require_fitted()
        if self.loadings is None:
            raise RuntimeError("M0-LRMRP loadings are not fitted")
        loadings = np.asarray(self.loadings, dtype=float)
        n_edges = int(self.backbone.static_coefficients.shape[1])
        if loadings.shape != (n_edges, self.rank):
            raise ValueError("loadings must have shape (edge, rank)")
        return self.backbone, loadings

    @staticmethod
    def _input_mode(pm: np.ndarray, requested: str) -> str:
        if requested not in {"auto", "zoh", "linear"}:
            raise ValueError("Pm interpolation must be auto, zoh, or linear")
        if requested != "auto":
            return requested
        return "zoh" if np.allclose(pm, pm[[0]], rtol=0.0, atol=1e-14) else "linear"

    def initial_modal_state(
        self, omega: np.ndarray, static: dict[str, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        _, loadings = self._require_fitted()
        incidence = np.asarray(static["incidence"], dtype=float)
        modal_input = loadings.T @ (incidence.T @ np.asarray(omega, dtype=float).reshape(-1))
        position = modal_input/self.angular_frequency**2
        return position, np.zeros_like(position)

    def modal_derivative(
        self, position: np.ndarray, velocity: np.ndarray, edge_speed: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        _, loadings = self._require_fitted()
        drive = loadings.T @ np.asarray(edge_speed, dtype=float).reshape(-1)
        angular = self.angular_frequency
        return velocity, drive-angular**2*position-2*self.damping_ratio*angular*velocity

    def vector_field_augmented(
        self,
        delta: np.ndarray,
        omega: np.ndarray,
        emf: np.ndarray,
        memory: np.ndarray,
        modal_position: np.ndarray,
        modal_velocity: np.ndarray,
        pm: np.ndarray,
        static: dict[str, np.ndarray],
        topology: dict[str, np.ndarray | float] | None = None,
    ) -> tuple[np.ndarray, ...]:
        backbone, loadings = self._require_fitted()
        dd, dw, dq, dz = backbone.vector_field_augmented(
            delta, omega, emf, memory, pm, static, topology
        )
        if modal_position.shape != (self.rank,) or modal_velocity.shape != (self.rank,):
            raise ValueError("modal state shape does not match selected rank")
        incidence = np.asarray(static["incidence"], dtype=float)
        mass = np.asarray(static["mass"], dtype=float).reshape(-1)
        edge_flow = loadings @ modal_velocity
        correction = coi_project_power(-incidence @ edge_flow, mass)
        edge_speed = incidence.T @ np.asarray(omega, dtype=float).reshape(-1)
        drq, drp = self.modal_derivative(modal_position, modal_velocity, edge_speed)
        return dd, dw+correction/mass, dq, dz, drq, drp

    def rollout_augmented(
        self,
        initial_state: dict[str, np.ndarray],
        time_grid: np.ndarray,
        exogenous: dict[str, np.ndarray],
        *,
        max_step: float | None = None,
    ) -> tuple[np.ndarray, ...]:
        backbone, _ = self._require_fitted()
        t = np.asarray(time_grid, dtype=float)
        pm = np.asarray(exogenous["Pm"], dtype=float)
        if len(t) != len(pm) or len(t) < 1:
            raise ValueError("time and Pm schedules must have the same positive length")
        step_limit = self.max_internal_step if max_step is None else float(max_step)
        if step_limit <= 0:
            raise ValueError("maximum integration step must be positive")
        input_mode = self._input_mode(pm, exogenous.get("interpolation", "auto"))
        mass = np.asarray(initial_state["mass"], dtype=float).reshape(-1)
        delta = coi_project(np.asarray(initial_state["delta0"], dtype=float).reshape(-1), mass)
        omega = coi_project(np.asarray(initial_state["omega0"], dtype=float).reshape(-1), mass)
        emf = backbone.backbone.initial_emf(delta, initial_state)
        memory = backbone.initial_memory(delta, omega, initial_state)
        position, velocity = self.initial_modal_state(omega, initial_state)
        topology = backbone.backbone._graph_topology(initial_state)
        state = (delta, omega, emf, memory, position, velocity)
        outputs = tuple([value.copy()] for value in state)

        def input_at(index: int, fraction: float) -> np.ndarray:
            if input_mode == "zoh":
                return pm[index]
            return pm[index]+fraction*(pm[index+1]-pm[index])

        def field(values: tuple[np.ndarray, ...], input_value: np.ndarray) -> tuple[np.ndarray, ...]:
            return self.vector_field_augmented(*values, input_value, initial_state, topology)

        for index in range(len(t)-1):
            interval = float(t[index+1]-t[index])
            if interval <= 0:
                raise ValueError("rollout time must be strictly increasing")
            substeps = max(1, int(np.ceil(interval/step_limit)))
            step = interval/substeps
            for substep in range(substeps):
                f0, fh, f1 = substep/substeps, (substep+0.5)/substeps, (substep+1)/substeps
                k1 = field(state, input_at(index, f0))
                k2 = field(tuple(x+0.5*step*k for x, k in zip(state, k1, strict=True)), input_at(index, fh))
                k3 = field(tuple(x+0.5*step*k for x, k in zip(state, k2, strict=True)), input_at(index, fh))
                k4 = field(tuple(x+step*k for x, k in zip(state, k3, strict=True)), input_at(index, f1))
                state = tuple(
                    x+step*(a+2*b+2*c+d)/6
                    for x, a, b, c, d in zip(state, k1, k2, k3, k4, strict=True)
                )
                state = (coi_project(state[0], mass), coi_project(state[1], mass), *state[2:])
                if not all(np.isfinite(value).all() for value in state):
                    raise FloatingPointError("nonfinite M0-LRMRP state")
                if np.min(state[2]) <= backbone.backbone.q_floor or np.max(state[2]) >= backbone.backbone.q_ceiling:
                    raise FloatingPointError("M0-LRMRP transient EMF left audited domain")
            for output, value in zip(outputs, state, strict=True):
                output.append(value.copy())
        return tuple(np.stack(output) for output in outputs)

    def rollout(
        self,
        initial_state: dict[str, np.ndarray],
        time_grid: np.ndarray,
        exogenous: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        state = self.rollout_augmented(initial_state, time_grid, exogenous)
        return state[0], state[1]

    @staticmethod
    def _edge_filter(
        edge_speed: np.ndarray,
        time_grid: np.ndarray,
        angular: float,
        damping: float,
    ) -> np.ndarray:
        speed = np.asarray(edge_speed, dtype=float)
        t = np.asarray(time_grid, dtype=float)
        position = speed[0]/angular**2
        velocity = np.zeros_like(position)
        output = [velocity.copy()]

        def derivative(q: np.ndarray, p: np.ndarray, drive: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return p, drive-angular**2*q-2*damping*angular*p

        for index, step in enumerate(np.diff(t)):
            if step <= 0:
                raise ValueError("teacher-forced time must be strictly increasing")
            u0, u1 = speed[index], speed[index+1]
            uh = 0.5*(u0+u1)
            k1 = derivative(position, velocity, u0)
            k2 = derivative(position+0.5*step*k1[0], velocity+0.5*step*k1[1], uh)
            k3 = derivative(position+0.5*step*k2[0], velocity+0.5*step*k2[1], uh)
            k4 = derivative(position+step*k3[0], velocity+step*k3[1], u1)
            position = position+step*(k1[0]+2*k2[0]+2*k3[0]+k4[0])/6
            velocity = velocity+step*(k1[1]+2*k2[1]+2*k3[1]+k4[1])/6
            output.append(velocity.copy())
        return np.stack(output)

    def _baseline_force_path(
        self, sample: dict[str, Any], steps: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.backbone is None:
            raise RuntimeError("backbone is required")
        target = sample["training_context"]["targets"]
        t = np.asarray(target["t"][:steps], dtype=float)
        delta = np.asarray(target["delta"][:steps], dtype=float)
        omega = np.asarray(target["omega"][:steps], dtype=float)
        pm = np.asarray(sample["training_context"]["Pm_observed"][:steps], dtype=float)
        static = sample["model_inputs"]["static"]
        emf = self.backbone.backbone.teacher_forced_emf(delta, t, static)
        incidence = np.asarray(static["incidence"], dtype=float)
        memory = self.backbone._first_order_hold_speed(omega @ incidence, t, self.backbone.taus)
        topology = self.backbone.backbone._graph_topology(static)
        mass = np.asarray(static["mass"], dtype=float).reshape(-1)
        force = []
        for index in range(steps):
            _, acceleration, _, _ = self.backbone.vector_field_augmented(
                delta[index], omega[index], emf[index], memory[index], pm[index], static, topology
            )
            force.append(mass*acceleration)
        return t, omega, np.stack(force), incidence

    def prepare_force_cache(
        self,
        samples: list[dict[str, Any]],
        *,
        horizon_steps: int = 1000,
        max_fit_steps: int = 0,
    ) -> list[dict[str, Any]]:
        """Cache candidate-invariant teacher paths before temporal/rank search."""
        cache: list[dict[str, Any]] = []
        for sample in samples:
            target = sample["training_context"]["targets"]
            steps = min(len(target["t"]), int(horizon_steps))
            if max_fit_steps > 0:
                steps = min(steps, int(max_fit_steps))
            if steps < 2:
                continue
            t, omega, baseline_force, incidence = self._baseline_force_path(sample, steps)
            cache.append({
                "event_id": str(sample["event_id"]),
                "t": t, "omega": omega, "baseline_force": baseline_force,
                "incidence": incidence, "static": sample["model_inputs"]["static"],
            })
        if not cache:
            raise ValueError("no candidate-invariant force paths were available")
        return cache

    def build_weak_system(
        self,
        samples: list[dict[str, Any]],
        *,
        windows: tuple[int, ...] = (1, 5, 20),
        horizon_steps: int = 1000,
        max_fit_steps: int = 0,
        max_windows_per_case_horizon: int = 8,
        prepared_force_cache: list[dict[str, Any]] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        if self.backbone is None:
            raise RuntimeError("backbone is required")
        signals, targets = [], []
        blocks = 0
        angular = self.angular_frequency
        cache = prepared_force_cache or self.prepare_force_cache(
            samples, horizon_steps=horizon_steps, max_fit_steps=max_fit_steps
        )
        expected_ids = {str(sample["event_id"]) for sample in samples}
        cache_ids = {str(item["event_id"]) for item in cache}
        if not cache_ids.issubset(expected_ids):
            raise ValueError("prepared force cache contains events outside the fit population")
        for item in cache:
            t = np.asarray(item["t"], dtype=float)
            omega = np.asarray(item["omega"], dtype=float)
            baseline_force = np.asarray(item["baseline_force"], dtype=float)
            incidence = np.asarray(item["incidence"], dtype=float)
            mass = np.asarray(item["static"]["mass"], dtype=float).reshape(-1)
            steps = len(t)
            filtered = self._edge_filter(omega @ incidence, t, angular, self.damping_ratio)
            dt = np.diff(t)
            filtered_prefix = np.concatenate((
                np.zeros((1, filtered.shape[1])),
                np.cumsum(0.5*(filtered[:-1]+filtered[1:])*dt[:, None], axis=0),
            ))
            force_prefix = np.concatenate((
                np.zeros((1, len(mass))),
                np.cumsum(0.5*(baseline_force[:-1]+baseline_force[1:])*dt[:, None], axis=0),
            ))
            for width in sorted({int(value) for value in windows if 0 < int(value) < steps}):
                possible = steps-width
                count = min(possible, max(1, int(max_windows_per_case_horizon)))
                starts = np.unique(np.linspace(0, possible-1, count, dtype=int))
                stops = starts+width
                signal = filtered_prefix[stops]-filtered_prefix[starts]
                actual = mass*(omega[stops]-omega[starts])
                baseline = force_prefix[stops]-force_prefix[starts]
                weight = 1.0/np.sqrt(max(len(starts)*len(mass), 1))
                signals.append(weight*signal)
                targets.append(weight*(actual-baseline))
                blocks += 1
        if not signals:
            raise ValueError("no low-rank modal weak windows were available")
        signal = np.concatenate(signals)
        target = np.concatenate(targets)
        return signal, target, {
            "n_case_horizon_blocks": blocks,
            "n_weak_windows": len(signal),
            "max_windows_per_case_horizon": int(max_windows_per_case_horizon),
        }

    @staticmethod
    def _initial_loading(
        signal: np.ndarray,
        target: np.ndarray,
        incidence: np.ndarray,
        rank: int,
        ridge: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_edges = incidence.shape[1]
        if rank == 0:
            return np.empty((n_edges, 0)), np.zeros((n_edges, n_edges))
        edge_target = -target @ np.linalg.pinv(incidence).T
        gram = signal.T@signal+max(float(ridge), 1e-12)*np.eye(n_edges)
        mapping_t = np.linalg.solve(gram, signal.T@edge_target)
        symmetric = 0.5*(mapping_t+mapping_t.T)
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
        positive = np.maximum(eigenvalues[-rank:], 0.0)
        loading = eigenvectors[:, -rank:]@np.diag(np.sqrt(positive))
        if np.linalg.norm(loading) <= np.finfo(float).eps:
            _, covariance_vectors = np.linalg.eigh(signal.T@signal)
            loading = covariance_vectors[:, -rank:]*1e-3
        return loading, symmetric

    def fit_fixed_rank(
        self,
        train: list[dict[str, Any]],
        *,
        frequency_hz: float,
        damping_ratio: float,
        rank: int,
        ridge: float | None = None,
        windows: tuple[int, ...] = (1, 5, 20),
        horizon_steps: int = 1000,
        max_fit_steps: int = 0,
        max_windows_per_case_horizon: int = 8,
        max_iterations: int = 500,
        prepared_system: tuple[np.ndarray, np.ndarray, dict[str, Any]] | None = None,
        prepared_force_cache: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        self.frequency_hz = float(frequency_hz)
        self.damping_ratio = float(damping_ratio)
        self.rank = int(rank)
        self.ridge = self.ridge if ridge is None else float(ridge)
        if self.rank == 0:
            if self.backbone is None:
                raise RuntimeError("backbone is required")
            n_edges = self.backbone.static_coefficients.shape[1]
            self.loadings = np.empty((n_edges, 0), dtype=float)
            diagnostics = {
                "solver_success": True, "solver_iterations": 0, "rank": 0,
                "active_dynamic_states": 0, "weak_residual_rmse": 0.0,
            }
            self.diagnostics["weak_fit"] = diagnostics
            return diagnostics
        signal, target, details = prepared_system or self.build_weak_system(
            train, windows=windows, horizon_steps=horizon_steps,
            max_fit_steps=max_fit_steps,
            max_windows_per_case_horizon=max_windows_per_case_horizon,
            prepared_force_cache=prepared_force_cache,
        )
        signal = np.asarray(signal, dtype=float)
        target = np.asarray(target, dtype=float)
        if signal.ndim != 2 or target.ndim != 2 or len(signal) != len(target):
            raise ValueError("prepared weak system must contain aligned 2D signal and target arrays")
        incidence = np.asarray(train[0]["model_inputs"]["static"]["incidence"], dtype=float)
        initial, unrestricted = self._initial_loading(
            signal, target, incidence, self.rank, self.ridge
        )
        scale = max(float(np.sqrt(np.mean(target**2))), np.finfo(float).eps)
        normalizer = float(target.size)*scale**2

        def objective(flat: np.ndarray) -> tuple[float, np.ndarray]:
            loading = flat.reshape(incidence.shape[1], self.rank)
            predicted = -(signal@loading)@loading.T@incidence.T
            residual = predicted-target
            loss = 0.5*float(np.sum(residual**2))/normalizer+0.5*self.ridge*float(np.sum(loading**2))
            gradient_h = -(signal.T@residual@incidence)/normalizer
            gradient = (gradient_h+gradient_h.T)@loading+self.ridge*loading
            return loss, gradient.ravel()

        result = minimize(
            objective, initial.ravel(), method="L-BFGS-B", jac=True,
            options={"maxiter": int(max_iterations), "ftol": 1e-12, "gtol": 1e-8, "maxls": 40},
        )
        self.loadings = result.x.reshape(incidence.shape[1], self.rank)
        predicted = -(signal@self.loadings)@self.loadings.T@incidence.T
        singular = np.linalg.svd(signal, compute_uv=False)
        tolerance = np.max(singular)*1e-10 if singular.size else np.inf
        transfer_eigenvalues = np.linalg.eigvalsh(self.loadings@self.loadings.T)
        diagnostics = {
            **details,
            "solver_success": bool(result.success),
            "solver_status": int(result.status),
            "solver_message": str(result.message),
            "solver_iterations": int(result.nit),
            "rank": int(self.rank),
            "active_dynamic_states": int(2*self.rank),
            "effective_signal_rank": int(np.sum(singular > tolerance)),
            "signal_condition_number": float(np.max(singular)/max(np.min(singular), np.finfo(float).eps)),
            "weak_residual_rmse": float(np.sqrt(np.mean((predicted-target)**2))),
            "loading_norm": float(np.linalg.norm(self.loadings)),
            "minimum_transfer_eigenvalue": float(np.min(transfer_eigenvalues)),
            "unrestricted_symmetric_positive_rank": int(np.sum(np.linalg.eigvalsh(unrestricted) > 1e-10)),
            "frequency_hz": self.frequency_hz,
            "damping_ratio": self.damping_ratio,
            "ridge": self.ridge,
        }
        self.diagnostics["weak_fit"] = diagnostics
        return diagnostics

    def fit(
        self,
        train: list[dict[str, Any]],
        val: list[dict[str, Any]],
        run_context: RunContext,
    ) -> FitResult:
        if not train or not val:
            raise ValueError("M0-LRMRP requires nonempty train and validation sets")
        weak = self.fit_fixed_rank(
            train,
            frequency_hz=self.frequency_hz,
            damping_ratio=self.damping_ratio,
            rank=self.rank,
            ridge=self.ridge,
            windows=run_context.integral_windows,
            horizon_steps=run_context.horizon_steps,
            max_fit_steps=run_context.fit_max_steps,
            max_windows_per_case_horizon=run_context.weak_max_windows_per_case_horizon,
        )
        errors = []
        for sample in val[: run_context.max_cases]:
            static, time_grid, exogenous = sample_rollout_inputs(sample)
            target = sample["training_context"]["targets"]
            steps = min(len(time_grid), int(run_context.horizon_steps))
            delta, omega = self.rollout(static, time_grid[:steps], {
                "Pm": exogenous["Pm"][:steps], "interpolation": exogenous["interpolation"]
            })
            error = np.concatenate((
                (delta-np.asarray(target["delta"][:steps])).ravel(),
                (omega-np.asarray(target["omega"][:steps])).ravel(),
            ))
            errors.append(float(np.sqrt(np.mean(error**2))))
        validation_loss = float(np.mean(errors))
        self.diagnostics["validation_mean_rmse"] = validation_loss
        return FitResult(
            method=self.method_name,
            status="fitted",
            train_case_ids=tuple(str(sample["event_id"]) for sample in train),
            val_case_ids=tuple(str(sample["event_id"]) for sample in val),
            best_validation_loss=validation_loss,
            details={"weak_fit": weak, **self.diagnostics},
        )

    def save(self, path: str | Path) -> None:
        backbone, loadings = self._require_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        backbone_path = path.with_name(path.stem+"_backbone.npz")
        backbone.save(backbone_path)
        np.savez_compressed(
            path, loadings=loadings, frequency_hz=self.frequency_hz,
            damping_ratio=self.damping_ratio, rank=self.rank, ridge=self.ridge,
            max_internal_step=self.max_internal_step, backbone_file=backbone_path.name,
        )
        path.with_suffix(".json").write_text(
            json.dumps({"method": self.method_name, "diagnostics": self.diagnostics}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> LowRankModalPortM0Model:
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            backbone = PortStructuredDiscoveryM0Model.load(path.with_name(str(data["backbone_file"])))
            model = cls(
                backbone=backbone, loadings=data["loadings"],
                frequency_hz=float(data["frequency_hz"]),
                damping_ratio=float(data["damping_ratio"]), rank=int(data["rank"]),
                ridge=float(data["ridge"]), max_internal_step=float(data["max_internal_step"]),
            )
        metadata = path.with_suffix(".json")
        if metadata.exists():
            model.diagnostics = json.loads(metadata.read_text(encoding="utf-8")).get("diagnostics", {})
        return model

    def convergence_audit(
        self,
        initial_state: dict[str, np.ndarray],
        time_grid: np.ndarray,
        exogenous: dict[str, np.ndarray],
    ) -> dict[str, float | bool]:
        trajectories = [
            self.rollout_augmented(initial_state, time_grid, exogenous, max_step=self.max_internal_step/divisor)
            for divisor in (1, 2, 4)
        ]
        errors = []
        for coarse, fine in pairwise(trajectories):
            component = [float(np.mean((a-b)**2)) for a, b in zip(coarse, fine, strict=True) if a.size]
            errors.append(float(np.sqrt(sum(component))))
        return {
            "dt_vs_dt2": errors[0], "dt2_vs_dt4": errors[1],
            "error_ratio": errors[0]/max(errors[1], np.finfo(float).eps),
            "convergent": bool(errors[1] <= errors[0]+1e-12),
        }

    def physics_audit(self) -> dict[str, Any]:
        try:
            backbone, loadings = self._require_fitted()
        except (RuntimeError, ValueError):
            return {"status": "not_fitted"}
        transfer = loadings@loadings.T
        minimum = float(np.min(np.linalg.eigvalsh(transfer))) if transfer.size else 0.0
        passed = bool(
            backbone.physics_audit().get("status") == "pass"
            and np.isfinite(loadings).all() and minimum >= -1e-10
            and self.frequency_hz > 0 and 0 < self.damping_ratio < 1
        )
        return {
            "status": "pass" if passed else "fail",
            "reciprocal_low_rank_port": passed,
            "rank": int(self.rank),
            "active_dynamic_states": int(2*self.rank),
            "finite_signed_loadings": bool(np.isfinite(loadings).all()),
            "minimum_transfer_eigenvalue": minimum,
            "positive_frequency": bool(self.frequency_hz > 0),
            "positive_damping_ratio": bool(self.damping_ratio > 0),
            "supply_identity": "Hdot=(L@p)^T*edge_speed-2*zeta*omega0*p^T*p",
        }
