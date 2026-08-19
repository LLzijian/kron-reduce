from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import lsq_linear

from kron_reduce.physics.coordinates import coi_project, coi_project_power

from .base import DynamicsModel, FitResult, RunContext, sample_rollout_inputs
from .port_structured_discovery import PortStructuredDiscoveryM0Model


class ModalResonantPortM0Model(DynamicsModel):
    """Add passive complex-pole edge states to a fitted port model.

    For every registered modal pair ``(f, zeta)`` and generator edge, the
    additional state obeys

    ``q_dot = p``
    ``p_dot = nu - omega0**2*q - 2*zeta*omega0*p``
    ``r = h*p``, with ``h >= 0`` and ``nu = B.T@omega``.

    The swing balance receives ``-B@r``.  Consequently the storage

    ``H_r = sum h/2 * (p**2 + omega0**2*q**2)``

    satisfies ``H_r_dot = sum h*p*nu - 2*h*zeta*omega0*p**2``.  The port term
    cancels the work of ``-B@r`` exactly, so the new state can supply phase and
    a complex-conjugate pole without becoming an unstructured energy source.
    For fixed ``f`` and ``zeta``, only ``h`` is learned and remains linear in
    the integral weak form.
    """

    method_name = "M0-MRPSD"
    DEFAULT_MODE_CANDIDATES = (
        (),
        ((0.4, 0.05),),
        ((0.7, 0.05),),
        ((1.0, 0.05),),
        ((1.3, 0.05),),
        ((1.6, 0.05),),
        ((1.0, 0.02),),
        ((1.0, 0.10),),
        ((1.0, 0.20),),
    )

    def __init__(
        self,
        backbone: PortStructuredDiscoveryM0Model | None = None,
        *,
        mode_pairs: tuple[tuple[float, float], ...] = (),
        coefficients: np.ndarray | None = None,
        mode_candidates: tuple[tuple[tuple[float, float], ...], ...] | None = None,
        ridge_values: tuple[float, ...] = (1e-8, 1e-6, 1e-4),
        max_internal_step: float = 0.01,
        diagnostics: dict[str, Any] | None = None,
    ) -> None:
        self.backbone = backbone
        self.mode_pairs = self._validate_mode_pairs(mode_pairs)
        self.coefficients = (
            None if coefficients is None else np.asarray(coefficients, dtype=float)
        )
        self.mode_candidates = tuple(
            self._validate_mode_pairs(candidate)
            for candidate in (mode_candidates or self.DEFAULT_MODE_CANDIDATES)
        )
        if () not in self.mode_candidates:
            raise ValueError("mode candidates must include the exact backbone fallback")
        self.ridge_values = tuple(float(value) for value in ridge_values)
        if any(not np.isfinite(value) or value < 0 for value in self.ridge_values):
            raise ValueError("ridge values must be finite and nonnegative")
        self.max_internal_step = float(max_internal_step)
        if not np.isfinite(self.max_internal_step) or self.max_internal_step <= 0:
            raise ValueError("max_internal_step must be finite and positive")
        self.diagnostics = diagnostics or {}

    @staticmethod
    def _validate_mode_pairs(
        pairs: tuple[tuple[float, float], ...] | list[tuple[float, float]],
    ) -> tuple[tuple[float, float], ...]:
        normalized = tuple((float(frequency), float(damping)) for frequency, damping in pairs)
        for frequency, damping in normalized:
            if not np.isfinite(frequency) or frequency <= 0:
                raise ValueError("resonance frequencies must be finite and positive")
            if not np.isfinite(damping) or damping <= 0 or damping >= 1:
                raise ValueError("resonance damping ratios must lie in (0, 1)")
        if len(set(normalized)) != len(normalized):
            raise ValueError("resonance mode pairs must be unique")
        return normalized

    def _require_fitted(
        self,
    ) -> tuple[PortStructuredDiscoveryM0Model, np.ndarray]:
        if self.backbone is None:
            raise RuntimeError("M0-MRPSD requires a fitted port-structured backbone")
        self.backbone._require_fitted()
        if self.coefficients is None:
            raise RuntimeError("M0-MRPSD resonance coefficients are not fitted")
        coefficient = np.asarray(self.coefficients, dtype=float)
        n_edges = self.backbone.static_coefficients.shape[1]
        if coefficient.shape != (len(self.mode_pairs), n_edges):
            raise ValueError("resonance coefficients must have shape (mode, edge)")
        return self.backbone, coefficient

    @property
    def angular_frequencies(self) -> np.ndarray:
        return 2.0 * np.pi * np.asarray([pair[0] for pair in self.mode_pairs])

    @property
    def damping_ratios(self) -> np.ndarray:
        return np.asarray([pair[1] for pair in self.mode_pairs], dtype=float)

    @staticmethod
    def _edge_speed(
        omega: np.ndarray, static: dict[str, np.ndarray]
    ) -> np.ndarray:
        incidence = np.asarray(static["incidence"], dtype=float)
        return incidence.T @ np.asarray(omega, dtype=float).reshape(-1)

    def initial_resonance(
        self, omega: np.ndarray, static: dict[str, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        speed = self._edge_speed(omega, static)
        angular = self.angular_frequencies
        position = (
            speed[None, :] / angular[:, None] ** 2
            if len(angular)
            else np.empty((0, len(speed)), dtype=float)
        )
        velocity = np.zeros_like(position)
        return position, velocity

    def _resonance_derivative(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        edge_speed: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        angular = self.angular_frequencies[:, None]
        damping = self.damping_ratios[:, None]
        return velocity, (
            edge_speed[None, :]
            - angular**2 * position
            - 2.0 * damping * angular * velocity
        )

    def vector_field_augmented(
        self,
        delta: np.ndarray,
        omega: np.ndarray,
        emf: np.ndarray,
        memory: np.ndarray,
        resonance_position: np.ndarray,
        resonance_velocity: np.ndarray,
        pm: np.ndarray,
        static: dict[str, np.ndarray],
        topology: dict[str, np.ndarray | float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        backbone, coefficients = self._require_fitted()
        dd, dw, dq, dz = backbone.vector_field_augmented(
            delta, omega, emf, memory, pm, static, topology
        )
        if resonance_position.shape != coefficients.shape or resonance_velocity.shape != coefficients.shape:
            raise ValueError("resonance state does not match learned mode-edge support")
        incidence = np.asarray(static["incidence"], dtype=float)
        mass = np.asarray(static["mass"], dtype=float).reshape(-1)
        edge_flow = np.sum(coefficients * resonance_velocity, axis=0)
        correction = coi_project_power(-incidence @ edge_flow, mass)
        edge_speed = incidence.T @ np.asarray(omega, dtype=float).reshape(-1)
        drq, drp = self._resonance_derivative(
            resonance_position, resonance_velocity, edge_speed
        )
        return dd, dw + correction / mass, dq, dz, drq, drp

    @staticmethod
    def _input_mode(pm: np.ndarray, requested: str) -> str:
        if requested not in {"auto", "zoh", "linear"}:
            raise ValueError("Pm interpolation must be auto, zoh, or linear")
        if requested != "auto":
            return requested
        return "zoh" if np.allclose(pm, pm[[0]], rtol=0.0, atol=1e-14) else "linear"

    def rollout_augmented(
        self,
        initial_state: dict[str, np.ndarray],
        time_grid: np.ndarray,
        exogenous: dict[str, np.ndarray],
        *,
        max_step: float | None = None,
    ) -> tuple[np.ndarray, ...]:
        backbone, coefficients = self._require_fitted()
        t = np.asarray(time_grid, dtype=float)
        pm = np.asarray(exogenous["Pm"], dtype=float)
        if len(t) != len(pm) or len(t) < 1:
            raise ValueError("time and Pm schedules must have the same positive length")
        step_limit = self.max_internal_step if max_step is None else float(max_step)
        if step_limit <= 0:
            raise ValueError("maximum integration step must be positive")
        input_mode = self._input_mode(pm, exogenous.get("interpolation", "auto"))
        mass = np.asarray(initial_state["mass"], dtype=float).reshape(-1)
        delta = coi_project(
            np.asarray(initial_state["delta0"], dtype=float).reshape(-1), mass
        )
        omega = coi_project(
            np.asarray(initial_state["omega0"], dtype=float).reshape(-1), mass
        )
        emf = backbone.backbone.initial_emf(delta, initial_state)
        memory = backbone.initial_memory(delta, omega, initial_state)
        position, velocity = self.initial_resonance(omega, initial_state)
        if position.shape != coefficients.shape:
            raise ValueError("selected resonance edge count does not match rollout graph")
        topology = backbone.backbone._graph_topology(initial_state)
        outputs = tuple(
            [value.copy()]
            for value in (delta, omega, emf, memory, position, velocity)
        )

        def input_at(index: int, fraction: float) -> np.ndarray:
            if input_mode == "zoh":
                return pm[index]
            return pm[index] + fraction * (pm[index + 1] - pm[index])

        def field(*state: np.ndarray, input_value: np.ndarray) -> tuple[np.ndarray, ...]:
            return self.vector_field_augmented(
                *state, input_value, initial_state, topology
            )

        state = (delta, omega, emf, memory, position, velocity)
        for index in range(len(t) - 1):
            interval = float(t[index + 1] - t[index])
            if interval <= 0:
                raise ValueError("rollout time must be strictly increasing")
            substeps = max(1, int(np.ceil(interval / step_limit)))
            step = interval / substeps
            for substep in range(substeps):
                f0 = substep / substeps
                fh = (substep + 0.5) / substeps
                f1 = (substep + 1.0) / substeps
                k1 = field(*state, input_value=input_at(index, f0))
                k2_state = tuple(value + 0.5 * step * slope for value, slope in zip(state, k1, strict=True))
                k2 = field(*k2_state, input_value=input_at(index, fh))
                k3_state = tuple(value + 0.5 * step * slope for value, slope in zip(state, k2, strict=True))
                k3 = field(*k3_state, input_value=input_at(index, fh))
                k4_state = tuple(value + step * slope for value, slope in zip(state, k3, strict=True))
                k4 = field(*k4_state, input_value=input_at(index, f1))
                state = tuple(
                    value + step * (a + 2.0 * b + 2.0 * c + d) / 6.0
                    for value, a, b, c, d in zip(state, k1, k2, k3, k4, strict=True)
                )
                state = (
                    coi_project(state[0], mass),
                    coi_project(state[1], mass),
                    *state[2:],
                )
                if not all(np.isfinite(value).all() for value in state):
                    raise FloatingPointError("nonfinite M0-MRPSD state")
                emf = state[2]
                transient = backbone.backbone
                if np.min(emf) <= transient.q_floor or np.max(emf) >= transient.q_ceiling:
                    raise FloatingPointError("M0-MRPSD transient EMF left audited domain")
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

    def teacher_forced_resonance(
        self,
        omega_path: np.ndarray,
        time_grid: np.ndarray,
        static: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        omega_path = np.asarray(omega_path, dtype=float)
        t = np.asarray(time_grid, dtype=float)
        incidence = np.asarray(static["incidence"], dtype=float)
        speed = omega_path @ incidence
        position, velocity = self.initial_resonance(omega_path[0], static)
        positions, velocities = [position.copy()], [velocity.copy()]
        for index, step in enumerate(np.diff(t)):
            if step <= 0:
                raise ValueError("teacher-forced time must be strictly increasing")
            speed0, speed1 = speed[index], speed[index + 1]
            speedh = 0.5 * (speed0 + speed1)
            k1 = self._resonance_derivative(position, velocity, speed0)
            k2 = self._resonance_derivative(
                position + 0.5 * step * k1[0],
                velocity + 0.5 * step * k1[1],
                speedh,
            )
            k3 = self._resonance_derivative(
                position + 0.5 * step * k2[0],
                velocity + 0.5 * step * k2[1],
                speedh,
            )
            k4 = self._resonance_derivative(
                position + step * k3[0], velocity + step * k3[1], speed1
            )
            position = position + step * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0]) / 6.0
            velocity = velocity + step * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1]) / 6.0
            positions.append(position.copy())
            velocities.append(velocity.copy())
        return np.stack(positions), np.stack(velocities)

    def _baseline_force_path(
        self, sample: dict[str, Any], steps: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        backbone = self.backbone
        if backbone is None:
            raise RuntimeError("backbone is required")
        target = sample["training_context"]["targets"]
        t = np.asarray(target["t"][:steps], dtype=float)
        delta = np.asarray(target["delta"][:steps], dtype=float)
        omega = np.asarray(target["omega"][:steps], dtype=float)
        pm = np.asarray(sample["training_context"]["Pm_observed"][:steps], dtype=float)
        static = sample["model_inputs"]["static"]
        emf = backbone.backbone.teacher_forced_emf(delta, t, static)
        incidence = np.asarray(static["incidence"], dtype=float)
        speed = omega @ incidence
        memory = backbone._first_order_hold_speed(speed, t, backbone.taus)
        topology = backbone.backbone._graph_topology(static)
        mass = np.asarray(static["mass"], dtype=float).reshape(-1)
        force = []
        for index in range(steps):
            _, acceleration, _, _ = backbone.vector_field_augmented(
                delta[index], omega[index], emf[index], memory[index], pm[index], static, topology
            )
            force.append(mass * acceleration)
        return t, omega, np.stack(force), incidence

    def fit_fixed_modes(
        self,
        train: list[dict[str, Any]],
        mode_pairs: tuple[tuple[float, float], ...],
        *,
        ridge: float = 1e-6,
        windows: tuple[int, ...] = (1, 5, 20),
        horizon_steps: int = 1000,
        max_fit_steps: int = 0,
    ) -> dict[str, Any]:
        if self.backbone is None:
            raise RuntimeError("backbone must be fitted before resonance discovery")
        self.mode_pairs = self._validate_mode_pairs(mode_pairs)
        n_edges = self.backbone.static_coefficients.shape[1]
        n_parameters = len(self.mode_pairs) * n_edges
        if n_parameters == 0:
            self.coefficients = np.empty((0, n_edges), dtype=float)
            return {
                "solver_success": True,
                "n_parameters": 0,
                "effective_rank": 0,
                "matrix_condition_number": 1.0,
                "weak_residual_rmse": 0.0,
            }
        rows: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        blocks = 0
        for sample in train:
            target = sample["training_context"]["targets"]
            steps = min(len(target["t"]), int(horizon_steps))
            if max_fit_steps > 0:
                steps = min(steps, int(max_fit_steps))
            if steps < 2:
                continue
            t, omega, baseline_force, incidence = self._baseline_force_path(sample, steps)
            static = sample["model_inputs"]["static"]
            mass = np.asarray(static["mass"], dtype=float).reshape(-1)
            _, velocity = self.teacher_forced_resonance(omega, t, static)
            design_path = -np.einsum(
                "ne,tme->tnme", incidence, velocity, optimize=True
            ).reshape(steps, len(mass), n_parameters)
            dt = np.diff(t)
            design_prefix = np.concatenate((
                np.zeros((1, len(mass), n_parameters)),
                np.cumsum(0.5*(design_path[:-1]+design_path[1:])*dt[:, None, None], axis=0),
            ))
            force_prefix = np.concatenate((
                np.zeros((1, len(mass))),
                np.cumsum(0.5*(baseline_force[:-1]+baseline_force[1:])*dt[:, None], axis=0),
            ))
            for width in sorted({int(value) for value in windows if int(value) > 0}):
                if width >= steps:
                    continue
                starts = np.arange(steps-width)
                stops = starts+width
                matrix = (design_prefix[stops]-design_prefix[starts]).reshape(-1, n_parameters)
                actual = mass*(omega[stops]-omega[starts])
                baseline = force_prefix[stops]-force_prefix[starts]
                vector = (actual-baseline).reshape(-1)
                weight = 1.0/np.sqrt(max(len(vector), 1))
                rows.append(weight*matrix)
                targets.append(weight*vector)
                blocks += 1
        if not rows:
            raise ValueError("no resonance weak-form windows were available")
        matrix = np.concatenate(rows)
        vector = np.concatenate(targets)
        scale = np.sqrt(np.mean(matrix**2, axis=0))
        informative = scale > max(np.max(scale)*1e-12, np.finfo(float).eps)
        scale = np.where(informative, scale, 1.0)
        normalized = matrix/scale
        augmented = np.vstack((normalized, np.sqrt(float(ridge))*np.eye(n_parameters)))
        augmented_y = np.concatenate((vector, np.zeros(n_parameters)))
        solve = lsq_linear(augmented, augmented_y, bounds=(0.0, np.inf), max_iter=2000)
        theta = solve.x/scale
        theta[~informative] = 0.0
        self.coefficients = theta.reshape(len(self.mode_pairs), n_edges)
        residual = matrix@theta-vector
        singular = np.linalg.svd(normalized[:, informative], compute_uv=False)
        tolerance = np.max(singular)*1e-10 if len(singular) else np.inf
        diagnostics = {
            "solver_success": bool(solve.success),
            "solver_iterations": int(solve.nit),
            "n_parameters": int(n_parameters),
            "n_active_parameters": int(np.sum(theta > np.max(theta, initial=0.0)*1e-6)),
            "effective_rank": int(np.sum(singular > tolerance)),
            "matrix_condition_number": float(np.max(singular)/max(np.min(singular), np.finfo(float).eps)),
            "weak_residual_rmse": float(np.sqrt(np.mean(residual**2))),
            "n_case_horizon_blocks": int(blocks),
            "mode_pairs": [list(pair) for pair in self.mode_pairs],
            "ridge": float(ridge),
        }
        self.diagnostics["weak_fit"] = diagnostics
        return diagnostics

    def _validation_summary(
        self, samples: list[dict[str, Any]], horizon_steps: int
    ) -> dict[str, float]:
        errors, drifts = [], []
        for sample in samples:
            static, t, exogenous = sample_rollout_inputs(sample)
            target = sample["training_context"]["targets"]
            steps = min(len(t), int(horizon_steps))
            delta, omega = self.rollout(
                static,
                t[:steps],
                {"Pm": exogenous["Pm"][:steps], "interpolation": exogenous["interpolation"]},
            )
            error = np.concatenate((
                (delta-np.asarray(target["delta"][:steps])).ravel(),
                (omega-np.asarray(target["omega"][:steps])).ravel(),
            ))
            errors.append(float(np.sqrt(np.mean(error**2))))
            mass = np.asarray(static["mass"], dtype=float)
            drifts.append(float(max(np.max(np.abs(delta@mass)), np.max(np.abs(omega@mass)))))
        return {
            "mean_rmse": float(np.mean(errors)),
            "median_rmse": float(np.median(errors)),
            "maximum_coi_drift": float(np.max(drifts)),
        }

    def fit(
        self, train: list[dict[str, Any]], val: list[dict[str, Any]], run_context: RunContext
    ) -> FitResult:
        if not train or not val:
            raise ValueError("M0-MRPSD requires nonempty train and validation sets")
        candidates: list[dict[str, Any]] = []
        best: tuple[float, tuple[tuple[float, float], ...], float, np.ndarray, dict[str, Any]] | None = None
        for pairs in self.mode_candidates:
            for ridge in self.ridge_values:
                weak = self.fit_fixed_modes(
                    train,
                    pairs,
                    ridge=ridge,
                    windows=run_context.integral_windows,
                    horizon_steps=run_context.horizon_steps,
                    max_fit_steps=run_context.fit_max_steps,
                )
                validation = self._validation_summary(val[:run_context.max_cases], run_context.horizon_steps)
                score = validation["mean_rmse"] + 1e-6*int(weak["n_active_parameters"])
                row = {"mode_pairs": [list(pair) for pair in pairs], "ridge": ridge, **weak, **validation, "score": score}
                candidates.append(row)
                if best is None or score < best[0]:
                    best = (score, pairs, ridge, self.coefficients.copy(), row)
        assert best is not None
        _, self.mode_pairs, ridge, self.coefficients, selected = best
        self.diagnostics.update({
            "selection_policy": "validation_rollout_then_complexity; modal gate belongs to workflow",
            "selected_ridge": ridge,
            "selected": selected,
            "candidates": candidates,
        })
        return FitResult(
            method=self.method_name,
            status="fitted",
            train_case_ids=tuple(str(sample["event_id"]) for sample in train),
            val_case_ids=tuple(str(sample["event_id"]) for sample in val),
            best_validation_loss=float(best[0]),
            details=self.diagnostics,
        )

    def save(self, path: str | Path) -> None:
        backbone, coefficients = self._require_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        backbone_path = path.with_name(path.stem+"_backbone.npz")
        backbone.save(backbone_path)
        np.savez_compressed(
            path,
            mode_pairs=np.asarray(self.mode_pairs, dtype=float).reshape(-1, 2),
            coefficients=coefficients,
            max_internal_step=self.max_internal_step,
            backbone_file=backbone_path.name,
        )
        path.with_suffix(".json").write_text(
            json.dumps({"method": self.method_name, "diagnostics": self.diagnostics}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "ModalResonantPortM0Model":
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            backbone = PortStructuredDiscoveryM0Model.load(path.with_name(str(data["backbone_file"])))
            model = cls(
                backbone=backbone,
                mode_pairs=tuple(tuple(map(float, pair)) for pair in data["mode_pairs"]),
                coefficients=data["coefficients"],
                max_internal_step=float(data["max_internal_step"]),
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
            self.rollout_augmented(
                initial_state, time_grid, exogenous,
                max_step=self.max_internal_step/divisor,
            )
            for divisor in (1, 2, 4)
        ]
        errors = []
        for coarse, fine in zip(trajectories[:-1], trajectories[1:], strict=True):
            errors.append(float(np.sqrt(sum(np.mean((left-right)**2) for left, right in zip(coarse, fine, strict=True)))))
        return {
            "dt_vs_dt2": errors[0],
            "dt2_vs_dt4": errors[1],
            "error_ratio": errors[0]/max(errors[1], np.finfo(float).eps),
            "convergent": bool(errors[1] <= errors[0]+1e-12),
        }

    def physics_audit(self) -> dict[str, Any]:
        try:
            backbone, coefficients = self._require_fitted()
        except (RuntimeError, ValueError):
            return {"status": "not_fitted"}
        angular = self.angular_frequencies[:, None]
        damping = self.damping_ratios[:, None]
        # Algebraic certificate for the registered port interconnection.  A
        # numerical chain-rule test is added by workflow audits on sampled
        # states and is intentionally not replaced by this exact identity.
        dissipation = 2.0*coefficients*damping*angular
        passed = bool(
            backbone.physics_audit().get("status") == "pass"
            and np.isfinite(coefficients).all()
            and np.all(coefficients >= 0)
            and np.isfinite(dissipation).all()
            and np.all(dissipation >= 0)
        )
        return {
            "status": "pass" if passed else "fail",
            "passive_complex_pole_module": passed,
            "nonnegative_coefficients": bool(np.all(coefficients >= 0)),
            "positive_frequencies": bool(np.all(self.angular_frequencies > 0)),
            "positive_damping_ratios": bool(np.all(self.damping_ratios > 0)),
            "minimum_resonance_dissipation": float(np.min(dissipation)) if dissipation.size else 0.0,
            "supply_identity": "Hdot=edge_velocity*edge_speed-2*h*zeta*omega0*edge_velocity^2",
        }
