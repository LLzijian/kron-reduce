from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear

from kron_reduce.physics.coordinates import coi_project, coi_project_power

from .base import DynamicsModel, FitResult, RunContext
from .m0 import M0Model, _pm_equilibrium, _trapz


class FirstOrderGovernor:
    """Causal per-node governor identified by one constrained weak solve."""

    def __init__(self, tau=None, gain=None):
        self.tau = None if tau is None else np.asarray(tau, dtype=np.float64)
        self.gain = None if gain is None else np.asarray(gain, dtype=np.float64)
        self.diagnostics = {}

    def fit(self, train, context: RunContext):
        n = train[0]["model_inputs"]["static"]["mass"].size
        matrices, vectors = [[] for _ in range(n)], [[] for _ in range(n)]
        for sample in train[: context.max_cases]:
            target = sample["training_context"]["targets"]
            t = np.asarray(target["t"])
            omega = np.asarray(target["omega"])
            dpm = np.asarray(sample["training_context"]["Pm_observed"]) - _pm_equilibrium(sample["model_inputs"]["static"])
            limit = min(len(t), max(context.horizon_steps * 5, max(context.integral_windows) + 1))
            for width in context.integral_windows:
                if width <= 0 or width >= limit:
                    continue
                for start in range(limit - width):
                    stop = start + width; segment = slice(start, stop + 1)
                    ipm, iw = _trapz(dpm[segment], t[segment]), _trapz(omega[segment], t[segment])
                    change = dpm[stop] - dpm[start]
                    for node in range(n):
                        matrices[node].append([-ipm[node], -iw[node]])
                        vectors[node].append(change[node])
        alpha, beta, residuals, iterations = np.empty(n), np.empty(n), [], 0
        for node in range(n):
            a, y = np.asarray(matrices[node]), np.asarray(vectors[node])
            augmented = np.vstack((a, 1e-3 * np.eye(2)))
            result = lsq_linear(augmented, np.concatenate((y, np.zeros(2))), bounds=(1e-8, np.inf), max_iter=1000)
            alpha[node], beta[node] = result.x
            residuals.extend((a @ result.x - y).tolist()); iterations += int(result.nit)
        self.tau = 1.0 / alpha
        self.gain = beta / alpha
        self.diagnostics = {
            "optimizer": "single_constrained_integral_least_squares",
            "solver_iterations": iterations,
            "weak_residual_rmse": float(np.sqrt(np.mean(np.asarray(residuals) ** 2))),
        }
        return self

    def derivative(self, delta_pm, omega):
        if self.tau is None or self.gain is None:
            raise RuntimeError("governor is not fitted")
        return (-delta_pm - self.gain * omega) / self.tau


class GovernorM0Model(DynamicsModel):
    method_name = "M0-governor"

    def __init__(self, base=None, governor=None):
        self.base = base or M0Model()
        self.governor = governor or FirstOrderGovernor()
        self.validation_improvement = float("nan")

    def fit(self, train, val, run_context):
        if self.base.damping is None:
            self.base.fit(train, val, run_context)
        self.governor.fit(train, run_context)
        baseline = self.base.validation_loss(val, run_context)
        losses = []
        for sample in val[: run_context.max_cases]:
            target = sample["training_context"]["targets"]
            steps = min(len(target["t"]), run_context.horizon_steps)
            d, w = self.rollout(sample["model_inputs"]["static"], target["t"][:steps], {})
            inc = np.asarray(sample["model_inputs"]["static"]["incidence"])
            losses.append(np.mean((((d-target["delta"][:steps])@inc)/self.base.validation_scales["edge_angle"])**2) + np.mean(((w-target["omega"][:steps])/self.base.validation_scales["omega"])**2))
        loss = float(np.mean(losses))
        self.validation_improvement = (baseline-loss)/max(baseline, np.finfo(float).eps)
        return FitResult(self.method_name, "completed", tuple(s["event_id"] for s in train[:run_context.max_cases]), tuple(s["event_id"] for s in val[:run_context.max_cases]), loss, {**self.governor.diagnostics, "relative_validation_improvement": self.validation_improvement, "promotion_requires_cross_fold_gate": True})

    def rollout(self, initial_state, time_grid, exogenous):
        t = np.asarray(time_grid, dtype=np.float64)
        mass = np.asarray(initial_state["mass"], dtype=np.float64)
        eqpm = _pm_equilibrium(initial_state)
        d = coi_project(np.asarray(initial_state["delta0"]).reshape(-1), mass)
        w = coi_project(np.asarray(initial_state["omega0"]).reshape(-1), mass)
        p = np.asarray(initial_state["Pm_clear"]).reshape(-1) - eqpm
        p = coi_project_power(p, mass)
        out_d, out_w = [d.copy()], [w.copy()]

        def field(di, wi, pi):
            absolute_pm = pi + eqpm
            return wi, self.base.force(di, wi, absolute_pm, initial_state)/mass, self.governor.derivative(pi, wi)

        for index in range(len(t)-1):
            interval = float(t[index+1]-t[index])
            if interval <= 0: raise ValueError("rollout time must be strictly increasing")
            substeps = max(1, int(np.ceil(interval/self.base.max_internal_step))); h=interval/substeps
            for _ in range(substeps):
                k1=field(d,w,p); k2=field(d+.5*h*k1[0],w+.5*h*k1[1],p+.5*h*k1[2]); k3=field(d+.5*h*k2[0],w+.5*h*k2[1],p+.5*h*k2[2]); k4=field(d+h*k3[0],w+h*k3[1],p+h*k3[2])
                d=coi_project(d+h*(k1[0]+2*k2[0]+2*k3[0]+k4[0])/6,mass)
                w=coi_project(w+h*(k1[1]+2*k2[1]+2*k3[1]+k4[1])/6,mass)
                p=coi_project_power(p+h*(k1[2]+2*k2[2]+2*k3[2]+k4[2])/6,mass)
            if not np.isfinite(d).all() or not np.isfinite(w).all(): raise FloatingPointError(f"nonfinite governor rollout at {index+1}")
            out_d.append(d.copy()); out_w.append(w.copy())
        return np.stack(out_d), np.stack(out_w)

    def save(self, path):
        path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,damping=self.base.damping,coupling=self.base.coupling,tau=self.governor.tau,gain=self.governor.gain,max_internal_step=self.base.max_internal_step)
        path.with_suffix(".json").write_text(json.dumps({"method":self.method_name,"relative_validation_improvement":self.validation_improvement,"diagnostics":self.governor.diagnostics},indent=2),encoding="utf-8")

    @classmethod
    def load(cls,path):
        with np.load(path) as data:
            return cls(M0Model(data["damping"],data["coupling"],max_internal_step=float(data["max_internal_step"])),FirstOrderGovernor(data["tau"],data["gain"]))

    def physics_audit(self):
        if self.governor.tau is None:return {"status":"not_fitted"}
        return {"status":"pass","causal_inputs_only":True,"minimum_tau":float(np.min(self.governor.tau)),"minimum_gain":float(np.min(self.governor.gain)),"promotion_requires_cross_fold_gate":True}
