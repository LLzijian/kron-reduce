from __future__ import annotations

import numpy as np


def numeric_physics_audit(model, sample: dict, tolerance: float = 1e-8) -> list[dict]:
    static = sample["model_inputs"]["static"]
    equilibrium = np.asarray(static["equilibrium_delta"])
    zeros = np.zeros_like(equilibrium)
    pm_zero = np.zeros_like(equilibrium)
    _, acceleration = model.vector_field(equilibrium, zeros, pm_zero, static)
    incidence = np.asarray(static["incidence"])
    balance = float(np.max(np.abs(np.sum(incidence, axis=0))))
    return [
        {"check": "equilibrium_residual_norm", "value": float(np.linalg.norm(acceleration)), "tolerance": tolerance, "status": "pass" if np.linalg.norm(acceleration) <= tolerance else "fail"},
        {"check": "incidence_internal_force_balance", "value": balance, "tolerance": tolerance, "status": "pass" if balance <= tolerance else "fail"},
    ]
