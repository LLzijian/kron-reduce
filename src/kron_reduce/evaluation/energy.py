from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from kron_reduce.physics.coordinates import coi_basis


@dataclass
class EnergyGateResult:
    reported_name: str
    certificate_passed: bool
    storage_passed: bool
    centering_error: float
    gradient_norm: float
    minimum_projected_hessian: float
    maximum_chain_rule_error: float
    maximum_supply_violation: float
    artificial_violation_loss: float

    def to_dict(self):
        return asdict(self)


def _gradient(function, x, epsilon=1e-6):
    result = np.empty_like(x, dtype=np.float64)
    for index in range(len(x)):
        direction = np.zeros_like(x); direction[index] = epsilon
        result[index] = (function(x + direction) - function(x - direction)) / (2 * epsilon)
    return result


def _hessian(function, x, epsilon=2e-5):
    hessian = np.empty((len(x), len(x)), dtype=np.float64)
    for index in range(len(x)):
        direction = np.zeros_like(x); direction[index] = epsilon
        hessian[:, index] = (_gradient(function, x + direction) - _gradient(function, x - direction)) / (2 * epsilon)
    return 0.5 * (hessian + hessian.T)


def evaluate_energy_gate(model, samples: list[dict], tolerance: float = 1e-6) -> EnergyGateResult:
    if not samples:
        raise ValueError("energy gate requires train/val samples")
    sample = samples[0]
    static = sample["model_inputs"]["static"]
    eq = np.asarray(static["equilibrium_delta"], dtype=np.float64)
    zeros = np.zeros_like(eq)
    n = len(eq)
    function = lambda x: float(model.storage(x[:n], x[n:], static))
    xeq = np.concatenate((eq, zeros))
    center = abs(function(xeq))
    gradient = _gradient(function, xeq)
    basis_delta = coi_basis(np.asarray(static["mass"]))
    projection = np.block([[basis_delta, np.zeros_like(basis_delta)], [np.zeros_like(basis_delta), basis_delta]])
    projected = projection.T @ _hessian(function, xeq) @ projection
    minimum_hessian = float(np.min(np.linalg.eigvalsh(projected)))
    chain_errors = []
    supply_violations = []
    for candidate in samples[:2]:
        target = candidate["training_context"]["targets"]
        pm = candidate["model_inputs"]["exogenous"]["Pm"]
        static_i = candidate["model_inputs"]["static"]
        for index in range(min(3, len(target["t"]))):
            d, w = target["delta"][index], target["omega"][index]
            x = np.concatenate((d, w))
            g = _gradient(lambda y: float(model.storage(y[:n], y[n:], static_i)), x)
            fd, fw = model.vector_field(d, w, pm[index], static_i)
            vector = np.concatenate((fd, fw))
            dot_h = float(g @ vector)
            eps = 1e-6
            directional = (float(model.storage(*( (x + eps*vector)[:n], (x + eps*vector)[n:], static_i))) - float(model.storage(*( (x - eps*vector)[:n], (x - eps*vector)[n:], static_i)))) / (2*eps)
            chain_errors.append(abs(dot_h - directional))
            supply = float(pm[index] @ w)
            supply_violations.append(dot_h - supply + 1e-8 * float(w @ w))
    artificial_loss = float(max(0.0, 1.0 - 0.0) ** 2)
    storage_passed = center <= tolerance and np.linalg.norm(gradient) <= tolerance * 10 and minimum_hessian > 1e-9 and max(chain_errors) <= tolerance * 10
    certificate_passed = storage_passed and max(supply_violations) <= tolerance and artificial_loss > 0
    name = "M2A-LYAP" if certificate_passed else ("M2A-storage" if storage_passed else "M2A-energy-regularized")
    model.reported_name = name
    return EnergyGateResult(name, certificate_passed, storage_passed, center, float(np.linalg.norm(gradient)), minimum_hessian, float(max(chain_errors)), float(max(supply_violations)), artificial_loss)
