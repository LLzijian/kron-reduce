from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ModuleSpec:
    """Typed, neutral physical-module contract used by AFSD v4."""

    name: str
    state: tuple[str, ...]
    input_ports: tuple[str, ...]
    output_ports: tuple[str, ...]
    units: dict[str, str]
    constitutive_law: str
    parameter_domain: dict[str, str]
    initial_state_policy: str
    equilibrium_anchor: str
    energy_or_storage: str
    dissipation_condition: str
    allowed_connections: tuple[str, ...]
    complexity_code_length: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def module_registry() -> dict[str, ModuleSpec]:
    """Return the preregistered v4 module grammar.

    Names deliberately describe ports and energetic roles rather than device
    semantics.  A later, independent audit may associate a selected module
    with a device class, but structure discovery cannot do so by itself.
    """

    specs = [
        ModuleSpec(
            "ROTOR_INERTIA",
            ("delta", "omega"),
            ("net_power",),
            ("omega",),
            {"M": "power*time^2/radian"},
            "M * domega = net_power",
            {"M": "strictly_positive"},
            "clearing_state",
            "delta=delta_eq, omega=0",
            "0.5 * omega.T @ M @ omega",
            "lossless_storage",
            ("node_power_balance",),
            2,
        ),
        ModuleSpec(
            "NODE_DAMPING",
            (),
            ("omega",),
            ("node_force",),
            {"D": "power*time/radian"},
            "node_force = D * omega",
            {"D": "nonnegative"},
            "stateless",
            "node_force(omega=0)=0",
            "none",
            "omega.T @ D @ omega >= 0",
            ("node_power_balance",),
            1,
        ),
        ModuleSpec(
            "EDGE_SIN_POTENTIAL",
            (),
            ("edge_angle",),
            ("edge_force",),
            {"K": "power"},
            "edge_force = K * (sin(eta)-sin(eta_eq))",
            {"K": "nonnegative"},
            "stateless",
            "edge_force(eta_eq)=0",
            "K * psi(eta; eta_eq)",
            "conservative_gradient",
            ("reciprocal_edge",),
            2,
        ),
        ModuleSpec(
            "EDGE_DAMPING",
            (),
            ("edge_speed",),
            ("edge_force",),
            {"D_e": "power*time/radian"},
            "edge_force = D_e * edge_speed",
            {"D_e": "nonnegative"},
            "stateless",
            "edge_force(edge_speed=0)=0",
            "none",
            "edge_speed.T @ D_e @ edge_speed >= 0",
            ("reciprocal_edge",),
            1,
        ),
        ModuleSpec(
            "STATIC_NONLINEAR_POTENTIAL",
            (),
            ("edge_angle",),
            ("edge_force",),
            {"kappa": "power"},
            "edge_force = gradient_eta anchored_potential(eta)",
            {"kappa": "nonnegative"},
            "stateless",
            "gradient_potential(eta_eq)=0",
            "anchored scalar edge potential",
            "conservative_gradient",
            ("reciprocal_edge",),
            3,
        ),
        ModuleSpec(
            "NODE_LAG_STORAGE",
            ("z_node",),
            ("omega",),
            ("node_force",),
            {"tau": "time", "gain": "power*time/radian"},
            "tau * dz = -z + gain * omega",
            {"tau": "strictly_positive", "gain": "nonnegative"},
            "causal_quasisteady_from_clearing_state",
            "z=0 at equilibrium",
            "0.5 * z.T @ z",
            "tau > 0 and passive port sign",
            ("node_power_balance",),
            4,
        ),
        ModuleSpec(
            "EDGE_LAG_STORAGE",
            ("z_edge",),
            ("edge_speed",),
            ("edge_force",),
            {"tau": "time", "gain": "power*time/radian"},
            "tau * dz = -z + gain * edge_speed",
            {"tau": "strictly_positive", "gain": "nonnegative"},
            "causal_quasisteady_from_clearing_state",
            "z=0 at equilibrium",
            "0.5 * z.T @ z",
            "tau > 0 and passive port sign",
            ("reciprocal_edge",),
            4,
        ),
        ModuleSpec(
            "TWO_TIMESCALE_EDGE_MEMORY",
            ("z_edge_fast", "z_edge_slow"),
            ("edge_speed",),
            ("edge_force",),
            {"tau_fast": "time", "tau_slow": "time", "gain": "power*time/radian"},
            "tau_q * dz_q = -z_q + gain_q * edge_speed",
            {
                "tau_fast": "strictly_positive",
                "tau_slow": "strictly_positive_ordered",
                "gain": "nonnegative",
            },
            "causal_quasisteady_from_clearing_state",
            "both states zero at equilibrium",
            "0.5 * (z_fast.T @ z_fast + z_slow.T @ z_slow)",
            "both time constants positive and passive port signs",
            ("reciprocal_edge",),
            7,
        ),
        ModuleSpec(
            "PM_CHANNEL_RELAXATION",
            ("z_pm",),
            ("observed_pm", "omega"),
            ("effective_pm",),
            {"tau_pm": "time", "gain": "dimensionless"},
            "tau_pm * dz_pm = -z_pm + observed_pm - gain * omega",
            {"tau_pm": "strictly_positive", "gain": "nonnegative"},
            "observed_pm_at_clearing",
            "effective_pm=pm_equilibrium at equilibrium",
            "input-channel stable storage",
            "stable relaxation; external supply is declared",
            ("external_pm_port",),
            5,
        ),
        ModuleSpec(
            "SATURATION_OR_DEADBAND",
            (),
            ("declared_port_signal",),
            ("bounded_port_signal",),
            {"threshold": "same_as_input", "slope": "dimensionless"},
            "anchored monotone piecewise-smooth constitutive map",
            {"threshold": "nonnegative", "slope": "nonnegative"},
            "stateless",
            "output(0)=0",
            "none",
            "monotone map at a declared dissipative or supply port",
            ("declared_node_or_edge_port",),
            4,
        ),
    ]
    return {spec.name: spec for spec in specs}


def validate_module_spec(spec: ModuleSpec) -> dict[str, Any]:
    errors: list[str] = []
    if not spec.name or not spec.input_ports or not spec.output_ports:
        errors.append("missing typed ports")
    if spec.complexity_code_length <= 0:
        errors.append("nonpositive description length")
    if not spec.equilibrium_anchor:
        errors.append("missing equilibrium anchor")
    if not spec.allowed_connections:
        errors.append("missing allowed connections")
    if any(
        "tau" in name and "positive" not in domain for name, domain in spec.parameter_domain.items()
    ):
        errors.append("time constant is not constrained positive")
    if spec.name in {"NODE_DAMPING", "EDGE_DAMPING"} and ">= 0" not in spec.dissipation_condition:
        errors.append("damping does not certify nonnegative dissipation")
    if (
        "POTENTIAL" in spec.name
        and "gradient" not in (spec.constitutive_law + spec.dissipation_condition).lower()
    ):
        errors.append("potential force is not generated by a gradient")
    return {"status": "PASS" if not errors else "FAIL", "errors": errors}


def validate_module_registry(registry: dict[str, ModuleSpec] | None = None) -> dict[str, Any]:
    registry = module_registry() if registry is None else registry
    expected = {
        "ROTOR_INERTIA",
        "NODE_DAMPING",
        "EDGE_SIN_POTENTIAL",
        "EDGE_DAMPING",
        "STATIC_NONLINEAR_POTENTIAL",
        "NODE_LAG_STORAGE",
        "EDGE_LAG_STORAGE",
        "TWO_TIMESCALE_EDGE_MEMORY",
        "PM_CHANNEL_RELAXATION",
        "SATURATION_OR_DEADBAND",
    }
    rows = {name: validate_module_spec(spec) for name, spec in registry.items()}
    missing = sorted(expected - set(registry))
    failures = sorted(name for name, row in rows.items() if row["status"] != "PASS")
    return {
        "status": "PASS" if not missing and not failures else "FAIL",
        "missing": missing,
        "invalid": failures,
        "modules": rows,
    }


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float).reshape(-1)
    right = np.asarray(right, dtype=float).reshape(-1)
    if len(left) != len(right) or len(left) < 3:
        return 0.0
    left = left - np.mean(left)
    right = right - np.mean(right)
    norm = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(left @ right / norm) if norm > 1e-14 else 0.0


def lag_correlation(residual: np.ndarray, signal: np.ndarray, max_lag: int = 20) -> dict[str, Any]:
    residual = np.asarray(residual, dtype=float).reshape(-1)
    signal = np.asarray(signal, dtype=float).reshape(-1)
    rows = []
    for lag in range(max(0, int(max_lag)) + 1):
        if lag == 0:
            value = _safe_correlation(residual, signal)
        elif lag < min(len(residual), len(signal)) - 2:
            value = _safe_correlation(residual[lag:], signal[:-lag])
        else:
            continue
        rows.append((lag, value))
    if not rows:
        return {"best_lag": 0, "correlation": 0.0, "memory_gain": 0.0}
    best_lag, best = max(rows, key=lambda row: abs(row[1]))
    zero = rows[0][1]
    return {
        "best_lag": int(best_lag),
        "correlation": float(best),
        "memory_gain": float(abs(best) - abs(zero)),
    }


def residual_module_proposal(
    residual: np.ndarray,
    *,
    node_speed: np.ndarray | None = None,
    edge_speed: np.ndarray | None = None,
    edge_angle: np.ndarray | None = None,
    pm_channel: np.ndarray | None = None,
    bootstrap_support: float = 1.0,
    threshold: float = 0.45,
) -> dict[str, Any]:
    """Map a reproducible residual signature to registered physical actions."""

    residual = np.asarray(residual, dtype=float).reshape(-1)
    evidence: dict[str, Any] = {"bootstrap_support": float(bootstrap_support)}
    if len(residual) < 8 or float(np.std(residual)) <= 1e-12 or bootstrap_support < 0.8:
        return {
            "status": "INSUFFICIENT_INFORMATION",
            "proposals": [],
            "evidence": evidence,
            "rejected_alternatives": ["all modules: residual not reproducible or not excited"],
        }

    correlations = {}
    for name, signal in {
        "node_speed": node_speed,
        "edge_speed": edge_speed,
        "edge_angle": edge_angle,
        "pm_channel": pm_channel,
    }.items():
        if signal is not None:
            correlations[name] = _safe_correlation(residual, np.asarray(signal))
    evidence["instantaneous_correlations"] = correlations

    lag_candidates = {}
    for name, signal in {
        "node_speed": node_speed,
        "edge_speed": edge_speed,
        "pm_channel": pm_channel,
    }.items():
        if signal is not None:
            lag_candidates[name] = lag_correlation(residual, np.asarray(signal))
    evidence["lag_correlations"] = lag_candidates

    proposals: list[str] = []
    best_instant = max(correlations, key=lambda key: abs(correlations[key]), default=None)
    best_lagged = max(
        lag_candidates,
        key=lambda key: lag_candidates[key]["memory_gain"],
        default=None,
    )
    lag_gain = lag_candidates.get(best_lagged, {}).get("memory_gain", 0.0)
    lag_value = abs(lag_candidates.get(best_lagged, {}).get("correlation", 0.0))

    if best_lagged is not None and lag_gain >= 0.01 and lag_value >= threshold:
        proposals.append(
            {
                "node_speed": "NODE_LAG_STORAGE",
                "edge_speed": "EDGE_LAG_STORAGE",
                "pm_channel": "PM_CHANNEL_RELAXATION",
            }[best_lagged]
        )
    elif best_instant is not None and abs(correlations[best_instant]) >= threshold:
        proposals.append(
            {
                "node_speed": "NODE_DAMPING",
                "edge_speed": "EDGE_DAMPING",
                "edge_angle": "STATIC_NONLINEAR_POTENTIAL",
                "pm_channel": "PM_CHANNEL_RELAXATION",
            }[best_instant]
        )

    sorted_lags = sorted(
        (
            row["best_lag"],
            abs(row["correlation"]),
        )
        for row in lag_candidates.values()
        if row["best_lag"] > 0 and abs(row["correlation"]) >= threshold
    )
    if len(sorted_lags) >= 2 and sorted_lags[-1][0] >= 2 * sorted_lags[0][0]:
        proposals = ["TWO_TIMESCALE_EDGE_MEMORY"] if best_lagged == "edge_speed" else proposals

    # Repeated clipping leaves many samples at a small set of residual levels.
    rounded = np.round(residual / max(np.std(residual), 1e-12), 2)
    _, counts = np.unique(rounded, return_counts=True)
    plateau_fraction = float(np.max(counts) / len(rounded))
    evidence["plateau_fraction"] = plateau_fraction
    if plateau_fraction >= 0.2 and best_instant is not None:
        proposals = ["SATURATION_OR_DEADBAND"]

    if not proposals:
        status = "NO_REGISTERED_MODULE_TRIGGER"
    else:
        status = "PROPOSED"
    considered = {
        "NODE_DAMPING",
        "EDGE_DAMPING",
        "STATIC_NONLINEAR_POTENTIAL",
        "NODE_LAG_STORAGE",
        "EDGE_LAG_STORAGE",
        "TWO_TIMESCALE_EDGE_MEMORY",
        "PM_CHANNEL_RELAXATION",
        "SATURATION_OR_DEADBAND",
    }
    return {
        "status": status,
        "proposals": proposals,
        "evidence": evidence,
        "rejected_alternatives": sorted(considered - set(proposals)),
    }
