"""Whole-instance interventions and physical traces, without fitting or claims."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any

import numpy as np

from kron_reduce.models.p1_calibration import core_fingerprint
from kron_reduce.models.prunable_energy_model import (
    EDGE_DAMPING,
    EDGE_MEMORY,
    NODE_HIGH_ORDER,
    NONLINEAR_POTENTIAL,
    RESONANT_MEMORY,
    PrunableEnergyModel,
)

_STATIC_PARAMETERS = {NONLINEAR_POTENTIAL: "potential_coefficients", EDGE_DAMPING: "edge_damping_coefficients", NODE_HIGH_ORDER: "node_high_order_gain"}


def checked_instance_ids(model: PrunableEnergyModel, instance_ids: Sequence[str]) -> tuple[str, ...]:
    known = [row.instance_id for row in model.active_instances]
    requested = tuple(instance_ids)
    if len(set(known)) != len(known) or len(set(requested)) != len(requested) or set(requested) - set(known):
        raise ValueError("interventions require unique, existing whole-instance IDs")
    for family in _STATIC_PARAMETERS:
        if sum(row.family == family for row in model.active_instances) > 1:
            raise ValueError("static module family does not support ambiguous multiple instances")
    return tuple(sorted(requested))


def zero_module_instances(model: PrunableEnergyModel, instance_ids: Sequence[str]) -> PrunableEnergyModel:
    """No refitting. Other parameters and their physical initialization stay fixed."""
    model._require_fitted()
    before = core_fingerprint(model)
    removed = checked_instance_ids(model, instance_ids)
    result = copy.deepcopy(model)
    for instance in result.active_instances:
        if instance.instance_id not in removed:
            continue
        if instance.family in _STATIC_PARAMETERS:
            name = _STATIC_PARAMETERS[instance.family]
            setattr(result, name, np.zeros_like(getattr(result, name)))
        else:
            family = "memory" if instance.family == EDGE_MEMORY else "resonant"
            index = next(index for index, row in enumerate(result._instances(instance.family)) if row.instance_id == instance.instance_id)
            factors = getattr(result, f"{family}_factors")
            # Preserve all other realized ports; remove both directions of this
            # connection and its now-unforced coordinates, not just its label.
            factors[index] = np.zeros((len(result.coupling), 0))
            getattr(result, f"{family}_gains")[index] = 0
    result._refresh_parameter_diagnostics()
    if core_fingerprint(model) != before:
        raise RuntimeError("zero intervention changed its reference")
    return result


def _hidden_columns(model: PrunableEnergyModel, hidden: np.ndarray) -> dict[str, np.ndarray]:
    n_nodes, n_edges = model._require_fitted()
    result, offset = {}, 0
    for name, shape in model._hidden_shapes(n_nodes, n_edges).items():
        count = int(np.prod(shape))
        result[name] = hidden[:, offset:offset + count]
        offset += count
    if offset != hidden.shape[1]:
        raise ValueError("physical audit hidden state does not match core")
    return result


def _components(model: PrunableEnergyModel, static: dict[str, np.ndarray], delta: np.ndarray, omega: np.ndarray, hidden: np.ndarray) -> dict[str, dict[str, Any]]:
    states = _hidden_columns(model, hidden)
    incidence = np.asarray(static["incidence"])
    theta = (delta - np.asarray(static["equilibrium_delta"])) @ incidence
    nu = omega @ incidence
    zeros = np.zeros(len(delta))
    output, offsets = {}, {EDGE_MEMORY: 0, RESONANT_MEMORY: 0}
    for instance in model.active_instances:
        family = instance.family
        if family == NONLINEAR_POTENTIAL:
            c = model.potential_coefficients
            log_cosh = np.logaddexp(theta / model.potential_scale, -theta / model.potential_scale) - np.log(2.)
            storage = np.sum(0.5*c[0]*theta**2 + 0.25*c[1]*theta**4 + c[2]*model.potential_scale*log_cosh, axis=1)
            force = c[0]*theta + c[1]*theta**3 + c[2]*np.tanh(theta/model.potential_scale)
            power, dissipation = np.sum(nu*force, axis=1), zeros
            norm = float(np.linalg.norm(c))
        elif family == EDGE_DAMPING:
            c = model.edge_damping_coefficients
            force = c[0]*nu + c[1]*nu*np.abs(nu) + c[2]*nu*theta**2
            storage, power = zeros, np.sum(nu*force, axis=1)
            dissipation, norm = power, float(np.linalg.norm(c))
        elif family == NODE_HIGH_ORDER:
            state = states["node"]
            storage = 0.5*model.node_tau*np.sum(state**2, axis=1)
            power = np.sum(omega*np.sqrt(model.node_high_order_gain)*state, axis=1)
            dissipation, norm = np.sum(state**2, axis=1), float(np.linalg.norm(model.node_high_order_gain))
        else:
            index = next(index for index, row in enumerate(model._instances(family)) if row.instance_id == instance.instance_id)
            factor = model._memory_factor(index, incidence.shape[1]) if family == EDGE_MEMORY else model._resonant_factor(index, incidence.shape[1])
            section = slice(offsets[family], offsets[family] + factor.shape[1])
            offsets[family] += factor.shape[1]
            norm = float(np.linalg.norm(factor @ factor.T))
            if family == EDGE_MEMORY:
                state = states["memory"][:, section]
                storage = 0.5*float(instance.time_constant)*np.sum(state**2, axis=1)
                power = np.sum(nu*(state @ factor.T), axis=1)
                dissipation = np.sum(state**2, axis=1)
            elif family == RESONANT_MEMORY:
                q, p = states["resonance_q"][:, section], states["resonance_p"][:, section]
                angular = 2*np.pi*float(instance.frequency_hz)
                storage = 0.5*np.sum(p**2 + angular**2*q**2, axis=1)
                power = np.sum(nu*(p @ factor.T), axis=1)
                dissipation = 2*float(instance.damping_ratio)*angular*np.sum(p**2, axis=1)
            else:
                raise ValueError(f"unregistered module family: {family}")
        output[instance.instance_id] = {"family": family, "storage": storage, "port_power": power, "dissipation": dissipation, "parameter_norm": norm, "storage_applicable": family != EDGE_DAMPING}
    return output


def module_physical_traces(model: PrunableEnergyModel, static: dict[str, np.ndarray], delta: np.ndarray, omega: np.ndarray, hidden: np.ndarray, *, difference_step: float = 1e-6) -> dict[str, dict[str, Any]]:
    """Finite-difference module storage along the actual complete core field."""
    checked_instance_ids(model, [row.instance_id for row in model.active_instances])
    n_nodes, _ = model._require_fitted()
    if delta.ndim != 2 or delta.shape != omega.shape or delta.shape[1] != n_nodes or hidden.ndim != 2 or hidden.shape[0] != delta.shape[0] or not all(np.isfinite(value).all() for value in (delta, omega, hidden)) or not np.isfinite(difference_step) or difference_step <= 0:
        raise ValueError("invalid full-state trajectory or difference step")
    clear = np.asarray(static["Pm_clear"]).reshape(-1)
    field = [model._flat_field(d, w, h, clear, static) for d, w, h in zip(delta, omega, hidden, strict=True)]
    derivative = tuple(np.stack([row[index] for row in field]) for index in range(3))
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        values = _components(model, static, delta, omega, hidden)
        plus = _components(model, static, *(value + difference_step*slope for value, slope in zip((delta, omega, hidden), derivative, strict=True)))
        minus = _components(model, static, *(value - difference_step*slope for value, slope in zip((delta, omega, hidden), derivative, strict=True)))
        for name, trace in values.items():
            trace["identity_error"] = np.abs((plus[name]["storage"] - minus[name]["storage"]) / (2*difference_step) - (trace["port_power"] - trace["dissipation"]))
            if not all(np.isfinite(trace[key]).all() for key in ("storage", "port_power", "dissipation", "identity_error")) or not np.isfinite(trace["parameter_norm"]):
                raise FloatingPointError("nonfinite module physical audit")
    return values


def summarize_physical_trace(traces: dict[str, dict[str, Any]], instance_ids: Sequence[str], stop: int, energy_scale: float) -> dict[str, Any]:
    if not np.isfinite(energy_scale) or energy_scale <= 0 or type(stop) is not int or stop < 1:
        raise ValueError("fixed positive energy scale and horizon required")
    ids = tuple(instance_ids)
    if len(set(ids)) != len(ids) or set(ids) - set(traces):
        raise ValueError("unknown or duplicate physical module selection")
    if any(stop > len(traces[name]["storage"]) for name in ids):
        raise ValueError("physical trace horizon is incomplete")
    selected = [traces[name] for name in ids]
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        storage = sum((row["storage"][:stop] for row in selected), np.zeros(stop))
        power = sum((row["port_power"][:stop] for row in selected), np.zeros(stop))
        dissipation = sum((row["dissipation"][:stop] for row in selected), np.zeros(stop))
        identity = max((float(np.max(row["identity_error"][:stop])) for row in selected), default=0.)
        storage_mean = float(np.mean(storage))
        port_abs_mean = float(np.mean(np.abs(power)))
        parameter_norm = float(np.linalg.norm([row["parameter_norm"] for row in selected]))
        result = {
            "storage_mean": storage_mean, "storage_q95": float(np.quantile(storage, .95)),
            "port_abs_mean": port_abs_mean,
            "sum_port_abs_mean": float(sum(np.mean(np.abs(row["port_power"][:stop])) for row in selected)),
            "dissipation_mean": float(np.mean(dissipation)), "identity_error_max": identity,
            "storage_normalized_mean": storage_mean/energy_scale,
            "port_normalized_abs_mean": port_abs_mean/energy_scale,
            "identity_error_normalized": identity/energy_scale,
            "energy_scale": energy_scale, "power_scale": energy_scale,
            "parameter_norm": parameter_norm,
            "storage_applicable": any(row["storage_applicable"] for row in selected),
            "pure_dissipation_protocol_conflict": bool(selected) and not any(row["storage_applicable"] for row in selected),
            "independent_support_verdict": "NOT_EVALUATED",
        }
    numeric = [value for value in result.values() if isinstance(value, (int, float)) and not isinstance(value, bool)]
    if not np.isfinite(numeric).all():
        raise FloatingPointError("nonfinite module physical summary")
    return result
