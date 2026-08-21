"""Small-signal modes of fitted physical models, never of trajectory snapshots."""

from __future__ import annotations

from typing import Any

import numpy as np

from kron_reduce.models.m0 import M0Model
from kron_reduce.models.port_structured_discovery import PortStructuredDiscoveryM0Model
from kron_reduce.models.transient_emf_m0 import TransientEmfM0Model

from .modal import (
    DmdMode,
    finite_difference_augmented_modes,
    linearized_passive_network_modes,
)


def physical_model_modes(
    model: M0Model | TransientEmfM0Model | PortStructuredDiscoveryM0Model,
    static: dict[str, Any],
    *,
    relative_step: float = 1e-6,
    min_frequency_hz: float = 0.01,
    max_frequency_hz: float = 10.0,
) -> tuple[list[DmdMode], dict[str, float | int | bool | str]]:
    """Extract observable quotient-space poles from a fitted physical model.

    This is deliberately a closed registry. Unsupported model families raise
    instead of silently linearizing an arbitrary predictor. The returned poles
    are properties of the fitted explicit equation; trajectory DMD remains a
    separate observation-support diagnostic.
    """
    mass = np.asarray(static["mass"], dtype=float).reshape(-1)
    incidence = np.asarray(static["incidence"], dtype=float)
    equilibrium = np.asarray(static["equilibrium_delta"], dtype=float).reshape(-1)
    if isinstance(model, M0Model):
        if model.damping is None or model.coupling is None:
            raise RuntimeError("M0 must be fitted before modal extraction")
        modes = linearized_passive_network_modes(
            mass,
            incidence,
            equilibrium,
            model.damping,
            model.coupling,
            min_frequency_hz=min_frequency_hz,
            max_frequency_hz=max_frequency_hz,
        )
        return modes, {
            "model_class": type(model).__name__,
            "linearization": "analytic_COI_quotient",
            "state_dimension": 2 * (len(mass) - 1),
            "hidden_state_dimension": 0,
            "all_poles_stable": bool(all(mode.decay_rate < 0 for mode in modes)),
        }

    if isinstance(model, PortStructuredDiscoveryM0Model):
        backbone, taus, _, memory_coefficients = model._require_fitted()
        q_equilibrium = backbone.initial_emf(equilibrium, static)
        zero_speed = np.zeros_like(mass)
        z_equilibrium = model.initial_memory(equilibrium, zero_speed, static)
        hidden_equilibrium = np.concatenate((q_equilibrium, z_equilibrium.reshape(-1)))
        topology = backbone._graph_topology(static)
        pm_equilibrium = np.asarray(static["Pm_equilibrium"], dtype=float).reshape(-1)
        z_shape = memory_coefficients.shape

        def field(
            delta: np.ndarray,
            omega: np.ndarray,
            hidden: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            q = hidden[: len(mass)]
            z = hidden[len(mass) :].reshape(z_shape)
            derivative = model.vector_field_augmented(
                delta,
                omega,
                q,
                z,
                pm_equilibrium,
                static,
                topology,
            )
            return (
                derivative[0],
                derivative[1],
                np.concatenate((derivative[2], derivative[3].reshape(-1))),
            )

        modes, diagnostics = finite_difference_augmented_modes(
            mass,
            equilibrium,
            hidden_equilibrium,
            field,
            relative_step=relative_step,
            min_frequency_hz=min_frequency_hz,
            max_frequency_hz=max_frequency_hz,
        )
        return modes, {
            **diagnostics,
            "model_class": type(model).__name__,
            "linearization": "central_difference_COI_quotient",
            "memory_time_scales": len(taus),
        }

    if isinstance(model, TransientEmfM0Model):
        q_equilibrium = model.initial_emf(equilibrium, static)
        topology = model._graph_topology(static)
        pm_equilibrium = np.asarray(static["Pm_equilibrium"], dtype=float).reshape(-1)

        def field(
            delta: np.ndarray,
            omega: np.ndarray,
            hidden: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            return model.vector_field_augmented(
                delta,
                omega,
                hidden,
                pm_equilibrium,
                static,
                topology,
            )

        modes, diagnostics = finite_difference_augmented_modes(
            mass,
            equilibrium,
            q_equilibrium,
            field,
            relative_step=relative_step,
            min_frequency_hz=min_frequency_hz,
            max_frequency_hz=max_frequency_hz,
        )
        return modes, {
            **diagnostics,
            "model_class": type(model).__name__,
            "linearization": "central_difference_COI_quotient",
        }

    raise TypeError(f"unregistered physical model family: {type(model).__name__}")
