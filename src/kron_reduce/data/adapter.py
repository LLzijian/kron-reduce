from __future__ import annotations

from typing import Any

import numpy as np

from kron_reduce.physics import CLUSTERED_7_PARTITION, aggregate_event_arrays, complete_incidence
from kron_reduce.physics.units import effective_mass

from .contracts import (
    IneligibleEventError,
    InputContract,
    ModelDataRequest,
    PmMode,
    RequestMode,
    ShortSequenceError,
)
from .schema import EventSequence


def _representation(
    event: EventSequence, inertia_prior: np.ndarray, representation: str
) -> dict[str, np.ndarray]:
    mass = effective_mass(inertia_prior)
    equilibrium_delta = event.equilibrium["delta0"]
    if representation == "generator_10":
        return {
            "delta": event.delta.copy(),
            "omega": event.omega.copy(),
            "Pm": event.Pm.copy(),
            "equilibrium_delta": equilibrium_delta.copy(),
            "mass": mass,
        }
    if representation == "clustered_7":
        return aggregate_event_arrays(
            event.delta,
            event.omega,
            event.Pm,
            equilibrium_delta,
            mass,
            CLUSTERED_7_PARTITION,
        )
    raise ValueError(f"unsupported representation: {representation}")


def adapt_event(
    event: EventSequence,
    *,
    inertia_prior: np.ndarray,
    request: ModelDataRequest,
    contract: InputContract | None = None,
    representation: str = "clustered_7",
) -> dict[str, Any]:
    contract = contract or InputContract()
    arrays = _representation(event, inertia_prior, representation)
    incidence, edges = complete_incidence(arrays["delta"].shape[1])
    decision_index = 0
    observed_context: dict[str, np.ndarray] | None = None
    if request.mode == RequestMode.PREFIX_STEPS:
        assert request.n_steps is not None
        if len(event.t) < request.n_steps:
            raise ShortSequenceError(
                f"{event.event_id}: requires {request.n_steps} real samples, has {len(event.t)}"
            )
        decision_index = request.n_steps - 1
        decision_time = float(event.t[decision_index])
        if (
            request.prospective
            and event.metadata.event_observed
            and event.metadata.event_time_s is not None
            and event.metadata.event_time_s <= decision_time
        ):
            raise IneligibleEventError(
                f"{event.event_id}: event occurred inside observation window"
            )
        observed_context = {
            "t": event.t[: request.n_steps].copy(),
            "delta": arrays["delta"][: request.n_steps].copy(),
            "omega": arrays["omega"][: request.n_steps].copy(),
            "Pm": arrays["Pm"][: request.n_steps].copy(),
        }
    t_future = event.t[decision_index:] - event.t[decision_index]
    observed_pm = arrays["Pm"][decision_index:].copy()
    if request.pm_mode == PmMode.ZOH:
        pm_schedule = np.repeat(arrays["Pm"][[decision_index]], len(t_future), axis=0)
        claim_type = "causal_zoh_rollout"
    else:
        pm_schedule = observed_pm.copy()
        claim_type = "conditional_rollout_not_causal_forecast"
    static = {
        "delta0": arrays["delta"][[decision_index]].copy(),
        "omega0": arrays["omega"][[decision_index]].copy(),
        "Pm0": arrays["Pm"][[decision_index]].copy(),
        "equilibrium_delta": arrays["equilibrium_delta"].copy(),
        "mass": arrays["mass"].copy(),
        "incidence": incidence,
    }
    for key in static:
        if key not in contract.allowed_static:
            raise ValueError(f"static field not allowed by contract: {key}")
    model_inputs: dict[str, Any] = {
        "static": static,
        "exogenous": {"t": t_future.copy(), "Pm": pm_schedule},
    }
    if observed_context is not None:
        model_inputs["observed_context"] = observed_context
    forbidden_present = set(contract.forbidden) & set(model_inputs)
    if forbidden_present:
        raise ValueError(f"forbidden model fields: {sorted(forbidden_present)}")
    return {
        "event_id": event.event_id,
        "model_inputs": model_inputs,
        "training_context": {
            "targets": {
                "t": t_future.copy(),
                "delta": arrays["delta"][decision_index:].copy(),
                "omega": arrays["omega"][decision_index:].copy(),
            },
            "Pm_observed": observed_pm,
            "label": event.metadata.outcome_label,
            "metadata": {
                "event_id": event.event_id,
                "fault_line": event.metadata.fault_line,
                "event_observed": event.metadata.event_observed,
                "event_time_s": event.metadata.event_time_s,
                "source_dataset": event.metadata.source_dataset,
            },
            "decision_index": decision_index,
            "claim_type": claim_type,
            "representation": representation,
            "edges": edges,
        },
    }
