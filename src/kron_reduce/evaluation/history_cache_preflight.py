"""Strict in-memory preflight for frozen A14 history caches.

Inputs are already decoded arrays. This module performs no filesystem access,
model I/O, RHS evaluation, rollout, prediction, or fitting.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import numpy as np

from kron_reduce.data.history_contract import CausalStateHistory
from kron_reduce.models.nonlinear_latent import (
    LearnedNonlinearModel,
    NonlinearFitSettings,
    NonlinearPassiveClosure,
    _features,
)

SEEDS = (324, 325, 326)
MODEL_ARRAY_SHAPES = {
    "mass": (10,),
    "edge_stiffness": (45,),
    "damping": (10, 10),
    "coupling": (10, 2),
    "latent_skew": (2, 2),
    "latent_dissipation": (2, 2),
    "encoder": (2, 80),
    "feature_scale": (80,),
    "target_scale": (20,),
}
MODEL_KEYS = frozenset((*MODEL_ARRAY_SHAPES, "metadata"))
SETTINGS_KEYS = frozenset({
    "latent_dimension", "backbone", "frames", "epochs", "learning_rate",
    "encoder_penalty", "history_weight", "horizons_s", "seed", "encoder_mode",
})
TRUTH_SHAPES = {
    "history": (258, 11, 20),
    "history_time": (258, 11),
    "constant_power": (258, 10),
    "equilibrium": (258, 10),
    "mass": (10,),
    "target_scale": (20,),
}


def _float_array(value: Any, name: str, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return array


def _metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, np.ndarray):
        if value.shape != () or value.dtype.kind != "U":
            raise ValueError("metadata must be a zero-dimensional Unicode string")
        value = str(value.item())
    if not isinstance(value, str):
        raise TypeError("metadata must be a JSON string")
    try:
        result = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("metadata is not valid JSON") from exc
    if not isinstance(result, dict) or set(result) != {"schema", "settings", "fit_record", "backbone"}:
        raise ValueError("metadata fields do not match the saved model schema")
    return result


def load_fixed_model(decoded: Mapping[str, Any], seed: int) -> LearnedNonlinearModel:
    """Construct a validated A14 model from already-decoded saved members."""
    if not isinstance(decoded, Mapping) or set(decoded) != MODEL_KEYS:
        raise ValueError("decoded model must contain exactly nine arrays and metadata")
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError("seed must be one of the three frozen A14 seeds")
    metadata = _metadata(decoded["metadata"])
    if metadata["schema"] != "nonlinear_passive_history.v1" or metadata["backbone"] != "sine":
        raise ValueError("model schema or backbone differs from the A14 contract")
    settings_data = metadata["settings"]
    if not isinstance(settings_data, dict) or set(settings_data) != SETTINGS_KEYS:
        raise ValueError("model settings do not match the registered schema")
    settings_data = dict(settings_data)
    settings_data["horizons_s"] = tuple(settings_data["horizons_s"])
    settings = NonlinearFitSettings(**settings_data)
    if (settings.seed != seed or settings.latent_dimension != 2 or settings.backbone != "sine"
            or settings.frames != 3 or settings.encoder_mode != "frames"):
        raise ValueError("checkpoint settings do not match the A14 seed/model contract")
    if not isinstance(metadata["fit_record"], dict):
        raise TypeError("fit_record must be a JSON object")

    arrays = {name: _float_array(decoded[name], name, shape)
              for name, shape in MODEL_ARRAY_SHAPES.items()}
    if (np.any(arrays["mass"] <= 0) or np.any(arrays["feature_scale"] <= 0)
            or np.any(arrays["target_scale"] <= 0)):
        raise ValueError("mass and normalization scales must be positive")
    closure = NonlinearPassiveClosure(
        **{name: arrays[name] for name in (
            "mass", "edge_stiffness", "damping", "coupling", "latent_skew",
            "latent_dissipation",
        )},
        backbone="sine",
    )
    return LearnedNonlinearModel(
        closure=closure,
        encoder=arrays["encoder"].copy(),
        feature_scale=arrays["feature_scale"].copy(),
        target_scale=arrays["target_scale"].copy(),
        settings=settings,
        fit_record=dict(metadata["fit_record"]),
    )


def _event_call(event_sink, counts, operation: str, identity: dict[str, int], function):
    attempted = {"operation": operation, **identity, "transition": "attempted"}
    counts[operation]["attempted"] += 1
    if event_sink is not None:
        event_sink(attempted)
    try:
        result = function()
    except Exception:
        counts[operation]["failed"] += 1
        if event_sink is not None:
            event_sink({"operation": operation, **identity, "transition": "failed"})
        raise
    counts[operation]["returned"] += 1
    if event_sink is not None:
        event_sink({"operation": operation, **identity, "transition": "returned"})
    return result


def _source_state(delta, omega, latent, mass):
    """Match PassiveLatentClosure._state checks without projecting or simulating."""
    delta = np.asarray(delta, dtype=np.float64)
    omega = np.asarray(omega, dtype=np.float64)
    latent = np.asarray(latent, dtype=np.float64)
    if (delta.shape != (10,) or omega.shape != (10,) or latent.shape != (2,)
            or not np.isfinite(np.r_[delta, omega, latent]).all()
            or max(abs(float(delta @ mass)), abs(float(omega @ mass))) > 1e-10):
        raise ValueError("cached origin is not a finite mass-weighted COI state")
    return np.r_[delta, omega, latent]


def preflight(truth: Mapping[str, Any], models: Mapping[int, LearnedNonlinearModel],
              controls: Mapping[tuple[int, float], Any], *, event_sink=None) -> dict[str, Any]:
    """Validate all 258 rows and three-seed cached starts without propagation."""
    if not isinstance(truth, Mapping) or set(truth) != set(TRUTH_SHAPES):
        raise ValueError("truth must contain exactly the six registered history members")
    arrays = {name: _float_array(truth[name], name, shape)
              for name, shape in TRUTH_SHAPES.items()}
    if np.any(arrays["mass"] <= 0) or np.any(arrays["target_scale"] <= 0):
        raise ValueError("truth mass and target_scale must be positive")
    if (type(models) is not dict or set(models) != set(SEEDS)
            or set(controls) != {(seed, step) for seed in SEEDS for step in (0.005, 0.0025)}):
        raise ValueError("models and controls must cover the frozen three seeds and two steps")
    control_arrays = {
        key: _float_array(value, f"control {key[0]}/{key[1]:g}", (258, 11, 22))
        for key, value in controls.items()
    }
    for seed in SEEDS:
        model = models[seed]
        if (not isinstance(model, LearnedNonlinearModel) or model.settings.seed != seed
                or model.settings.latent_dimension != 2 or model.settings.frames != 3
                or model.settings.backbone != "sine" or model.settings.encoder_mode != "frames"
                or model.closure.backbone != "sine"):
            raise ValueError("model seed identity mismatch")
        model_arrays = {
            "mass": model.closure.mass,
            "edge_stiffness": model.closure.edge_stiffness,
            "damping": model.closure.damping,
            "coupling": model.closure.coupling,
            "latent_skew": model.closure.latent_skew,
            "latent_dissipation": model.closure.latent_dissipation,
            "encoder": model.encoder,
            "feature_scale": model.feature_scale,
            "target_scale": model.target_scale,
        }
        for name, shape in MODEL_ARRAY_SHAPES.items():
            _float_array(model_arrays[name], f"model {seed} {name}", shape)
        if (np.any(model.closure.mass <= 0) or np.any(model.feature_scale <= 0)
                or np.any(model.target_scale <= 0)):
            raise ValueError("model mass and normalization scales must be positive")
        if (not np.array_equal(model.closure.mass, arrays["mass"])
                or not np.array_equal(model.target_scale, arrays["target_scale"])):
            raise ValueError("model mass and target_scale must exactly match the truth cache")

    times = arrays["history_time"]
    if (np.any(times[:, 0] >= 0) or np.any(times[:, -1] != 0.0)
            or np.any(np.diff(times, axis=1) <= 0)):
        raise ValueError("history_time must be increasing, negative-origin, and end exactly at zero")
    intervals = np.diff(times, axis=1)
    expected_counts = {0.005: 20, 0.0025: 40}
    step_counts = {}
    for step, expected in expected_counts.items():
        count = np.maximum(1, np.ceil(intervals / step - 1e-12)).astype(np.int64)
        total_per_event = np.sum(count, axis=1)
        if not np.all(total_per_event == expected):
            raise ValueError(
                f"history clock does not have exactly {expected} total substeps at {step:g}s; "
                f"observed={np.unique(total_per_event).tolist()}"
            )
        step_counts[str(step)] = count

    encoder_z = np.empty((3, 258, 2), dtype=np.float64)
    counts = {
        name: {"attempted": 0, "returned": 0, "failed": 0}
        for name in ("features", "encoder")
    }
    event_stats = []

    for model_index, seed in enumerate(SEEDS):
        model = models[seed]
        for event_index in range(258):
            history = CausalStateHistory(
                time=times[event_index],
                delta=arrays["history"][event_index, :, :10],
                omega=arrays["history"][event_index, :, 10:],
                constant_power=arrays["constant_power"][event_index],
            )
            identity = {"seed": seed, "event_index": event_index}
            feature = _event_call(
                event_sink, counts, "features", identity,
                lambda history=history, model=model, event_index=event_index: _features(
                    history, arrays["equilibrium"][event_index], model.settings.frames,
                    model.settings.encoder_mode,
                ),
            )
            encoded = _event_call(
                event_sink, counts, "encoder", identity,
                lambda model=model, feature=feature: model.encoder @ (feature / model.feature_scale),
            )
            encoded = _float_array(encoded, "encoder output", (2,))
            encoder_z[model_index, event_index] = encoded

            observed_origin = arrays["history"][event_index, 0]
            _source_state(observed_origin[:10], observed_origin[10:], encoded, arrays["mass"])
            histories = {}
            for step in (0.005, 0.0025):
                path = control_arrays[(seed, step)]
                row = path[event_index]
                if not np.allclose(row[0, :20], observed_origin, rtol=0.0, atol=1e-12):
                    raise ValueError("cached visible origin differs from the observed history start")
                if not np.allclose(row[0, 20:], encoded, rtol=0.0, atol=1e-12):
                    raise ValueError("cached hidden origin differs from the original encoder mapping")
                _source_state(row[0, :10], row[0, 10:20], row[0, 20:], arrays["mass"])
                histories[step] = row
            fine, coarse = histories[0.0025], histories[0.005]
            qv_difference = np.abs((fine[:, :20] - coarse[:, :20]) / arrays["target_scale"])
            row_qv_max = float(np.max(qv_difference))
            fine_hidden_norm = np.linalg.norm(fine[:, 20:], axis=1)
            row_hidden_difference = np.linalg.norm(fine[:, 20:] - coarse[:, 20:], axis=1)
            row_hidden_max = float(np.max(row_hidden_difference))
            row_fine_hidden_peak = float(np.max(fine_hidden_norm))
            if row_qv_max > 1e-5:
                raise ValueError("cached visible paths exceed the normalized step-difference bound")
            if row_hidden_max > 1e-5 * (1.0 + row_fine_hidden_peak):
                raise ValueError("cached hidden paths exceed the relative step-difference bound")
            event_stats.append({
                "seed": seed,
                "event_index": event_index,
                "visible_max_normalized_step_difference": row_qv_max,
                "hidden_max_step_difference_norm": row_hidden_max,
                "fine_hidden_peak_norm": row_fine_hidden_peak,
            })

    return {
        "encoder_z": encoder_z,
        "event_stats": event_stats,
        "call_counts": counts,
        "step_counts_per_event": step_counts,
        "population_size": 258,
        "model_seed_order": list(SEEDS),
        "rhs_calls": 0,
        "rollout_calls": 0,
    }
