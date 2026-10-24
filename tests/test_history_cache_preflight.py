"""Synthetic in-memory preflight tests; no payload or propagation is used."""
from __future__ import annotations

import json
import sys

import numpy as np
import pytest

import kron_reduce.evaluation.history_cache_preflight as preflight_module


def _projection(mass):
    return np.eye(10) - np.outer(np.ones(10), mass) / mass.sum()


def _decoded_model(seed):
    mass = np.linspace(1.0, 2.0, 10)
    mass /= mass.sum()
    coupling = (np.eye(10) - np.ones((10, 10)) / 10) @ np.column_stack(
        (np.linspace(-0.3, 0.3, 10), np.cos(np.arange(10)))
    )
    damping = 0.02 * (np.eye(10) - np.ones((10, 10)) / 10)
    arrays = {
        "mass": mass,
        "edge_stiffness": np.full(45, 0.4),
        "damping": damping,
        "coupling": coupling,
        "latent_skew": np.array([[0.0, -0.2], [0.2, 0.0]]),
        "latent_dissipation": np.diag([0.1, 0.15]),
        "encoder": np.full((2, 80), 0.002 * (seed / 324)),
        "feature_scale": np.linspace(0.8, 1.2, 80),
        "target_scale": np.linspace(0.7, 1.3, 20),
    }
    settings = {
        "latent_dimension": 2,
        "backbone": "sine",
        "frames": 3,
        "epochs": 240,
        "learning_rate": 0.001,
        "encoder_penalty": 0.0001,
        "history_weight": 1.0,
        "horizons_s": [1.0, 3.0],
        "seed": seed,
        "encoder_mode": "frames",
    }
    meta = {"schema": "nonlinear_passive_history.v1", "settings": settings,
            "fit_record": {"synthetic": True}, "backbone": "sine"}
    return {**arrays, "metadata": json.dumps(meta, separators=(",", ":"))}


def _synthetic_population(monkeypatch=None):
    mass = np.linspace(1.0, 2.0, 10)
    mass /= mass.sum()
    qproj = _projection(mass)
    target_scale = np.linspace(0.7, 1.3, 20)
    time = np.arange(11, dtype=np.float64) * 0.01 - 0.1
    history = np.empty((258, 11, 20))
    power = np.empty((258, 10))
    equilibrium = np.empty((258, 10))
    for i in range(258):
        phase = i * 0.003
        raw_q = np.column_stack([
            0.02 * np.sin(np.arange(11) * 0.12 + phase + j * 0.07)
            for j in range(10)
        ])
        raw_v = np.column_stack([
            0.01 * np.cos(np.arange(11) * 0.14 + phase + j * 0.11)
            for j in range(10)
        ])
        history[i] = np.c_[raw_q @ qproj.T, raw_v @ qproj.T]
        power[i] = qproj @ (np.linspace(-0.01, 0.01, 10) * (1 + phase))
        equilibrium[i] = qproj @ np.linspace(-0.05, 0.05, 10)
    truth = {
        "history": history,
        "history_time": np.broadcast_to(time, (258, 11)).copy(),
        "constant_power": power,
        "equilibrium": equilibrium,
        "mass": mass.copy(),
        "target_scale": target_scale.copy(),
    }
    models = {
        seed: preflight_module.load_fixed_model(_decoded_model(seed), seed)
        for seed in preflight_module.SEEDS
    }

    # Derive the cached origin only from synthetic visible history and the
    # registered original feature mapping, then hold it fixed across both
    # synthetic step sizes.
    controls = {}
    model_module = sys.modules[preflight_module.LearnedNonlinearModel.__module__]
    for seed, model in models.items():
        paths = np.zeros((258, 11, 22), dtype=np.float64)
        paths[:, :, :20] = history
        for i in range(258):
            h = model_module.CausalStateHistory(
                time=truth["history_time"][i],
                delta=history[i, :, :10],
                omega=history[i, :, 10:],
                constant_power=power[i],
            )
            features = model_module._features(h, equilibrium[i], 3, "frames")
            paths[i, :, 20:] = model.encoder @ (features / model.feature_scale)
        controls[(seed, 0.005)] = paths.copy()
        controls[(seed, 0.0025)] = paths.copy()
    return truth, models, controls


def test_fixed_model_loader_validates_exact_schema_and_members():
    model = preflight_module.load_fixed_model(_decoded_model(324), 324)
    assert model.settings.seed == 324
    assert model.closure.n_nodes == 10
    assert model.closure.latent_dimension == 2
    bad = _decoded_model(324)
    bad["metadata"] = json.dumps({**json.loads(bad["metadata"]), "unexpected": 1})
    with pytest.raises(ValueError):
        preflight_module.load_fixed_model(bad, 324)
    bad = _decoded_model(324)
    bad["extra"] = np.ones(1)
    with pytest.raises(ValueError):
        preflight_module.load_fixed_model(bad, 324)


def test_preflight_checks_all_258_events_and_records_774_feature_encoder_calls(monkeypatch):
    truth, models, controls = _synthetic_population()
    model_type = type(models[324].closure)

    def forbidden(*args, **kwargs):
        raise AssertionError("preflight must not run RHS or rollout")

    monkeypatch.setattr(model_type, "rhs", forbidden)
    monkeypatch.setattr(model_type, "rollout", forbidden)
    transitions = []
    result = preflight_module.preflight(truth, models, controls, event_sink=transitions.append)
    assert result["encoder_z"].shape == (3, 258, 2)
    assert len(result["event_stats"]) == 774
    assert result["call_counts"] == {
        "features": {"attempted": 774, "returned": 774, "failed": 0},
        "encoder": {"attempted": 774, "returned": 774, "failed": 0},
    }
    assert result["step_counts_per_event"]["0.005"].shape == (258, 10)
    assert np.all(np.sum(result["step_counts_per_event"]["0.005"], axis=1) == 20)
    assert np.all(np.sum(result["step_counts_per_event"]["0.0025"], axis=1) == 40)
    assert result["rhs_calls"] == result["rollout_calls"] == 0
    assert len(transitions) == 4 * 774
    assert transitions[0]["transition"] == "attempted"
    assert transitions[1]["transition"] == "returned"
    assert transitions[2]["operation"] == "encoder"


def test_preflight_refuses_clock_control_and_model_binding_mismatches():
    truth, models, controls = _synthetic_population()
    bad_truth = dict(truth)
    bad_truth["history_time"] = truth["history_time"].copy()
    bad_truth["history_time"][0, 3] += 0.001
    with pytest.raises(ValueError, match="substeps"):
        preflight_module.preflight(bad_truth, models, controls)

    bad_controls = dict(controls)
    bad_controls[(324, 0.005)] = controls[(324, 0.005)].copy()
    bad_controls[(324, 0.005)][0, 0, 20] += 0.1
    with pytest.raises(ValueError, match="hidden origin"):
        preflight_module.preflight(truth, models, bad_controls)

    mismatched = dict(models)
    model = models[324]
    model.target_scale = model.target_scale.copy()
    model.target_scale[0] += 1e-15
    with pytest.raises(ValueError, match="exactly match"):
        preflight_module.preflight(truth, mismatched, controls)


def test_short_positive_interval_counts_one_original_rollout_substep():
    truth, models, controls = _synthetic_population()
    widths = np.array([1e-16, 0.0126, 0.0126, *([0.01] * 7)], dtype=np.float64)
    assert len(widths) == 10
    clock = -float(np.sum(widths)) + np.r_[0.0, np.cumsum(widths)]
    assert clock[-1] == 0.0 and np.all(np.diff(clock) > 0.0)
    truth["history_time"] = np.broadcast_to(clock, (258, 11)).copy()

    # The raw ceil formula gives the planned aggregate totals 20/40 because
    # the tiny positive interval rounds to zero; the rollout's max(1, ceil)
    # rule actually requires 21/41 and must be rejected before feature mapping.
    raw_coarse = np.ceil(widths / 0.005 - 1e-12).astype(int)
    raw_fine = np.ceil(widths / 0.0025 - 1e-12).astype(int)
    assert int(raw_coarse.sum()) == 20
    assert int(raw_fine.sum()) == 40
    assert int(np.maximum(1, raw_coarse).sum()) == 21
    assert int(np.maximum(1, raw_fine).sum()) == 41
    with pytest.raises(ValueError, match="21"):
        preflight_module.preflight(truth, models, controls)


def test_cache_origin_is_history_first_sample_not_last_observation():
    truth, models, controls = _synthetic_population()
    assert not np.array_equal(truth["history"][0, 0], truth["history"][0, -1])
    assert np.array_equal(controls[(324, 0.005)][0, 0, :20], truth["history"][0, 0])
    preflight_module.preflight(truth, models, controls)

    wrong = dict(controls)
    wrong[(324, 0.005)] = controls[(324, 0.005)].copy()
    wrong[(324, 0.005)][0, 0, :20] = truth["history"][0, -1]
    with pytest.raises(ValueError, match="history start"):
        preflight_module.preflight(truth, models, wrong)


def test_sink_attempt_failure_prevents_feature_call(monkeypatch):
    truth, models, controls = _synthetic_population()
    calls = []
    original = preflight_module._features

    def track(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(preflight_module, "_features", track)

    def fail_sink(_event):
        raise OSError("synthetic journal refusal")

    with pytest.raises(OSError, match="journal refusal"):
        preflight_module.preflight(truth, models, controls, event_sink=fail_sink)
    assert calls == []


def test_all_model_priors_are_checked_before_any_feature_mapping(monkeypatch):
    truth, models, controls = _synthetic_population()
    models[324].encoder[0, 0] = np.nan
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(1)
        raise AssertionError("feature mapping started before priors were checked")

    monkeypatch.setattr(preflight_module, "_features", forbidden)
    with pytest.raises(ValueError, match="model 324 encoder"):
        preflight_module.preflight(truth, models, controls)
    assert calls == []


def test_feature_failure_is_journaled_and_not_retried(monkeypatch):
    truth, models, controls = _synthetic_population()
    calls = []

    def fail_feature(*args, **kwargs):
        calls.append(1)
        raise FloatingPointError("synthetic feature failure")

    monkeypatch.setattr(preflight_module, "_features", fail_feature)
    transitions = []
    with pytest.raises(FloatingPointError, match="feature failure"):
        preflight_module.preflight(truth, models, controls, event_sink=transitions.append)
    assert calls == [1]
    assert [row["transition"] for row in transitions] == ["attempted", "failed"]
