"""Synthetic tests for physical full-start refinement and checkpoint snapshots."""

from dataclasses import replace

import numpy as np
import pytest
from test_paired_training import tiny_train

from kron_reduce.models.nonlinear_latent import fit_nonlinear_model


def make_full_start(mass, rows, settings):
    baseline = fit_nonlinear_model(rows, mass, settings)
    c = baseline.closure
    return {
        "mass": c.mass.copy(),
        "edge_stiffness": c.edge_stiffness.copy(),
        "damping": c.damping.copy(),
        "coupling": c.coupling.copy(),
        "latent_skew": c.latent_skew.copy(),
        "latent_dissipation": c.latent_dissipation.copy(),
        "encoder": baseline.encoder.copy(),
        "feature_scale": baseline.feature_scale.copy(),
        "target_scale": baseline.target_scale.copy(),
    }


def test_full_start_snapshots_are_updated_fresh_adam_states(monkeypatch):
    import kron_reduce.models.nonlinear_latent as module

    mass, _, rows, settings = tiny_train()
    settings = replace(settings, epochs=3)
    full = make_full_start(mass, rows, settings)

    def forbidden(*args, **kwargs):
        raise AssertionError("full_start must bypass the integral initializer")

    monkeypatch.setattr(module, "_integral_start", forbidden)
    snapshots = []

    def receive(step, model, state):
        snapshots.append((step, model, {k: np.array(v, copy=True) for k, v in state.items()}))

    result = fit_nonlinear_model(
        rows, mass, settings, full_start=full, checkpoint_steps=(1, 3), checkpoint_callback=receive
    )
    assert [step for step, _, _ in snapshots] == [1, 3]
    first = snapshots[0][1]
    first_arrays = [first.encoder.copy(), first.closure.edge_stiffness.copy(), first.closure.damping.copy()]
    assert first.fit_record["actual_optimizer_steps"] == 1
    assert result.fit_record["actual_optimizer_steps"] == 3
    assert np.array_equal(np.asarray(first.fit_record["initial_parameters"]["encoder"]), full["encoder"])
    assert first.fit_record["full_start_roundtrip_errors"]
    assert max(first.fit_record["full_start_roundtrip_errors"].values()) <= 1e-12
    assert not np.array_equal(result.encoder, first.encoder)
    assert all(np.array_equal(old, now) for old, now in zip(
        first_arrays, (first.encoder, first.closure.edge_stiffness, first.closure.damping), strict=True
    ))
    for expected_step, (_, _, state) in zip((1, 3), snapshots, strict=True):
        assert int(state["completed_optimizer_steps"]) == expected_step
        for name in ("raw", "port", "skew", "decay", "encoder"):
            assert np.all(np.isfinite(state[f"parameter__{name}"]))
            adam_step = state[f"adam_step__{name}"]
            assert np.all(adam_step == expected_step)
            assert np.all(np.isfinite(state[f"adam_exp_avg__{name}"]))
            assert np.all(np.isfinite(state[f"adam_exp_avg_sq__{name}"]))
    assert result.fit_record["full_start_sha256"]


def test_full_start_rejects_mixed_modes_bad_memory_roles_and_checkpoints():
    mass, _, rows, settings = tiny_train()
    full = make_full_start(mass, rows, settings)
    paired = {name: full[name] for name in ("mass", "edge_stiffness", "damping", "feature_scale", "target_scale")}
    with pytest.raises(ValueError, match="mutually exclusive"):
        fit_nonlinear_model(rows, mass, settings, full_start=full, paired_start=paired)
    with pytest.raises(ValueError, match="train only"):
        fit_nonlinear_model([replace(rows[0], role="val")], mass, settings, full_start=full)
    bad_skew = dict(full, latent_skew=np.eye(1))
    with pytest.raises(ValueError, match="representable"):
        fit_nonlinear_model(rows, mass, settings, full_start=bad_skew)
    bad_port = dict(full, coupling=full["coupling"] + np.ones_like(full["coupling"]))
    with pytest.raises(ValueError, match="representable"):
        fit_nonlinear_model(rows, mass, settings, full_start=bad_port)
    for steps in ((0,), (2, 1), (1, 1), (settings.epochs + 1,)):
        with pytest.raises(ValueError, match="checkpoint"):
            fit_nonlinear_model(rows, mass, settings, full_start=full, checkpoint_steps=steps, checkpoint_callback=lambda *a: None)
