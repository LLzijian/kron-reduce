"""Synthetic contract tests for paired nonlinear training."""

from dataclasses import replace

import numpy as np
import pytest

from kron_reduce.data.history_contract import CausalStateHistory
from kron_reduce.models.nonlinear_latent import (
    EquilibriumLearningSample,
    NonlinearFitSettings,
    NonlinearPassiveClosure,
    fit_nonlinear_model,
)
from kron_reduce.physics.coordinates import coi_basis, coi_project


def tiny_train():
    mass = np.array([1.0, 1.2, 1.4])
    closure = NonlinearPassiveClosure(
        mass,
        np.array([1.2, 0.9, 1.1]),
        np.eye(3) * 0.15 - np.ones((3, 3)) * 0.05,
        np.zeros((3, 1)),
        np.zeros((1, 1)),
        np.eye(1) * 0.3,
    )
    clock = np.arange(4) * 0.01
    rows = []
    for i in range(2):
        q0 = coi_project(np.array([0.01 * (i + 1), -0.02, 0.01]), mass)
        v0 = coi_project(np.array([0.03, -0.01 * (i + 1), -0.02]), mass)
        equilibrium = np.zeros(3)
        path = closure.rollout(q0, v0, np.array([0.05 * (i + 1)]), clock, np.zeros(3), equilibrium)
        history_clock = np.array([-0.02, -0.01, 0.0])
        history = CausalStateHistory(
            history_clock,
            np.stack([q0 * 0.8, q0 * 0.9, q0]),
            np.stack([v0 * 0.8, v0 * 0.9, v0]),
            np.zeros(3),
        )
        rows.append(EquilibriumLearningSample(
            f"s{i}", "synthetic", f"line{i}", "train", history, clock,
            path[:, :3], path[:, 3:6], equilibrium,
        ))
    settings = NonlinearFitSettings(latent_dimension=1, epochs=2, horizons_s=(0.03,))
    return mass, closure, tuple(rows), settings


def starting_point(mass, closure, rows, settings):
    baseline = fit_nonlinear_model(rows, mass, settings)
    return {
        "mass": mass.copy(),
        "edge_stiffness": closure.edge_stiffness.copy(),
        "damping": closure.damping.copy(),
        "feature_scale": baseline.feature_scale.copy(),
        "target_scale": baseline.target_scale.copy(),
    }


def test_paired_freeze_and_joint_seed_initialization(monkeypatch):
    import torch

    import kron_reduce.models.nonlinear_latent as module

    mass, closure, rows, settings = tiny_train()
    start = starting_point(mass, closure, rows, settings)

    def forbidden(*args, **kwargs):
        raise AssertionError("paired path must not use integral initialization")

    monkeypatch.setattr(module, "_integral_start", forbidden)
    captured = []
    original_clip = torch.nn.utils.clip_grad_norm_

    def capture_clip(parameters, *args, **kwargs):
        params = list(parameters)
        captured.append((params, [p.grad.detach().clone() if p.grad is not None else None for p in params]))
        return original_clip(params, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", capture_clip)
    frozen = fit_nonlinear_model(rows, mass, settings, paired_start=start, freeze_backbone=True)
    joint = fit_nonlinear_model(rows, mass, settings, paired_start=start, freeze_backbone=False)
    assert frozen.fit_record["initial_parameter_sha256"] == joint.fit_record["initial_parameter_sha256"]
    assert np.allclose(frozen.closure.edge_stiffness, start["edge_stiffness"], atol=1e-12, rtol=0)
    assert np.allclose(frozen.closure.damping, start["damping"], atol=1e-12, rtol=0)
    initial = frozen.fit_record["initial_parameters"]
    assert np.array_equal(frozen.encoder, np.asarray(initial["encoder"])) is False
    initial_coupling = coi_basis(np.ones(len(mass))) @ np.asarray(initial["port"])
    assert not np.array_equal(frozen.closure.coupling, initial_coupling)
    assert not np.allclose(joint.closure.edge_stiffness, start["edge_stiffness"], atol=1e-12, rtol=0)
    assert not np.allclose(joint.closure.damping, start["damping"], atol=1e-12, rtol=0)
    raw_count = len(initial["raw"])
    total_count = sum(np.asarray(v).size for v in initial.values())
    assert frozen.fit_record["optimizer_updated_scalars"] == total_count - raw_count
    assert joint.fit_record["optimizer_updated_scalars"] == total_count
    assert len(captured) == 2 * settings.epochs
    for params, grads in captured:
        assert len(params) == 5
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        raw_norm = torch.linalg.vector_norm(grads[0])
        memory_norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in grads[1:]]))
        full_norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in grads]))
        assert torch.isfinite(raw_norm) and torch.isfinite(memory_norm) and torch.isfinite(full_norm)
    for row in frozen.fit_record["loss_trace"]:
        assert all(np.isfinite(row[key]) for key in ("objective", "future_loss", "history_loss", "full_gradient_norm"))
        assert "gradient_clipped" in row and "optimizer_updated_scalars" in row
        norm = np.hypot(row["backbone_gradient_norm"], row["memory_gradient_norm"])
        assert row["full_gradient_norm"] == pytest.approx(norm, abs=1e-12)
        assert row["optimizer_updated_scalars"] == total_count - raw_count
    for row in joint.fit_record["loss_trace"]:
        assert row["optimizer_updated_scalars"] == total_count
    assert frozen.fit_record["parameter_count"] > 0


@pytest.mark.parametrize("role", ["val", "audit"])
def test_paired_fit_rejects_nontrain(role):
    mass, closure, rows, settings = tiny_train()
    start = starting_point(mass, closure, rows, settings)
    with pytest.raises(ValueError, match="train only"):
        fit_nonlinear_model([replace(rows[0], role=role)], mass, settings, paired_start=start, freeze_backbone=True)


def test_paired_fit_rejects_bad_scale_or_backbone():
    mass, closure, rows, settings = tiny_train()
    start = starting_point(mass, closure, rows, settings)
    bad_scale = dict(start, target_scale=start["target_scale"] * 2)
    with pytest.raises(ValueError):
        fit_nonlinear_model(rows, mass, settings, paired_start=bad_scale, freeze_backbone=True)
    bad_backbone = dict(start, damping=np.eye(3))
    with pytest.raises(ValueError):
        fit_nonlinear_model(rows, mass, settings, paired_start=bad_backbone, freeze_backbone=True)

