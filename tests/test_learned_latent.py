from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from kron_reduce.models.causal_latent import CausalStateHistory, PassiveLatentClosure
from kron_reduce.models.learned_latent import (
    LatentFitSettings,
    LatentLearningSample,
    LearnedLatentModel,
    fit_latent_model,
    history_features,
    refit_latent_model,
    visible_scales,
)
from kron_reduce.physics.coordinates import coi_basis, coi_project


def _samples() -> tuple[PassiveLatentClosure, tuple[LatentLearningSample, ...]]:
    n = 3
    laplacian = np.eye(n) - np.ones((n, n)) / n
    model = PassiveLatentClosure(
        np.ones(n),
        2 * laplacian,
        0.2 * laplacian,
        coi_basis(np.ones(n))[:, :1],
        np.zeros((1, 1)),
        np.ones((1, 1)),
    )
    rng = np.random.default_rng(324)
    samples = []
    for index in range(4):
        q, v = coi_project(rng.normal(0, 0.05, n), model.mass), np.zeros(n)
        path = model.rollout(q, v, rng.normal(size=1), np.arange(31) * 0.01, np.zeros(n))
        history = CausalStateHistory(
            np.arange(11) * 0.01 - 0.1, path[:11, :n], path[:11, n : 2 * n], np.zeros(n)
        )
        samples.append(
            LatentLearningSample(
                f"e{index}",
                "A" if index < 2 else "B",
                f"L{index}",
                "train",
                history,
                np.arange(21) * 0.01,
                path[10:, :n],
                path[10:, n : 2 * n],
            )
        )
    return model, tuple(samples)


def test_learned_latent_training_rejects_val_and_audit() -> None:
    truth, samples = _samples()
    for role in ("val", "audit"):
        with pytest.raises(ValueError, match="train samples only"):
            fit_latent_model(
                [replace(samples[0], role=role)], truth.mass, LatentFitSettings(epochs=2)
            )
        with pytest.raises(ValueError, match="train only"):
            visible_scales([replace(samples[0], role=role)])


def test_shared_encoder_fit_passivity_determinism_and_checkpoint(tmp_path: Path) -> None:
    truth, samples = _samples()
    settings = LatentFitSettings(latent_dimension=1, epochs=8)
    first = fit_latent_model(samples, truth.mass, settings)
    second = fit_latent_model(samples, truth.mass, settings)
    assert np.array_equal(first.encoder, second.encoder)
    assert np.array_equal(first.closure.generator(), second.closure.generator())
    assert first.fit_record["hidden_targets_read"] is False
    assert (
        first.fit_record["loss_trace"][-1]["train_objective"]
        < first.fit_record["loss_trace"][0]["train_objective"]
    )
    state = first.predict(samples[0].history, samples[0].forecast_time)[-1]
    assert first.closure.energy_identity(state, np.zeros(3))["identity_error"] < 1e-12
    path = tmp_path / "learned.npz"
    first.save(path)
    restored = LearnedLatentModel.load(path)
    assert np.array_equal(
        first.predict(samples[0].history, samples[0].forecast_time),
        restored.predict(samples[0].history, samples[0].forecast_time),
    )
    with pytest.raises(FileExistsError):
        first.save(path)


def test_refit_keeps_train_normalization_and_rejects_line_overlap() -> None:
    truth, samples = _samples()
    train, val = samples[:2], tuple(replace(sample, role="val") for sample in samples[2:])
    settings = LatentFitSettings(latent_dimension=0, epochs=2)
    initial = fit_latent_model(train, truth.mass, settings)
    final = refit_latent_model(train, val, truth.mass, settings, initial.target_scale)
    assert np.array_equal(initial.feature_scale, final.feature_scale)
    assert np.array_equal(initial.target_scale, final.target_scale)
    assert final.fit_record["fitted_roles"] == ["train", "val"]
    assert final.encoder.shape[0] == 0
    with pytest.raises(ValueError, match="fault lines overlap"):
        refit_latent_model(
            train,
            [
                replace(
                    val[0], source_dataset=train[0].source_dataset, fault_line=train[0].fault_line
                )
            ],
            truth.mass,
            settings,
            initial.target_scale,
        )


def test_encoder_input_contains_only_allowed_history() -> None:
    _, samples = _samples()
    sample = samples[0]
    feature = history_features(sample.history, 5)
    changed_targets = replace(
        sample, target_delta=sample.target_delta + np.r_[0.0, np.ones(20)][:, None]
    )
    assert np.array_equal(feature, history_features(changed_targets.history, 5))
    assert feature.shape == (5 * 2 * 3 + 3,)
