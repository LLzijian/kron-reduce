import numpy as np
import pytest

from kron_reduce.models.conditional_encoder import (
    causal_affine_maps,
    solve_conditional_encoder,
)
from kron_reduce.models.machine_local import DiagonalMemoryModel
from kron_reduce.models.nonlinear_latent import NonlinearPassiveClosure, complete_incidence


def _problem(seed=19):
    rng = np.random.default_rng(seed)
    n, channels, latent, features = 4, 3, 2, 3
    f = rng.normal(size=(n, features))
    d = rng.normal(size=(n, channels, latent))
    y = rng.normal(size=(n, n, channels))
    weights = np.array([0.1, 0.2, 0.3, 0.4])
    scale = np.array([0.7, 1.4, 2.0])
    return f, y, d, weights, scale


def _direct_weighted_ridge(f, y, d, weights, scale, penalty):
    count, channels = len(f), d.shape[1]
    design, target, weight = [], [], []
    for b in range(count):
        for t in range(count):
            for channel in range(channels):
                design.append(np.kron(d[t, channel] / scale[channel], f[b]))
                target.append(y[b, t, channel] / scale[channel])
                weight.append(weights[t] / (count * channels))
    design, target, weight = np.asarray(design), np.asarray(target), np.asarray(weight)
    augmented = np.vstack((design * np.sqrt(weight[:, None]),
                           np.sqrt(penalty) * np.eye(design.shape[1])))
    augmented_target = np.r_[target * np.sqrt(weight), np.zeros(design.shape[1])]
    return np.linalg.lstsq(augmented, augmented_target, rcond=None)[0].reshape(
        d.shape[2], f.shape[1]
    )


def test_conditional_encoder_matches_direct_weighted_ridge_and_normalization():
    f, y, d, weights, scale = _problem()
    penalty = 0.17
    spent = []
    result = solve_conditional_encoder(
        f, y, d, weights, scale, penalty=penalty,
        spend=lambda operation, amount=1: spent.append((operation, amount)),
    )
    direct = _direct_weighted_ridge(f, y, d, weights, scale, penalty)

    np.testing.assert_allclose(result["encoder"], direct, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(result["system"] @ result["encoder"].reshape(-1),
                               result["rhs"].reshape(-1), atol=1e-12)
    np.testing.assert_allclose(result["feature_gram"], f.T @ f / len(f), atol=1e-14)
    assert spent == [("matrix_products", 4), ("linear_solves", 1), ("matrix_products", 1)]


def test_conditional_encoder_rank_deficient_features_remain_finite_with_positive_penalty():
    f, y, d, weights, scale = _problem()
    f[:, 1] = f[:, 0]
    result = solve_conditional_encoder(
        f, y, d, weights, scale, penalty=0.03, spend=lambda *_: None,
    )
    assert np.isfinite(result["encoder"]).all()
    assert np.isfinite(result["optimality_gap_upper_bound"])
    assert np.linalg.norm(result["stationarity_residual"]) < 1e-12


@pytest.mark.parametrize("change", [
    lambda f, y, d, w, s: (f[:-1], y, d, w, s),
    lambda f, y, d, w, s: (f, y, d, w * 2, s),
    lambda f, y, d, w, s: (f, y, d, w, np.zeros_like(s)),
])
def test_conditional_encoder_rejects_malformed_train_arrays(change):
    args = _problem()
    with pytest.raises(ValueError, match="train-only"):
        solve_conditional_encoder(
            *change(*args), penalty=0.1, spend=lambda *_: None,
        )


def _linear_model(seed=5):
    n = 3
    incidence = complete_incidence(n)
    mass = np.array([1.0, 1.3, 1.7])
    damping = 0.06 * incidence @ incidence.T
    backbone = NonlinearPassiveClosure(
        mass, np.array([0.25, 0.31, 0.22]), damping,
        np.zeros((n, 0)), np.zeros((0, 0)), np.zeros((0, 0)), "linear",
    )
    rng = np.random.default_rng(seed)
    return DiagonalMemoryModel(
        backbone, port_kind="local", port=np.array([0.2, 0.3, 0.4]),
        tau=np.array([0.12, 0.3, 0.8]),
        encoder=rng.normal(scale=0.05, size=(n, 8*n)),
        feature_scale=np.linspace(0.8, 1.2, 8*n),
    )


def test_causal_affine_maps_reconstruct_core_history_reset_and_future():
    model = _linear_model()
    rng = np.random.default_rng(23)
    q = np.zeros((2, 11, 3))
    v = np.zeros_like(q)
    power = np.zeros((2, 3))
    equilibrium = np.zeros_like(power)
    features = rng.normal(scale=0.1, size=(2, 24))
    count = 7
    h0, f0, hm, fm = causal_affine_maps(
        model, q, v, power, equilibrium, count, max_step=0.01,
        spend=lambda *_: None,
    )
    z0 = (features / model.feature_scale.detach().numpy()) @ model.encoder.detach().numpy().T
    expected_history = h0 + np.einsum("tsr,br->bts", hm, z0)
    expected_future = f0 + np.einsum("tsr,br->bts", fm, z0)

    actual_history, actual_future = model.causal_rollout(
        q, v, power, equilibrium, features, count, max_step=0.01,
        return_history=True,
    )
    actual_history, actual_future = actual_history.detach().numpy(), actual_future.detach().numpy()
    np.testing.assert_allclose(actual_history, expected_history, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(actual_future, expected_future, rtol=2e-12, atol=2e-12)
    np.testing.assert_array_equal(actual_future[:, 0, :6], np.zeros((2, 6)))
    np.testing.assert_array_equal(actual_future[:, 0, 6:], actual_history[:, -1, 6:])


def test_conditional_encoder_accepts_train_arrays_only():
    f, y, d, weights, scale = _problem()
    with pytest.raises(TypeError):
        solve_conditional_encoder(
            f, y, d, weights, scale, penalty=0.1,
            spend=lambda *_: None, eval_features=f,
        )
