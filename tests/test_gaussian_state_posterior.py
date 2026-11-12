import numpy as np
import pytest

from kron_reduce.evaluation.gaussian_state_posterior import condition_initial_state


def _counting_spend():
    counts = {}

    def spend(operation, amount=1):
        counts[operation] = counts.get(operation, 0) + amount

    return spend, counts


def test_scalar_multiframe_posterior_matches_closed_form_and_budget():
    design = np.array([[[1.0]], [[2.0]], [[-0.5]]])
    residual = np.array([[[0.2], [-0.1], [0.4]], [[-0.3], [0.5], [0.1]]])
    prior = np.array([[2.5]])
    noise = np.array([[0.4]])
    spend, counts = _counting_spend()

    result = condition_initial_state(design, residual, prior, noise, spend=spend)

    precision = 1/prior[0, 0] + np.sum(design[:, 0, 0]**2)/noise[0, 0]
    expected_covariance = np.array([[1/precision]])
    expected_mean = (residual[:, :, 0] @ design[:, 0, 0] / noise[0, 0])[:, None] / precision
    np.testing.assert_allclose(result["precision"], [[precision]], rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(result["covariance"], expected_covariance, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(result["mean"], expected_mean, rtol=1e-13, atol=1e-13)
    assert counts == {"cholesky": 3, "linear_solves": 4, "matrix_products": 2*len(design)+8}
    assert result["factor_reconstruction_error"] <= 1e-12
    assert result["solve_backward_error"] <= 1e-12
    assert result["covariance_symmetry_error"] <= 1e-12


def test_correlated_noise_posterior_matches_weighted_precision_formula():
    design = np.array([
        [[1.0, 0.2], [-0.3, 0.8]],
        [[0.5, -0.7], [1.1, 0.4]],
        [[-0.2, 1.2], [0.6, -0.1]],
    ])
    residual = np.array([
        [[0.2, -0.5], [0.1, 0.7], [-0.3, 0.4]],
        [[-0.6, 0.3], [0.8, -0.2], [0.5, 0.1]],
    ])
    prior = np.array([[1.4, 0.25], [0.25, 0.8]])
    noise = np.array([[0.5, 0.12], [0.12, 0.35]])
    spend, counts = _counting_spend()

    result = condition_initial_state(design, residual, prior, noise, spend=spend)

    prior_precision = np.linalg.solve(prior, np.eye(2))
    noise_precision = np.linalg.solve(noise, np.eye(2))
    expected_precision = prior_precision + sum(
        frame.T @ noise_precision @ frame for frame in design
    )
    expected_rhs = np.stack([
        sum(design[t].T @ noise_precision @ residual[event, t] for t in range(len(design)))
        for event in range(len(residual))
    ], axis=1)
    expected_covariance = np.linalg.solve(expected_precision, np.eye(2))
    expected_mean = np.linalg.solve(expected_precision, expected_rhs).T

    np.testing.assert_allclose(result["precision"], expected_precision, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(result["rhs"], expected_rhs, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(result["covariance"], expected_covariance, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(result["mean"], expected_mean, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(result["covariance"], result["covariance"].T, atol=1e-14)
    np.testing.assert_allclose(result["prior_cholesky"] @ result["prior_cholesky"].T,
                               prior, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(result["noise_cholesky"] @ result["noise_cholesky"].T,
                               noise, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(result["precision_cholesky"] @ result["precision_cholesky"].T,
                               expected_precision, rtol=1e-13, atol=1e-13)
    assert counts == {"cholesky": 3, "linear_solves": 4, "matrix_products": 2*len(design)+8}
    assert result["factor_reconstruction_error"] <= 1e-12
    assert result["solve_backward_error"] <= 1e-12
    assert result["covariance_symmetry_error"] <= 1e-12


@pytest.mark.parametrize("problem", ["design_shape", "residual_shape", "prior_shape", "noise_shape"])
def test_posterior_rejects_invalid_shapes(problem):
    design = np.ones((2, 2, 1))
    residual = np.ones((3, 2, 2))
    prior = np.eye(1)
    noise = np.eye(2)
    if problem == "design_shape":
        design = np.ones((2, 2))
    elif problem == "residual_shape":
        residual = np.ones((3, 3, 2))
    elif problem == "prior_shape":
        prior = np.eye(2)
    else:
        noise = np.eye(1)

    with pytest.raises(ValueError, match="invalid finite Gaussian observation problem"):
        condition_initial_state(design, residual, prior, noise, spend=lambda *_: None)


@pytest.mark.parametrize("target", ["design", "residual", "prior", "noise"])
def test_posterior_rejects_nonfinite_inputs(target):
    design = np.ones((2, 2, 1))
    residual = np.ones((3, 2, 2))
    prior = np.eye(1)
    noise = np.eye(2)
    arrays = {"design": design, "residual": residual, "prior": prior, "noise": noise}
    arrays[target].flat[0] = np.inf

    with pytest.raises(ValueError, match="invalid finite Gaussian observation problem"):
        condition_initial_state(arrays["design"], arrays["residual"], arrays["prior"],
                                arrays["noise"], spend=lambda *_: None)


@pytest.mark.parametrize("target", ["prior", "noise"])
def test_posterior_rejects_nonsymmetric_covariance_inputs(target):
    design = np.ones((2, 2, 1))
    residual = np.ones((3, 2, 2))
    prior = np.eye(1)
    noise = np.eye(2)
    if target == "prior":
        prior = np.array([[1.0, 0.2], [0.0, 1.0]])
    else:
        noise = np.array([[1.0, 0.2], [0.0, 1.0]])

    with pytest.raises(ValueError, match="invalid finite Gaussian observation problem"):
        condition_initial_state(design, residual, prior, noise, spend=lambda *_: None)


@pytest.mark.parametrize("target", ["prior", "noise"])
def test_posterior_rejects_non_positive_definite_covariance_inputs(target):
    design = np.ones((2, 2, 1))
    residual = np.ones((3, 2, 2))
    prior = np.array([[1.0]])
    noise = np.eye(2)
    if target == "prior":
        prior[0, 0] = -0.2
    else:
        noise = np.array([[1.0, 1.5], [1.5, 1.0]])

    with pytest.raises(np.linalg.LinAlgError):
        condition_initial_state(design, residual, prior, noise, spend=lambda *_: None)
