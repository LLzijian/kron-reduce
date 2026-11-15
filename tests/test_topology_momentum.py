import numpy as np

from kron_reduce.evaluation.topology_momentum import memory_integrals, project_residual


def _phi2(x):
    return (np.expm1(x) - x) / x**2


def _phi3(x):
    return (np.expm1(x) - x - x**2 / 2) / x**3


def test_memory_integrals_for_constant_piecewise_linear_drive():
    step, rate, drive = .2, .5, 1.4
    velocity = np.full((4, 1), 2.0, dtype=np.float64)
    coupling = np.array([[.7]], dtype=np.float64)
    free, driven = memory_integrals(velocity, coupling, np.array([rate]), step)

    interval = np.arange(3) * step
    expected_free = -np.expm1(-rate * step) / rate
    expected_driven = drive / rate * (
        step - (np.exp(-rate * interval) - np.exp(-rate * (interval + step))) / rate
    )
    assert free.shape == driven.shape == (3, 1)
    assert np.allclose(free[:, 0], expected_free * np.exp(-rate * interval), atol=1e-14, rtol=0)
    assert np.allclose(driven[:, 0], expected_driven, atol=1e-14, rtol=0)


def test_memory_integral_for_linear_drive_matches_closed_form():
    step, rate = .3, 1.7
    velocity = np.array([[.25], [1.15]], dtype=np.float64)
    coupling = np.array([[2.0]], dtype=np.float64)
    free, driven = memory_integrals(velocity, coupling, np.array([rate]), step)

    initial_drive = velocity[0, 0] * coupling[0, 0]
    drive_slope = ((velocity[1, 0] - velocity[0, 0]) / step) * coupling[0, 0]
    x = -rate * step
    expected = initial_drive * step**2 * _phi2(x) + drive_slope * step**3 * _phi3(x)
    assert free.shape == driven.shape == (1, 1)
    np.testing.assert_allclose(free[0, 0], -np.expm1(x) / rate, atol=1e-14, rtol=0)
    np.testing.assert_allclose(driven[0, 0], expected, atol=1e-14, rtol=0)


def test_project_residual_reports_minimum_norm_solution_when_rank_deficient():
    operator = np.array([[1., 2.], [2., 4.], [0., 0.]], dtype=np.float64)
    rhs = np.array([1., 0., 1.], dtype=np.float64)

    result = project_residual(operator, rhs)

    assert result["rank"] == 1
    np.testing.assert_allclose(result["singular"], [5., 0.], atol=1e-14, rtol=0)
    np.testing.assert_allclose(result["state"], [.04, .08], atol=1e-14, rtol=0)
    np.testing.assert_allclose(result["perpendicular"], rhs - operator @ result["state"], atol=1e-14, rtol=0)
    np.testing.assert_allclose(operator.T @ result["perpendicular"], 0., atol=1e-14, rtol=0)


def test_project_residual_is_orthogonal_for_full_rank_operator():
    operator = np.array([[1., 0.], [0., 2.], [1., 1.], [2., -1.]], dtype=np.float64)
    rhs = np.array([2., -1., .5, 3.], dtype=np.float64)

    result = project_residual(operator, rhs)

    assert result["rank"] == 2
    np.testing.assert_allclose(result["perpendicular"], rhs - operator @ result["state"], atol=1e-13, rtol=0)
    np.testing.assert_allclose(operator.T @ result["perpendicular"], 0., atol=1e-13, rtol=0)
