import numpy as np

from kron_reduce.evaluation import topology_modal_momentum as modal_module


def _reference_blocks():
    n = 10
    incidence = np.zeros((n, n - 1), dtype=np.float64)
    for edge in range(n - 1):
        incidence[edge, edge] = 1.0
        incidence[edge + 1, edge] = -1.0
    stiffness = np.arange(1.0, float(n), dtype=np.float64)
    damping = incidence @ np.diag(np.linspace(0.3, 0.9, n - 1)) @ incidence.T
    return (
        np.ones(n, dtype=np.float64),
        incidence,
        stiffness,
        damping,
        np.zeros((n, 0), dtype=np.float64),
        np.zeros(0, dtype=np.float64),
    )


def test_modal_probe_returns_balanced_unit_highest_mode_and_damping_eta():
    blocks = _reference_blocks()
    probe = modal_module.modal_probe(blocks)
    mode = probe["mode"]
    damping = blocks[3]

    assert probe["positive_modes"] == 9
    assert probe["relative_eigengap"] >= 1e-8
    np.testing.assert_allclose(np.linalg.norm(mode), 1.0, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(np.sum(mode), 0.0, rtol=0.0, atol=1e-10)
    assert mode[probe["sign_pivot"]] > 0.0
    np.testing.assert_allclose(probe["eta"], 0.5 * mode @ damping @ mode, rtol=0.0, atol=1e-14)
    assert probe["eta"] > 0.0
    np.testing.assert_allclose(probe["root"] @ probe["root"], damping, rtol=1e-9, atol=1e-11)


def test_modal_residuals_add_drag_and_score_future_only_with_one_shared_svd(monkeypatch):
    blocks = _reference_blocks()
    probe = modal_module.modal_probe(blocks)
    n = 10
    intervals = 13
    future_intervals = intervals - 10
    path = np.zeros((intervals + 1, 2 * n), dtype=np.float64)
    path[:, :n] = np.arange(intervals + 1, dtype=np.float64)[:, None] * 0.002
    path[:, n:] = np.arange(intervals + 1, dtype=np.float64)[:, None] * probe["mode"][None, :] * 0.03
    power = np.zeros(n, dtype=np.float64)
    equilibrium = np.zeros(n, dtype=np.float64)

    operator = np.zeros((intervals * n, n), dtype=np.float64)
    operator[: 10 * n] = np.tile(np.eye(n), (10, 1))
    operator[10 * n :, 0] = 1.0
    rhs = np.full(intervals * n, 1000.0, dtype=np.float64)
    rhs[10 * n :] = np.linspace(-0.4, 0.8, future_intervals * n)

    def controlled_momentum_system(*_args):
        return operator.copy(), rhs.copy(), {}

    force_calls = []

    def controlled_force_directions(_path, _blocks, _step, points, _root):
        force_calls.append(points)
        result = np.zeros((intervals * n, 3), dtype=np.float64)
        result[:, 0] = 0.4
        result[:, 1] = -0.15
        result[:, 2] = 0.7
        return result

    monkeypatch.setattr(modal_module, "momentum_system", controlled_momentum_system)
    monkeypatch.setattr(modal_module, "_force_directions", controlled_force_directions)
    svd_inputs = []
    real_svd = np.linalg.svd

    def counted_svd(matrix, *args, **kwargs):
        svd_inputs.append(np.array(matrix, copy=True))
        return real_svd(matrix, *args, **kwargs)

    monkeypatch.setattr(modal_module.np.linalg, "svd", counted_svd)
    operations = {}

    def spend(name, count=1):
        operations[name] = operations.get(name, 0) + count

    starts = np.stack(
        (np.linspace(-0.2, 0.25, n), np.linspace(0.3, -0.15, n))
    )
    step = 1.0
    result = modal_module.modal_residuals(
        path, power, equilibrium, blocks, step, spend, starts, probe=probe
    )

    cutoff = 10 * n
    full_intervals = len(path) - 1
    normalization = full_intervals / future_intervals
    projection = np.eye(n) - blocks[0][:, None] / blocks[0].sum()
    mode = probe["mode"]
    modal_axis = projection @ mode
    modal_velocity = 0.5 * (path[:-1, n:] + path[1:, n:]) @ mode
    modal_rows = (step * probe["eta"] * modal_velocity)[:, None] * modal_axis[None, :]
    row_scale = 1.0 / (blocks[0] * step * np.sqrt(full_intervals))
    modal_direction = (modal_rows * row_scale[None, :]).ravel()[cutoff:]
    old_future = (rhs - 0.4 - (-0.15))[cutoff:]
    new_future = old_future + modal_direction
    future_operator = operator[cutoff:]

    expected = {}
    for label, start in zip(("0.0025", "0.00125"), starts, strict=True):
        old_residual = old_future - future_operator @ start
        new_residual = new_future - future_operator @ start
        expected[label] = [old_residual @ old_residual * normalization,
                           new_residual @ new_residual * normalization]
    unit_future_mode = np.ones(len(old_future), dtype=np.float64)
    unit_future_mode /= np.linalg.norm(unit_future_mode)
    old_free = old_future - unit_future_mode * (unit_future_mode @ old_future)
    new_free = new_future - unit_future_mode * (unit_future_mode @ new_future)
    expected["free"] = [old_free @ old_free * normalization,
                        new_free @ new_free * normalization]

    assert result["rank"] == 1
    assert result["future_intervals"] == future_intervals
    assert result["normalization"] == normalization
    assert result["quadrature_gap"] == {"spatial_anchor": 0.0}
    for key, values in expected.items():
        np.testing.assert_allclose(result["sse"][key], values, rtol=2e-13, atol=2e-13)
    assert len(svd_inputs) == 1
    np.testing.assert_array_equal(svd_inputs[0], future_operator)
    assert force_calls == [4, 8]
    assert operations == {"weak_intervals": full_intervals,
                          "quad_check_intervals": full_intervals,
                          "svd": 1}
