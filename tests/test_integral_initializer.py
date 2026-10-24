"""Small synthetic tests for the A14 fixed integral initializer kernel."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from kron_reduce.evaluation import integral_initializer as canonical_kernel

HERE = Path(__file__).resolve().parent
LEGACY_PATH = HERE / "fixtures" / "integral_reference.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def kernels():
    return canonical_kernel, _load("a13_integral_kernel", LEGACY_PATH)


def _fixture():
    n = 10
    mass = np.linspace(1.0, 2.8, n, dtype=np.float64)
    mass /= mass.sum()
    qproj = np.eye(n) - np.outer(np.ones(n), mass)
    # Two independent mass-centered columns make the synthetic integral design
    # identifiable without any source data or trajectory integration.
    coupling = qproj @ np.column_stack((np.linspace(-1, 1, n), np.cos(np.arange(n))))
    damping = 0.03 * np.eye(n) + 0.002 * (np.ones((n, n)) - np.eye(n))
    closure = SimpleNamespace(
        mass=mass,
        edge_stiffness=np.linspace(0.4, 1.4, n * (n - 1) // 2),
        damping=damping,
        coupling=coupling,
        latent_skew=np.array([[0.0, -0.35], [0.35, 0.0]]),
        latent_dissipation=np.diag([0.12, 0.2]),
        backbone="sine",
    )
    # Deliberately negative-origin and nonuniform; the kernel must translate
    # to tau[0] without replacing the interval widths by a nominal clock.
    widths = np.array([.0092, .0105, .0087, .0111, .0098, .0102, .0089, .0113, .0094, .0109])
    time = -float(widths.sum()) + np.r_[0.0, np.cumsum(widths)]
    t = np.arange(11, dtype=np.float64)
    q = np.column_stack([0.05 * np.sin(.2 * t + j * .1) for j in range(n)]) @ qproj.T
    v = np.column_stack([0.03 * np.cos(.17 * t + j * .2) for j in range(n)]) @ qproj.T
    p = qproj @ np.linspace(-.04, .05, n)
    eq = qproj @ np.linspace(-.1, .1, n)
    scales = np.linspace(.7, 1.3, n)
    return closure, q, v, time, p, eq, scales


def _assert_tree_close(left, right, *, atol=1e-12):
    assert set(left) == set(right)
    for key in left:
        a, b = left[key], right[key]
        if isinstance(a, dict) and isinstance(b, dict):
            _assert_tree_close(a, b, atol=atol)
        elif isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            np.testing.assert_allclose(a, b, rtol=0.0, atol=atol, err_msg=key)
        elif isinstance(a, (float, np.floating)) and isinstance(b, (float, np.floating)):
            assert abs(float(a) - float(b)) <= atol, key
        else:
            assert a == b, key


def test_a14_matches_a13_and_preserves_nonuniform_negative_clock(kernels):
    new, old = kernels
    closure, q, v, time, p, eq, scales = _fixture()
    assert time[0] < 0 and time[-1] == pytest.approx(0.0, abs=1e-16)
    assert len(np.unique(np.diff(time))) == 10

    new_ledger = new.Ledger()
    cache = new.prepare_static(closure, scales, owner_key="seed-324", ledger=new_ledger)
    got = new.build(q, v, time, closure, p, eq, cache=cache,
                    owner_key="seed-324", ledger=new_ledger)
    expected = old.build(q, v, time, closure, p, eq, scales, ledger=old.Ledger())

    for key in ("L", "Ltilde", "g2", "g4", "b2", "b4", "zh2", "zh4",
                "zTobs2", "zTobs4", "kin", "dynamic_residual2", "dynamic_residual4"):
        np.testing.assert_allclose(got[key], expected[key], rtol=0.0, atol=1e-12, err_msg=key)
    np.testing.assert_allclose(got["system"]["h"], np.diff(time), rtol=0.0, atol=1e-15)
    assert got["status"] == expected["status"] == "complete"

    snap = new_ledger.snapshot()
    counts = snap.get("counts", snap.get("returned", {}))
    assert counts["expm"] == 10
    assert counts["svd"] == 3  # two cached static factorizations plus this slot's L
    assert counts["ls"] == 2
    assert counts["force"] == 60


def test_static_cache_is_reused_and_bound_to_owner_and_closure(kernels):
    new, _ = kernels
    closure, q, v, time, p, eq, scales = _fixture()
    ledger = new.Ledger()
    cache = new.prepare_static(closure, scales, owner_key="seed-325", ledger=ledger)
    prepared = ledger.snapshot()
    assert prepared.get("counts", prepared.get("returned", {}))["svd"] == 2

    new.build(q, v, time, closure, p, eq, cache=cache,
              owner_key="seed-325", ledger=ledger)
    after_one = ledger.snapshot().get("counts", ledger.snapshot().get("returned", {}))
    assert after_one["svd"] == 3
    new.build(q, v, time, closure, p, eq, cache=cache,
              owner_key="seed-325", ledger=ledger)
    after_two = ledger.snapshot().get("counts", ledger.snapshot().get("returned", {}))
    assert after_two["svd"] == 4  # no repeated static SVDs

    before = ledger.snapshot()
    with pytest.raises((ValueError, RuntimeError)):
        new.build(q, v, time, closure, p, eq, cache=cache,
                  owner_key="seed-326", ledger=ledger)
    assert ledger.snapshot() == before  # reject before the first new operator

    changed = SimpleNamespace(**vars(closure))
    changed.coupling = closure.coupling.copy()
    changed.coupling[0, 0] += 1e-5
    with pytest.raises((ValueError, RuntimeError)):
        new.build(q, v, time, changed, p, eq, cache=cache,
                  owner_key="seed-325", ledger=ledger)
    assert ledger.snapshot() == before


def test_rank_deficient_slot_does_not_return_candidate_or_solve(kernels):
    new, _ = kernels
    closure, q, v, time, p, eq, scales = _fixture()
    closure.coupling = closure.coupling[:, :1] @ np.array([[1.0, 0.0]])
    # Keep the rank-one port direction invariant so this particular L is
    # deficient; a separate test below covers rank-one PC with full-rank L.
    closure.latent_skew = np.zeros((2, 2))
    closure.latent_dissipation = np.zeros((2, 2))
    ledger = new.Ledger()
    cache = new.prepare_static(closure, scales, owner_key="rank-one", ledger=ledger)
    result = new.build(q, v, time, closure, p, eq, cache=cache,
                       owner_key="rank-one", ledger=ledger)
    assert result["status"] == "unresolved"
    for key in ("zh2", "zh4", "zTobs2", "zTobs4", "candidate_zh"):
        assert result[key] is None
    snap = ledger.snapshot()
    counts = snap.get("counts", snap.get("returned", {}))
    assert counts.get("ls", 0) == 0
    assert counts["force"] == 60


def test_rank_one_pc_can_still_have_full_rank_dynamic_integral_design(kernels):
    new, _ = kernels
    closure, q, v, time, p, eq, scales = _fixture()
    closure.coupling = closure.coupling[:, :1] @ np.array([[1.0, 0.0]])
    ledger = new.Ledger()
    cache = new.prepare_static(closure, scales, owner_key="rank-one-port", ledger=ledger)
    result = new.build(q, v, time, closure, p, eq, cache=cache,
                       owner_key="rank-one-port", ledger=ledger)
    assert cache.data["C_svd"]["rank"] == 1
    assert result["system"]["rank"] == 2
    assert result["status"] == "complete"
    assert result["candidate_zh"] is not None
    assert ledger.snapshot()["returned"]["ls"] == 2


def test_ledger_journal_failure_is_fail_closed_before_and_after_operator():
    kernel = canonical_kernel
    calls = []

    def fail_attempt(entry):
        if entry["transition"] == "attempted":
            raise OSError("attempt journal unavailable")

    ledger = kernel.Ledger(event_sink=fail_attempt)
    with pytest.raises(kernel.OperatorJournalError):
        ledger.call("force", lambda: calls.append("first"), kind="probe")
    assert calls == []
    assert ledger.snapshot()["attempted"]["force"] == 1
    assert ledger.snapshot()["returned"].get("force", 0) == 0
    assert ledger.snapshot()["poisoned"] is True
    with pytest.raises(kernel.OperatorJournalError):
        ledger.call("force", lambda: calls.append("retry"), kind="probe")
    assert calls == []

    def fail_return(entry):
        if entry["transition"] == "returned":
            raise OSError("return journal unavailable")

    returned_ledger = kernel.Ledger(event_sink=fail_return)
    with pytest.raises(kernel.OperatorJournalError):
        returned_ledger.call("force", lambda: calls.append("completed"), kind="probe")
    assert calls == ["completed"]
    assert returned_ledger.snapshot()["returned"]["force"] == 1
    assert returned_ledger.snapshot()["poisoned"] is True
    with pytest.raises(kernel.OperatorJournalError):
        returned_ledger.call("force", lambda: calls.append("retry-after-return"), kind="probe")
    assert calls == ["completed"]


def test_operator_budget_refuses_before_callable_and_poisons_ledger():
    kernel = canonical_kernel
    calls = []
    ledger = kernel.Ledger(limits={"expm": 0})
    with pytest.raises(kernel.OperatorBudgetError):
        ledger.call("expm", lambda: calls.append("dispatched"), kind="over-budget")
    assert calls == []
    assert ledger.snapshot()["attempted"].get("expm", 0) == 0
    assert ledger.snapshot()["returned"].get("expm", 0) == 0
    assert ledger.snapshot()["poisoned"] is True
    with pytest.raises(kernel.OperatorJournalError):
        ledger.call("force", lambda: calls.append("after-poison"), kind="blocked")
    assert calls == []


@pytest.mark.parametrize("category", ["expm", "svd", "force", "ls"])
def test_nonfinite_operator_return_is_local_numerical_failure(category, monkeypatch):
    kernel = canonical_kernel
    closure, q, v, time, p, eq, scales = _fixture()
    cache = kernel.prepare_static(closure, scales, owner_key="seed-324", ledger=kernel.Ledger())
    ledger = kernel.Ledger(limits={"expm": 10, "svd": 1, "force": 60, "ls": 2})
    original_call = ledger.call

    def injected_call(operator_category, function, **kwargs):
        def output():
            value = function()
            if operator_category != category:
                return value
            if operator_category == "svd":
                u, singular, vh = value
                return u, np.full_like(singular, np.nan), vh
            return np.full_like(value, np.nan)

        return original_call(operator_category, output, **kwargs)

    monkeypatch.setattr(ledger, "call", injected_call)
    with pytest.raises(kernel.KernelNumericalError) as caught:
        kernel.build(q, v, time, closure, p, eq, cache=cache,
                     owner_key="seed-324", ledger=ledger)
    partial = caught.value.partial_record
    assert partial["status"] == "incomplete_numerical_failure"
    assert partial["static_model_sha256"] == cache.model_sha256
    matrix = partial["matrix_system"]
    if category == "expm":
        assert len(matrix["exponential_returns"]) == 1
        assert np.isnan(matrix["exponential_returns"][0]).all()
        assert matrix["blocks"] == []
    else:
        assert len(matrix["exponential_returns"]) == 10
        assert matrix["L"].shape == matrix["Ltilde"].shape == (100, 2)
    if category == "svd":
        assert np.isnan(matrix["svd_return"]["singular_values"]).all()
    elif category == "force":
        assert len(partial["w_nodes"]) == 2
        assert np.isnan(partial["force_quadrature"][0]["force_returns"][0]).all()
    elif category == "ls":
        rhs = partial["right_hand_sides"]["2"]
        assert rhs["b"].shape == (10, 10) and rhs["btilde"].shape == (100,)
        assert np.isnan(rhs["ls_return"]).all()
        assert sum(len(r["force_returns"]) for r in partial["force_quadrature"]) == 60
    snap = ledger.snapshot()
    assert snap["attempted"][category] == snap["returned"][category] == 1
    assert snap["failed"].get(category, 0) == 0  # the operator returned an invalid value
    assert snap["poisoned"] is False  # no journal or budget failure occurred
    assert issubclass(kernel.KernelNumericalError, FloatingPointError)
    assert not issubclass(kernel.KernelNumericalError, ValueError)

    # A distinct, predeclared slot can use the unchanged static cache. This is
    # not a retry of the failed slot and does not replace its retained ledger.
    next_ledger = kernel.Ledger()
    result = kernel.build(q, v, time, closure, p, eq, cache=cache,
                          owner_key="seed-324", ledger=next_ledger)
    assert result.status == "complete"
    assert next_ledger.snapshot()["returned"] == {"expm": 10, "svd": 1, "force": 60, "ls": 2}


@pytest.mark.parametrize("invalid", ["history", "owner", "model"])
def test_input_and_cache_failures_remain_value_errors_before_operators(invalid):
    kernel = canonical_kernel
    closure, q, v, time, p, eq, scales = _fixture()
    cache = kernel.prepare_static(closure, scales, owner_key="seed-324", ledger=kernel.Ledger())
    owner = "seed-324"
    if invalid == "history":
        q[0, 0] = np.nan
    elif invalid == "owner":
        owner = "seed-325"
    else:
        closure.coupling = closure.coupling.copy()
        closure.coupling[0, 0] += 1e-5
    ledger = kernel.Ledger()
    with pytest.raises(ValueError) as caught:
        kernel.build(q, v, time, closure, p, eq, cache=cache, owner_key=owner, ledger=ledger)
    assert type(caught.value) is ValueError
    assert not hasattr(caught.value, "partial_record")
    assert ledger.snapshot()["entries"] == []


def test_second_ls_failure_preserves_first_solution_and_both_right_hand_sides(monkeypatch):
    kernel = canonical_kernel
    closure, q, v, time, p, eq, scales = _fixture()
    cache = kernel.prepare_static(closure, scales, owner_key="seed-324", ledger=kernel.Ledger())
    ledger = kernel.Ledger()
    original_call = ledger.call
    solve_attempts = 0

    def injected_call(category, function, **kwargs):
        nonlocal solve_attempts
        if category == "ls":
            solve_attempts += 1
            if solve_attempts == 2:
                def fail():
                    raise FloatingPointError("synthetic second solve failure")
                return original_call(category, fail, **kwargs)
        return original_call(category, function, **kwargs)

    monkeypatch.setattr(ledger, "call", injected_call)
    with pytest.raises(FloatingPointError) as caught:
        kernel.build(q, v, time, closure, p, eq, cache=cache,
                     owner_key="seed-324", ledger=ledger)
    partial = caught.value.partial_record
    assert set(partial["right_hand_sides"]) == {"2", "4"}
    assert partial["result"]["zh2"].shape == (2,)
    assert partial["result"]["dynamic_residual2"].shape == (100,)
    assert partial["right_hand_sides"]["4"]["btilde"].shape == (100,)
    assert "ls_return" not in partial["right_hand_sides"]["4"]
    assert "zh4" not in partial["result"]
    snap = ledger.snapshot()
    assert snap["attempted"]["ls"] == 2 and snap["returned"]["ls"] == 1
    assert snap["failed"]["ls"] == 1 and solve_attempts == 2
    assert snap["returned"]["expm"] == 10 and snap["returned"]["force"] == 60


def test_static_second_svd_failure_preserves_first_decomposition_without_replay(monkeypatch):
    kernel = canonical_kernel
    closure, _, _, _, _, _, scales = _fixture()
    ledger = kernel.Ledger(limits={"svd": 2})
    original_call = ledger.call

    def injected_call(category, function, **kwargs):
        def output():
            u, singular, vh = function()
            if kwargs["kind"] == "O":
                singular = np.full_like(singular, np.nan)
            return u, singular, vh
        return original_call(category, output, **kwargs)

    monkeypatch.setattr(ledger, "call", injected_call)
    with pytest.raises(kernel.KernelNumericalError) as caught:
        kernel.prepare_static(closure, scales, owner_key="seed-324", ledger=ledger)
    partial = caught.value.partial_record
    assert partial["PC"].shape == (10, 2) and partial["O"].shape == (20, 2)
    assert partial["C_svd"].rank == 2
    assert np.isnan(partial["O_decomposition"]["svd_return"]["singular_values"]).all()
    snap = ledger.snapshot()
    assert snap["attempted"] == snap["returned"] == {"svd": 2}
    assert snap["poisoned"] is False
    next_ledger = kernel.Ledger(limits={"svd": 2})
    next_cache = kernel.prepare_static(closure, scales, owner_key="seed-325", ledger=next_ledger)
    assert next_cache.owner_key == "seed-325"
    assert next_ledger.snapshot()["returned"] == {"svd": 2}
