"""Synthetic-only evidence for A17 pure-array process composition."""
from collections import Counter

import numpy as np
import pytest

from kron_reduce.evaluation import input_consistency_paths as paths
from kron_reduce.models.nonlinear_latent import NonlinearPassiveClosure
from kron_reduce.physics.coordinates import coi_project


class Dispatch:
    def __init__(self, fail=None):
        self.calls = []
        self.counts = Counter()
        self.fail = fail

    def __call__(self, category, function):
        self.calls.append(category)
        self.counts[category] += 1
        value = function()
        if self.fail == (category, self.counts[category]):
            error = OSError("synthetic terminal journal failure")
            error.partial_record = {"original_dispatch_evidence": True}
            raise error
        return value


def integral_fixture():
    U = np.zeros((100, 2))
    U[0, 0] = U[1, 1] = 1.
    singular = np.array([3., 2.])
    Vh = np.array([[0., -1.], [1., 0.]])
    B = (U * singular) @ Vh
    z = np.array([.2, -.1])
    old_rhs = B @ z
    saved_z = Vh.T @ ((U.T @ old_rhs) / singular)
    PhiT, wT = np.array([[.9, .1], [-.1, .9]]), np.array([.01, -.02])
    time = np.linspace(0., .1, 11)
    integral = {
        "Ltilde": B, "U": U, "s": singular, "Vh": Vh, "btilde4": old_rhs,
        "c": np.ones(10), "S": np.ones(10), "h": np.diff(time), "time": time,
        "zh2": saved_z.copy(), "zh4": saved_z.copy(), "zTobs2": PhiT @ saved_z + wT,
        "zTobs4": PhiT @ saved_z + wT, "PhiT": PhiT, "wT": wT,
    }
    args = {"rank": 2, "threshold": float(64 * np.finfo(float).eps * 100 * singular[0]),
            "natural_norm": 0., "determinant_lower": 36., "trace_upper": 13.}
    return integral, B @ np.array([.03, -.04]), args


def assert_complete_caps(record, caps):
    assert record["status"] == "complete"
    assert record["success_caps"] == caps
    for category, expected in caps.items():
        counts = record["call_counts"].get(category, {})
        for field in ("dispatch_attempted", "callback_attempted", "callback_returned", "dispatch_returned"):
            assert counts.get(field, 0) == expected
        assert counts.get("callback_failed", 0) == counts.get("dispatch_failed", 0) == 0
        assert counts.get("dispatch_contract_failed", 0) == 0


def model_fixture():
    mass = np.linspace(.1, 1.3, 10)
    projector = np.eye(10) - np.ones((10, 10)) / 10
    coupling = np.column_stack((np.arange(10) - 4.5, (-1.) ** np.arange(10))) * .01
    return NonlinearPassiveClosure(
        mass, np.full(45, .04), .03 * projector, coupling,
        np.array([[0., .6], [-.6, 0.]]), np.diag([.2, .4]))


def path_args(model, *, linear=False):
    clock = np.linspace(0, .1, 11) if linear else np.array([0., .01, .02])
    pattern = np.arange(10, dtype=float) - 4.5
    return {
        "closure": model, "q": coi_project(np.linspace(-.01, .01, 10), model.mass),
        "v": coi_project(np.sin(np.arange(10)) * .01, model.mass), "z": np.array([.2, -.1]),
        "clock": clock,
        "power": np.arange(11)[:, None] * pattern * .001 if linear else pattern * .001,
        "equilibrium": np.linspace(-.05, .05, 10), "max_step": .005, "linear": linear,
        "target_scale": np.ones(20),
    }


def test_initialization_order_exact_solution_no_new_factorizations(monkeypatch):
    integral, impulse, args = integral_fixture()
    snapshot = {name: value.copy() for name, value in integral.items()}
    for method in ("svd", "solve", "lstsq", "pinv", "inv", "qr", "eig", "eigvalsh"):
        monkeypatch.setattr(np.linalg, method, lambda *a, **k: pytest.fail("new factorization or solver"))
    dispatch = Dispatch()
    record = paths.estimate_initial(integral, impulse, dispatch=dispatch, **args)
    assert dispatch.calls == ["initializers", "decomposition_check", "ls", "ls_diagnostics", "quadrature_proxy"]
    assert_complete_caps(record, paths.initializer_success_caps())
    expected_rhs = integral["btilde4"] + impulse
    projected = integral["U"].T @ expected_rhs
    coefficients = projected / integral["s"]
    assert np.array_equal(record["ls"]["rhs"], expected_rhs)
    assert np.array_equal(record["ls"]["projected_rhs"], projected)
    assert np.array_equal(record["ls"]["coefficients"], coefficients)
    assert np.array_equal(record["z0"], integral["Vh"].T @ coefficients)
    assert np.allclose(record["z0"], [.23, -.14], rtol=0, atol=1e-16)
    assert record["derived_actual"] == {"observation_map_attempted": 1, "observation_map_returned": 1}
    assert record["diagnostic_operation_counts"]["decomposition_check"]["matmul_returned"] == 3
    assert record["diagnostic_operation_counts"]["ls_diagnostics"]["vector_norms_returned"] == 3
    assert record["diagnostic_operation_counts"]["quadrature_proxy"]["matmul_returned"] == 1
    assert all(np.array_equal(integral[key], old) for key, old in snapshot.items())


def test_zero_impulse_retains_old_solution_without_second_solve():
    integral, _, args = integral_fixture()
    dispatch = Dispatch()
    record = paths.estimate_initial(integral, np.zeros(100), dispatch=dispatch, **args)
    assert np.array_equal(record["z0"], integral["zh4"])
    assert dispatch.counts["ls"] == 1


@pytest.mark.parametrize("bad", ["rank", "threshold", "singular_order", "shape", "missing_member", "complex", "nonfinite"])
def test_source_contract_failures_are_value_errors_before_diagnostics_or_ls(bad):
    integral, impulse, args = integral_fixture()
    if bad == "rank":
        args["rank"] = True
    elif bad == "threshold":
        args["threshold"] = np.nextafter(args["threshold"], np.inf)
    elif bad == "singular_order":
        integral["s"] = integral["s"][::-1]
    elif bad == "shape":
        integral["wT"] = np.zeros(3)
    elif bad == "missing_member":
        del integral["time"]
    elif bad == "complex":
        integral["btilde4"] = integral["btilde4"].astype(complex) + 1j
    else:
        impulse[0] = np.nan
    dispatch = Dispatch()
    with pytest.raises(ValueError) as caught:
        paths.estimate_initial(integral, impulse, dispatch=dispatch, **args)
    assert dispatch.calls == ["initializers"]
    assert caught.value.input_consistency_paths_partial_record["status"] == "global_failure"


@pytest.mark.parametrize("gate", ["decomposition_check", "ls_diagnostics", "quadrature_proxy"])
def test_failed_numeric_gate_keeps_returned_checks_and_stops_downstream(gate):
    integral, impulse, args = integral_fixture()
    if gate == "decomposition_check":
        integral["Ltilde"][3, 0] = .01
    elif gate == "ls_diagnostics":
        args["determinant_lower"] = 1e-15
    else:
        integral["zh2"] += .1
    dispatch = Dispatch()
    with pytest.raises(FloatingPointError) as caught:
        paths.estimate_initial(integral, impulse, dispatch=dispatch, **args)
    record = caught.value.input_consistency_paths_partial_record
    assert record["checks"][gate]["qualified"] is False
    assert record["status"] == "numerical_failure"
    assert dispatch.calls[-1] == gate
    assert dispatch.counts["ls"] == (gate != "decomposition_check")
    assert record["derived_actual"]["observation_map_returned"] == (gate == "quadrature_proxy")
    if gate != "decomposition_check":
        assert "z0" in record["ls"]


@pytest.mark.parametrize("category", ["decomposition_check", "ls", "ls_diagnostics", "quadrature_proxy", "initializers"])
def test_initializer_terminal_failure_retains_actual_return_without_recovery(category):
    integral, impulse, args = integral_fixture()
    dispatch = Dispatch((category, 1))
    with pytest.raises(OSError) as caught:
        paths.estimate_initial(integral, impulse, dispatch=dispatch, **args)
    error = caught.value
    assert error.partial_record == {"original_dispatch_evidence": True}
    record = error.input_consistency_paths_partial_record
    assert record["call_counts"][category]["callback_returned"] == 1
    assert record["call_counts"][category]["dispatch_returned"] == 0
    if category == "ls":
        assert set(record["kernel_partial"]["intermediates"]) == {"rhs", "projected_rhs", "coefficients", "z0"}
    elif category in record["checks"]:
        assert record["checks"][category]["qualified"] is True
    else:
        assert "z0" in record
    assert dispatch.counts["ls"] <= 1


@pytest.mark.parametrize("linear", [False, True])
@pytest.mark.parametrize("step", [.005, .0025])
def test_path_complete_counts_physical_inputs_and_no_extra_constructors(linear, step, monkeypatch):
    model = model_fixture()
    args = path_args(model, linear=linear)
    args["max_step"] = step
    reference = None
    if not linear:
        reference = model.rollout(args["q"], args["v"], args["z"], args["clock"],
                                  args["power"], args["equilibrium"], max_step=step)
    expected_updates = (20 if linear else 4) * (2 if step == .0025 else 1)
    seen_inputs, actual_rhs = [], []

    class Recording:
        n_nodes, latent_dimension, mass = 10, 2, model.mass

        def rhs(self, state, power, equilibrium):
            actual_rhs.append(state.copy())
            return model.rhs(state, power, equilibrium)

        def energy_identity(self, state, power, equilibrium):
            seen_inputs.append(power.copy())
            return model.energy_identity(state, power, equilibrium)

    args["closure"] = Recording()
    monkeypatch.setattr(np.linalg, "eigvalsh", lambda *a, **k: pytest.fail("new structural check"))
    dispatch = Dispatch()
    record = paths.evaluate_fixed_path(dispatch=dispatch, **args)
    assert record["derived_actual"]["successful_rk4_updates"] == expected_updates
    caps = paths.path_success_caps(point_count=len(args["clock"]), rk4_updates=expected_updates, linear=linear)
    assert_complete_caps(record, caps)
    assert len(actual_rhs) == 4 * expected_updates
    assert all(np.array_equal(value, args["power"][i] if linear else args["power"])
               for i, value in enumerate(seen_inputs))
    assert record["checks"]["physical_judgement"]["qualified"] is True
    assert np.array_equal(record["path"][0], np.r_[args["q"], args["v"], args["z"]])
    if reference is not None:
        assert np.array_equal(record["path"], reference)


@pytest.mark.parametrize("category,index", [("path_preflight", 1), ("input_eval", 3), ("integration_rhs", 6),
                                             ("rollouts", 1), ("physical_rhs", 2),
                                             ("physical_judgement", 1), ("path_evaluations", 1)])
def test_path_terminal_failure_retains_every_completed_phase(category, index):
    args = path_args(model_fixture(), linear=True)
    dispatch = Dispatch((category, index))
    with pytest.raises(OSError) as caught:
        paths.evaluate_fixed_path(dispatch=dispatch, **args)
    record = caught.value.input_consistency_paths_partial_record
    assert caught.value.partial_record == {"original_dispatch_evidence": True}
    counts = record["call_counts"][category]
    assert counts["callback_returned"] == index and counts["dispatch_returned"] == index - 1
    if category == "path_preflight":
        assert record["kernel_partial"]["phase"] == "preflight"
        assert "returned" in record["kernel_partial"]
    elif category in {"input_eval", "integration_rhs"}:
        current = record["kernel_partial"]["current"]
        assert current["stages"][current["stage"]]["power"].shape == (10,)
        if category == "integration_rhs":
            assert current["stages"][current["stage"]]["returned_rhs"].shape == (22,)
            assert record["derived_actual"]["successful_rk4_updates"] == 1
    else:
        assert record["path"].shape == (11, 22)
        assert record["derived_actual"]["successful_rk4_updates"] == 20
    if category == "physical_rhs":
        assert len(record["physical_points"]) == 2
        assert "energy_identity_return" in record["physical_points"][1]


def test_physical_numeric_gate_retains_complete_path_and_checks():
    model = model_fixture()

    class BadPhysical:
        n_nodes, latent_dimension, mass = 10, 2, model.mass
        rhs = model.rhs

        def energy_identity(self, *args):
            return {"energy_rate": 2e-8, "supply_minus_dissipation": 0., "identity_error": 2e-8}

    dispatch = Dispatch()
    with pytest.raises(FloatingPointError) as caught:
        paths.evaluate_fixed_path(dispatch=dispatch, **path_args(BadPhysical()))
    record = caught.value.input_consistency_paths_partial_record
    assert record["path"].shape == (3, 22)
    assert len(record["physical_points"]) == 3
    assert record["checks"]["physical_judgement"]["qualified"] is False
    assert dispatch.counts["rollouts"] == 1


def test_physical_exception_preserves_previous_records_and_no_repeat():
    model = model_fixture()
    calls = []

    class FailingPhysical:
        n_nodes, latent_dimension, mass = 10, 2, model.mass
        rhs = model.rhs

        def energy_identity(self, *args):
            calls.append(1)
            if len(calls) == 2:
                raise FloatingPointError("synthetic energy overflow")
            return model.energy_identity(*args)

    dispatch = Dispatch()
    with pytest.raises(FloatingPointError) as caught:
        paths.evaluate_fixed_path(dispatch=dispatch, **path_args(FailingPhysical()))
    record = caught.value.input_consistency_paths_partial_record
    assert len(calls) == 2 and dispatch.counts["physical_judgement"] == 0
    assert "energy_identity_return" in record["physical_points"][0]
    assert record["physical_points"][1] == {"index": 1}
    assert record["path"].shape == (3, 22)


@pytest.mark.parametrize("bad", ["qshape", "complex", "scale", "mode", "step", "coi"])
def test_path_contract_failure_is_global_and_calls_no_rhs(bad):
    args = path_args(model_fixture())
    if bad == "qshape":
        args["q"] = np.zeros(9)
    elif bad == "complex":
        args["q"] = args["q"].astype(complex) + 1j
    elif bad == "scale":
        args["target_scale"][0] = 0.
    elif bad == "mode":
        args["linear"] = 1
    elif bad == "step":
        args["max_step"] = .001
    else:
        args["q"] = np.ones(10)
    dispatch = Dispatch()
    with pytest.raises(ValueError) as caught:
        paths.evaluate_fixed_path(dispatch=dispatch, **args)
    assert dispatch.counts["integration_rhs"] == dispatch.counts["physical_rhs"] == 0
    assert caught.value.input_consistency_paths_partial_record["status"] == "global_failure"


def test_reset_keeps_each_history_hidden_endpoint_and_only_two_energy_calls():
    model = model_fixture()
    observed = np.zeros(20)
    origins = []
    for hidden in ([.2, -.1], [.21, -.11]):
        path = np.zeros((11, 22))
        path[-1, 20:] = hidden
        dispatch = Dispatch()
        record = paths.reset_path_origin(model, path, observed, np.zeros(10), dispatch=dispatch)
        assert_complete_caps(record, paths.reset_success_caps())
        reset = record["reset"]
        assert np.array_equal(reset["origin"][20:], hidden)
        assert reset["reset_energy_change"] == reset["energy_after"] - reset["energy_before"]
        origins.append(reset["origin"])
    assert not np.array_equal(origins[0][20:], origins[1][20:])


@pytest.mark.parametrize("category,index", [("energy_values", 1), ("energy_values", 2), ("visible_resets", 1)])
def test_reset_terminal_failure_preserves_energy_prefix(category, index):
    path = np.zeros((11, 22))
    path[-1, 20:] = [.2, -.1]
    dispatch = Dispatch((category, index))
    with pytest.raises(OSError) as caught:
        paths.reset_path_origin(model_fixture(), path, np.zeros(20), np.zeros(10), dispatch=dispatch)
    record = caught.value.input_consistency_paths_partial_record
    assert caught.value.partial_record == {"original_dispatch_evidence": True}
    if category == "energy_values":
        assert "energy_before" in record["kernel_partial"]
        assert ("energy_after" in record["kernel_partial"]) == (index == 2)
    else:
        assert "reset_energy_change" in record["reset"]
    assert dispatch.counts["energy_values"] == (index if category == "energy_values" else 2)


def test_dispatch_cannot_silently_skip_or_repeat_real_process():
    integral, impulse, args = integral_fixture()
    with pytest.raises(RuntimeError, match="actual single"):
        paths.estimate_initial(integral, impulse, dispatch=lambda category, fn: None, **args)
    calls = Counter()

    def repeat(category, function):
        calls[category] += 1
        value = function()
        if category == "ls":
            function()
        return value

    with pytest.raises(RuntimeError, match="twice") as caught:
        paths.estimate_initial(integral, impulse, dispatch=repeat, **args)
    assert caught.value.input_consistency_paths_partial_record["call_counts"]["ls"]["callback_returned"] == 1
    assert calls["ls_diagnostics"] == 0


@pytest.mark.parametrize("where", ["before", "after"])
def test_dispatcher_numerical_exception_is_global_with_original_chain(where):
    integral, impulse, args = integral_fixture()
    original = FloatingPointError("synthetic dispatcher bug, not a numerical result")
    original.partial_record = {"journal_evidence": 7}

    def dispatch(category, function):
        if category == "ls" and where == "before":
            raise original
        value = function()
        if category == "ls" and where == "after":
            raise original
        return value

    with pytest.raises(RuntimeError, match="outside the numerical callback") as caught:
        paths.estimate_initial(integral, impulse, dispatch=dispatch, **args)
    error = caught.value
    assert error.__cause__ is original
    assert error.partial_record is original.partial_record
    record = error.input_consistency_paths_partial_record
    assert record["status"] == "global_failure"
    assert record["call_counts"]["ls"]["callback_returned"] == (where == "after")
    assert "ls_diagnostics" not in record["call_counts"]


def test_actual_ls_overflow_is_local_and_retains_all_intermediates():
    integral, impulse, args = integral_fixture()
    integral["btilde4"][:] = 1e308
    impulse[:] = 1e308
    dispatch = Dispatch()
    with np.errstate(over="ignore", invalid="ignore"), pytest.raises(FloatingPointError) as caught:
        paths.estimate_initial(integral, impulse, dispatch=dispatch, **args)
    record = caught.value.input_consistency_paths_partial_record
    assert record["status"] == "numerical_failure"
    assert set(record["kernel_partial"]["intermediates"]) == {"rhs", "projected_rhs", "coefficients", "z0"}
    assert not np.isfinite(record["kernel_partial"]["intermediates"]["rhs"]).all()
    assert record["call_counts"]["ls"]["callback_failed"] == 1
    assert dispatch.counts["ls_diagnostics"] == 0


@pytest.mark.parametrize("linear,clock", [(True, np.arange(11) * .02),
                                          (False, np.array([0., 10.])),
                                          (False, np.array([0., .02]))])
def test_clock_with_unregistered_update_count_is_rejected_before_rhs(linear, clock):
    args = path_args(model_fixture(), linear=linear)
    args["clock"] = clock
    dispatch = Dispatch()
    with pytest.raises(ValueError, match="frozen") as caught:
        paths.evaluate_fixed_path(dispatch=dispatch, **args)
    assert dispatch.counts["integration_rhs"] == dispatch.counts["input_eval"] == 0
    record = caught.value.input_consistency_paths_partial_record
    assert record["derived_actual"]["successful_rk4_updates"] == 0
    assert "preflight_return" in record


def test_four_returned_rhs_values_are_not_a_successful_rk4_update():
    model = model_fixture()
    rhs_calls = []

    class Diverging:
        n_nodes, latent_dimension, mass = 10, 2, model.mass

        def rhs(self, state, power, equilibrium):
            rhs_calls.append(1)
            return np.full(22, np.nan) if len(rhs_calls) == 4 else np.zeros(22)

    dispatch = Dispatch()
    with pytest.raises(FloatingPointError) as caught:
        paths.evaluate_fixed_path(dispatch=dispatch, **path_args(Diverging()))
    record = caught.value.input_consistency_paths_partial_record
    assert len(rhs_calls) == 4
    assert record["call_counts"]["integration_rhs"]["callback_returned"] == 4
    assert record["derived_actual"]["successful_rk4_updates"] == 0
    assert record["kernel_partial"]["current"]["stage"] == "k4"
    assert dispatch.counts["physical_rhs"] == 0


def test_reset_numeric_overflow_keeps_both_returned_energies():
    class OverflowingEnergy:
        def energy(self, state, equilibrium):
            return 1e308 if state[0] else -1e308

    observed = np.zeros(20)
    observed[0] = 1.
    dispatch = Dispatch()
    with pytest.raises(FloatingPointError) as caught:
        paths.reset_path_origin(OverflowingEnergy(), np.zeros((11, 22)), observed,
                                np.zeros(10), dispatch=dispatch)
    record = caught.value.input_consistency_paths_partial_record
    assert record["status"] == "numerical_failure"
    assert dispatch.counts["energy_values"] == 2
    assert record["kernel_partial"]["energy_before"] == -1e308
    assert record["kernel_partial"]["energy_after"] == 1e308
