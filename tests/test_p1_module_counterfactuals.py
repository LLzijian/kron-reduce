from __future__ import annotations

import numpy as np
import pytest

from kron_reduce.models import (
    EDGE_DAMPING,
    EDGE_MEMORY,
    NODE_HIGH_ORDER,
    NONLINEAR_POTENTIAL,
    RESONANT_MEMORY,
    CandidateBlockInstance,
    PrunableEnergyModel,
)
from kron_reduce.models.p1_calibration import core_fingerprint
from kron_reduce.models.p1_module_counterfactuals import (
    module_physical_traces,
    summarize_physical_trace,
    zero_module_instances,
)
from kron_reduce.physics import coi_project, complete_incidence


def _all_modules():
    factor = ((0.3,), (-0.1,), (0.2,))
    instances = (
        CandidateBlockInstance("potential", NONLINEAR_POTENTIAL),
        CandidateBlockInstance("damper", EDGE_DAMPING),
        CandidateBlockInstance("node", NODE_HIGH_ORDER, time_constant=.2),
        CandidateBlockInstance("memory1", EDGE_MEMORY, time_constant=.2, factor=factor, port_rank=1),
        CandidateBlockInstance("memory2", EDGE_MEMORY, time_constant=.7, factor=factor, port_rank=1),
        CandidateBlockInstance("oscillator", RESONANT_MEMORY, frequency_hz=1., damping_ratio=.1, factor=factor, port_rank=1),
    )
    model = PrunableEnergyModel(block_instances=instances)
    layout = model._parameter_layout(3, 3)
    theta = np.full(layout["all"].stop, .1)
    theta[layout["node_damping"]], theta[layout["coupling"]] = [.2, .3, .4], [.9, 1.1, .8]
    model._assign_parameters(theta, 3, 3)
    mass = np.array([1., 2., 3.])
    static = {"mass": mass, "incidence": complete_incidence(3)[0], "equilibrium_delta": coi_project(np.array([.02, -.01, .03]), mass), "Pm_equilibrium": np.zeros(3), "delta0": coi_project(np.array([.12, -.08, .04]), mass), "omega0": coi_project(np.array([.1, -.03, .02]), mass), "Pm_clear": np.array([.01, -.02, .01])}
    return model, static


def test_whole_zero_preserves_all_other_coefficients_and_removes_both_port_directions():
    model, static = _all_modules()
    before = core_fingerprint(model)
    zero = zero_module_instances(model, ["memory1"])
    assert core_fingerprint(model) == before
    assert zero.active_instances == model.active_instances
    assert zero._memory_factor(0, 3).shape == (3, 0)
    assert np.all(zero.memory_gains[0] == 0)
    assert np.array_equal(zero.memory_gains[1], model.memory_gains[1])
    assert np.array_equal(zero.memory_factors[1], model.memory_factors[1])
    assert np.array_equal(zero.node_damping, model.node_damping)
    assert np.array_equal(zero.coupling, model.coupling)
    full_hidden = model.initial_hidden(static["omega0"], static)
    zero_hidden = zero.initial_hidden(static["omega0"], static)
    assert len(full_hidden["memory"]) == 2 and len(zero_hidden["memory"]) == 1
    time = np.arange(31)*.01
    pm = np.repeat(static["Pm_clear"][None, :], len(time), axis=0)
    full = model.rollout(static, time, {"Pm": pm, "interpolation": "zoh"})
    changed = zero.rollout(static, time, {"Pm": pm, "interpolation": "zoh"})
    assert np.linalg.norm(full[1] - changed[1]) > 1e-5


def test_joint_zero_is_actual_b2_dynamics_not_aggregation_of_module_labels():
    model, static = _all_modules()
    zero = zero_module_instances(model, [row.instance_id for row in model.active_instances])
    b2 = PrunableEnergyModel(active_blocks=())
    b2._assign_parameters(np.r_[model.node_damping, model.coupling], 3, 3)
    time = np.arange(31)*.01
    pm = np.repeat(static["Pm_clear"][None, :], len(time), axis=0)
    actual = zero.rollout(static, time, {"Pm": pm, "interpolation": "zoh"})
    expected = b2.rollout(static, time, {"Pm": pm, "interpolation": "zoh"})
    assert np.array_equal(actual[0], expected[0]) and np.array_equal(actual[1], expected[1])


def test_each_nonzero_module_storage_and_power_match_actual_complete_vector_field():
    model, static = _all_modules()
    time = np.arange(101)*.01
    pm = np.repeat(static["Pm_clear"][None, :], len(time), axis=0)
    trajectory = model.rollout_augmented(static, time, {"Pm": pm, "interpolation": "zoh"})
    traces = module_physical_traces(model, static, *trajectory)
    assert set(traces) == {row.instance_id for row in model.active_instances}
    for name, trace in traces.items():
        assert trace["parameter_norm"] > 0
        assert np.max(trace["identity_error"]) < 1e-8
        assert np.mean(np.abs(trace["port_power"])) > 1e-10
        assert np.min(trace["dissipation"]) >= 0
        if name == "damper":
            assert np.array_equal(trace["storage"], np.zeros(len(time)))
        else:
            assert np.max(trace["storage"]) > 1e-7
    b2 = PrunableEnergyModel(active_blocks=())
    b2._assign_parameters(np.r_[model.node_damping, model.coupling], 3, 3)
    for index in (0, 10, 100):
        delta, omega, hidden = (value[index] for value in trajectory)
        optional = sum(trace["storage"][index] for trace in traces.values())
        assert model.storage(delta, omega, static, hidden) - b2.storage(delta, omega, static) == pytest.approx(optional, abs=1e-12)
        base_acceleration = b2._flat_field(delta, omega, np.zeros(0), static["Pm_clear"], static)[1]
        full_acceleration = model._flat_field(delta, omega, hidden, static["Pm_clear"], static)[1]
        actual_outflow = omega @ (static["mass"]*(base_acceleration - full_acceleration))
        assert actual_outflow == pytest.approx(sum(trace["port_power"][index] for trace in traces.values()), abs=1e-12)
    damper = summarize_physical_trace(traces, ["damper"], 101, .01)
    assert damper["pure_dissipation_protocol_conflict"] is True
    assert damper["storage_mean"] == 0 and damper["dissipation_mean"] > 0
    assert damper["independent_support_verdict"] == "NOT_EVALUATED"


def test_redundant_port_basis_does_not_inflate_physical_parameter_norm():
    model, static = _all_modules()
    equivalent = zero_module_instances(model, [])
    factor = equivalent.memory_factors[0]
    equivalent.memory_factors[0] = np.concatenate([factor/np.sqrt(2), factor/np.sqrt(2)], axis=1)
    time = np.arange(11)*.01
    pm = np.repeat(static["Pm_clear"][None, :], len(time), axis=0)
    traces = []
    for candidate in (model, equivalent):
        trajectory = candidate.rollout_augmented(static, time, {"Pm": pm, "interpolation": "zoh"})
        traces.append(module_physical_traces(candidate, static, *trajectory)["memory1"])
    assert traces[0]["parameter_norm"] == pytest.approx(traces[1]["parameter_norm"], abs=1e-12)
    assert np.allclose(traces[0]["storage"], traces[1]["storage"], atol=1e-12)


@pytest.mark.parametrize("ids", [["absent"], ["memory1", "memory1"]])
def test_unknown_or_partial_name_interventions_reject(ids):
    model, _ = _all_modules()
    with pytest.raises(ValueError, match="whole-instance"):
        zero_module_instances(model, ids)


def test_physical_summary_rejects_finite_inputs_that_overflow_after_normalization():
    traces = {
        "module": {
            "storage": np.ones(2),
            "port_power": np.ones(2),
            "dissipation": np.ones(2),
            "identity_error": np.zeros(2),
            "parameter_norm": 1.0,
            "storage_applicable": True,
        }
    }
    with pytest.raises(FloatingPointError):
        summarize_physical_trace(traces, ["module"], 2, np.nextafter(0.0, 1.0))
