import json

import numpy as np
import pytest

from kron_reduce.evaluation.machine_local_topology import make_model, scoring
from kron_reduce.evaluation.machine_local_topology_learning import selected_model
from kron_reduce.evaluation.machine_local_topology_transfer import transfer_cases
from kron_reduce.evaluation.machine_local_topology_truth import parameters
from project_harness.evidence import sha256_file


def _plan():
    return {
        "seed": 123,
        "ids": [f"transfer-{i:02d}" for i in range(24)],
        "cohorts": {"train": 12, "val": 6, "test": 6},
        "speed_rad_s": [10.5, 12.0],
    }


def test_transfer_cases_are_deterministic_balanced_and_physical_at_origin():
    plan = _plan()
    first, second = transfer_cases(plan), transfer_cases(plan)

    assert first["ids"] == plan["ids"]
    assert first["roles"] == [role for role, count in plan["cohorts"].items() for _ in range(count)]
    assert first["excitation"] == [label for count in plan["cohorts"].values()
                                    for _ in range(count // 2) for label in ("low", "high")]
    for key in ("q", "v", "power", "a", "z"):
        np.testing.assert_array_equal(first[key], second[key])
    assert first["ids"] == second["ids"]
    assert first["roles"] == second["roles"]
    assert first["excitation"] == second["excitation"]

    mass = parameters()[0]
    for key in ("q", "v"):
        np.testing.assert_allclose(first[key] @ mass, 0., atol=1e-12, rtol=0.)
    for key in ("power", "a"):
        np.testing.assert_allclose(first[key].sum(axis=1), 0., atol=1e-12, rtol=0.)
    assert np.max(np.abs(first["v"][np.array(first["excitation"]) == "high"])) > 10.5
    assert np.max(np.abs(first["q"])) < np.pi


def test_transfer_cases_require_registered_roster():
    plan = _plan()
    plan["ids"][0] = "unexpected-id"
    try:
        transfer_cases(plan)
    except ValueError as error:
        assert "roster" in str(error)
    else:
        raise AssertionError("unregistered transfer IDs must be rejected")


def test_failed_reference_invalidates_test_trajectory_and_accuracy():
    params = parameters()
    model = make_model(params, "T0M0", "cpu")
    scale = np.ones(20)
    result = scoring(model, {"ids": ["unchecked"], "roles": ["test"]}, np.zeros((2, 1, 801, 20)),
                     np.zeros((2, 1, 11, 20)), np.zeros((1, 811, 20)), np.zeros((1, 10)),
                     np.zeros((1, 10)), params[0], scale, scale,
                     [{"label": "stable", "trigger_index": None, "reference_qualified": False}], True)
    test = result["roles"]["test"]
    assert test["stability"]["accuracy"] is None
    assert test["stability"]["uncovered"] == test["stability"]["numerical_failures"] == 1
    assert test["pooled_nrmse"]["8"] is None


@pytest.mark.parametrize("support", ["T0M0", "T1M0"])
def test_selected_model_loads_hashed_no_memory_equation_and_rejects_mismatch(tmp_path, support):
    equation = tmp_path / "results" / "frozen" / support / "selected-equation.json"
    equation.parent.mkdir(parents=True)
    equation.write_text(json.dumps({"stiffness_scale": 1.02, "damping_scale": .98,
                                    "port_amplitude": [], "tau": []}), encoding="utf-8")
    source = {"run_id": "frozen", "equation_sha256": {support: sha256_file(equation)}}

    model = selected_model(tmp_path, source, support, parameters(), device="cpu")
    assert model.memory_dimension == 0
    assert model.topology_dimension == (9 if support.startswith("T1") else 0)

    source["equation_sha256"][support] = "0" * 64
    with pytest.raises(RuntimeError, match="frozen topology equation differs"):
        selected_model(tmp_path, source, support, parameters(), device="cpu")
