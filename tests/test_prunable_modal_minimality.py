from __future__ import annotations

import json

import numpy as np
import pytest

import kron_reduce.models.prunable_energy_model as prunable_energy_module
from kron_reduce.models import (
    EDGE_MEMORY,
    RESONANT_MEMORY,
    CandidateBlockInstance,
    PrunableEnergyModel,
    RunContext,
)
from kron_reduce.models.prunable_energy_model import _minimal_port_factor


def _static() -> dict[str, np.ndarray]:
    return {
        "mass": np.ones(3),
        "incidence": np.asarray([[1.0, 0.0], [-1.0, 1.0], [0.0, -1.0]]),
        "equilibrium_delta": np.zeros(3),
        "Pm_equilibrium": np.zeros(3),
        "delta0": np.asarray([0.03, -0.01, -0.02]),
        "omega0": np.asarray([0.02, -0.01, -0.01]),
    }


def _ready_model(instance: CandidateBlockInstance | None = None) -> PrunableEnergyModel:
    model = PrunableEnergyModel(block_instances=() if instance is None else (instance,))
    model.node_damping = np.asarray([0.2, 0.25, 0.3])
    model.coupling = np.asarray([1.0, 0.8])
    model.potential_coefficients = np.zeros((3, 2))
    model.edge_damping_coefficients = np.zeros((3, 2))
    model.node_high_order_gain = np.zeros(3)
    model.memory_factors = []
    model.memory_gains = np.zeros((0, 2))
    model.resonant_factors = []
    model.resonant_gains = np.zeros((0, 2))
    return model


def test_zero_edge_port_has_no_hidden_state_and_matches_removed_rollout() -> None:
    instance = CandidateBlockInstance(
        "edge_memory:zero",
        EDGE_MEMORY,
        time_constant=0.2,
        factor=((0.0, 0.0), (0.0, 0.0)),
        port_rank=2,
    )
    with_zero = _ready_model(instance)
    with_zero.memory_factors = [np.zeros((2, 2))]
    with_zero.memory_gains = np.zeros((1, 2))
    without = _ready_model()
    static = _static()
    time = np.linspace(0.0, 0.1, 11)
    exogenous = {"Pm": np.zeros((len(time), 3)), "interpolation": "zoh"}

    assert with_zero._hidden_shapes(3, 2)["memory"] == (0,)
    assert with_zero._port_realization(0, EDGE_MEMORY, 2).zero_port
    delta_zero, omega_zero = with_zero.rollout(static, time, exogenous)
    delta_removed, omega_removed = without.rollout(static, time, exogenous)
    assert np.allclose(delta_zero, delta_removed)
    assert np.allclose(omega_zero, omega_removed)

    pruned = with_zero.prune([], [], RunContext())
    assert pruned.active_instances == ()
    assert pruned.pruning_trace[0]["action"] == "automatic_zero_port_removal"
    assert pruned.pruning_trace[0]["modal_protection_bypassed"] is True


def test_rank_one_three_column_factor_is_minimally_realized_without_operator_change() -> None:
    factor = np.asarray([[1.0, 2.0, 3.0], [-0.5, -1.0, -1.5]])
    realization = _minimal_port_factor(factor)
    assert realization.proposed_rank == 3
    assert realization.realized_rank == 1
    assert realization.discarded_rank == 2
    assert np.allclose(realization.factor @ realization.factor.T, factor @ factor.T)

    instance = CandidateBlockInstance(
        "edge_memory:rank_one",
        EDGE_MEMORY,
        time_constant=0.2,
        factor=tuple(tuple(row) for row in factor),
        port_rank=3,
    )
    model = _ready_model(instance)
    model.memory_factors = [factor]
    model.memory_gains = np.diag(factor @ factor.T)[None, :]
    assert model.reciprocal_connections()[EDGE_MEMORY][0].shape == (2, 1)
    assert model._hidden_shapes(3, 2)["memory"] == (1,)

    compressed = _ready_model(instance)
    compressed.memory_factors = [realization.factor]
    compressed.memory_gains = np.diag(factor @ factor.T)[None, :]
    time = np.linspace(0.0, 0.1, 11)
    exogenous = {"Pm": np.zeros((len(time), 3)), "interpolation": "zoh"}
    raw_delta, raw_omega = model.rollout(_static(), time, exogenous)
    compressed_delta, compressed_omega = compressed.rollout(_static(), time, exogenous)
    assert np.allclose(raw_delta, compressed_delta)
    assert np.allclose(raw_omega, compressed_omega)


def test_legacy_checkpoint_recomputes_port_rank_without_changing_response(tmp_path) -> None:
    factor = np.asarray([[1.0, 2.0, 3.0], [-0.5, -1.0, -1.5]])
    instance = CandidateBlockInstance(
        "edge_memory:legacy_rank_one",
        EDGE_MEMORY,
        time_constant=0.2,
        factor=tuple(tuple(row) for row in factor),
        port_rank=3,
    )
    model = _ready_model(instance)
    model.memory_factors = [factor]
    model.memory_gains = np.diag(factor @ factor.T)[None, :]
    path = tmp_path / "legacy.npz"
    model.save(path)
    metadata_path = path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["format_version"] = 2
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with np.load(path) as checkpoint:
        arrays = {name: checkpoint[name] for name in checkpoint.files}
    arrays["memory_factor_0"] = factor
    np.savez_compressed(path, **arrays)

    loaded = PrunableEnergyModel.load(path)
    assert loaded.active_instances[0].port_rank == 3
    assert loaded._port_realization(0, EDGE_MEMORY, 2).realized_rank == 1
    time = np.linspace(0.0, 0.1, 11)
    exogenous = {"Pm": np.zeros((len(time), 3)), "interpolation": "zoh"}
    expected_delta, expected_omega = model.rollout(_static(), time, exogenous)
    actual_delta, actual_omega = loaded.rollout(_static(), time, exogenous)
    assert np.allclose(actual_delta, expected_delta)
    assert np.allclose(actual_omega, expected_omega)


def test_refinement_rejects_unsuccessful_solver(monkeypatch) -> None:
    model = _ready_model()
    static = _static()
    time = np.linspace(0.0, 0.1, 11)
    exogenous = {"Pm": np.zeros((len(time), 3)), "interpolation": "zoh"}
    delta, omega = model.rollout(static, time, exogenous)
    sample = {
        "event_id": "refinement-gate",
        "model_inputs": {"static": static, "exogenous": {"t": time, "Pm": exogenous["Pm"]}},
        "training_context": {
            "claim_type": "causal_zoh_rollout",
            "targets": {"t": time, "delta": delta, "omega": omega},
        },
    }

    class UnsuccessfulResult:
        success = False
        nfev = 1

        def __init__(self, initial: np.ndarray) -> None:
            self.x = initial

    def unsuccessful_solver(_residual, initial, **_kwargs):
        return UnsuccessfulResult(np.asarray(initial))

    monkeypatch.setattr(prunable_energy_module, "least_squares", unsuccessful_solver)
    audit = model.refine_closed_loop(
        [sample], RunContext(horizon_steps=len(time)), maximum_cases=1, maximum_evaluations=1
    )
    assert audit["solver_success"] is False
    assert audit["status"] == "rejected"


def test_tiny_resonant_port_does_not_normalize_an_unobservable_mode() -> None:
    instance = CandidateBlockInstance(
        "resonant_memory:tiny",
        RESONANT_MEMORY,
        frequency_hz=1.4,
        damping_ratio=0.1,
        factor=((1e-12,), (0.0,)),
        port_rank=1,
    )
    model = _ready_model(instance)
    model.resonant_factors = [np.asarray(instance.factor)]
    model.resonant_gains = np.asarray([[1e-24, 0.0]])
    spectrum = model.modal_spectrum(_static())
    tiny_poles = [
        pole
        for pole in spectrum["internal_poles"]
        if pole["positive_imaginary"] and abs(pole["frequency_hz"] - 1.4) < 0.02
    ]
    assert tiny_poles and tiny_poles[0]["speed_visibility"] < 1e-3
    assert not [
        mode
        for mode in spectrum["equation_reportable_modes"]
        if abs(mode["frequency_hz"] - 1.4) < 0.02
    ]


def test_nonpositive_energy_and_zero_visibility_threshold_refuse_reporting(monkeypatch) -> None:
    model = _ready_model()
    with pytest.raises(ValueError, match="finite and positive"):
        model.modal_spectrum(_static(), minimum_speed_visibility=0.0)

    def nonpositive_energy(_static_argument):
        return -np.eye(4), {
            "energy_hessian_coordinate_system": "test",
            "energy_hessian_min_eigenvalue": -1.0,
            "energy_hessian_positive_definite": False,
        }

    monkeypatch.setattr(model, "_modal_energy_hessian", nonpositive_energy)
    spectrum = model.modal_spectrum(_static())
    assert spectrum["equation_reportable_modes"] == []
    assert all(not pole["energy_normalization_available"] for pole in spectrum["internal_poles"])


@pytest.mark.parametrize("entry", [np.nan, np.inf, -np.inf])
def test_minimal_port_factor_rejects_nonfinite_entries(entry: float) -> None:
    with pytest.raises(ValueError, match="finite values"):
        _minimal_port_factor(np.asarray([[entry]]))


def test_zero_resonant_port_does_not_create_its_configured_pole() -> None:
    instance = CandidateBlockInstance(
        "resonant_memory:zero",
        RESONANT_MEMORY,
        frequency_hz=1.4,
        damping_ratio=0.1,
        factor=((0.0,), (0.0,)),
        port_rank=1,
    )
    model = _ready_model(instance)
    model.resonant_factors = [np.zeros((2, 1))]
    model.resonant_gains = np.zeros((1, 2))
    spectrum = model.modal_spectrum(_static())
    assert model._hidden_shapes(3, 2)["resonance_q"] == (0,)
    assert not [
        pole
        for pole in spectrum["internal_poles"]
        if pole["positive_imaginary"] and abs(pole["frequency_hz"] - 1.4) < 0.02
    ]
