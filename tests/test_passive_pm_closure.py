from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from kron_reduce.data.p1_causal import P1_TIME, P1EventRole, P1RoleManifest
from kron_reduce.data.p1_pm_targets import PmObservationBatch, PmReadPurpose
from kron_reduce.models import EDGE_MEMORY, CandidateBlockInstance, PrunableEnergyModel
from kron_reduce.models.passive_pm_closure import (
    ClosureAugmentedModel,
    PassivePmClosure,
    exponential_linear_filter,
    fit_passive_pm_candidates,
    mass_quotient_basis,
)
from kron_reduce.physics import complete_incidence


def _manifest() -> P1RoleManifest:
    records = (
        P1EventRole("fit", "source", "line", "fit", 1001),
        P1EventRole("selection", "source", "selection_line", "selection", 1001),
        P1EventRole("audit", "source", "audit_line", "internal_audit", 1001),
    )
    return P1RoleManifest(records, "F1", hashlib.sha256(b"passive closure fixture").hexdigest())


def _sample(row: P1EventRole, manifest: P1RoleManifest, *, velocity: np.ndarray | None = None) -> dict:
    mass = np.array([1.0, 2.0, 3.0])
    basis = mass_quotient_basis(mass)
    velocity = np.zeros((1001, 2)) if velocity is None else velocity
    omega = velocity @ basis.T
    delta = np.zeros_like(omega)
    incidence, _ = complete_incidence(3)
    static = {
        "delta0": delta[[0]].copy(), "omega0": omega[[0]].copy(),
        "Pm_clear": np.zeros((1, 3)), "Pm0": np.zeros((1, 3)),
        "equilibrium_delta": np.zeros(3), "Pm_equilibrium": np.zeros(3),
        "mass": mass, "incidence": incidence,
    }
    return {
        "event_id": row.event_id,
        "model_inputs": {"static": static, "exogenous": {"t": P1_TIME.copy(), "Pm": np.zeros((1001, 3))}},
        "training_context": {
            "targets": {"t": P1_TIME.copy(), "delta": delta, "omega": omega},
            "metadata": {"event_id": row.event_id, "source_dataset": row.source_dataset, "fault_line": row.fault_line},
            "role": row.role, "role_manifest_sha256": manifest.source_sha256,
            "claim_type": "causal_zoh_rollout", "representation": "generator_10",
        },
    }


def _core(*, memory: bool = False) -> PrunableEnergyModel:
    if memory:
        factor = np.array([[0.2], [0.1], [-0.1]])
        instance = CandidateBlockInstance("memory", EDGE_MEMORY, time_constant=0.2, factor=tuple(map(tuple, factor)), port_rank=1)
        model = PrunableEnergyModel(block_instances=(instance,))
        model.memory_factors = [factor]
        model.memory_gains = np.sum(factor**2, axis=1)[None, :]
    else:
        model = PrunableEnergyModel(active_blocks=())
        model.memory_factors, model.memory_gains = [], np.zeros((0, 3))
    model.node_damping = np.array([0.2, 0.25, 0.3])
    model.coupling = np.array([1.0, 0.8, 0.9])
    model.potential_coefficients = np.zeros((3, 3))
    model.edge_damping_coefficients = np.zeros((3, 3))
    model.node_high_order_gain = np.zeros(3)
    model.resonant_factors, model.resonant_gains = [], np.zeros((0, 3))
    model.max_internal_step = 0.01
    return model


def _constant_pm(static: dict, time: np.ndarray) -> dict:
    clear = static["Pm_clear"].reshape(-1)
    return {"Pm": np.broadcast_to(clear, (len(time), len(clear))).copy(), "interpolation": "zoh"}


def _observations(manifest: P1RoleManifest, pm: np.ndarray, *, role: str = "fit") -> PmObservationBatch:
    event_ids = manifest.ids(role)
    values = {event: pm.copy() for event in event_ids}
    return PmObservationBatch(role, PmReadPurpose.FIT_CLOSURE_TARGET, manifest.source_sha256, event_ids, values, (), None)


def test_mass_quotient_port_and_storage_identity() -> None:
    mass = np.array([1.0, 2.0, 3.0])
    basis = mass_quotient_basis(mass)
    closure = PassivePmClosure(mass, np.array([0.4, 0.9]), 0.7, basis=basis)
    omega, z = np.array([0.2, -0.3, 0.1]), np.array([0.4, -0.5])
    increment, zdot = closure.field(omega, z)
    assert np.allclose(basis.T @ (mass[:, None] * basis), np.eye(2), atol=1e-12)
    assert np.allclose(basis.T @ mass, 0, atol=1e-12)
    assert abs(increment.sum()) < 1e-12
    assert omega @ increment + z @ zdot == pytest.approx(-closure.decay * (z @ z), abs=1e-12)


def test_zero_kappa_preserves_memory_core_rollout_exactly() -> None:
    manifest = _manifest()
    sample = _sample(manifest.records[0], manifest, velocity=np.column_stack((0.2 * np.sin(P1_TIME), 0.1 * np.cos(P1_TIME))))
    core = _core(memory=True)
    static, time = sample["model_inputs"]["static"], P1_TIME[:101]
    exogenous = _constant_pm(static, time)
    reference = core.rollout_augmented(static, time, exogenous)
    augmented = ClosureAugmentedModel(core, PassivePmClosure(static["mass"], np.zeros(2), 1.0))
    actual = augmented.rollout_augmented(static, time, exogenous)
    assert augmented.closure.n_states == 0
    assert np.array_equal(actual[0], reference[0])
    assert np.array_equal(actual[1], reference[1])
    assert np.array_equal(actual[2], reference[2])
    assert np.linalg.norm(reference[2]) > 0


def test_coupled_core_and_closure_supply_identity() -> None:
    manifest = _manifest()
    sample = _sample(manifest.records[0], manifest, velocity=np.array([[0.1, -0.2]]) .repeat(1001, axis=0))
    static = sample["model_inputs"]["static"]
    core = _core(memory=True)
    closure = PassivePmClosure(static["mass"], np.array([0.3, 0.6]), 0.8)
    model = ClosureAugmentedModel(core, closure)
    omega = static["omega0"].reshape(-1)
    hidden = core._pack_hidden(core.initial_hidden(omega, static))
    state = (static["delta0"].reshape(-1), omega, hidden, np.array([0.2, -0.1]))
    assert model.supply_identity_error(static, state) < 1e-8


def test_augmented_rk4_has_fourth_order_convergence() -> None:
    manifest = _manifest()
    sample = _sample(manifest.records[0], manifest, velocity=np.column_stack((0.2 * np.sin(P1_TIME), 0.1 * np.cos(0.7 * P1_TIME))))
    static = sample["model_inputs"]["static"]
    model = ClosureAugmentedModel(_core(memory=True), PassivePmClosure(static["mass"], np.array([0.4, 0.2]), 0.9))
    time = np.arange(0.0, 0.81, 0.08)
    exogenous = _constant_pm(static, time)
    reference = model.rollout_augmented(static, time, exogenous, max_step=0.00125)
    errors = []
    for step in (0.04, 0.02, 0.01):
        result = model.rollout_augmented(static, time, exogenous, max_step=step)
        errors.append(np.sqrt(sum(np.sum((got - want) ** 2) for got, want in zip(result[:4], reference[:4], strict=True))))
    ratios = np.asarray(errors[:-1]) / np.asarray(errors[1:])
    assert np.all((ratios > 8.0) & (ratios < 32.0))


def test_exponential_linear_filter_matches_constant_and_linear_integrals() -> None:
    time = np.array([0.0, 0.1, 0.4, 0.7, 1.0])
    decay = 1.3
    values = np.column_stack((np.full(len(time), 2.0), 3.0 * time))
    filtered = exponential_linear_filter(time, values, decay)
    expected_constant = 2 * (1 - np.exp(-decay * time)) / decay
    expected_linear = 3 * (time / decay - (1 - np.exp(-decay * time)) / decay**2)
    assert np.allclose(filtered[:, 0], expected_constant, atol=1e-13)
    assert np.allclose(filtered[:, 1], expected_linear, atol=1e-13)


def test_fit_recovers_known_nonnegative_kappa_and_records_clipping() -> None:
    manifest = _manifest()
    velocity = np.column_stack((0.2 * np.sin(0.8 * P1_TIME), 0.15 * (1 - np.cos(0.5 * P1_TIME))))
    sample = _sample(manifest.records[0], manifest, velocity=velocity)
    mass, basis = sample["model_inputs"]["static"]["mass"], mass_quotient_basis(np.array([1.0, 2.0, 3.0]))
    filtered = exponential_linear_filter(P1_TIME, velocity, 1.0)
    known = np.array([0.35, 0.7])
    pm = -(filtered * known) @ (mass[:, None] * basis).T
    candidates = fit_passive_pm_candidates([sample], _observations(manifest, pm), manifest)
    recovered = next((closure, record) for closure, record in candidates if record["decay"] == 1.0)
    assert np.allclose(recovered[0].kappa, known, atol=1e-12)
    zero = fit_passive_pm_candidates([_sample(manifest.records[0], manifest)], _observations(manifest, np.zeros((1001, 3))), manifest)
    assert all(closure.n_states == 0 for closure, _ in zero)
    negative_pm = filtered @ (mass[:, None] * basis).T
    clipped = next((closure, record) for closure, record in fit_passive_pm_candidates([sample], _observations(manifest, negative_pm), manifest) if record["decay"] == 1.0)
    assert np.array_equal(clipped[0].kappa, np.zeros(2))
    assert clipped[1]["nonpassive_unconstrained_coordinates"] == [0, 1]


def test_only_fit_causal_samples_can_fit_and_feature_attacks_reject() -> None:
    manifest = _manifest()
    fit = _sample(manifest.records[0], manifest)
    observations = _observations(manifest, np.zeros((1001, 3)))
    selection = _sample(manifest.records[1], manifest)
    with pytest.raises(ValueError):
        fit_passive_pm_candidates([selection], observations, manifest)
    selection_observations = PmObservationBatch(
        "selection", PmReadPurpose.FIT_CLOSURE_TARGET, manifest.source_sha256,
        manifest.ids("selection"), {"selection": np.zeros((1001, 3))}, (), None,
    )
    with pytest.raises(ValueError):
        fit_passive_pm_candidates([fit], selection_observations, manifest)
    future = _sample(manifest.records[0], manifest)
    future["model_inputs"]["exogenous"]["Pm"][2, 0] = 0.1
    with pytest.raises(ValueError):
        fit_passive_pm_candidates([future], observations, manifest)
    source = _sample(manifest.records[0], manifest)
    source["model_inputs"]["static"]["source_dataset"] = "leak"
    with pytest.raises(ValueError):
        fit_passive_pm_candidates([source], observations, manifest)
    assert fit["event_id"] == "fit"


def test_closure_save_load_refuses_overwrite_and_claimed_fields(tmp_path) -> None:
    closure = PassivePmClosure(np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.2]), 1.0)
    path = tmp_path / "closure.json"
    closure.save(path)
    loaded = PassivePmClosure.load(path)
    assert loaded.as_dict() == closure.as_dict()
    with pytest.raises(FileExistsError):
        closure.save(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["level_6_claimed"] = True
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="contract"):
        PassivePmClosure.load(path)
