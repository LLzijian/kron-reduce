from __future__ import annotations

import copy
import hashlib
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pytest

import kron_reduce.models.p1_calibration as calibration
from kron_reduce.data.p1_causal import (
    P1_TIME,
    P1CausalReader,
    P1EventRole,
    P1RoleManifest,
    validate_p1_samples,
)
from kron_reduce.evaluation.pm_closure_diagnostics import validate_balanced_loss_weights
from kron_reduce.models import EDGE_MEMORY, CandidateBlockInstance, PrunableEnergyModel
from kron_reduce.models.p1_calibration import (
    CalibrationRolloutError,
    P1CalibrationObjective,
    P1CalibrationSettings,
    PhysicalGroupScaling,
    calibrate_loss_pair,
    core_fingerprint,
    measure_gradient_audit,
)
from kron_reduce.physics import complete_incidence


def _manifest(fit_specs=(('a', 'source_a', 'line_a'), ('b', 'source_b', 'line_b'))) -> P1RoleManifest:
    records = [P1EventRole(event, source, line, "fit", 1001) for event, source, line in fit_specs]
    records.extend((P1EventRole("select", "source_a", "line_selection", "selection", 1001), P1EventRole("internal", "source_a", "line_internal", "internal_audit", 1001)))
    return P1RoleManifest(tuple(records), "F1", hashlib.sha256(b"synthetic fixture role manifest").hexdigest())


def _sample(row: P1EventRole, manifest: P1RoleManifest, n_nodes: int = 3) -> dict:
    mass = np.ones(n_nodes)
    incidence, _ = complete_incidence(n_nodes)
    shape = np.arange(n_nodes, dtype=float) - (n_nodes - 1) / 2
    delta = np.sin(P1_TIME[:, None]) * shape * 0.03
    omega = np.cos(P1_TIME[:, None]) * shape * 0.03
    static = {
        "delta0": delta[[0]].copy(), "omega0": omega[[0]].copy(),
        "Pm_clear": np.zeros((1, n_nodes)), "Pm0": np.zeros((1, n_nodes)),
        "equilibrium_delta": np.zeros(n_nodes), "Pm_equilibrium": np.zeros(n_nodes),
        "mass": mass, "incidence": incidence,
    }
    return {
        "event_id": row.event_id,
        "model_inputs": {"static": static, "exogenous": {"t": P1_TIME.copy(), "Pm": np.zeros((1001, n_nodes))}},
        "training_context": {
            "targets": {"t": P1_TIME.copy(), "delta": delta, "omega": omega},
            "metadata": {"event_id": row.event_id, "source_dataset": row.source_dataset, "fault_line": row.fault_line},
            "role": row.role, "role_manifest_sha256": manifest.source_sha256,
            "claim_type": "causal_zoh_rollout", "representation": "generator_10",
        },
    }


def _fit_samples(manifest: P1RoleManifest) -> list[dict]:
    return [_sample(row, manifest) for row in manifest.records if row.role == "fit"]


def _ready_model() -> PrunableEnergyModel:
    model = PrunableEnergyModel(active_blocks=())
    model.node_damping = np.array([0.2, 0.25, 0.3])
    model.coupling = np.array([1.0, 0.8, 0.9])
    model.potential_coefficients = np.zeros((3, 3))
    model.edge_damping_coefficients = np.zeros((3, 3))
    model.node_high_order_gain = np.zeros(3)
    model.memory_factors, model.resonant_factors = [], []
    model.memory_gains, model.resonant_gains = np.zeros((0, 3)), np.zeros((0, 3))
    return model


def _h5(path: Path, manifest: P1RoleManifest) -> None:
    with h5py.File(path, "w") as handle:
        for row in manifest.records:
            group = handle.create_group(f"events/{row.event_id}")
            group.attrs["outcome_label"] = "MUST_NOT_READ"
            group.attrs["event_time_s"] = "MUST_NOT_READ"
            group.create_dataset("postclear/t", data=P1_TIME)
            group.create_dataset("postclear/delta", data=np.zeros((1001, 10)))
            group.create_dataset("postclear/omega", data=np.zeros((1001, 10)))
            pm = np.full((1001, 10), np.nan)
            pm[0] = 0
            group.create_dataset("postclear/Pm", data=pm)
            group.create_dataset("postclear/Pe", data=np.full((1001, 10), np.nan))
            group.create_dataset("equilibrium/delta0", data=np.zeros(10))
            group.create_dataset("equilibrium/Pm0", data=np.zeros(10))


def test_frozen_real_role_metadata_only(root) -> None:
    run = root / "results/paper_ieee39_modular_energy_p1_a03_candidate_acquired_f1_seed324-candidate_acquired-20260831T130327Z-93d29d92"
    path = run / "data_role_manifest.csv"
    # This reads saved role metadata, never the canonical index, arrays, or labels.
    import json

    expected = json.loads((run / "artifact_hashes.json").read_text())[path.name]
    manifest = P1RoleManifest.from_csv(path, fold="F1", expected_sha256=expected)
    assert [len(manifest.ids(role)) for role in ("fit", "selection", "internal_audit")] == [110, 25, 43]
    assert set(manifest.role_id_hashes) == {"fit", "selection", "internal_audit"}


@pytest.mark.parametrize("role,ids", [("test", ("a",)), ("external_validation", ("a",)), ("fit", ("a",)), ("fit", ("a", "internal")), ("fit", ("a", "a", "b"))])
def test_complete_request_authorized_before_any_file_open(tmp_path, role, ids) -> None:
    reader = P1CausalReader(tmp_path / "must_not_open.h5", np.ones(10), _manifest())
    with pytest.raises(ValueError):
        reader.read_role(role, event_ids=ids)
    assert reader.accesses == []


def test_fold_and_hash_are_checked_before_trajectory_reads(tmp_path) -> None:
    with pytest.raises(ValueError, match="F1"):
        P1RoleManifest.from_csv(tmp_path / "missing.csv", fold="F2", expected_sha256="0" * 64)
    path = tmp_path / "roles.csv"
    path.write_text("changed")
    with pytest.raises(ValueError, match="hash"):
        P1RoleManifest.from_csv(path, fold="F1", expected_sha256="0" * 64)


def test_duplicate_ids_and_cross_role_lines_rejected() -> None:
    manifest = _manifest()
    with pytest.raises(ValueError, match="duplicate"):
        P1RoleManifest(manifest.records + (manifest.records[0],), "F1", manifest.source_sha256)
    changed = list(manifest.records)
    changed[-1] = P1EventRole("internal", "source_a", "line_a", "internal_audit", 1001)
    with pytest.raises(ValueError, match="leakage"):
        P1RoleManifest(tuple(changed), "F1", manifest.source_sha256)


def test_actual_hdf5_reads_only_clearing_pm_and_allowed_fields(tmp_path, monkeypatch) -> None:
    manifest = _manifest()
    path = tmp_path / "synthetic.h5"
    _h5(path, manifest)
    reads = []
    original = h5py.Dataset.__getitem__

    def traced(dataset, selection):
        reads.append((dataset.name, selection))
        return original(dataset, selection)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", traced)
    def forbidden_attribute(_self, _key):
        raise AssertionError("the causal reader must not read label/event attributes")

    monkeypatch.setattr(h5py.AttributeManager, "__getitem__", forbidden_attribute)
    reader = P1CausalReader(path, np.ones(10), manifest)
    samples = reader.read_role("fit")
    assert len(samples) == 2
    assert len(reads) == 12  # Six explicitly allowed datasets for each event.
    assert {name.split("/")[2] for name, _ in reads} == {"a", "b"}
    assert all(selection == 0 for name, selection in reads if name.endswith("/Pm"))
    assert all("Pe" not in name and "label" not in name for name, _ in reads)
    assert all(np.isfinite(sample["model_inputs"]["exogenous"]["Pm"]).all() for sample in samples)
    assert all("Pm_observed" not in sample["training_context"] for sample in samples)


@pytest.mark.parametrize("attack", ["future_pm", "label", "source_input", "short", "wrong_role", "initial_state", "missing"])
def test_causal_contract_rejects_attacks_before_calibration(attack) -> None:
    manifest = _manifest()
    samples = _fit_samples(manifest)
    first = samples[0]
    if attack == "future_pm":
        first["model_inputs"]["exogenous"]["Pm"][10, :2] = (1, -1)
    elif attack == "label":
        first["training_context"]["label"] = "stable"
    elif attack == "source_input":
        first["model_inputs"]["static"]["source_dataset"] = "source_a"
    elif attack == "short":
        first["training_context"]["targets"]["t"] = P1_TIME[:-1]
    elif attack == "wrong_role":
        first["training_context"]["role"] = "internal_audit"
    elif attack == "initial_state":
        first["model_inputs"]["static"]["omega0"] += 0.1
    else:
        samples.pop()
    with pytest.raises(ValueError):
        P1CalibrationObjective(samples, manifest, balanced=True)


def test_loss_cells_have_complete_hierarchical_weights_and_same_design() -> None:
    manifest = _manifest((('a', 's1', 'l1'), ('a2', 's1', 'l1'), ('b', 's1', 'l2'), ('c', 's2', 'l3')))
    samples = _fit_samples(manifest)
    balanced = P1CalibrationObjective(samples, manifest, balanced=True)
    original = P1CalibrationObjective(samples, manifest, balanced=False)
    assert original.common_design == balanced.common_design
    assert len(balanced.cells) == len(samples) * 3 * 2
    for depth in range(5):
        groups = defaultdict(lambda: defaultdict(float))
        for cell in balanced.cells:
            key = cell.registered_key
            groups[key[:depth]][key[depth]] += cell.weight
        assert all(np.allclose(list(values.values()), next(iter(values.values()))) for values in groups.values())
    assert sum(cell.weight for cell in balanced.cells) == pytest.approx(1)
    assert sum(cell.weight for cell in original.cells if cell.source == "s1") == pytest.approx(0.75)
    assert sum(cell.weight for cell in balanced.cells if cell.source == "s1") == pytest.approx(0.5)


def _predictions(samples: list[dict], *, offset=0.01) -> dict:
    result = {}
    for sample in samples:
        target = sample["training_context"]["targets"]
        shape = np.arange(target["omega"].shape[1], dtype=float)
        shape -= shape.mean()
        result[sample["event_id"]] = (target["delta"] + offset * shape, target["omega"] + offset * shape)
    return result


def test_duplicate_source_events_do_not_increase_balanced_loss() -> None:
    first = _manifest()
    doubled = _manifest((('a', 'source_a', 'line_a'), ('a2', 'source_a', 'line_a'), ('b', 'source_b', 'line_b')))
    losses = []
    for manifest in (first, doubled):
        samples = _fit_samples(manifest)
        predictions = _predictions(samples)
        # All targets equal here, so the shared legacy scales also remain equal.
        objective = P1CalibrationObjective(samples, manifest, balanced=True)
        residual = objective.assemble(predictions)
        losses.append(residual @ residual)
    assert losses[0] == pytest.approx(losses[1])


def test_duplicate_angle_coordinates_do_not_suppress_balanced_speed_weight() -> None:
    manifest = _manifest()
    samples = _fit_samples(manifest)
    losses = []
    for multiplier in (1, 3):
        changed = copy.deepcopy(samples)
        for sample in changed:
            sample["model_inputs"]["static"]["incidence"] = np.tile(sample["model_inputs"]["static"]["incidence"], (1, multiplier))
        objective = P1CalibrationObjective(changed, manifest, balanced=True)
        assert sum(cell.weight for cell in objective.cells if cell.output == "omega") == pytest.approx(0.5)
        residual = objective.assemble(_predictions(changed))
        losses.append(residual @ residual)
    assert losses[0] == pytest.approx(losses[1])


@pytest.mark.parametrize("attack", ["missing", "short", "nan"])
def test_failed_predictions_cannot_be_dropped_or_replaced_by_zero(attack) -> None:
    manifest = _manifest()
    samples = _fit_samples(manifest)
    objective = P1CalibrationObjective(samples, manifest, balanced=True)
    predictions = _predictions(samples)
    if attack == "missing":
        predictions.pop("a")
    elif attack == "short":
        predictions["a"] = tuple(array[:-1] for array in predictions["a"])
    else:
        predictions["a"][0][10, 0] = np.nan
    with pytest.raises(CalibrationRolloutError):
        objective.assemble(predictions)


def test_physical_group_scaling_scales_port_operator_not_factor_squared() -> None:
    model = _ready_model()
    instance = CandidateBlockInstance("memory", EDGE_MEMORY, time_constant=0.2, factor=((0.2,), (0.1,), (-0.1,)), port_rank=1)
    model.active_instances = (instance,)
    model.active_blocks = (EDGE_MEMORY,)
    model.memory_factors = [np.asarray(instance.factor)]
    model.memory_gains = np.sum(model.memory_factors[0] ** 2, axis=1)[None, :]
    before = core_fingerprint(model)
    scaling = PhysicalGroupScaling(model)
    changed = scaling.assign(np.log(np.array([1., 1., 4.])))
    assert np.allclose(changed.memory_factors[0] @ changed.memory_factors[0].T, 4 * model.memory_factors[0] @ model.memory_factors[0].T)
    assert core_fingerprint(model) == before
    assert core_fingerprint(scaling.assign(np.zeros(3))) == before


def test_real_rk4_calibration_pair_uses_same_start_and_does_not_mutate_it() -> None:
    manifest = _manifest((('a', 'source_a', 'line_a'),))
    samples = _fit_samples(manifest)
    model = _ready_model()
    static = samples[0]["model_inputs"]["static"]
    delta, omega = model.rollout(static, P1_TIME, {"Pm": np.zeros((1001, 3)), "interpolation": "zoh"})
    samples[0]["training_context"]["targets"].update(delta=delta, omega=omega)
    before = core_fingerprint(model)
    pair = calibrate_loss_pair(model, samples, manifest, settings=P1CalibrationSettings(max_nfev=1))
    assert pair[0][1]["common_design_sha256"] == pair[1][1]["common_design_sha256"]
    for fitted, record in pair:
        assert record["status"] == "calibration_accepted"
        assert record["baseline_loss"] == pytest.approx(0, abs=1e-20)
        assert record["initial_core_fingerprint"] == before
        assert record["scientific_acceptance"] is False
        assert core_fingerprint(fitted) == before
    assert core_fingerprint(model) == before


def test_unsuccessful_solver_restores_parameters_even_with_smaller_loss(monkeypatch) -> None:
    manifest = _manifest((('a', 'source_a', 'line_a'),))
    samples = _fit_samples(manifest)
    model = _ready_model()
    model.diagnostics = {"must_remain_unchanged": True}
    teacher = PhysicalGroupScaling(model).assign(np.log(np.array([1.1, 1.2])))
    target_delta, target_omega = teacher.rollout(samples[0]["model_inputs"]["static"], P1_TIME, {"Pm": np.zeros((1001, 3)), "interpolation": "zoh"})
    samples[0]["training_context"]["targets"].update(delta=target_delta, omega=target_omega)

    class Result:
        success = False
        nfev = 1
        x = np.log(np.array([1.1, 1.2]))

    monkeypatch.setattr(calibration, "least_squares", lambda *args, **kwargs: Result())
    before = core_fingerprint(model)
    pair = calibrate_loss_pair(model, samples, manifest)
    assert all(record["status"] == "calibration_rejected" for _, record in pair)
    assert all(record["candidate_loss"] < record["baseline_loss"] for _, record in pair)
    assert all(core_fingerprint(fitted) == before for fitted, _ in pair)
    assert all(fitted.diagnostics == model.diagnostics and fitted.validation_scales == model.validation_scales for fitted, _ in pair)


def test_real_solver_reduces_nonzero_synthetic_error_without_changing_modules() -> None:
    manifest = _manifest((('a', 'source_a', 'line_a'),))
    samples = _fit_samples(manifest)
    model = _ready_model()
    teacher = PhysicalGroupScaling(model).assign(np.log(np.array([1.1, 0.9])))
    delta, omega = teacher.rollout(samples[0]["model_inputs"]["static"], P1_TIME, {"Pm": np.zeros((1001, 3)), "interpolation": "zoh"})
    samples[0]["training_context"]["targets"].update(delta=delta, omega=omega)
    pair = calibrate_loss_pair(model, samples, manifest)
    for fitted, record in pair:
        assert record["status"] == "calibration_accepted"
        assert record["baseline_loss"] > 1e-4
        assert record["candidate_loss"] < record["baseline_loss"] * 1e-4
        assert fitted.active_instances == model.active_instances
        assert record["final_group_scales"]["node_damping"] == pytest.approx(1.1, abs=1e-4)
        assert record["final_group_scales"]["coupling"] == pytest.approx(0.9, abs=1e-4)


def test_actual_model_residual_gradient_agrees_with_complete_objective() -> None:
    manifest = _manifest((('a', 'source_a', 'line_a'),))
    objective = P1CalibrationObjective(_fit_samples(manifest), manifest, balanced=True)
    coordinates = PhysicalGroupScaling(_ready_model())

    def residual(point):
        return objective.evaluate(coordinates.assign(point))

    audit, check = measure_gradient_audit(objective, residual, np.zeros(len(coordinates.names)), coordinates.names)
    for group, total in audit.total_gradients.items():
        assert sum(item.gradient[0] for item in audit.measurements if item.parameter_group_id == group) == pytest.approx(total[0], abs=1e-10)
    assert check["relative_error"] < 1e-5
    assert core_fingerprint(coordinates.model) == core_fingerprint(coordinates.reference)


def test_prediction_failure_rejects_every_cell_and_preserves_start(monkeypatch) -> None:
    manifest = _manifest((('a', 'source_a', 'line_a'),))
    model = _ready_model()
    before = core_fingerprint(model)

    def failed_rollout(*args, **kwargs):
        raise FloatingPointError("synthetic integrator failure")

    monkeypatch.setattr(PrunableEnergyModel, "rollout", failed_rollout)
    pair = calibrate_loss_pair(model, _fit_samples(manifest), manifest)
    for fitted, record in pair:
        assert record["status"] == "calibration_rejected"
        assert "synthetic integrator failure" in record["failure"]
        assert "candidate_loss" not in record
        assert len(record["final_cell_weights"]) == 6
        assert core_fingerprint(fitted) == before


def test_nonpassive_start_is_not_certified_by_positive_rescaling() -> None:
    model = _ready_model()
    model.node_damping[0] = -1
    with pytest.raises(ValueError, match="nonnegative"):
        PhysicalGroupScaling(model)


def test_scalar_objective_and_cell_vector_gradients_agree() -> None:
    manifest = _manifest()
    samples = _fit_samples(manifest)
    objective = P1CalibrationObjective(samples, manifest, balanced=True)
    base = objective.assemble(_predictions(samples))
    # A controlled residual with opposing parameter directions catches abs-sums.
    column = np.linspace(-1, 1, len(base)) / np.sqrt(len(base))

    def residual(point):
        return base * (1 + point[0]) + column * point[1]

    audit, check = measure_gradient_audit(objective, residual, np.array([0.2, -0.3]), ("node", "edge"))
    result = validate_balanced_loss_weights(final_cell_weights=objective.cells, registered_cell_keys=[cell.registered_key for cell in objective.cells], gradient_audit=audit)
    assert "gradient_vector_not_closed:node" not in result.violations
    assert "gradient_vector_not_closed:edge" not in result.violations
    assert "gradient_share_not_matched_to_l2_contribution" not in result.violations
    assert check["relative_error"] < 1e-8


def test_internal_audit_cannot_be_used_for_fit_or_scale_estimation() -> None:
    manifest = _manifest()
    samples = [_sample(row, manifest) for row in manifest.records if row.role == "internal_audit"]
    validate_p1_samples(samples, manifest, "internal_audit")
    with pytest.raises(ValueError, match="complete registered role"):
        P1CalibrationObjective(samples, manifest, balanced=True)


def test_targets_are_copied_and_not_mutated_by_caller_after_preparation() -> None:
    manifest = _manifest()
    samples = _fit_samples(manifest)
    objective = P1CalibrationObjective(samples, manifest, balanced=True)
    old = objective.samples[0]["training_context"]["targets"]["delta"].copy()
    samples[0]["training_context"]["targets"]["delta"][:] = 99
    assert np.array_equal(objective.samples[0]["training_context"]["targets"]["delta"], old)


@pytest.mark.parametrize("bad_scale", [float("inf"), float("nan"), 0.0])
def test_invalid_training_scale_cannot_manufacture_zero_loss(monkeypatch, bad_scale) -> None:
    manifest = _manifest()
    monkeypatch.setattr(PrunableEnergyModel, "_training_scales", staticmethod(lambda _samples: {"omega": bad_scale, "edge_angle": 1.0}))
    with pytest.raises(ValueError, match="training scales"):
        P1CalibrationObjective(_fit_samples(manifest), manifest, balanced=True)


def test_finite_predictions_with_overflowing_squared_loss_are_rejected(monkeypatch) -> None:
    manifest = _manifest((('a', 'source_a', 'line_a'),))
    model = _ready_model()
    before = core_fingerprint(model)
    monkeypatch.setattr(PrunableEnergyModel, "rollout", lambda *_args, **_kwargs: (np.full((1001, 3), 1e200), np.full((1001, 3), 1e200)))
    pair = calibrate_loss_pair(model, _fit_samples(manifest), manifest)
    for fitted, record in pair:
        assert record["status"] == "calibration_rejected"
        assert record["failure"] == "nonfinite squared objective"
        assert core_fingerprint(fitted) == before
