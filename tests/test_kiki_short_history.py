import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from kron_reduce.data.history_contract import CausalStateHistory
from kron_reduce.models.learned_latent import history_features
from kron_reduce.models.nonlinear_latent import (
    EquilibriumLearningSample,
    LearnedNonlinearModel,
    NonlinearFitSettings,
    _features,
    fit_nonlinear_model,
    fixed_quadratic_basis,
)
from kron_reduce.physics.coordinates import coi_project
from kron_reduce.workflows.kiki_short_history import (
    CANDIDATES,
    SEEDS,
    ShortHistoryContract,
    candidate_settings,
    classify_los,
    real_evaluation_prerequisite,
    require_real_evaluation,
    score_fixed_predictions,
    select_structure,
    stability_refinement,
    synthetic_observability,
    synthetic_truth,
    trajectory_refinement,
)
from project_harness.report_storage import locate_report


def samples():
    closure = synthetic_truth(backbone="sine")
    rng = np.random.default_rng(12)
    output = []
    for i in range(2):
        q, v = [coi_project(rng.normal(0, 0.1, 10), closure.mass) for _ in range(2)]
        path = closure.rollout(q, v, np.array([0.3, -0.2]), np.arange(16) * 0.01,
                               np.zeros(10), np.zeros(10))
        history = CausalStateHistory(np.arange(11)*0.01 - 0.1, path[:11, :10],
                                     path[:11, 10:20], np.zeros(10))
        output.append(EquilibriumLearningSample(
            f"s{i}", "synthetic", f"line{i}", "train", history, np.arange(6)*0.01,
            path[10:, :10], path[10:, 10:20], np.zeros(10)))
    return closure, tuple(output)


def test_fixed_basis_is_data_independent_and_signed():
    basis = fixed_quadratic_basis()
    np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=1e-14)
    assert all(col[np.flatnonzero(abs(col) > 1e-14)[0]] > 0 for col in basis.T)
    assert not basis.flags.writeable
    s = (np.arange(11) - 5)/5
    for col in (np.ones(11), s, s*s):
        np.testing.assert_allclose(basis @ basis.T @ col, col, atol=1e-14)
    _, data = samples()
    row = data[0]
    features = _features(row.history, row.equilibrium_delta, 3, "fixed_quadratic")
    np.testing.assert_allclose(features[:30], (basis.T @ row.history.delta).reshape(-1))
    assert len(features) == 80
    with pytest.raises(ValueError, match="three"):
        NonlinearFitSettings(frames=11, encoder_mode="fixed_quadratic")


def test_legacy_feature_path_and_new_contract():
    _, data = samples()
    s = data[0]
    assert np.array_equal(_features(s.history, s.equilibrium_delta, 3),
                          np.r_[history_features(s.history, 3), s.equilibrium_delta])
    contract = ShortHistoryContract()
    assert (contract.origin_s, contract.history_count, contract.splice_offset) == (0.1, 11, 10)
    contract.validate_history(s.history)
    with pytest.raises(ValueError):
        ShortHistoryContract(power_index=0)


@pytest.mark.parametrize("candidate", list(CANDIDATES))
def test_actual_trainable_parameter_counts_and_checkpoint(candidate, tmp_path):
    closure, data = samples()
    settings = replace(candidate_settings(candidate, 324), epochs=1, horizons_s=(0.05,))
    model = fit_nonlinear_model(data, closure.mass, settings, allow_partial_horizons=True)
    assert model.fit_record["parameter_count"] == CANDIDATES[candidate]["parameter_count"]
    assert model.fit_record["actual_optimizer_steps"] == 1
    assert model.fit_record["training_rhs_event_evaluations"] == 4*2*(10+5)
    model.save(tmp_path / "model.npz")
    restored = LearnedNonlinearModel.load(tmp_path / "model.npz")
    row = data[0]
    assert np.array_equal(model.predict(row.history, row.forecast_time, row.equilibrium_delta),
                          restored.predict(row.history, row.forecast_time, row.equilibrium_delta))
    checks = trajectory_refinement(model, row, ShortHistoryContract(), model.target_scale)
    assert checks["qualified"]
    assert checks["history_initialization_refined_too"]


def records():
    return [{"candidate": c, "seed": s, "status": "completed", "actual_optimizer_steps": 80,
             "parameter_count": CANDIDATES[c]["parameter_count"], "val_main_nrmse": 1.0}
            for c in CANDIDATES for s in SEEDS]


def test_selection_median_ties_and_no_classification_routing():
    rows = records()
    assert select_structure(rows)["selected"] == "A"
    for r in rows:
        if r["candidate"] in {"C", "D"}:
            r["val_main_nrmse"] = 0.8
        r["fn"] = 0 if r["candidate"] == "C" else 100
    assert select_structure(rows)["selected"] == "D"
    for r in rows:
        if r["candidate"] == "D" and r["seed"] == 325:
            r["val_main_nrmse"] = 100
    assert select_structure(rows)["selected"] == "D"  # median, not mean or best seed


def test_failed_seed_is_retained_and_baseline_failure_incomplete():
    rows = records()
    rows[-1]["val_main_nrmse"] = float("nan")
    rows[0]["status"] = "failed"
    result = select_structure(rows)
    assert not result["candidate_table"][-1]["eligible"]
    assert len(result["candidate_table"][-1]["seeds"]) == 3
    assert not result["baseline_comparison_complete"]
    assert result["selected"] == "B"
    with pytest.raises(ValueError, match="12"):
        select_structure(rows[:-1])
    with pytest.raises(ValueError, match="12"):
        select_structure(rows + [rows[0]])


def test_los_and_fifth_point_and_contract_splice():
    contract = ShortHistoryContract()
    mass, equilibrium = np.ones(10), np.zeros(10)
    history = CausalStateHistory(np.arange(11)*0.01-0.1, np.zeros((11, 10)),
                                 np.zeros((11, 10)), np.zeros(10))
    prediction = np.zeros((801, 20))
    prediction[1:6, :2] = [2.0, -2.0]
    angle_only = classify_los(history, prediction, equilibrium, mass, 1.0, contract)
    assert not angle_only["predicted_los"]
    prediction[1:6, 10:12] = [0.03, -0.03]
    both = classify_los(history, prediction, equilibrium, mass, 1.0, contract)
    assert both["trigger_index"] == 15
    assert both["predicted_los_time_forecast_s"] == 0.05
    assert both["predicted_los_time_record_s"] == 0.15
    prediction[:, :10] = 0
    assert not classify_los(history, prediction, equilibrium, mass, 1.0, contract)["predicted_los"]


def test_refinement_subtracts_10_and_checks_finite_after_trigger():
    fine = np.zeros((801, 20))
    coarse = fine.copy()
    fc = {"predicted_los": True, "trigger_index": 15}
    cc = {"predicted_los": True, "trigger_index": 17}
    check = stability_refinement(fine, coarse, fc, cc, ShortHistoryContract())
    assert check["qualified"]
    assert check["comparison_future_points_including_origin"] == 8
    fine[-1, 0] = np.nan
    assert not stability_refinement(fine, coarse, fc, cc, ShortHistoryContract())["qualified"]


def test_short_target_weights_and_nonfinite_score():
    _, data = samples()
    short = replace(data[1], forecast_time=data[1].forecast_time[:3],
                    target_delta=data[1].target_delta[:3], target_omega=data[1].target_omega[:3])
    population = (data[0], short)
    predictions = {s.event_id: np.c_[s.target_delta, s.target_omega] + (i+1)
                   for i, s in enumerate(population)}
    result = score_fixed_predictions(predictions, population, np.ones(20))
    assert result["available"]["multi_horizon_normalized_rmse"] == pytest.approx(np.sqrt(2.5))
    assert result["complete3_common"]["multi_horizon_normalized_rmse"] is None
    predictions[short.event_id][-1, 0] = np.nan
    result = score_fixed_predictions(predictions, population, np.ones(20))
    assert result["available"]["multi_horizon_normalized_rmse"] is None
    assert result["available"]["horizons"]["3.0"]["failed_event_count"] == 1


def test_known_system_observability_and_unobservable_counterexample():
    coupled, _ = synthetic_observability(observable=True)
    disconnected, _ = synthetic_observability(observable=False)
    assert all(r["rank"] == 2 and r["reconstruction_max_error"] < 1e-10 for r in coupled)
    assert all(r["rank"] == 0 and r["status"] == "REFUSE_HIDDEN_STATE_IDENTIFICATION"
               for r in disconnected)
    assert all(r["noise_gain_l2"] > 0 for r in coupled)
    assert all(r["worst_noise_identity_error"] < 1e-10 for r in coupled)


def test_no_envelope_means_no_real_evaluation_even_for_diagnostics():
    gate = real_evaluation_prerequisite()
    assert gate["status"] == "BLOCKED_NO_CERTIFIED_ENVELOPE"
    assert gate["certified_envelopes"] == []
    assert not gate["real_data_evaluation_allowed"]
    assert not gate["out_of_envelope_diagnostics_allowed"]
    with pytest.raises(PermissionError, match="100ms"):
        require_real_evaluation()


def test_synthetic_runner_accepts_evidence_without_real_promotion(tmp_path, monkeypatch):
    """Exercise actual artifact emission, runner binding, validation and state update."""
    from kron_reduce.workflows import kiki_short_history
    from project_harness import runner
    from project_harness.evidence import (
        EvidenceContext,
        validate_stage_evidence,
        verify_artifact_directory,
    )
    from project_harness.loop import load_state

    repository = Path(__file__).resolve().parents[1]
    workflow_name = "a04_short_history_synthetic_prerequisite_v1"
    experiment_name = "a04_short_history_synthetic_prerequisite"
    sources = [
        "configs/project_manifest.json", "configs/research_policy.json",
        f"configs/workflows/{workflow_name}.json",
        f"configs/experiments/{experiment_name}.toml",
        "src/kron_reduce/workflows/kiki_short_history.py",
        "src/kron_reduce/models/nonlinear_latent.py",
        "src/kron_reduce/workflows/kiki_stability_comparison.py",
        "src/kron_reduce/physics/coordinates.py",
        "docs/protocols/kiki_a04_envelope_prerequisite_20261023_zh.md",
        "data/synthetic_benchmark_registry_v1/checksums.sha256",
        "data/synthetic_benchmark_registry_v1/README.md",
        "data/synthetic_benchmark_registry_v1/registry.json",
    ]
    for relative in sources:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repository / locate_report(repository, relative), target)
    # The isolated harness fixture has no Git repository; numerical work and
    # the real evidence validator are deliberately not mocked.
    monkeypatch.setattr(runner, "git_head", lambda _root: "a" * 40)
    monkeypatch.setattr(kiki_short_history, "git_head", lambda _root: "a" * 40)
    result_dir = runner.run_experiment(tmp_path, experiment_name, "synthetic_audit")
    verify_artifact_directory(result_dir, strict=True)
    state = load_state(tmp_path, workflow_name)
    evidence = state["history"][-1]["evidence"]
    workflow = json.loads((tmp_path / f"configs/workflows/{workflow_name}.json").read_text())
    validate_stage_evidence(EvidenceContext(tmp_path, workflow, "synthetic_audit"), evidence)
    assert state["current_stage"] == "synthetic_audit"
    assert evidence["scientific_verdict"] == "ACCEPT"
    assert evidence["mechanism_status"] == "MECHANISM_VERIFIED"
    assert evidence["promotion_decision"] == "KEEP_BASELINE"
    assert evidence["real_data_improvement_demonstrated"] is False
    assert not (result_dir / "RUN_FAILED.json").exists()


@pytest.mark.parametrize("candidate", list(CANDIDATES))
@pytest.mark.parametrize("step", [0.01, 0.005])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_batch_forecast_matches_original_equations(candidate, step, batch_size):
    from kron_reduce.workflows.kiki_short_history import forecast_population

    closure, data = samples()
    settings = candidate_settings(candidate, 324)
    if candidate == "A":
        closure = replace(closure, coupling=np.zeros((10, 0)), latent_skew=np.zeros((0, 0)),
                          latent_dissipation=np.zeros((0, 0)))
    features = _features(data[0].history, data[0].equilibrium_delta, settings.frames, settings.encoder_mode)
    model = LearnedNonlinearModel(closure, np.full((settings.latent_dimension, len(features)), 0.01),
                                  np.ones(len(features)), np.ones(20), settings, {})
    population = data[:batch_size]
    paths, historical = forecast_population(model, [s.history for s in population],
                                            [s.equilibrium_delta for s in population], 0.05, max_step=step)
    for i, sample in enumerate(population):
        direct = model.predict(sample.history, sample.forecast_time, sample.equilibrium_delta, max_step=step)
        _, history = model.initialize(sample.history, sample.equilibrium_delta, max_step=step)
        np.testing.assert_allclose(paths[i], direct, rtol=0, atol=1e-13)
        np.testing.assert_allclose(historical[i], history, rtol=0, atol=1e-13)


def test_progress_callback_does_not_change_fit():
    closure, data = samples()
    settings = replace(candidate_settings("D", 324), epochs=1, horizons_s=(0.05,))
    steps = []
    first = fit_nonlinear_model(data, closure.mass, settings, allow_partial_horizons=True)
    second = fit_nonlinear_model(data, closure.mass, settings, allow_partial_horizons=True,
                                 progress_callback=steps.append)
    assert steps == [1]
    assert np.array_equal(first.encoder, second.encoder)
    for name in ("edge_stiffness", "damping", "coupling", "latent_skew", "latent_dissipation"):
        assert np.array_equal(getattr(first.closure, name), getattr(second.closure, name))


def test_support_is_train_only_and_outside_points_are_audited():
    from kron_reduce.workflows.kiki_short_history import construct_support, support_membership

    _, data = samples()
    support = construct_support(data)
    assert not support["certified"]
    assert all(support_membership(support, s.history, s.equilibrium_delta)["inside_empirical_support"] for s in data)
    changed = replace(data[0].history, constant_power=np.arange(10)*1000.0)
    audit = support_membership(support, changed, data[0].equilibrium_delta)
    assert not audit["inside_empirical_support"]
    assert audit["delivery_status"] == "REFUSE_NO_CERTIFIED_ENVELOPE"
    with pytest.raises(ValueError, match="train only"):
        construct_support([replace(data[0], role="val")])


def test_development_approval_is_exact_and_not_a_delivery_certificate(tmp_path):
    import tomllib

    from kron_reduce.workflows.kiki_short_history import development_authorization

    repository = Path(__file__).resolve().parents[1]
    for path in ("configs/project_manifest.json", "configs/research_policy.json",
                 "docs/protocols/kiki_a04_development_authorization_20261023_zh.md"):
        target = tmp_path/path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repository / locate_report(repository, path), target)
    config = tomllib.loads((repository/"configs/experiments/paper_kiki_a04_short_history.toml").read_text())
    # Explicit approved snapshot from the execution commit. The active policy
    # is consumed; copying it must never reactivate production authorization.
    snapshot = json.loads((repository/"tests/fixtures/a04_development_approval_20261023.json").read_text(encoding="utf-8"))
    policy_path = tmp_path/"configs/research_policy.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["development_evaluations"][snapshot["authorization_id"]] = snapshot["approval"]
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    manifest = json.loads((tmp_path/"configs/project_manifest.json").read_text())
    experiment = manifest["experiments"]["paper_kiki_a04_short_history"]
    assert development_authorization(tmp_path, experiment, config)["deployment_predictions_allowed"] is False
    assert real_evaluation_prerequisite()["status"] == "BLOCKED_NO_CERTIFIED_ENVELOPE"
    with pytest.raises(PermissionError):
        development_authorization(tmp_path, {**experiment, "workflow": "another"}, config)
    with pytest.raises(PermissionError):
        development_authorization(tmp_path, experiment, {**config, "sources": ["another"]})
    path = tmp_path/"configs/research_policy.json"
    policy = json.loads(path.read_text(encoding="utf-8"))
    policy["development_evaluations"][config["development_authorization_id"]]["source_test_payload_allowed"] = 0
    path.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(PermissionError):
        development_authorization(tmp_path, experiment, config)


def test_consumed_development_approval_cannot_be_reused(tmp_path):
    import tomllib

    from kron_reduce.workflows.kiki_short_history import development_authorization
    from project_harness.evidence import sha256_file

    repository = Path(__file__).resolve().parents[1]
    for relative in ("configs/project_manifest.json", "configs/research_policy.json"):
        target = tmp_path/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repository / locate_report(repository, relative), target)
    policy = json.loads((tmp_path/"configs/research_policy.json").read_text(encoding="utf-8"))
    authorization = policy["development_evaluations"]["kiki_a04_development_20261023"]
    assert authorization["status"] == "consumed"
    assert authorization["final_fit_attempts"] == 12
    assert authorization["final_confirmed_optimizer_steps"] == 960
    receipt_sha = authorization["result_receipt_sha256"]
    receipt_path = locate_report(repository, authorization["result_receipt_path"], receipt_sha)
    assert sha256_file(repository / receipt_path) == receipt_sha
    config = tomllib.loads((repository/"configs/experiments/paper_kiki_a04_short_history.toml").read_text(encoding="utf-8"))
    manifest = json.loads((tmp_path/"configs/project_manifest.json").read_text(encoding="utf-8"))
    # Refuse before even opening the approval protocol or any source payload.
    with pytest.raises(PermissionError, match="frozen A04 scientific plan"):
        development_authorization(tmp_path, manifest["experiments"]["paper_kiki_a04_short_history"], config)


def test_budget_interruption_and_completed_recovery_do_not_retrain(tmp_path, monkeypatch):
    from kron_reduce.workflows import kiki_short_history as workflow

    monkeypatch.setattr(workflow, "shared_budget_directory", lambda root: root/"shared_budget")
    first = tmp_path/"results/first"
    first.mkdir(parents=True)
    ledger = workflow.FitBudgetLedger(tmp_path, "test", {"fixed": True}, "first", first)
    assert ledger.claim("A", 324)
    ledger.progress("A", 324, 1)
    assert ledger.claim("B", 324)
    for epoch in range(1, 81):
        ledger.progress("B", 324, epoch)
    receipt = first/"fit_record.json"
    receipt.write_text(json.dumps({"candidate": "B", "seed": 324, "status": "completed",
                                  "actual_optimizer_steps": 80, "parameter_count": 238}))
    checkpoint = first/"model.npz"
    checkpoint.write_bytes(b"test-checkpoint")
    ledger.finish("B", 324, receipt, checkpoint)
    second = tmp_path/"results/second"
    second.mkdir()
    resumed = workflow.FitBudgetLedger(tmp_path, "test", {"fixed": True}, "second", second, "first")
    assert not resumed.claim("A", 324)
    assert resumed.recover("A", 324)[0]["actual_optimizer_steps"] is None
    assert not resumed.claim("B", 324)
    assert resumed.recover("B", 324)[1] == checkpoint.resolve()
    with pytest.raises(PermissionError):
        resumed.progress("A", 324, 2)
    checkpoint.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash"):
        resumed.recover("B", 324)
    third = tmp_path/"results/third"
    third.mkdir()
    with pytest.raises(PermissionError, match="scientific plan changed"):
        workflow.FitBudgetLedger(tmp_path, "test", {"fixed": False}, "third", third, "first")
    assert resumed.snapshot()["optimizer_steps_reserved"] == 160


def test_real_controller_freezes_before_labels_and_resumes_completed_fit(tmp_path, monkeypatch):
    """Synthetic payloads and mocked optimization exercise the actual real run controller."""
    import tomllib

    from kron_reduce.workflows import kiki_short_history as w
    from project_harness.evidence import sha256_file, verify_artifact_directory

    repository = Path(__file__).resolve().parents[1]
    copied = ["AGENTS.md", "configs/input_contracts.json", "configs/research_policy.json",
              "configs/experiments/paper_kiki_a04_short_history.toml",
              "docs/protocols/kiki_a04_development_authorization_20261023_zh.md",
              "docs/protocols/kiki_a04_short_history_design_20261023_zh.md",
              *[f"src/kron_reduce/{p}" for p in (
                  "models/nonlinear_latent.py", "models/learned_latent.py", "models/causal_latent.py",
                  "data/kiki_history.py", "data/history_contract.py", "physics/coordinates.py",
                  "workflows/kiki_short_history.py", "workflows/kiki_history_learning.py",
                  "workflows/kiki_stability_comparison.py")]]
    for relative in copied:
        target = tmp_path/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repository / locate_report(repository, relative), target)
    closure, original = samples()
    train = tuple(replace(s, source_dataset="ieee39_postfault") for s in original)
    val = (replace(train[0], event_id="val0", fault_line="val-line", role="val"),)
    rows = [{"event_id": s.event_id, "raw_id": s.event_id, "source_dataset": s.source_dataset,
             "fault_line": s.fault_line, "role": s.role, "source_role": "train", "declared_length": 16}
            for s in (*train, *val)]
    raw = tmp_path/"fixtures/ieee39_postfault"
    raw.mkdir(parents=True)
    (raw/"splits.json").write_text("{}")
    bindings = {"ieee39_postfault": {"splits.json": sha256_file(raw/"splits.json")}}
    allocation_path = tmp_path/"allocation.json"
    allocation_path.write_text(json.dumps({"rows": rows, "source_bindings": bindings}))
    config_path = tmp_path/"configs/experiments/paper_kiki_a04_short_history.toml"
    config = tomllib.loads(config_path.read_text())
    text = config_path.read_text().replace(config["frozen_allocation_path"], "allocation.json")
    text = text.replace(config["frozen_allocation_sha256"], sha256_file(allocation_path))
    text = text.replace(config["source_root"], "fixtures")
    config_path.write_text(text)
    approval = {"authorization_id": "test_once", "approval": {"fixed": True}}
    monkeypatch.setattr(w, "development_authorization", lambda *_: approval)
    monkeypatch.setattr(w, "allocate_roles", lambda *_: (rows, bindings))
    monkeypatch.setattr(w, "shared_budget_directory", lambda root: root/"shared_budget")
    monkeypatch.setattr(w, "git_head", lambda _: "a"*40)
    monkeypatch.setattr(w, "status_lines", lambda _: [])
    active_dir = [None]

    class Provider:
        def __init__(self, source_root, source_rows, source_bindings, **kwargs):
            assert kwargs == {"history_count": 11, "power_index": 10}
            self.source_root, self.rows, self.bindings = source_root, source_rows, source_bindings
            self.coverage = [{**r, "history_eligible": True} for r in source_rows]
            self.static = {"ieee39_postfault": {"omega_s": 2*np.pi*60}}

        def load(self, role):
            if role == "val":
                assert (active_dir[0]/"support_construction_freeze.json").exists()
            return closure.mass, train if role == "train" else val

        def access_audit(self):
            return {"source_val_ids_read": [], "test_ids_read": []}

    monkeypatch.setattr(w, "KikiHistorySamples", Provider)
    fitted = []

    def fitter(population, mass, settings, *, target_scale_samples, allow_partial_horizons, progress_callback):
        assert population == train and target_scale_samples == train and allow_partial_horizons
        assert (active_dir[0]/"scientific_plan.json").exists()
        candidate = "A" if settings.latent_dimension == 0 else "D" if settings.encoder_mode == "fixed_quadratic" else "C" if settings.frames == 11 else "B"
        fitted.append((candidate, settings.seed))
        for epoch in range(1, 81):
            progress_callback(epoch)
        true_closure = closure if candidate != "A" else replace(closure, coupling=np.zeros((10, 0)),
            latent_skew=np.zeros((0, 0)), latent_dissipation=np.zeros((0, 0)))
        width = len(_features(train[0].history, train[0].equilibrium_delta, settings.frames, settings.encoder_mode))
        return LearnedNonlinearModel(true_closure, np.zeros((settings.latent_dimension, width)),
            np.ones(width), w.visible_scales(train), settings,
            {"actual_optimizer_steps": 80, "parameter_count": CANDIDATES[candidate]["parameter_count"],
             "fitted_event_ids": sorted(s.event_id for s in train)})

    monkeypatch.setattr(w, "fit_nonlinear_model", fitter)
    calls = [0]
    original_score = w.population_scores

    def scorer(*args, **kwargs):
        calls[0] += 1
        if calls[0] == 1:
            raise RuntimeError("injected engineering failure after completed checkpoint")
        return original_score(*args, **kwargs)

    monkeypatch.setattr(w, "population_scores", scorer)
    monkeypatch.setattr(w, "numerical_population", lambda model, population, *_: [
        {"event_id": s.event_id, "qualified": True} for s in population])

    def metadata(provider):
        freeze = json.loads((active_dir[0]/"selection_freeze.json").read_text())
        assert len(freeze["checkpoint_hashes"]) == 12
        assert len(fitted) == 12
        return {r["event_id"]: {"label": "stable", "true_event_time_record_s": None,
                                "previous_200ms_eligible": False} for r in provider.rows}

    monkeypatch.setattr(w, "truth_metadata", metadata)
    monkeypatch.setattr(w, "stability_population", lambda model, population, *_: (
        {"summary": {"fn": 0, "fp": 0}, "rows": [{"event_id": s.event_id, "qualified": True} for s in population]},
        {"placeholder_synthetic_test_only": np.zeros(1)}))
    experiment = {"data_policy": "kiki_original_train_only", "config": "paper_kiki_a04_short_history.toml"}
    first = tmp_path/"results/first"
    first.mkdir(parents=True)
    active_dir[0] = first
    with pytest.raises(RuntimeError, match="injected engineering"):
        w.run(root=tmp_path, experiment=experiment, stage="short_history_learning", result_dir=first, run_id="first")
    assert fitted == [("A", 324)]
    (first/"RUN_FAILED.json").write_text('{"test_injected": true}')
    config_path.write_text(config_path.read_text()+'\nresume_from_run = "first"\n')
    second = tmp_path/"results/second"
    second.mkdir()
    active_dir[0] = second
    w.run(root=tmp_path, experiment=experiment, stage="short_history_learning", result_dir=second, run_id="second")
    assert len(fitted) == len(set(fitted)) == 12
    verify_artifact_directory(second, strict=True)
    summary = json.loads((second/"learning_summary.json").read_text())
    assert summary["budget"]["optimizer_steps_reserved"] == 960
    assert summary["delivery_predictions_issued"] == 0
    assert summary["promotion_decision"] == "KEEP_BASELINE"
    assert (first/"RUN_FAILED.json").exists()
