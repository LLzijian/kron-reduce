import pytest

from kron_reduce.evaluation.kiki_machine_local_topology_radius import _validate_result
from kron_reduce.workflows.kiki_machine_local_topology_radius import (
    _execution_plan,
    _result_file,
    momentum_damping_freeze,
    radius_freeze,
)
from project_harness.evidence import sha256_file


def anchor_record():
    return {"phase": "freeze", "fitting_closed": True, "selection_closed": True,
            "input_contract": "measured_history_100ms_zoh_v1",
            "preprocessing_path": "preprocessing.json", "preprocessing_sha256": "prep-hash",
            "settings": {"prior_weight": 1e-6, "eval_batch": 32},
            "training": {"groups": {"T0M1": {"selected_update": 12}}}}


def test_radius_only_freeze_preserves_weight_and_quadratic_prior():
    anchor = anchor_record()
    checkpoint = {"path": "old-weights.npz", "sha256": "old-weight-hash"}
    freeze = radius_freeze(anchor, {"path": "freeze.json", "sha256": "anchor-hash"}, checkpoint)
    assert freeze["checkpoint_path"] == checkpoint["path"] and freeze["checkpoint_sha256"] == checkpoint["sha256"]
    assert freeze["preprocessing_path"] == anchor["preprocessing_path"]
    assert freeze["preprocessing_sha256"] == anchor["preprocessing_sha256"]
    assert freeze["settings"]["prior_weight"]/freeze["memory_radius_multiplier"]**2 == 1e-6
    assert freeze["settings"]["eval_batch"] == anchor["settings"]["eval_batch"]


def test_momentum_damping_freeze_retains_the_original_history_contract():
    anchor = anchor_record()
    checkpoint = {"path": "old-weights.npz", "sha256": "old-weight-hash"}
    factor = 0.3436767364716119
    freeze = momentum_damping_freeze(anchor, {"path": "freeze.json", "sha256": "anchor-hash"}, checkpoint, factor)
    assert freeze["memory_radius_multiplier"] == 1.
    assert freeze["checkpoint_sha256"] == checkpoint["sha256"]
    assert freeze["preprocessing_sha256"] == anchor["preprocessing_sha256"]
    assert freeze["settings"] == anchor["settings"] | {"damping_multiplier": factor, "matched_train_witnesses": True}


def test_radius_freeze_rejects_inexact_anchor():
    anchor = anchor_record()
    anchor["training"]["groups"]["T0M1"]["selected_update"] = 0
    with pytest.raises(RuntimeError, match="exact selected T0M1 anchor"):
        radius_freeze(anchor, {}, {})


def test_reference_file_uses_artifact_hash_without_git_tracking(tmp_path):
    result = tmp_path/"results"/"reference.json"
    result.parent.mkdir()
    result.write_text("{}", encoding="utf-8")
    binding = {"path": "results/reference.json", "sha256": sha256_file(result)}
    assert _result_file(tmp_path, binding) == result
    result.write_text("[]", encoding="utf-8")
    with pytest.raises(RuntimeError, match="frozen result artifact differs"):
        _result_file(tmp_path, binding)


def test_execution_plan_leaves_reader_roles_for_each_cohort():
    resources = {"files": {"static": "asset"}, "rosters": {"train": [], "val": [], "test": []}}
    source = {"task3_root": "task3", "sources": resources | {"allowed_roles": ["train", "val"]}}
    plan = _execution_plan({"stage": "test"}, source)
    assert plan == {"stage": "test", "task3_root": "task3", "sources": resources}


def test_radius_counts_missing_initializer_with_full_failure_record():
    common = {"allocated": 1, "tp": 0, "fp": 0, "fn": 0, "uncovered": 0}
    events = [{"id": "sample", "label": "stable"}]
    rows = [{"id": "sample", "truth_label": "stable"}]
    ref = {"stability_events": rows, "stability": common | {"tn": 1, "unresolved": 0},
           "initialization": [{"id": "sample", "step": step, "hidden_start_energy": .5}
                              for step in (.0025, .00125)]}
    failed = {"stability_events": rows, "stability": common | {"tn": 0, "unresolved": 1}, "initialization": []}
    counts = _validate_result(failed, ref, events, 2.)
    assert counts["reference"]["0.0025"]["at_radius"] == 1
    assert counts["R2"]["0.0025"]["missing"] == 1
    assert counts["R2"]["0.00125"]["fraction_of_assessed"] is None
