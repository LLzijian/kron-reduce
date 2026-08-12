from __future__ import annotations

import csv
import hashlib
import json
import tomllib
from collections import Counter, defaultdict
from collections.abc import Iterable
from itertools import product
from pathlib import Path
from typing import Any

from kron_reduce.data import (
    CanonicalStore,
    ModelDataRequest,
    adapt_event,
    fit_stratification_policy,
    load_split,
    stratify_samples,
)
from kron_reduce.data.audit import audit_model_sample
from kron_reduce.evaluation.selection_validity import (
    nested_fault_line_folds,
    source_excitation_overlap_audit,
    structure_selection_ids,
)

MODULES = ("conservative", "dissipative", "memory")
TERMINAL_ELIGIBILITY_STATES = {
    "SOURCE_EXCITATION_CONFOUNDED",
    "INSUFFICIENT_INFORMATION",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in materialized for key in row})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in materialized:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )


def _write_hashes(result_dir: Path) -> None:
    artifacts = {
        str(path.relative_to(result_dir)).replace("\\", "/"): _sha256(path)
        for path in sorted(result_dir.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    _json(result_dir / "artifact_hashes.json", artifacts)


def _verify_hashes(result_dir: Path) -> list[str]:
    manifest = json.loads(
        (result_dir / "artifact_hashes.json").read_text(encoding="utf-8")
    )
    return [
        relative
        for relative, expected in manifest.items()
        if not (result_dir / relative).is_file()
        or _sha256(result_dir / relative) != expected
    ]


def _latest_result(root: Path, experiment: str, stage: str) -> Path:
    prefix = f"{experiment}-{stage}-"
    candidates = sorted(
        (
            path
            for path in (root / "results").glob(f"{prefix}*")
            if path.is_dir() and not (path / "RUN_FAILED.json").exists()
        ),
        key=lambda path: path.name,
    )
    if not candidates:
        raise FileNotFoundError(f"no completed v3 result for {experiment}/{stage}")
    result = candidates[-1]
    if "m0_afsd_v3" not in result.name:
        raise RuntimeError("v3 stages may consume only v3 stage artifacts")
    return result


def _validate_scope(config: dict[str, Any]) -> None:
    expected = {
        "protocol_revision": "m0_afsd_v3_joint_selection_validity",
        "fold": "F1",
        "seed": 324,
        "pm_mode": "observed",
        "test_access": "forbidden",
        "other_folds_access": "forbidden",
        "canonical_mutation": "forbidden",
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"invalid v3 locked scope: {mismatches}")
    if int(config["bootstrap_draws"]) < 5000:
        raise RuntimeError("v3 requires at least 5000 fault-line bootstrap draws")


def candidate_lattice(config: dict[str, Any]) -> dict[str, Any]:
    """Return the complete preregistered family without fitting a model."""

    spatial = [
        {
            "representation": f"adaptive_k{count}",
            "spatial_coordinates": int(count),
            "partition_policy": "train_only_data_driven",
        }
        for count in config["cluster_counts"]
    ]
    spatial.extend(
        [
            {
                "representation": "clustered_7_fixed",
                "spatial_coordinates": 7,
                "partition_policy": "fixed_control",
            },
            {
                "representation": "generator_10",
                "spatial_coordinates": 10,
                "partition_policy": "raw_control",
            },
        ]
    )
    b3_grid = [
        {
            "alpha": float(alpha),
            "tau_s": float(tau),
            "exact_B2_fallback": float(alpha) == 0.0,
        }
        for alpha in config["b3_alpha_candidates"]
        for tau in config["b3_tau_candidates"]
    ]
    memory_grid = [
        {
            "rank": int(rank),
            "taus_s": [float(value) for value in taus],
            "time_scale_count": len(taus),
        }
        for rank in config["memory_rank_candidates"]
        for taus in config["memory_tau_mode_candidates"]
    ]
    module_lattice = [
        {
            "code": "SS-"
            + (
                "".join(
                    symbol
                    for name, symbol, enabled in zip(
                        MODULES, "CDM", flags, strict=True
                    )
                    if enabled
                )
                or "zero"
            ),
            "enabled_modules": [
                name
                for name, enabled in zip(MODULES, flags, strict=True)
                if enabled
            ],
        }
        for flags in product((False, True), repeat=3)
    ]
    return {
        "spatial_candidates": spatial,
        "short_screen": {
            "model": "B2",
            "horizon_steps": int(config["spatial_shortlist_horizon"]),
            "within_best_fraction": float(config["spatial_shortlist_tolerance"]),
            "forced_controls": list(config["required_spatial_controls"]),
        },
        "dynamic_families": ["B2", "B3", "BS", "B4", "BM-r", "PSD"],
        "b3_grid": b3_grid,
        "memory_grid": memory_grid,
        "same_support_module_lattice": module_lattice,
        "parameterizations": list(config["parameterizations"]),
        "post_fit_complexity_tuple": [
            "total_state_count",
            "active_parameters",
            "active_edges",
            "spatial_coordinates",
        ],
        "active_state_definition": (
            "only states in the fitted vector field that influence reconstructed output"
        ),
        "intra_cluster_modes_first_round": False,
        "intra_cluster_trigger": {
            "raw10_loss_advantage": float(config["intra_cluster_trigger_loss"]),
            "minimum_intra_cluster_energy": float(
                config["intra_cluster_trigger_energy"]
            ),
            "minimum_bootstrap_support": float(
                config["minimum_intra_cluster_mode_support"]
            ),
            "next_round_ranks": list(config["intra_cluster_rank_candidates"]),
        },
    }


def hypothesis_registry(config: dict[str, Any], run_id: str) -> dict[str, Any]:
    lattice = candidate_lattice(config)
    return {
        "run_id": run_id,
        "protocol": "m0_afsd_v3_joint_selection_validity",
        "round": 1,
        "registered_before_data_qualification": True,
        "hypotheses": [
            {
                "id": "H0-overlap",
                "statement": (
                    "F1 train contains source-balanced Q-A excitation cells that can "
                    "separate source from excitation effects"
                ),
                "falsifier": (
                    "any represented amplitude-richness cell has fewer than five "
                    "fault-line clusters for either source, or no directly comparable cell exists"
                ),
                "terminal_if_falsified": "SOURCE_EXCITATION_CONFOUNDED",
            },
            {
                "id": "H1-joint-branch",
                "statement": (
                    "nested fault-line CV identifies a stable minimum joint spatial and dynamic branch"
                ),
                "falsifier": "the 2%/CI/fold-agreement/physics/modal gates fail",
            },
            {
                "id": "H2-parameters",
                "statement": (
                    "effective inertia, damping, coupling and memory parameters are identifiable"
                ),
                "falsifier": (
                    "rank, condition, sign stability or cross-fold drift gate fails"
                ),
                "terminal_if_falsified": "PARAMETER_NOT_IDENTIFIABLE",
            },
        ],
        "qualification_policy": {
            "structure_order_partition_support": "Q-A F1-train only",
            "parameter_robustness_and_extrapolation": "Q-A and Q-B after support freeze",
            "excluded": "Q-C",
        },
        "nested_cv": {
            "outer_fault_line_folds": int(config["outer_fault_line_folds"]),
            "inner_fault_line_folds": int(config["inner_fault_line_folds"]),
            "primary_weighting": "equal_fault_line",
            "validation_role": "confirmation_only_after_rules_frozen",
        },
        "candidate_lattice": lattice,
        "statistics": {
            "cluster_bootstrap_draws": int(config["bootstrap_draws"]),
            "random_sign_draws": int(config["random_sign_draws"]),
            "exact_sign_flip_max_clusters": 15,
            "knockout_multiplicity": "Holm",
        },
        "advancement_gates": {
            "minimum_10s_improvement": float(config["minimum_model_improvement"]),
            "bootstrap_ci_lower": ">0",
            "outer_fold_direction_agreement": "at least 4/5",
            "failure_rate": 0.0,
            "maximum_coi_drift": float(config["maximum_coi_drift"]),
            "rk4_steps": ["dt", "dt/2", "dt/4"],
            "dynamic_family_agreement": "at least 4/5",
            "spatial_count_agreement": "at least 4/5 within one node",
        },
        "modal_absolute_gates": {
            "median_frequency_relative_error": float(
                config["maximum_modal_frequency_relative_error"]
            ),
            "median_damping_absolute_error": float(
                config["maximum_modal_damping_absolute_error"]
            ),
            "median_MAC": float(config["minimum_modal_mac"]),
        },
        "allowed_terminal_states": [
            "JOINT_EFFECTIVE_BRANCH_IDENTIFIED",
            "SOURCE_EXCITATION_CONFOUNDED",
            "PARAMETER_NOT_IDENTIFIABLE",
            "MODEL_CLASS_INADEQUATE",
            "INSUFFICIENT_INFORMATION",
        ],
        "claim_boundaries": config["claim_boundaries"],
    }


def _run_hypothesis_registration(
    config_path: Path,
    config: dict[str, Any],
    result_dir: Path,
    run_id: str,
) -> None:
    (result_dir / "frozen_config.toml").write_bytes(config_path.read_bytes())
    _json(result_dir / "hypothesis_registry.json", hypothesis_registry(config, run_id))
    _json(result_dir / "candidate_lattice.json", candidate_lattice(config))
    _json(
        result_dir / "lineage.json",
        {
            "run_id": run_id,
            "stage": "hypothesis_registration",
            "v1_results_read": [],
            "v2_results_read": [],
            "test_ids_read": [],
            "other_folds_read": [],
            "canonical_files_modified": [],
            "config_sha256": _sha256(config_path),
        },
    )
    _write_hashes(result_dir)


def _load_data(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    dataset_dir = root / "data" / str(config["dataset"])
    store = CanonicalStore.open(dataset_dir)
    split = load_split(
        dataset_dir,
        view=str(config["view"]),
        split_name=str(config["split_name"]),
        fold="F1",
    )
    minimum = int(config["minimum_real_steps"])
    train_ids = [
        event_id
        for event_id in split.train_ids
        if store.metadata(event_id).valid_length >= minimum
    ]
    val_ids = [
        event_id
        for event_id in split.val_ids
        if store.metadata(event_id).valid_length >= minimum
    ]
    request = ModelDataRequest.initial_state_rollout("observed")
    inertia = store.dataset_metadata.M_prior

    def adapt(ids: list[str]) -> list[dict[str, Any]]:
        samples = [
            adapt_event(
                store.load_event(event_id),
                inertia_prior=inertia,
                request=request,
                representation="generator_10",
            )
            for event_id in ids
        ]
        for sample in samples:
            audit_model_sample(sample)
        return samples

    raw_train = adapt(train_ids)
    raw_val = adapt(val_ids)
    policy, train_rows = fit_stratification_policy(
        raw_train,
        report_horizons=tuple(int(value) for value in config["report_horizons"]),
        minimum_structure_steps=int(config["minimum_structure_steps"]),
        hankel_delays=int(config["hankel_delays"]),
        hankel_relative_floor=float(config["hankel_relative_floor"]),
        maximum_hankel_condition=float(config["maximum_hankel_condition"]),
    )
    val_rows = stratify_samples(raw_val, policy)
    return {
        "store": store,
        "split": split,
        "raw_train": raw_train,
        "raw_val": raw_val,
        "policy": policy,
        "train_rows": train_rows,
        "val_rows": val_rows,
    }


def _qualification_rows(data: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    store = data["store"]
    rows = []
    for role, strata in (("train", data["train_rows"]), ("validation", data["val_rows"])):
        for row in strata:
            metadata = store.metadata(str(row["event_id"]))
            rows.append(
                {
                    **row,
                    "role": role,
                    "source_dataset": metadata.source_dataset,
                    "fault_line": metadata.fault_line,
                    "valid_length": int(metadata.valid_length),
                    "qualification_use": (
                        "structure_order_partition_support"
                        if role == "train" and row["quality_grade"] == "Q-A"
                        else "post_freeze_parameter_or_extrapolation"
                        if row["quality_grade"] == "Q-B"
                        else "confirmation_only"
                        if role == "validation" and row["quality_grade"] == "Q-A"
                        else "excluded"
                    ),
                }
            )
    included = {str(row["event_id"]) for row in rows}
    for role, ids in (("train", data["split"].train_ids), ("validation", data["split"].val_ids)):
        for event_id in ids:
            if event_id in included:
                continue
            metadata = store.metadata(event_id)
            rows.append(
                {
                    "event_id": event_id,
                    "role": role,
                    "source_dataset": metadata.source_dataset,
                    "fault_line": metadata.fault_line,
                    "valid_length": int(metadata.valid_length),
                    "n_real_steps": int(metadata.valid_length),
                    "quality_grade": "Q-C",
                    "amplitude_level": "A-uncertain",
                    "richness_level": "R-uncertain",
                    "structure_selection_eligible": False,
                    "coefficient_fit_eligible": False,
                    "qualification_use": "excluded",
                    "exclusion_reason": (
                        f"valid_length_below_{int(config['minimum_real_steps'])}"
                    ),
                }
            )
    return sorted(rows, key=lambda row: (str(row["role"]), str(row["event_id"])))


def _fold_balance(folds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for outer in sorted({int(row["outer_fold"]) for row in folds}):
        subset = [row for row in folds if int(row["outer_fold"]) == outer]
        for role in ("development", "evaluation"):
            selected = [row for row in subset if row["outer_role"] == role]
            source_counts = Counter(str(row["source_dataset"]) for row in selected)
            output.append(
                {
                    "outer_fold": outer,
                    "outer_role": role,
                    "n_events": len(selected),
                    "n_fault_line_clusters": len(
                        {str(row["fault_line"]) for row in selected}
                    ),
                    "source_event_counts": dict(source_counts),
                }
            )
    return output


def _source_controls(rows: list[dict[str, Any]]) -> dict[str, Any]:
    qa = [
        row
        for row in rows
        if row.get("role") == "train"
        and row.get("quality_grade") == "Q-A"
        and row.get("structure_selection_eligible") is True
    ]
    sources = sorted({str(row["source_dataset"]) for row in qa})
    counts = {
        source: {
            "events": sum(str(row["source_dataset"]) == source for row in qa),
            "fault_line_clusters": len(
                {
                    str(row["fault_line"])
                    for row in qa
                    if str(row["source_dataset"]) == source
                }
            ),
        }
        for source in sources
    }
    return {
        "shared_parameter_model": {"eligible_QA_events": len(qa)},
        "separate_source_parameter_models": counts,
        "leave_one_source_out_routes": [
            {"train_source": source, "held_out_sources": [s for s in sources if s != source]}
            for source in sources
        ],
        "role": (
            "heterogeneity controls only; they cannot repair missing source-by-excitation overlap"
        ),
    }


def _run_eligibility_audit(
    root: Path,
    config_path: Path,
    config: dict[str, Any],
    result_dir: Path,
    run_id: str,
) -> None:
    hypothesis = _latest_result(
        root, str(config["hypothesis_experiment"]), "hypothesis_registration"
    )
    if _verify_hashes(hypothesis):
        raise RuntimeError("hypothesis artifacts changed before eligibility audit")
    data = _load_data(root, config)
    rows = _qualification_rows(data, config)
    overlap = source_excitation_overlap_audit(
        rows,
        minimum_fault_line_clusters=int(config["minimum_source_cell_fault_lines"]),
    )
    selection_ids = structure_selection_ids(rows)
    qa_fault_lines = {
        str(row["fault_line"])
        for row in rows
        if str(row["event_id"]) in selection_ids
    }
    folds: list[dict[str, Any]] = []
    fold_error = None
    try:
        folds = nested_fault_line_folds(
            rows,
            outer_folds=int(config["outer_fault_line_folds"]),
            inner_folds=int(config["inner_fault_line_folds"]),
        )
    except ValueError as error:
        fold_error = str(error)
    if fold_error or len(qa_fault_lines) < int(config["outer_fault_line_folds"]):
        status = "INSUFFICIENT_INFORMATION"
    elif overlap["status"] != "PASS":
        status = "SOURCE_EXCITATION_CONFOUNDED"
    else:
        status = "PASS"
    decision = {
        "run_id": run_id,
        "status": status,
        "q_a_train_events": len(selection_ids),
        "q_a_train_fault_line_clusters": len(qa_fault_lines),
        "source_overlap_status": overlap["status"],
        "nested_fold_status": "PASS" if not fold_error else "FAIL",
        "nested_fold_error": fold_error,
        "controlled_joint_lattice_permitted": status == "PASS",
        "q_b_structure_use_permitted": False,
        "interpretation": (
            "joint branch selection preconditions satisfied"
            if status == "PASS"
            else "the preregistered validity gate forbids a joint order or excitation-effect claim"
        ),
    }
    _csv(result_dir / "qualification_table.csv", rows)
    _json(result_dir / "stratification_policy.json", data["policy"].as_dict())
    _json(result_dir / "source_excitation_overlap.json", overlap)
    _csv(result_dir / "source_excitation_table.csv", overlap["table"])
    _csv(result_dir / "nested_fault_line_folds.csv", folds)
    _csv(result_dir / "fold_balance.csv", _fold_balance(folds) if folds else [])
    _json(result_dir / "source_parameter_controls.json", _source_controls(rows))
    _json(result_dir / "eligibility_decision.json", decision)
    _json(
        result_dir / "selection_population.json",
        {
            "structure_selection_event_ids": sorted(selection_ids),
            "q_b_post_freeze_only_event_ids": sorted(
                str(row["event_id"])
                for row in rows
                if row.get("role") == "train" and row.get("quality_grade") == "Q-B"
            ),
            "test_ids_read": [],
        },
    )
    _json(
        result_dir / "lineage.json",
        {
            "run_id": run_id,
            "stage": "eligibility_audit",
            "parent_hypothesis_result": hypothesis.name,
            "parent_hash_manifest_sha256": _sha256(
                hypothesis / "artifact_hashes.json"
            ),
            "v1_results_read": [],
            "v2_results_read": [],
            "test_ids_read": [],
            "other_folds_read": [],
            "canonical_files_modified": [],
            "offline_only_fields": ["source_dataset", "fault_line", "valid_length"],
            "model_features_added": [],
            "config_sha256": _sha256(config_path),
        },
    )
    _write_hashes(result_dir)


def _run_controlled_experiment(
    root: Path,
    config: dict[str, Any],
    result_dir: Path,
    run_id: str,
) -> None:
    eligibility = _latest_result(
        root, str(config["eligibility_experiment"]), "eligibility_audit"
    )
    if _verify_hashes(eligibility):
        raise RuntimeError("eligibility artifacts changed before controlled stage")
    decision = json.loads(
        (eligibility / "eligibility_decision.json").read_text(encoding="utf-8")
    )
    inventory = candidate_lattice(config)
    if decision["status"] in TERMINAL_ELIGIBILITY_STATES:
        status = "NOT_RUN_BY_PREREGISTERED_ELIGIBILITY_GATE"
        branch_result = decision["status"]
    elif decision["status"] == "PASS":
        # The current registered F1 round is a selection-validity audit.  A
        # positive eligibility result would authorize the expensive nested
        # lattice in a fresh immutable controlled run; it may never silently
        # fall back to the old single-validation selector.
        status = "AUTHORIZED_FOR_NESTED_JOINT_SELECTION"
        branch_result = "PENDING_FRESH_CONTROLLED_RUN"
    else:
        raise RuntimeError(f"unknown eligibility state: {decision['status']}")
    controlled = {
        "run_id": run_id,
        "status": status,
        "eligibility_status": decision["status"],
        "scientific_result": branch_result,
        "candidate_lattice_fitted": False,
        "reason": (
            "source/excitation validity is a precondition, so fitting the branch lattice would create invalid order evidence"
            if decision["status"] in TERMINAL_ELIGIBILITY_STATES
            else "a new immutable compute run is required after eligibility authorization"
        ),
        "q_b_used_for_structure_order_partition_or_support": False,
        "validation_used_for_rule_change": False,
        "test_ids_read": [],
    }
    _json(result_dir / "joint_candidate_inventory.json", inventory)
    _json(result_dir / "controlled_experiment.json", controlled)
    _json(
        result_dir / "controlled_selection_marker.json",
        {
            "rules_frozen": True,
            "selected_branch": None,
            "terminal_validity_result": branch_result,
            "conditional_rollout": True,
            "Pm_policy": "observed_schedule",
            "test_access": "forbidden",
        },
    )
    _json(
        result_dir / "lineage.json",
        {
            "run_id": run_id,
            "stage": "controlled_experiment",
            "parent_eligibility_result": eligibility.name,
            "v1_results_read": [],
            "v2_results_read": [],
            "test_ids_read": [],
            "other_folds_read": [],
            "canonical_files_modified": [],
        },
    )
    _write_hashes(result_dir)


def _run_residual_diagnosis(
    root: Path,
    config: dict[str, Any],
    result_dir: Path,
    run_id: str,
) -> None:
    eligibility = _latest_result(
        root, str(config["eligibility_experiment"]), "eligibility_audit"
    )
    controlled = _latest_result(
        root, str(config["controlled_experiment"]), "controlled_experiment"
    )
    if _verify_hashes(eligibility) or _verify_hashes(controlled):
        raise RuntimeError("upstream v3 artifacts changed before residual diagnosis")
    overlap = json.loads(
        (eligibility / "source_excitation_overlap.json").read_text(encoding="utf-8")
    )
    control = json.loads(
        (controlled / "controlled_experiment.json").read_text(encoding="utf-8")
    )
    deficits = []
    minimum = int(overlap["minimum_fault_line_clusters"])
    for row in overlap["table"]:
        if int(row["n_fault_line_clusters"]) < minimum:
            deficits.append(
                {
                    **row,
                    "additional_fault_line_clusters_needed": minimum
                    - int(row["n_fault_line_clusters"]),
                }
            )
    diagnosis = {
        "run_id": run_id,
        "status": "COMPLETED",
        "controlled_status": control["status"],
        "residual_type": (
            "design_support_residual"
            if overlap["status"] != "PASS"
            else "model_residual_pending_authorized_compute"
        ),
        "underpowered_comparable_cells": deficits,
        "missing_overlap": not bool(overlap["comparable_cells"]),
        "scientific_interpretation": (
            "the present data cannot distinguish excitation-dependent order from source-dependent dynamics"
            if overlap["status"] != "PASS"
            else "eligibility passed; model residual analysis belongs to the authorized compute run"
        ),
        "highest_information_next_experiment": (
            "increase Q-A fault-line overlap within the existing source-by-amplitude-by-richness cells, or improve a preregistered quality estimator without relabeling outcomes"
            if overlap["status"] != "PASS"
            else "run the frozen 5x4 nested joint candidate lattice"
        ),
        "q_b_may_be_promoted_post_hoc": False,
        "test_ids_read": [],
    }
    _json(result_dir / "residual_diagnosis.json", diagnosis)
    _csv(result_dir / "source_cell_deficits.csv", deficits)
    _json(
        result_dir / "lineage.json",
        {
            "run_id": run_id,
            "stage": "residual_diagnosis",
            "parent_eligibility_result": eligibility.name,
            "parent_controlled_result": controlled.name,
            "v1_results_read": [],
            "v2_results_read": [],
            "test_ids_read": [],
            "other_folds_read": [],
            "canonical_files_modified": [],
        },
    )
    _write_hashes(result_dir)


def _folds_are_isolated(rows: list[dict[str, Any]]) -> bool:
    for outer in sorted({int(row["outer_fold"]) for row in rows}):
        subset = [row for row in rows if int(row["outer_fold"]) == outer]
        evaluation = {
            str(row["fault_line"])
            for row in subset
            if row["outer_role"] == "evaluation"
        }
        development = {
            str(row["fault_line"])
            for row in subset
            if row["outer_role"] == "development"
        }
        if evaluation & development:
            return False
        by_inner: dict[int, set[str]] = defaultdict(set)
        for row in subset:
            if row["outer_role"] == "development":
                by_inner[int(row["inner_fold"])].add(str(row["fault_line"]))
        inner_sets = list(by_inner.values())
        if any(
            inner_sets[left] & inner_sets[right]
            for left in range(len(inner_sets))
            for right in range(left + 1, len(inner_sets))
        ):
            return False
    return True


def _run_adversarial_review(
    root: Path,
    config: dict[str, Any],
    result_dir: Path,
    run_id: str,
) -> None:
    hypothesis = _latest_result(
        root, str(config["hypothesis_experiment"]), "hypothesis_registration"
    )
    eligibility = _latest_result(
        root, str(config["eligibility_experiment"]), "eligibility_audit"
    )
    controlled = _latest_result(
        root, str(config["controlled_experiment"]), "controlled_experiment"
    )
    residual = _latest_result(
        root, str(config["residual_experiment"]), "residual_diagnosis"
    )
    qualification = []
    with (eligibility / "qualification_table.csv").open(encoding="utf-8-sig") as handle:
        qualification = list(csv.DictReader(handle))
    folds = []
    fold_path = eligibility / "nested_fault_line_folds.csv"
    if fold_path.stat().st_size:
        with fold_path.open(encoding="utf-8-sig") as handle:
            folds = list(csv.DictReader(handle))
    population = json.loads(
        (eligibility / "selection_population.json").read_text(encoding="utf-8")
    )
    selection_ids = set(population["structure_selection_event_ids"])
    q_b_ids = set(population["q_b_post_freeze_only_event_ids"])
    eligibility_decision = json.loads(
        (eligibility / "eligibility_decision.json").read_text(encoding="utf-8")
    )
    controlled_decision = json.loads(
        (controlled / "controlled_experiment.json").read_text(encoding="utf-8")
    )
    lineage_values = [
        json.loads((path / "lineage.json").read_text(encoding="utf-8"))
        for path in (hypothesis, eligibility, controlled, residual)
    ]
    eligibility_lineage = lineage_values[1]
    marker = json.loads(
        (controlled / "controlled_selection_marker.json").read_text(encoding="utf-8")
    )
    gates = {
        "all_upstream_hashes_immutable": all(
            not _verify_hashes(path)
            for path in (hypothesis, eligibility, controlled, residual)
        ),
        "hypothesis_precedes_eligibility": eligibility_lineage.get(
            "parent_hypothesis_result"
        )
        == hypothesis.name,
        "only_QA_train_enters_structure_selection": all(
            row["role"] == "train" and row["quality_grade"] == "Q-A"
            for row in qualification
            if row["event_id"] in selection_ids
        ),
        "QB_excluded_from_structure_selection": not bool(selection_ids & q_b_ids),
        "fault_line_outer_and_inner_isolation": (
            _folds_are_isolated(folds)
            if folds
            else eligibility_decision["status"] == "INSUFFICIENT_INFORMATION"
            and eligibility_decision["nested_fold_status"] == "FAIL"
        ),
        "observed_Pm_marked_conditional": marker.get("conditional_rollout") is True
        and marker.get("Pm_policy") == "observed_schedule"
        and config["claim_boundaries"]["observed_pm_is_conditional"],
        "eligibility_gate_honored": (
            eligibility_decision["status"] not in TERMINAL_ELIGIBILITY_STATES
            or controlled_decision["candidate_lattice_fitted"] is False
        ),
        "no_v1_or_v2_results_read": all(
            not row.get("v1_results_read") and not row.get("v2_results_read")
            for row in lineage_values
        ),
        "test_and_other_folds_locked": all(
            not row.get("test_ids_read") and not row.get("other_folds_read")
            for row in lineage_values
        ),
        "canonical_data_unchanged_by_workflow": all(
            not row.get("canonical_files_modified") for row in lineage_values
        ),
        "terminal_state_is_preregistered": eligibility_decision["status"]
        in ({"PASS"} | TERMINAL_ELIGIBILITY_STATES),
    }
    status = "PASS" if all(gates.values()) else "FAIL"
    review = {
        "run_id": run_id,
        "status": status,
        "decision": "ACCEPTABLE" if status == "PASS" else "REJECT",
        "eligibility_status": eligibility_decision["status"],
        "gates": gates,
        "adversarial_findings": [
            {
                "severity": "critical",
                "finding": (
                    "a source-confounded Q-A design cannot support an excitation-causes-order claim"
                ),
                "mitigation": "fail-closed eligibility gate",
                "mitigation_applied": eligibility_decision["status"]
                in TERMINAL_ELIGIBILITY_STATES,
            },
            {
                "severity": "major",
                "finding": "Q-B may improve parameter fit but cannot supply order evidence",
                "mitigation": "separate immutable Q-B post-freeze ID list",
                "mitigation_applied": not bool(selection_ids & q_b_ids),
            },
        ],
        "test_ids_read": [],
    }
    _json(result_dir / "adversarial_review.json", review)
    (result_dir / "adversarial_review.md").write_text(
        "# M0-AFSD v3 对抗审查\n\n"
        f"**结论：{review['decision']}**\n\n"
        + "\n".join(
            f"- {'PASS' if value else 'FAIL'} `{key}`"
            for key, value in gates.items()
        )
        + "\n",
        encoding="utf-8",
    )
    _json(
        result_dir / "lineage.json",
        {
            "run_id": run_id,
            "stage": "adversarial_review",
            "v1_results_read": [],
            "v2_results_read": [],
            "test_ids_read": [],
            "other_folds_read": [],
            "canonical_files_modified": [],
        },
    )
    _write_hashes(result_dir)
    if status != "PASS":
        raise RuntimeError("v3 adversarial review rejected the evidence chain")


def _run_acceptance(
    root: Path,
    config: dict[str, Any],
    result_dir: Path,
    run_id: str,
) -> None:
    eligibility = _latest_result(
        root, str(config["eligibility_experiment"]), "eligibility_audit"
    )
    controlled = _latest_result(
        root, str(config["controlled_experiment"]), "controlled_experiment"
    )
    residual = _latest_result(
        root, str(config["residual_experiment"]), "residual_diagnosis"
    )
    adversarial = _latest_result(
        root, str(config["adversarial_experiment"]), "adversarial_review"
    )
    decision = json.loads(
        (eligibility / "eligibility_decision.json").read_text(encoding="utf-8")
    )
    review = json.loads(
        (adversarial / "adversarial_review.json").read_text(encoding="utf-8")
    )
    immutable = all(
        not _verify_hashes(path)
        for path in (eligibility, controlled, residual, adversarial)
    )
    if review["status"] == "PASS" and immutable and decision["status"] in TERMINAL_ELIGIBILITY_STATES:
        outcome = decision["status"]
        acceptance_decision = "ACCEPT_NEGATIVE_CONCLUSION"
        status = "PASS"
    elif review["status"] == "PASS" and immutable and decision["status"] == "PASS":
        outcome = "INSUFFICIENT_INFORMATION"
        acceptance_decision = "REJECT_PENDING_AUTHORIZED_NESTED_COMPUTE"
        status = "FAIL"
    else:
        outcome = "UNRESOLVED"
        acceptance_decision = "REJECT"
        status = "FAIL"
    acceptance = {
        "run_id": run_id,
        "round": 1,
        "status": status,
        "decision": acceptance_decision,
        "scientific_conclusion": outcome,
        "joint_effective_branch_selected": False,
        "reason": (
            "the Q-A source-by-excitation support required for a valid joint order comparison is absent or insufficient"
            if outcome in TERMINAL_ELIGIBILITY_STATES
            else "the nested joint-selection evidence is not complete"
        ),
        "valid_claim": (
            "current F1 data do not identify an excitation-dependent change of effective order independently of source"
            if outcome == "SOURCE_EXCITATION_CONFOUNDED"
            else "current evidence is insufficient for a joint branch claim"
        ),
        "invalid_claims": [
            "higher excitation causes higher dynamic order",
            "B2, B3 or memory is absent from the physical system",
            "the learned graph is the IEEE39 physical topology",
            "observed-Pm rollout is an autonomous causal forecast",
        ],
        "test_ids_read": [],
        "other_folds_read": [],
        "merge_gate_requires_explicit_approval": True,
    }
    _json(result_dir / "acceptance.json", acceptance)
    (result_dir / "acceptance.md").write_text(
        "# M0-AFSD v3 联合选择有效性验收\n\n"
        f"**决定：{acceptance_decision}**\n\n"
        f"科学终态：`{outcome}`。\n\n"
        + acceptance["reason"]
        + "\n",
        encoding="utf-8",
    )
    _json(
        result_dir / "lineage.json",
        {
            "run_id": run_id,
            "stage": "acceptance",
            "v1_results_read": [],
            "v2_results_read": [],
            "test_ids_read": [],
            "other_folds_read": [],
            "canonical_files_modified": [],
        },
    )
    _write_hashes(result_dir)
    if status != "PASS":
        raise RuntimeError("v3 acceptance requires the authorized nested compute run")


def run(
    *,
    root: Path,
    experiment: dict[str, Any],
    stage: str,
    result_dir: Path,
    run_id: str,
) -> None:
    config_path = root / "configs" / "experiments" / str(experiment["config"])
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    _validate_scope(config)
    if stage == "hypothesis_registration":
        _run_hypothesis_registration(config_path, config, result_dir, run_id)
    elif stage == "eligibility_audit":
        _run_eligibility_audit(root, config_path, config, result_dir, run_id)
    elif stage == "controlled_experiment":
        _run_controlled_experiment(root, config, result_dir, run_id)
    elif stage == "residual_diagnosis":
        _run_residual_diagnosis(root, config, result_dir, run_id)
    elif stage == "adversarial_review":
        _run_adversarial_review(root, config, result_dir, run_id)
    elif stage == "acceptance":
        _run_acceptance(root, config, result_dir, run_id)
    else:
        raise ValueError(f"unsupported M0-AFSD v3 stage: {stage}")
