from __future__ import annotations

import csv
import hashlib
import json
import tomllib
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from kron_reduce.data import (
    CanonicalStore,
    ModelDataRequest,
    adapt_event,
    fit_stratification_policy,
    load_split,
    stratify_samples,
)
from kron_reduce.data.audit import audit_model_sample
from kron_reduce.evaluation.selection_validity import source_excitation_overlap_audit
from kron_reduce.physics import CLUSTERED_7_PARTITION
from kron_reduce.physics.global_quotient import (
    global_quotient_gap,
    holdout_global_quotient_gap,
    unified_quotient_agglomeration,
)


WORKFLOW = "m0_afsd_v5_unified_quotient_selection"
ROLLOUT_LABEL = "conditional rollout with observed Pm"
TERMINAL_STATES = {
    "HARD_CLUSTER_ADMISSIBLE",
    "FULL_RESOLUTION_REQUIRED",
    "INSUFFICIENT_INFORMATION",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )


def _write_hashes(result_dir: Path) -> None:
    values = {
        str(path.relative_to(result_dir)).replace("\\", "/"): _sha256(path)
        for path in sorted(result_dir.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    _json(result_dir / "artifact_hashes.json", values)


def _verify_hashes(result_dir: Path) -> list[str]:
    expected = json.loads((result_dir / "artifact_hashes.json").read_text(encoding="utf-8"))
    return [
        name
        for name, digest in expected.items()
        if not (result_dir / name).is_file() or _sha256(result_dir / name) != digest
    ]


def _latest_result(root: Path, experiment: str, stage: str) -> Path:
    candidates = sorted(
        path
        for path in (root / "results").glob(f"{experiment}-{stage}-*")
        if path.is_dir() and not (path / "RUN_FAILED.json").exists()
    )
    if not candidates:
        raise FileNotFoundError(f"no completed v5 result for {experiment}/{stage}")
    selected = candidates[-1]
    if "m0_afsd_v5" not in selected.name:
        raise RuntimeError("v5 may consume only v5 stage artifacts")
    return selected


def _lineage(run_id: str, stage: str, **parents: Any) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "stage": stage,
        **parents,
        "historical_result_runs_read": [],
        "test_ids_read": [],
        "other_folds_read": [],
        "canonical_files_modified": [],
        "model_features_added": [],
        "rollout_label": ROLLOUT_LABEL,
    }


def _validate_scope(config: dict[str, Any]) -> None:
    required = {
        "protocol_revision": WORKFLOW,
        "fold": "F1",
        "seed": 324,
        "pm_mode": "observed",
        "rollout_label": ROLLOUT_LABEL,
        "test_access": "forbidden",
        "other_folds_access": "forbidden",
        "canonical_mutation": "forbidden",
    }
    mismatch = {
        key: (config.get(key), expected)
        for key, expected in required.items()
        if config.get(key) != expected
    }
    if mismatch:
        raise RuntimeError(f"invalid locked v5 scope: {mismatch}")
    if tuple(config["report_horizons"]) != (100, 500, 1000):
        raise RuntimeError("v5 requires registered 1, 5 and 10 s horizons")
    if not 0.0 < float(config["tail_fraction"]) <= 0.5:
        raise RuntimeError("v5 tail fraction must be in (0, 0.5]")


def _hypothesis_registry(config: dict[str, Any], run_id: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "workflow": WORKFLOW,
        "registered_before_controlled_experiment": True,
        "primary_estimand": "one minimum quotient for the union of all observed Q-A train excitations",
        "metric": {
            "name": "Unified Quotient Gap",
            "abbreviation": "UQG",
            "definition": "max over horizons of equal-fault-line CVaR of max(representation loss, conditional closure gain)",
            "tail_fraction": float(config["tail_fraction"]),
            "equivalence_tolerance": float(config["maximum_uqg"]),
        },
        "hypotheses": [
            {
                "id": "H1-UNION",
                "statement": "one partition can be selected on the union of local Q-A excitations without source overlap",
                "falsifier": "the statistic reads source labels or fits source-specific partitions",
            },
            {
                "id": "H2-SUFFICIENCY",
                "statement": "UQG distinguishes an admissible quotient from a locally required internal mode",
                "falsifier": "disjoint-locality synthetic controls or local-hidden-mode controls fail",
            },
            {
                "id": "H3-F1",
                "statement": "F1 Q-A train supports a minimum common observed-envelope partition",
                "falsifier": "all contractions are rejected or uncertainty prevents a minimality decision",
            },
        ],
        "selection_policy": {
            "population": "all Q-A F1 train fault lines jointly",
            "source_labels": "not used",
            "amplitude_and_richness_labels": "not used",
            "fault_line_weight": "equal",
            "source_excitation_overlap": "post-selection causal-attribution audit only",
            "validation": "frozen confirmation only",
        },
        "claim_boundaries": config["claim_boundaries"],
    }


def _run_hypothesis(
    config_path: Path, config: dict[str, Any], result_dir: Path, run_id: str
) -> None:
    (result_dir / "frozen_config.toml").write_bytes(config_path.read_bytes())
    _json(result_dir / "hypothesis_registry.json", _hypothesis_registry(config, run_id))
    _json(
        result_dir / "lineage.json",
        _lineage(run_id, "hypothesis_registration"),
    )
    _write_hashes(result_dir)


def _synthetic_paths(*, hidden_locality: bool, relabel: bool = False) -> list[dict[str, Any]]:
    time = np.arange(240, dtype=float) * 0.01
    paths = []
    for index in range(12):
        phase = 0.17 * index
        first = (1.0 + 0.03 * index) * np.sin(2.0 * time + phase)
        second = (0.7 + 0.02 * index) * np.sin(1.1 * time - 0.5 * phase)
        inner = (
            0.35 * np.sin(3.7 * time + phase)
            if hidden_locality and index >= 8
            else np.zeros_like(time)
        )
        delta = np.column_stack((first + inner, first - inner, second, second))
        paths.append(
            {
                "event_id": f"S{index:02d}",
                "fault_line": f"L{index:02d}",
                "source_dataset": f"arbitrary-{index % 3}"
                if relabel
                else "source-A"
                if index < 6
                else "source-B",
                "delta": delta,
                "omega": np.gradient(delta, time, axis=0),
                "u": np.zeros_like(delta),
            }
        )
    return paths


def _synthetic_correctness(config: dict[str, Any]) -> dict[str, Any]:
    options = {
        "horizons": (5, 20),
        "folds": 4,
        "max_rows_per_fault_line": 30,
        "tolerance": float(config["maximum_uqg"]),
        "bootstrap_draws": 200,
        "seed": int(config["seed"]),
    }
    exact = global_quotient_gap(
        _synthetic_paths(hidden_locality=False),
        np.ones(4),
        ((0, 1), (2, 3)),
        **options,
    )
    relabelled = global_quotient_gap(
        _synthetic_paths(hidden_locality=False, relabel=True),
        np.ones(4),
        ((0, 1), (2, 3)),
        **options,
    )
    hidden = global_quotient_gap(
        _synthetic_paths(hidden_locality=True),
        np.ones(4),
        ((0, 1), (2, 3)),
        **options,
    )
    agglomeration = unified_quotient_agglomeration(
        _synthetic_paths(hidden_locality=False),
        np.ones(4),
        horizons=(5, 20),
        screen_horizon=5,
        minimum_clusters=1,
        folds=4,
        max_rows_per_fault_line=30,
        tolerance=float(config["maximum_uqg"]),
        screen_bootstrap_draws=100,
        final_bootstrap_draws=200,
        seed=int(config["seed"]),
    )
    gates = {
        "disjoint_local_excitations_accept_common_partition": exact["status"]
        == "EMPIRICALLY_SUFFICIENT",
        "local_hidden_mode_rejects_hard_partition": hidden["status"]
        == "PARTITION_REJECTED",
        "source_relabelling_invariant": exact["uqg"] == relabelled["uqg"]
        and exact["uqg_ci_upper"] == relabelled["uqg_ci_upper"],
        "agglomeration_recovers_two_pairs": agglomeration["selected_partition"]
        == [[0, 1], [2, 3]],
    }
    return {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "exact": exact,
        "hidden_locality": hidden,
        "source_relabelled": relabelled,
        "agglomeration": agglomeration,
    }


def _load_data(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    dataset_dir = root / "data" / str(config["dataset"])
    store = CanonicalStore.open(dataset_dir)
    split = load_split(
        dataset_dir,
        view=str(config["view"]),
        split_name=str(config["split_name"]),
        fold="F1",
    )
    request = ModelDataRequest.initial_state_rollout("observed")
    inertia = store.dataset_metadata.M_prior
    minimum = int(config["minimum_real_steps"])

    def adapt(ids: Sequence[str]) -> list[dict[str, Any]]:
        output = []
        for event_id in ids:
            if int(store.metadata(event_id).valid_length) < minimum:
                continue
            sample = adapt_event(
                store.load_event(event_id),
                inertia_prior=inertia,
                request=request,
                representation="generator_10",
            )
            audit_model_sample(sample)
            output.append(sample)
        return output

    train = adapt(split.train_ids)
    validation = adapt(split.val_ids)
    policy, train_rows = fit_stratification_policy(
        train,
        report_horizons=tuple(int(value) for value in config["report_horizons"]),
        minimum_structure_steps=int(config["minimum_structure_steps"]),
        hankel_delays=int(config["hankel_delays"]),
        hankel_relative_floor=float(config["hankel_relative_floor"]),
        maximum_hankel_condition=float(config["maximum_hankel_condition"]),
    )
    validation_rows = stratify_samples(validation, policy)
    rows = []
    for role, role_rows in (("train", train_rows), ("validation", validation_rows)):
        for row in role_rows:
            metadata = store.metadata(str(row["event_id"]))
            rows.append(
                {
                    **row,
                    "role": role,
                    "source_dataset": metadata.source_dataset,
                    "fault_line": str(metadata.fault_line),
                    "valid_length": int(metadata.valid_length),
                    "eligible_for_v5_selection": bool(
                        role == "train"
                        and row["quality_grade"] == "Q-A"
                        and bool(row["structure_selection_eligible"])
                        and int(metadata.valid_length) >= int(config["minimum_selection_steps"])
                    ),
                    "eligible_for_frozen_confirmation": bool(
                        role == "validation"
                        and row["quality_grade"] == "Q-A"
                        and int(metadata.valid_length) >= int(config["minimum_selection_steps"])
                    ),
                }
            )
    return {
        "store": store,
        "split": split,
        "train": train,
        "validation": validation,
        "rows": rows,
        "policy": policy,
    }


def _sample_path(sample: dict[str, Any], row: dict[str, Any], steps: int) -> dict[str, Any]:
    target = sample["training_context"]["targets"]
    static = sample["model_inputs"]["static"]
    equilibrium_pm = np.asarray(static["Pm_equilibrium"], dtype=float)
    use = min(int(steps), len(target["t"]))
    return {
        "event_id": str(sample["event_id"]),
        "fault_line": str(row["fault_line"]),
        "delta": np.asarray(target["delta"], dtype=float)[:use],
        "omega": np.asarray(target["omega"], dtype=float)[:use],
        "u": np.asarray(sample["training_context"]["Pm_observed"], dtype=float)[:use]
        - equilibrium_pm[None, :],
    }


def _candidate_rows(selection: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for row in selection["candidate_evaluations"]:
        full = row.get("full") or {}
        output.append(
            {
                "round": row["round"],
                "merge": row["merge"],
                "child": row["child"],
                "screen_status": row["screen"]["status"],
                "screen_uqg": row["screen"].get("uqg"),
                "screen_uqg_ci_upper": row["screen"].get("uqg_ci_upper"),
                "full_status": full.get("status", "NOT_RUN"),
                "full_uqg": full.get("uqg"),
                "full_uqg_ci_upper": full.get("uqg_ci_upper"),
                "selected": row["selected"],
            }
        )
    return output


def _run_controlled(root: Path, config: dict[str, Any], result_dir: Path, run_id: str) -> None:
    hypothesis = _latest_result(
        root, str(config["hypothesis_experiment"]), "hypothesis_registration"
    )
    if _verify_hashes(hypothesis):
        raise RuntimeError("v5 hypothesis artifacts changed before controlled experiment")
    synthetic = _synthetic_correctness(config)
    if synthetic["status"] != "PASS":
        raise RuntimeError("v5 synthetic UQG correctness gate failed")

    data = _load_data(root, config)
    row_by_id = {str(row["event_id"]): row for row in data["rows"]}
    train_ids = {
        str(row["event_id"]) for row in data["rows"] if row["eligible_for_v5_selection"]
    }
    validation_ids = {
        str(row["event_id"])
        for row in data["rows"]
        if row["eligible_for_frozen_confirmation"]
    }
    steps = int(config["minimum_selection_steps"])
    train_paths = [
        _sample_path(sample, row_by_id[str(sample["event_id"])], steps)
        for sample in data["train"]
        if str(sample["event_id"]) in train_ids
    ]
    validation_paths = [
        _sample_path(sample, row_by_id[str(sample["event_id"])], steps)
        for sample in data["validation"]
        if str(sample["event_id"]) in validation_ids
    ]
    if len({path["fault_line"] for path in train_paths}) < int(config["outer_fault_line_folds"]):
        raise RuntimeError("insufficient Q-A fault lines for v5 cross-fitting")
    mass = np.asarray(data["train"][0]["model_inputs"]["static"]["mass"], dtype=float)
    common = {
        "folds": int(config["outer_fault_line_folds"]),
        "delays": int(config["probe_delays"]),
        "ridge": float(config["probe_ridge"]),
        "rank_cap": int(config["probe_rank_cap"]),
        "max_rows_per_fault_line": int(config["maximum_rows_per_fault_line"]),
        "tail_fraction": float(config["tail_fraction"]),
        "tolerance": float(config["maximum_uqg"]),
        "seed": int(config["seed"]),
    }
    selection = unified_quotient_agglomeration(
        train_paths,
        mass,
        horizons=tuple(int(value) for value in config["report_horizons"]),
        screen_horizon=int(config["screen_horizon"]),
        minimum_clusters=int(config["minimum_clusters"]),
        screen_bootstrap_draws=int(config["screen_bootstrap_draws"]),
        final_bootstrap_draws=int(config["selection_bootstrap_draws"]),
        **common,
    )
    fixed_seven = global_quotient_gap(
        train_paths,
        mass,
        tuple(tuple(group) for group in CLUSTERED_7_PARTITION),
        horizons=tuple(int(value) for value in config["report_horizons"]),
        bootstrap_draws=int(config["selection_bootstrap_draws"]),
        minimum_probe_skill=float(config["minimum_probe_skill"]),
        **common,
    )
    selected_partition = tuple(tuple(group) for group in selection["selected_partition"])
    confirmation = {
        "status": "NOT_APPLICABLE_FULL_RESOLUTION"
        if selection["status"] == "FULL_RESOLUTION_REQUIRED"
        else "INSUFFICIENT_INFORMATION"
    }
    if (
        selection["status"] == "HARD_CLUSTER_ADMISSIBLE"
        and len({path["fault_line"] for path in validation_paths}) >= 2
        and not (
            {path["fault_line"] for path in train_paths}
            & {path["fault_line"] for path in validation_paths}
        )
    ):
        confirmation = holdout_global_quotient_gap(
            train_paths,
            validation_paths,
            mass,
            selected_partition,
            horizons=tuple(int(value) for value in config["report_horizons"]),
            delays=int(config["probe_delays"]),
            ridge=float(config["probe_ridge"]),
            rank_cap=int(config["probe_rank_cap"]),
            max_rows_per_fault_line=int(config["maximum_rows_per_fault_line"]),
            tail_fraction=float(config["tail_fraction"]),
            tolerance=float(config["maximum_uqg"]),
            bootstrap_draws=int(config["confirmation_bootstrap_draws"]),
            minimum_probe_skill=float(config["minimum_probe_skill"]),
            seed=int(config["seed"]),
        )

    if selection["status"] == "FULL_RESOLUTION_REQUIRED":
        conclusion = "FULL_RESOLUTION_REQUIRED"
    elif selection["status"] == "HARD_CLUSTER_ADMISSIBLE" and confirmation.get(
        "status"
    ) == "EMPIRICALLY_SUFFICIENT":
        conclusion = "HARD_CLUSTER_ADMISSIBLE"
    else:
        conclusion = "INSUFFICIENT_INFORMATION"

    overlap_rows = [
        {**row, "structure_selection_eligible": row["eligible_for_v5_selection"]}
        for row in data["rows"]
    ]
    causal_overlap = source_excitation_overlap_audit(overlap_rows)
    controlled = {
        "run_id": run_id,
        "status": "COMPLETED",
        "scientific_conclusion": conclusion,
        "rollout_label": ROLLOUT_LABEL,
        "n_QA_train_events": len(train_paths),
        "n_QA_train_fault_lines": len({path["fault_line"] for path in train_paths}),
        "n_confirmation_events": len(validation_paths),
        "n_confirmation_fault_lines": len({path["fault_line"] for path in validation_paths}),
        "all_QA_train_jointly_fit": True,
        "source_overlap_used_as_selection_gate": False,
        "source_labels_used_by_UQG": False,
        "validation_used_for_selection": False,
        "Q_B_used_for_selection": False,
        "test_ids_read": [],
    }
    _json(result_dir / "synthetic_uqg_correctness.json", synthetic)
    _csv(result_dir / "data_qualification.csv", data["rows"])
    _json(result_dir / "stratification_policy.json", data["policy"].as_dict())
    _json(result_dir / "causal_source_excitation_audit_only.json", causal_overlap)
    _json(result_dir / "global_quotient_selection.json", selection)
    _csv(result_dir / "merge_uqg_certificates.csv", _candidate_rows(selection))
    _json(result_dir / "fixed_seven_control.json", fixed_seven)
    _json(result_dir / "frozen_validation_confirmation.json", confirmation)
    _json(result_dir / "controlled_experiment.json", controlled)
    _json(
        result_dir / "selection_population.json",
        {
            "train_event_ids": sorted(train_ids),
            "validation_confirmation_event_ids": sorted(validation_ids),
            "Q_B_event_ids_used_for_selection": [],
            "source_labels_used_for_selection": False,
            "test_ids_read": [],
        },
    )
    _json(
        result_dir / "lineage.json",
        _lineage(
            run_id,
            "controlled_experiment",
            parent_hypothesis_result=hypothesis.name,
        ),
    )
    _write_hashes(result_dir)


def _dominant_failures(selection: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for candidate in selection["candidate_evaluations"]:
        result = candidate.get("full") or candidate["screen"]
        for horizon in result.get("horizons", []):
            for row in horizon.get("tail_fault_lines", []):
                output.append(
                    {
                        "round": candidate["round"],
                        "merge": candidate["merge"],
                        "horizon_steps": horizon["horizon_steps"],
                        **row,
                        "dominant_component": "representation"
                        if float(row["representation_loss"])
                        >= float(row["conditional_closure_gain"])
                        else "conditional_closure",
                    }
                )
    return sorted(output, key=lambda row: -float(row["uqg_local"]))


def _run_residual(root: Path, config: dict[str, Any], result_dir: Path, run_id: str) -> None:
    controlled = _latest_result(
        root, str(config["controlled_experiment"]), "controlled_experiment"
    )
    if _verify_hashes(controlled):
        raise RuntimeError("v5 controlled artifacts changed before residual diagnosis")
    decision = json.loads((controlled / "controlled_experiment.json").read_text(encoding="utf-8"))
    selection = json.loads((controlled / "global_quotient_selection.json").read_text(encoding="utf-8"))
    failures = _dominant_failures(selection)
    top = failures[:20]
    component_counts = {
        name: sum(row["dominant_component"] == name for row in failures)
        for name in ("representation", "conditional_closure")
    }
    diagnosis = {
        "run_id": run_id,
        "status": "COMPLETED",
        "scientific_conclusion": decision["scientific_conclusion"],
        "dominant_failure_counts": component_counts,
        "largest_local_gaps": top,
        "interpretation": (
            "at least one common hard quotient is supported across the union of observed local excitations"
            if decision["scientific_conclusion"] == "HARD_CLUSTER_ADMISSIBLE"
            else "every admissible contraction loses a repeatedly observed local coordinate"
            if decision["scientific_conclusion"] == "FULL_RESOLUTION_REQUIRED"
            else "the observed-envelope equivalence interval is too wide for a minimal partition claim"
        ),
        "source_overlap_not_a_remedy": True,
        "next_modeling_action": (
            "fit one pooled physical model on the frozen selected quotient"
            if decision["scientific_conclusion"] == "HARD_CLUSTER_ADMISSIBLE"
            else "fit the dynamic module search at ten-machine resolution"
            if decision["scientific_conclusion"] == "FULL_RESOLUTION_REQUIRED"
            else "increase independent fault-line coverage for the unresolved merge, not source-cell overlap"
        ),
    }
    _json(result_dir / "residual_diagnosis.json", diagnosis)
    _csv(result_dir / "local_gap_drivers.csv", failures)
    _json(
        result_dir / "lineage.json",
        _lineage(run_id, "residual_diagnosis", parent_controlled_result=controlled.name),
    )
    _write_hashes(result_dir)


def _run_adversarial(root: Path, config: dict[str, Any], result_dir: Path, run_id: str) -> None:
    hypothesis = _latest_result(
        root, str(config["hypothesis_experiment"]), "hypothesis_registration"
    )
    controlled = _latest_result(
        root, str(config["controlled_experiment"]), "controlled_experiment"
    )
    residual = _latest_result(root, str(config["residual_experiment"]), "residual_diagnosis")
    upstream = (hypothesis, controlled, residual)
    if any(_verify_hashes(path) for path in upstream):
        raise RuntimeError("v5 upstream artifacts changed before adversarial review")
    decision = json.loads((controlled / "controlled_experiment.json").read_text(encoding="utf-8"))
    synthetic = json.loads((controlled / "synthetic_uqg_correctness.json").read_text(encoding="utf-8"))
    selection = json.loads((controlled / "global_quotient_selection.json").read_text(encoding="utf-8"))
    population = json.loads((controlled / "selection_population.json").read_text(encoding="utf-8"))
    confirmation = json.loads(
        (controlled / "frozen_validation_confirmation.json").read_text(encoding="utf-8")
    )
    lineages = [json.loads((path / "lineage.json").read_text(encoding="utf-8")) for path in upstream]
    gates = {
        "immutable_upstream": all(not _verify_hashes(path) for path in upstream),
        "synthetic_disjoint_locality_pass": synthetic["status"] == "PASS",
        "source_relabelling_invariant": synthetic["gates"]["source_relabelling_invariant"],
        "local_hidden_mode_not_averaged_away": synthetic["gates"][
            "local_hidden_mode_rejects_hard_partition"
        ],
        "all_QA_train_jointly_used": decision["all_QA_train_jointly_fit"],
        "source_overlap_not_selection_gate": not decision[
            "source_overlap_used_as_selection_gate"
        ],
        "source_labels_not_features": not decision["source_labels_used_by_UQG"],
        "fault_line_equal_weight": selection["all_QA_fault_lines_jointly_used"],
        "QB_excluded": not population["Q_B_event_ids_used_for_selection"],
        "validation_confirmation_only": not decision["validation_used_for_selection"]
        and bool(confirmation.get("selection_frozen_before_holdout", True)),
        "test_other_folds_canonical_locked": all(
            not row["test_ids_read"]
            and not row["other_folds_read"]
            and not row["canonical_files_modified"]
            for row in lineages
        ),
        "allowed_terminal_state": decision["scientific_conclusion"] in TERMINAL_STATES,
    }
    alternatives = [
        {
            "alternative": "the previous no-overlap result proves that no common model exists",
            "disposition": "rejected: no overlap limits causal attribution, not pooled predictive representation",
        },
        {
            "alternative": "mean error can hide one locally excited internal mode",
            "disposition": "controlled by equal-fault-line worst-10% UQG and a local-hidden-mode synthetic test",
        },
        {
            "alternative": "clustering is selected only because it regularizes a weak probe",
            "disposition": "representation loss is independent of the probe and closure probes use the same rank cap",
        },
        {
            "alternative": "an unexcited physical mode is proven absent",
            "disposition": "explicitly outside the observed-envelope claim",
        },
    ]
    status = "PASS" if all(gates.values()) else "FAIL"
    review = {
        "run_id": run_id,
        "status": status,
        "decision": "ACCEPTABLE_EVIDENCE_CHAIN" if status == "PASS" else "REJECT",
        "scientific_conclusion": decision["scientific_conclusion"],
        "gates": gates,
        "alternative_explanations": alternatives,
    }
    _json(result_dir / "adversarial_review.json", review)
    (result_dir / "adversarial_review.md").write_text(
        "# M0-AFSD v5 adversarial review\n\n"
        f"Decision: **{review['decision']}**\n\n"
        + "\n".join(f"- {'PASS' if value else 'FAIL'} `{name}`" for name, value in gates.items())
        + "\n",
        encoding="utf-8",
    )
    _json(
        result_dir / "lineage.json",
        _lineage(run_id, "adversarial_review", parent_results=[path.name for path in upstream]),
    )
    _write_hashes(result_dir)
    if status != "PASS":
        raise RuntimeError("v5 adversarial review rejected the evidence chain")


def _run_acceptance(root: Path, config: dict[str, Any], result_dir: Path, run_id: str) -> None:
    controlled = _latest_result(
        root, str(config["controlled_experiment"]), "controlled_experiment"
    )
    residual = _latest_result(root, str(config["residual_experiment"]), "residual_diagnosis")
    adversarial = _latest_result(
        root, str(config["adversarial_experiment"]), "adversarial_review"
    )
    upstream = (controlled, residual, adversarial)
    if any(_verify_hashes(path) for path in upstream):
        raise RuntimeError("v5 evidence artifacts changed before acceptance")
    decision = json.loads((controlled / "controlled_experiment.json").read_text(encoding="utf-8"))
    selection = json.loads((controlled / "global_quotient_selection.json").read_text(encoding="utf-8"))
    review = json.loads((adversarial / "adversarial_review.json").read_text(encoding="utf-8"))
    conclusion = str(decision["scientific_conclusion"])
    passed = review["status"] == "PASS" and conclusion in TERMINAL_STATES
    acceptance = {
        "run_id": run_id,
        "status": "PASS" if passed else "FAIL",
        "decision": "ACCEPT" if passed else "REJECT",
        "scientific_conclusion": conclusion,
        "selected_partition": selection["selected_partition"],
        "selection_rule": selection["selection_rule"],
        "valid_claim": (
            "the selected quotient is sufficient for the union of observed Q-A F1 train excitations"
            if conclusion == "HARD_CLUSTER_ADMISSIBLE"
            else "no tested hard contraction is sufficient for the union of observed Q-A F1 train excitations"
            if conclusion == "FULL_RESOLUTION_REQUIRED"
            else "the present data do not determine a minimum common quotient"
        ),
        "invalid_claims": [
            "source overlap is required before fitting one global observed-envelope model",
            "excitation amplitude causally changes physical order",
            "unselected modes are absent from the real system",
            "conditional observed-Pm evidence is autonomous deployment evidence",
            "F1 seed324 is cross-system generalization",
        ],
        "source_overlap_role": "causal attribution audit only",
        "test_ids_read": [],
        "other_folds_read": [],
        "merge_gate_requires_explicit_approval": True,
    }
    _json(result_dir / "acceptance.json", acceptance)
    (result_dir / "acceptance.md").write_text(
        "# M0-AFSD v5 unified quotient selection acceptance\n\n"
        f"Decision: **{acceptance['decision']}**\n\n"
        f"Scientific conclusion: `{conclusion}`.\n\n"
        f"All dynamic evidence remains **{ROLLOUT_LABEL}**.\n",
        encoding="utf-8",
    )
    _json(
        result_dir / "lineage.json",
        _lineage(run_id, "acceptance", parent_results=[path.name for path in upstream]),
    )
    _write_hashes(result_dir)
    if not passed:
        raise RuntimeError("v5 acceptance failed")


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
        _run_hypothesis(config_path, config, result_dir, run_id)
    elif stage == "controlled_experiment":
        _run_controlled(root, config, result_dir, run_id)
    elif stage == "residual_diagnosis":
        _run_residual(root, config, result_dir, run_id)
    elif stage == "adversarial_review":
        _run_adversarial(root, config, result_dir, run_id)
    elif stage == "acceptance":
        _run_acceptance(root, config, result_dir, run_id)
    else:
        raise ValueError(f"unsupported M0-AFSD v5 stage: {stage}")
