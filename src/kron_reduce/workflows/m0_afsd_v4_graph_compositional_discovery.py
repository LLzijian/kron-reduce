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
from kron_reduce.evaluation.compositional_search import (
    SearchState,
    add_module,
    deterministic_beam_search,
    exhaustive_oracle,
    module_subset_states,
    oracle_agreement,
    paired_module_mcts_trigger,
)
from kron_reduce.evaluation.selection_validity import (
    nested_fault_line_folds,
    source_excitation_overlap_audit,
    structure_selection_ids,
)
from kron_reduce.models.compositional_modules import (
    module_registry,
    residual_module_proposal,
    validate_module_registry,
)
from kron_reduce.physics.graph_quotient import (
    beam_graph_contraction,
    dynamic_distance_matrix,
    input_heterogeneity,
    internal_mode_basis,
    intra_cluster_energy,
    projection_certificate,
)

WORKFLOW = "m0_afsd_v4_graph_compositional_discovery"
ROLLOUT_LABEL = "conditional rollout with observed Pm"
TERMINAL_STATES = {
    "SOURCE_EXCITATION_CONFOUNDED",
    "INSUFFICIENT_EXCITATION",
    "INSUFFICIENT_INFORMATION",
    "MODEL_CLASS_INADEQUATE",
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


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_hashes(result_dir: Path) -> None:
    artifacts = {
        str(path.relative_to(result_dir)).replace("\\", "/"): _sha256(path)
        for path in sorted(result_dir.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    _json(result_dir / "artifact_hashes.json", artifacts)


def _verify_hashes(result_dir: Path) -> list[str]:
    manifest = json.loads((result_dir / "artifact_hashes.json").read_text(encoding="utf-8"))
    return [
        relative
        for relative, expected in manifest.items()
        if not (result_dir / relative).is_file() or _sha256(result_dir / relative) != expected
    ]


def _latest_result(root: Path, experiment: str, stage: str) -> Path:
    prefix = f"{experiment}-{stage}-"
    candidates = sorted(
        path
        for path in (root / "results").glob(f"{prefix}*")
        if path.is_dir() and not (path / "RUN_FAILED.json").exists()
    )
    if not candidates:
        raise FileNotFoundError(f"no completed v4 result for {experiment}/{stage}")
    selected = candidates[-1]
    if "m0_afsd_v4" not in selected.name:
        raise RuntimeError("v4 may consume only v4 stage artifacts")
    return selected


def _lineage(run_id: str, stage: str, **parents: Any) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "stage": stage,
        **parents,
        "historical_v1_v2_v3_results_read": [],
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
        raise RuntimeError(f"invalid locked v4 scope: {mismatch}")
    if int(config["search_bootstrap_draws"]) < 500:
        raise RuntimeError("v4 search requires at least 500 cluster bootstraps")
    if int(config["final_bootstrap_draws"]) < 5000:
        raise RuntimeError("v4 final audit requires at least 5000 cluster bootstraps")
    if list(config["beam_width_candidates"]) != [3, 5, 10]:
        raise RuntimeError("v4 Beam widths must remain preregistered as 3, 5 and 10")


def _hypothesis_registry(config: dict[str, Any], run_id: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "workflow": WORKFLOW,
        "registered_before_controlled_experiment": True,
        "rollout_label": ROLLOUT_LABEL,
        "hypotheses": [
            {
                "id": "H1-GRAPH-QUOTIENT",
                "statement": "Q-A F1 train supports a lower-dimensional predictive graph quotient",
                "falsifier": "CPIG, closure, modal, input or stability certificate rejects every contraction",
            },
            {
                "id": "H2-INTERNAL-MODES",
                "statement": "a failed hard quotient can be repaired by a small mass-orthogonal internal basis",
                "falsifier": "full resolution remains necessary or the internal basis is bootstrap-unstable",
            },
            {
                "id": "H3-MODULE-LOCATION",
                "statement": "repeatable residual signatures locate a registered static or dynamic module",
                "falsifier": "signatures are unstable, observationally equivalent, or outside the grammar",
            },
            {
                "id": "H4-GRAPH-SUPPORT",
                "statement": "a sparse reciprocal dynamic-equivalent edge support is identifiable",
                "falsifier": "support, Jaccard, knockout or reciprocity gates fail",
            },
            {
                "id": "H5-SEARCH",
                "statement": "finite Beam search matches a small exhaustive oracle with fewer full fits",
                "falsifier": "oracle disagreement, order dependence, or unresolved pair synergy",
            },
            {
                "id": "H6-DESIGN",
                "statement": "source and excitation have enough Q-A overlap for a joint F1 structure claim",
                "falsifier": "any represented cell has fewer than five fault-line clusters per source",
                "terminal_if_falsified": "SOURCE_EXCITATION_CONFOUNDED",
            },
        ],
        "qualification": {
            "Q-A": "partition, order, graph and module support",
            "Q-B": "post-freeze robustness and extrapolation only",
            "Q-C": "excluded",
        },
        "selection_order": [
            "data_qualification",
            "physics_and_numerics",
            "10s_rollout",
            "1s_and_5s_rollout",
            "relative_modes",
            "absolute_modes",
            "structure_stability",
            "description_length",
        ],
        "beam_widths": list(config["beam_width_candidates"]),
        "MCTS_policy": "disabled unless all preregistered Beam widths miss a paired-module oracle case",
        "stopping_states": sorted(
            {
                "JOINT_GRAPH_MODULE_BRANCH_IDENTIFIED",
                "HARD_CLUSTER_ADMISSIBLE",
                "CLUSTER_PLUS_INTERNAL_MODES_IDENTIFIED",
                "FULL_RESOLUTION_REQUIRED",
                "STRUCTURE_EQUIVALENCE_CLASS",
                "SOURCE_EXCITATION_CONFOUNDED",
                "INSUFFICIENT_EXCITATION",
                "PARTITION_UNSTABLE",
                "JOINT_SELECTION_UNSTABLE",
                "PARAMETER_NOT_IDENTIFIABLE",
                "GRAPH_SUPPORT_NOT_IDENTIFIABLE",
                "MODEL_CLASS_INADEQUATE",
                "SEARCH_METHOD_INADEQUATE",
                "INSUFFICIENT_INFORMATION",
            }
        ),
        "claim_boundaries": config["claim_boundaries"],
    }


def _run_hypothesis(
    config_path: Path, config: dict[str, Any], result_dir: Path, run_id: str
) -> None:
    (result_dir / "frozen_config.toml").write_bytes(config_path.read_bytes())
    _json(result_dir / "hypothesis_registry.json", _hypothesis_registry(config, run_id))
    registry = module_registry()
    _json(
        result_dir / "physical_module_grammar.json",
        {
            "status": validate_module_registry(registry)["status"],
            "neutral_semantics_only": True,
            "modules": {name: spec.to_dict() for name, spec in registry.items()},
        },
    )
    _json(
        result_dir / "registered_search_actions.json",
        {
            "allowed": [
                "add_module",
                "remove_module",
                "replace_module",
                "split_timescale",
                "merge_timescale",
                "add_edge",
                "remove_edge",
                "merge_nodes",
                "split_cluster",
                "add_internal_mode",
                "remove_internal_mode",
            ],
            "arbitrary_state_equations": "forbidden",
        },
    )
    _json(result_dir / "lineage.json", _lineage(run_id, "hypothesis_registration"))
    _write_hashes(result_dir)


def _synthetic_paths(*, hidden: bool, heterogeneous_input: bool = False) -> list[dict[str, Any]]:
    result = []
    time = np.arange(180, dtype=float) * 0.01
    for index in range(8):
        phase = 0.23 * index
        common_a = np.sin(2.0 * time + phase)
        common_b = np.cos(1.2 * time - phase)
        inner = 0.25 * np.sin(5.0 * time + phase) if hidden else np.zeros_like(time)
        delta = np.column_stack((common_a + inner, common_a - inner, common_b, common_b))
        omega = np.gradient(delta, time, axis=0)
        forcing = np.zeros_like(delta)
        forcing[:, 0] = np.sin(0.8 * time)
        forcing[:, 1] = forcing[:, 0] if not heterogeneous_input else -forcing[:, 0]
        result.append({"delta": delta, "omega": omega, "u": forcing, "fault_line": f"S{index}"})
    return result


def _synthetic_graph_gates(config: dict[str, Any]) -> dict[str, Any]:
    mass = np.array([2.0, 3.0, 5.0, 7.0])
    partition = ((0, 1), (2, 3))
    coherent = _synthetic_paths(hidden=False)
    hidden = _synthetic_paths(hidden=True)
    exact = intra_cluster_energy(coherent, mass, partition)
    nonaggregate = intra_cluster_energy(hidden, mass, partition)
    basis, basis_audit = internal_mode_basis(hidden, mass, partition, 1)
    heterogeneous = input_heterogeneity(
        _synthetic_paths(hidden=False, heterogeneous_input=True), mass, partition
    )
    projection = projection_certificate(mass, partition)
    full = projection_certificate(mass, ((0,), (1,), (2,), (3,)))
    distance = np.array(
        [[0.0, 0.1, 4.0, 4.0], [0.1, 0.0, 4.0, 4.0], [4.0, 4.0, 0.0, 0.1], [4.0, 4.0, 0.1, 0.0]]
    )

    def certificate(candidate):
        groups = {frozenset(group) for group in candidate}
        admissible = groups.issubset(
            {
                frozenset({0}),
                frozenset({1}),
                frozenset({2}),
                frozenset({3}),
                frozenset({0, 1}),
                frozenset({2, 3}),
            }
        )
        return {"status": "HARD_CLUSTER_ADMISSIBLE" if admissible else "FULL_RESOLUTION_REQUIRED"}

    contraction = beam_graph_contraction(distance, certificate, beam_width=3, minimum_clusters=1)
    gates = {
        "G1_projector_algebra": projection["status"] == "PASS",
        "G1_full_resolution_exact": full["full_resolution_exact"] is True,
        "G1_exact_partition": max(exact.values()) < 1e-12,
        "G2_nonaggregable_rejected": nonaggregate["delta"] > 0.01,
        "G2_internal_mode_recovered": basis_audit["status"] == "completed"
        and basis.shape == (4, 1),
        "G2_input_heterogeneity_detected": heterogeneous > 0.4,
        "G2_contraction_oracle": contraction["best_partition"] == [[0, 1], [2, 3]],
    }
    return {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "projector": projection,
        "exact_partition_intra_energy": exact,
        "nonaggregable_intra_energy": nonaggregate,
        "input_heterogeneity": heterogeneous,
        "internal_mode_audit": basis_audit,
        "contraction": contraction,
    }


def _search_evaluator(target: tuple[str, ...], *, pair_synergy: bool = False):
    def evaluate(state: SearchState, tier: int) -> dict[str, Any]:
        selected = set(state.active_modules)
        truth = set(target)
        if pair_synergy and truth and not truth.issubset(selected):
            error = 2.0 + 0.1 * len(selected)
        else:
            error = float(len(selected ^ truth))
        return {
            "hard_gate": "PASS",
            "heldout_error": error,
            "description_length": len(selected),
            "tier": int(tier),
        }

    return evaluate


def _synthetic_module_search_gates(config: dict[str, Any]) -> dict[str, Any]:
    time = np.linspace(0.0, 8.0, 800)
    signal = np.sin(2.0 * time) + 0.2 * np.sin(4.0 * time)
    lagged = np.roll(signal, 8)
    lagged[:8] = 0.0
    static = residual_module_proposal(signal, edge_angle=signal)
    memory = residual_module_proposal(lagged, edge_speed=signal)
    unexcited = residual_module_proposal(np.zeros_like(signal), edge_speed=signal)
    saturation = residual_module_proposal(
        np.clip(np.linspace(-2, 2, len(signal)), -0.4, 0.4),
        node_speed=np.linspace(-2, 2, len(signal)),
    )
    initial = SearchState(((0,), (1,)))
    modules = ("EDGE_DAMPING", "EDGE_LAG_STORAGE", "NODE_DAMPING")
    evaluator = _search_evaluator(("EDGE_LAG_STORAGE",))

    def actions(state):
        for module in modules:
            if module not in state.active_modules:
                yield add_module(state, module)

    beam = deterministic_beam_search(initial, actions, evaluator, beam_width=3, max_depth=3)
    oracle = exhaustive_oracle(module_subset_states(initial, modules), evaluator)
    comparison = oracle_agreement(beam, oracle)
    pair_modules = tuple(f"M{index}" for index in range(12))
    mcts = paired_module_mcts_trigger(
        initial,
        pair_modules,
        _search_evaluator(("M8", "M9"), pair_synergy=True),
        beam_widths=tuple(int(value) for value in config["beam_width_candidates"]),
    )
    gates = {
        "G3_registry": validate_module_registry()["status"] == "PASS",
        "G3_static_location": static["proposals"] == ["STATIC_NONLINEAR_POTENTIAL"],
        "G3_edge_memory_location": memory["proposals"] == ["EDGE_LAG_STORAGE"],
        "G3_unexcited_rejected": unexcited["status"] == "INSUFFICIENT_INFORMATION",
        "G4_saturation_not_polynomial": saturation["proposals"] == ["SATURATION_OR_DEADBAND"],
        "G5_beam_matches_oracle": comparison["status"] == "PASS",
        "G5_fewer_full_fits": comparison["full_fit_reduction"] > 0,
        "G5_mcts_trigger_detected": mcts["status"] == "MCTS_TRIGGERED",
    }
    return {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "module_proposals": {
            "static": static,
            "edge_memory": memory,
            "unexcited": unexcited,
            "saturation": saturation,
        },
        "beam": beam,
        "oracle": oracle,
        "oracle_comparison": comparison,
        "mcts_trigger_audit": mcts,
    }


def _load_data(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    dataset_dir = root / "data" / str(config["dataset"])
    store = CanonicalStore.open(dataset_dir)
    split = load_split(
        dataset_dir, view=str(config["view"]), split_name=str(config["split_name"]), fold="F1"
    )
    minimum = int(config["minimum_real_steps"])
    train_ids = [
        event_id for event_id in split.train_ids if store.metadata(event_id).valid_length >= minimum
    ]
    validation_ids = [
        event_id for event_id in split.val_ids if store.metadata(event_id).valid_length >= minimum
    ]
    request = ModelDataRequest.initial_state_rollout("observed")
    inertia = store.dataset_metadata.M_prior

    def adapt(ids: Sequence[str]) -> list[dict[str, Any]]:
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

    train = adapt(train_ids)
    validation = adapt(validation_ids)
    policy, train_rows = fit_stratification_policy(
        train,
        report_horizons=tuple(int(value) for value in config["report_horizons"]),
        minimum_structure_steps=int(config["minimum_structure_steps"]),
        hankel_delays=int(config["hankel_delays"]),
        hankel_relative_floor=float(config["hankel_relative_floor"]),
        maximum_hankel_condition=float(config["maximum_hankel_condition"]),
    )
    return {
        "store": store,
        "split": split,
        "train": train,
        "validation": validation,
        "policy": policy,
        "train_rows": train_rows,
        "validation_rows": stratify_samples(validation, policy),
    }


def _qualification_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    store = data["store"]
    for role, strata in (("train", data["train_rows"]), ("validation", data["validation_rows"])):
        for row in strata:
            metadata = store.metadata(str(row["event_id"]))
            rows.append(
                {
                    **row,
                    "role": role,
                    "source_dataset": metadata.source_dataset,
                    "fault_line": metadata.fault_line,
                    "valid_length": int(metadata.valid_length),
                    "valid_horizon": max(
                        horizon
                        for horizon in (0, 100, 500, 1000)
                        if int(metadata.valid_length) >= horizon + 1
                    ),
                    "qualification_use": (
                        "structure_partition_graph_module_support"
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
                    "valid_horizon": 0,
                    "quality_grade": "Q-C",
                    "amplitude_level": "A-uncertain",
                    "richness_level": "R-uncertain",
                    "structure_selection_eligible": False,
                    "coefficient_fit_eligible": False,
                    "qualification_use": "excluded",
                    "exclusion_reason": "insufficient_valid_length",
                }
            )
    return sorted(rows, key=lambda row: (str(row["role"]), str(row["event_id"])))


def _sample_path(
    sample: dict[str, Any], fault_line: str, maximum_steps: int = 1001
) -> dict[str, Any]:
    target = sample["training_context"]["targets"]
    steps = min(int(maximum_steps), len(target["t"]))
    equilibrium_pm = np.asarray(sample["model_inputs"]["static"]["Pm_equilibrium"], dtype=float)
    return {
        "event_id": str(sample["event_id"]),
        "fault_line": str(fault_line),
        "delta": np.asarray(target["delta"], dtype=float)[:steps],
        "omega": np.asarray(target["omega"], dtype=float)[:steps],
        "u": np.asarray(sample["training_context"]["Pm_observed"], dtype=float)[:steps]
        - equilibrium_pm[None, :],
    }


def _f1_preliminary_spatial_proposals(
    data: dict[str, Any], rows: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    selection_ids = structure_selection_ids(rows)
    row_by_id = {str(row["event_id"]): row for row in rows}
    samples = [sample for sample in data["train"] if str(sample["event_id"]) in selection_ids]
    if not samples:
        return {"status": "INSUFFICIENT_EXCITATION", "tree": [], "paths": []}
    paths = [
        _sample_path(sample, str(row_by_id[str(sample["event_id"])]["fault_line"]))
        for sample in samples
    ]
    mass = np.asarray(samples[0]["model_inputs"]["static"]["mass"], dtype=float)
    distance, distance_audit = dynamic_distance_matrix(paths, mass, modal_rank=4)
    full = tuple((node,) for node in range(len(mass)))
    proposals = []
    for left in range(len(mass)):
        for right in range(left + 1, len(mass)):
            partition = tuple(
                sorted(
                    [
                        tuple(sorted((left, right))),
                        *[(node,) for node in range(len(mass)) if node not in {left, right}],
                    ],
                    key=lambda group: (group[0], len(group), group),
                )
            )
            energy = intra_cluster_energy(paths, mass, partition)
            input_ratio = input_heterogeneity(paths, mass, partition)
            preliminary = max(energy.values()) <= float(
                config["maximum_intra_cluster_energy"]
            ) and input_ratio <= float(config["maximum_input_heterogeneity"])
            proposals.append(
                {
                    "parent": [list(group) for group in full],
                    "child": [list(group) for group in partition],
                    "merge": [left, right],
                    "dynamic_distance": float(distance[left, right]),
                    "rho_delta": float(energy["delta"]),
                    "rho_omega": float(energy["omega"]),
                    "rho_Pm": float(input_ratio),
                    "preliminary_screen": "PASS" if preliminary else "REJECT",
                    "complete_certificate": False,
                    "selection_permitted": False,
                }
            )
    return {
        "status": "PROPOSAL_ONLY_NOT_A_SELECTION",
        "reason": "source-excitation validity is evaluated before CPIG/closure/model selection",
        "n_QA_train_events": len(samples),
        "n_QA_fault_lines": len({path["fault_line"] for path in paths}),
        "distance_audit": distance_audit,
        "tree": sorted(proposals, key=lambda row: (row["dynamic_distance"], row["merge"])),
        "selection_event_ids": sorted(selection_ids),
    }


def _run_controlled(root: Path, config: dict[str, Any], result_dir: Path, run_id: str) -> None:
    hypothesis = _latest_result(
        root, str(config["hypothesis_experiment"]), "hypothesis_registration"
    )
    if _verify_hashes(hypothesis):
        raise RuntimeError("v4 hypothesis artifacts changed before controlled experiment")
    graph_gates = _synthetic_graph_gates(config)
    module_gates = _synthetic_module_search_gates(config)
    if graph_gates["status"] != "PASS" or module_gates["status"] != "PASS":
        raise RuntimeError("synthetic correctness gates rejected AFSD v4")

    data = _load_data(root, config)
    rows = _qualification_rows(data)
    overlap = source_excitation_overlap_audit(
        rows, minimum_fault_line_clusters=int(config["minimum_source_cell_fault_lines"])
    )
    selection_ids = structure_selection_ids(rows)
    try:
        folds = nested_fault_line_folds(
            rows,
            outer_folds=int(config["outer_fault_line_folds"]),
            inner_folds=int(config["inner_fault_line_folds"]),
        )
        fold_status = "PASS"
    except ValueError:
        folds = []
        fold_status = "FAIL"
    if not selection_ids or fold_status != "PASS":
        scientific_status = "INSUFFICIENT_INFORMATION"
    elif overlap["status"] != "PASS":
        scientific_status = "SOURCE_EXCITATION_CONFOUNDED"
    else:
        scientific_status = "PASS"

    preliminary = _f1_preliminary_spatial_proposals(data, rows, config)
    fitting_permitted = scientific_status == "PASS"
    controlled = {
        "run_id": run_id,
        "status": "COMPLETED",
        "scientific_status": scientific_status,
        "synthetic_graph_gate": graph_gates["status"],
        "synthetic_module_search_gate": module_gates["status"],
        "f1_joint_fit_permitted": fitting_permitted,
        "f1_joint_fit_performed": False,
        "reason": (
            "joint F1 graph/module fitting is invalid because source and excitation are not separable"
            if scientific_status == "SOURCE_EXCITATION_CONFOUNDED"
            else "joint fitting requires a subsequent immutable compute round"
            if scientific_status == "PASS"
            else "Q-A nested-design prerequisites are insufficient"
        ),
        "rollout_label": ROLLOUT_LABEL,
        "Q_B_used_for_structure": False,
        "validation_used_for_structure_generation": False,
        "test_ids_read": [],
    }
    _json(result_dir / "synthetic_graph_correctness.json", graph_gates)
    _json(result_dir / "synthetic_module_search_correctness.json", module_gates)
    _json(
        result_dir / "beam_mcts_search_tree.json",
        {"beam": module_gates["beam"], "mcts": module_gates["mcts_trigger_audit"]},
    )
    _json(result_dir / "exhaustive_oracle_comparison.json", module_gates["oracle_comparison"])
    _json(
        result_dir / "search_compute_comparison.json",
        {
            "beam": module_gates["beam"]["fit_counts"],
            "oracle_full_fits": module_gates["oracle"]["full_fits"],
        },
    )
    _json(result_dir / "module_proposal_and_rejection_log.json", module_gates["module_proposals"])
    _csv(result_dir / "data_qualification_and_confounding.csv", rows)
    _json(result_dir / "stratification_policy.json", data["policy"].as_dict())
    _json(result_dir / "source_excitation_overlap.json", overlap)
    _csv(result_dir / "source_excitation_cells.csv", overlap["table"])
    _csv(result_dir / "nested_fault_line_folds.csv", folds)
    _json(result_dir / "graph_contraction_tree.json", preliminary)
    _csv(result_dir / "merge_cpig_and_certificates.csv", preliminary.get("tree", []))
    _json(
        result_dir / "spatial_resolution_selection_path.json",
        {"status": "NOT_SELECTED", "gate": scientific_status, "proposal_only": True},
    )
    _json(
        result_dir / "internal_modes_and_participation.json",
        {"status": "NOT_RUN_BY_DESIGN_GATE", "gate": scientific_status},
    )
    _json(
        result_dir / "consensus_dynamic_graph.json",
        {"status": "NOT_IDENTIFIED", "gate": scientific_status},
    )
    _json(
        result_dir / "module_and_edge_support.json",
        {"status": "NOT_IDENTIFIED", "gate": scientific_status},
    )
    _json(
        result_dir / "explicit_final_equations.json",
        {"status": "NOT_IDENTIFIED", "gate": scientific_status},
    )
    _csv(result_dir / "parameters_and_confidence_intervals.csv", [])
    _json(
        result_dir / "rollout_1s_5s_10s.json",
        {
            "status": "NOT_RUN_BY_DESIGN_GATE",
            "rollout_label": ROLLOUT_LABEL,
            "gate": scientific_status,
        },
    )
    _csv(result_dir / "modal_table.csv", [])
    _csv(result_dir / "failure_cases.csv", [])
    _json(result_dir / "controlled_experiment.json", controlled)
    _json(
        result_dir / "selection_population.json",
        {
            "structure_selection_event_ids": sorted(selection_ids),
            "Q_B_post_freeze_only_event_ids": sorted(
                str(row["event_id"])
                for row in rows
                if row["role"] == "train" and row["quality_grade"] == "Q-B"
            ),
            "validation_event_ids_used_for_confirmation": [],
            "test_ids_read": [],
        },
    )
    _json(
        result_dir / "lineage.json",
        _lineage(run_id, "controlled_experiment", parent_hypothesis_result=hypothesis.name),
    )
    _write_hashes(result_dir)


def _run_residual(root: Path, config: dict[str, Any], result_dir: Path, run_id: str) -> None:
    controlled = _latest_result(root, str(config["controlled_experiment"]), "controlled_experiment")
    if _verify_hashes(controlled):
        raise RuntimeError("controlled artifacts changed before residual diagnosis")
    decision = json.loads((controlled / "controlled_experiment.json").read_text(encoding="utf-8"))
    overlap = json.loads(
        (controlled / "source_excitation_overlap.json").read_text(encoding="utf-8")
    )
    minimum = int(overlap["minimum_fault_line_clusters"])
    deficits = [
        {
            **row,
            "additional_fault_line_clusters_needed": max(
                0, minimum - int(row["n_fault_line_clusters"])
            ),
        }
        for row in overlap["table"]
        if int(row["n_fault_line_clusters"]) < minimum
    ]
    comparable = len(overlap["comparable_cells"])
    total_cells = len({(row["amplitude_level"], row["richness_level"]) for row in overlap["table"]})
    diagnosis = {
        "run_id": run_id,
        "status": "COMPLETED",
        "scientific_status": decision["scientific_status"],
        "residual_class": "experimental-design support residual"
        if decision["scientific_status"] == "SOURCE_EXCITATION_CONFOUNDED"
        else "model residual",
        "direct_evidence": {
            "eligible_events": overlap["n_eligible_events"],
            "eligible_fault_line_clusters": overlap["n_eligible_fault_line_clusters"],
            "comparable_cells": comparable,
            "represented_cells": total_cells,
            "failing_source_cells": len(deficits),
        },
        "confidence_statement": {
            "estimand": "existence of preregistered source-balanced excitation cells in the finite F1 design",
            "interval": [comparable, comparable],
            "type": "exact finite-population design bound; resampling cannot create absent source-cell support",
        },
        "alternative_explanations": [
            "data-source dynamics rather than excitation may explain apparent branch changes",
            "quality filtering may preferentially retain different excitation ranges by source",
            "unexcited within-cluster or device modes may remain observationally absent",
        ],
        "why_more_compute_cannot_resolve": "optimization and additional bootstrap draws cannot create missing Q-A fault-line clusters in an empty or underpowered source-by-excitation cell",
        "required_new_information": "at least five independent Q-A fault-line clusters per source in every represented amplitude-richness cell, or an independently justified design that removes the source comparison claim",
        "highest_information_next_experiment": "collect or qualify source-overlapping fault-line excitations before joint graph/module selection",
        "Q_B_promotion_forbidden": True,
    }
    _json(result_dir / "residual_diagnosis.json", diagnosis)
    _csv(result_dir / "source_cell_deficits.csv", deficits)
    _json(
        result_dir / "lineage.json",
        _lineage(run_id, "residual_diagnosis", parent_controlled_result=controlled.name),
    )
    _write_hashes(result_dir)


def _folds_isolated(rows: list[dict[str, str]]) -> bool:
    for outer in {row["outer_fold"] for row in rows}:
        subset = [row for row in rows if row["outer_fold"] == outer]
        evaluation = {row["fault_line"] for row in subset if row["outer_role"] == "evaluation"}
        development = {row["fault_line"] for row in subset if row["outer_role"] == "development"}
        if evaluation & development:
            return False
    return True


def _run_adversarial(root: Path, config: dict[str, Any], result_dir: Path, run_id: str) -> None:
    hypothesis = _latest_result(
        root, str(config["hypothesis_experiment"]), "hypothesis_registration"
    )
    controlled = _latest_result(root, str(config["controlled_experiment"]), "controlled_experiment")
    residual = _latest_result(root, str(config["residual_experiment"]), "residual_diagnosis")
    paths = (hypothesis, controlled, residual)
    if any(_verify_hashes(path) for path in paths):
        raise RuntimeError("upstream v4 artifacts changed before adversarial review")
    decision = json.loads((controlled / "controlled_experiment.json").read_text(encoding="utf-8"))
    population = json.loads((controlled / "selection_population.json").read_text(encoding="utf-8"))
    rows = _read_csv(controlled / "data_qualification_and_confounding.csv")
    selection_ids = set(population["structure_selection_event_ids"])
    folds = _read_csv(controlled / "nested_fault_line_folds.csv")
    lineages = [json.loads((path / "lineage.json").read_text(encoding="utf-8")) for path in paths]
    synthetic_graph = json.loads(
        (controlled / "synthetic_graph_correctness.json").read_text(encoding="utf-8")
    )
    synthetic_module = json.loads(
        (controlled / "synthetic_module_search_correctness.json").read_text(encoding="utf-8")
    )
    gates = {
        "immutable_upstream_artifacts": all(not _verify_hashes(path) for path in paths),
        "synthetic_graph_gates_pass": synthetic_graph["status"] == "PASS",
        "synthetic_module_and_search_gates_pass": synthetic_module["status"] == "PASS",
        "Beam_matches_small_oracle": synthetic_module["oracle_comparison"]["status"] == "PASS",
        "fewer_full_fits_than_oracle": synthetic_module["oracle_comparison"]["full_fit_reduction"]
        > 0,
        "only_QA_train_selected": all(
            row["role"] == "train" and row["quality_grade"] == "Q-A"
            for row in rows
            if row["event_id"] in selection_ids
        ),
        "QB_never_selected": not bool(
            selection_ids & set(population["Q_B_post_freeze_only_event_ids"])
        ),
        "fault_line_split_isolated": _folds_isolated(folds)
        if folds
        else decision["scientific_status"] == "INSUFFICIENT_INFORMATION",
        "validation_not_used_for_structure": decision["validation_used_for_structure_generation"]
        is False,
        "observed_Pm_is_conditional": decision["rollout_label"] == ROLLOUT_LABEL,
        "source_confound_gate_honored": decision["scientific_status"]
        != "SOURCE_EXCITATION_CONFOUNDED"
        or decision["f1_joint_fit_performed"] is False,
        "no_historical_results_read": all(
            not row["historical_v1_v2_v3_results_read"] for row in lineages
        ),
        "test_other_folds_canonical_locked": all(
            not row["test_ids_read"]
            and not row["other_folds_read"]
            and not row["canonical_files_modified"]
            for row in lineages
        ),
    }
    findings = [
        {
            "alternative": "clustering gain is regularization rather than reduction",
            "disposition": "not estimable because global F1 selection was blocked before fit",
        },
        {
            "alternative": "memory absorbs an incorrect mechanical input",
            "disposition": "observed Pm is explicitly declared; no memory module was fit on F1",
        },
        {
            "alternative": "graph support is a data-source marker",
            "disposition": "critical and unresolved; source-excitation gate prevents a graph claim",
        },
        {
            "alternative": "node and edge states are input-output equivalent",
            "disposition": "synthetic grammar returns structure equivalence unless independent ports separate them; no F1 semantic claim",
        },
        {
            "alternative": "low CPIG reflects weak excitation",
            "disposition": "no F1 CPIG sufficiency certificate was issued",
        },
        {
            "alternative": "observed Pm hides deployment error",
            "disposition": "all artifacts carry the conditional rollout label and make no autonomous claim",
        },
    ]
    status = "PASS" if all(gates.values()) else "FAIL"
    review = {
        "run_id": run_id,
        "status": status,
        "decision": "ACCEPTABLE_EVIDENCE_CHAIN" if status == "PASS" else "REJECT",
        "scientific_status": decision["scientific_status"],
        "gates": gates,
        "alternative_explanations": findings,
    }
    _json(result_dir / "adversarial_review.json", review)
    (result_dir / "adversarial_review.md").write_text(
        "# M0-AFSD v4 adversarial review\n\n"
        f"Decision: **{review['decision']}**\n\n"
        + "\n".join(f"- {'PASS' if value else 'FAIL'} `{name}`" for name, value in gates.items())
        + "\n",
        encoding="utf-8",
    )
    _json(
        result_dir / "lineage.json",
        _lineage(run_id, "adversarial_review", parent_results=[path.name for path in paths]),
    )
    _write_hashes(result_dir)
    if status != "PASS":
        raise RuntimeError("v4 adversarial review rejected the evidence chain")


def _run_acceptance(root: Path, config: dict[str, Any], result_dir: Path, run_id: str) -> None:
    controlled = _latest_result(root, str(config["controlled_experiment"]), "controlled_experiment")
    residual = _latest_result(root, str(config["residual_experiment"]), "residual_diagnosis")
    adversarial = _latest_result(root, str(config["adversarial_experiment"]), "adversarial_review")
    if any(_verify_hashes(path) for path in (controlled, residual, adversarial)):
        raise RuntimeError("v4 evidence artifacts changed before acceptance")
    decision = json.loads((controlled / "controlled_experiment.json").read_text(encoding="utf-8"))
    diagnosis = json.loads((residual / "residual_diagnosis.json").read_text(encoding="utf-8"))
    review = json.loads((adversarial / "adversarial_review.json").read_text(encoding="utf-8"))
    outcome = str(decision["scientific_status"])
    if review["status"] == "PASS" and outcome in TERMINAL_STATES:
        status = "PASS"
        acceptance_decision = "ACCEPT_NEGATIVE_OR_INFORMATION_LIMITED_CONCLUSION"
    elif review["status"] == "PASS" and outcome == "PASS":
        status = "FAIL"
        outcome = "INSUFFICIENT_INFORMATION"
        acceptance_decision = "REJECT_PENDING_F1_JOINT_COMPUTE"
    else:
        status = "FAIL"
        outcome = "UNRESOLVED"
        acceptance_decision = "REJECT"
    acceptance = {
        "run_id": run_id,
        "status": status,
        "decision": acceptance_decision,
        "scientific_conclusion": outcome,
        "joint_graph_module_branch_identified": False,
        "rejected_hypotheses": ["H6-DESIGN"] if outcome == "SOURCE_EXCITATION_CONFOUNDED" else [],
        "direct_evidence": diagnosis["direct_evidence"],
        "confidence_interval_or_bound": diagnosis["confidence_statement"],
        "alternative_explanations": diagnosis["alternative_explanations"],
        "why_more_compute_cannot_resolve": diagnosis["why_more_compute_cannot_resolve"],
        "required_new_observation_or_excitation": diagnosis["required_new_information"],
        "valid_claim": "the present Q-A F1 design cannot identify a joint graph/module branch independently of data source"
        if outcome == "SOURCE_EXCITATION_CONFOUNDED"
        else "the present evidence does not support a v4 branch claim",
        "invalid_claims": [
            "a lower-dimensional quotient has been identified on F1",
            "an unselected module is absent from the physical system",
            "a learned dynamic edge is a physical IEEE39 transmission line",
            "a neutral lag state is a governor, exciter, PSS, RL, RC, or transient EMF",
            "conditional rollout with observed Pm is autonomous causal deployment",
        ],
        "synthetic_algorithm_correctness": "PASS",
        "rollout_label": ROLLOUT_LABEL,
        "test_ids_read": [],
        "other_folds_read": [],
        "merge_gate_requires_explicit_approval": True,
    }
    _json(result_dir / "acceptance.json", acceptance)
    (result_dir / "acceptance.md").write_text(
        "# M0-AFSD v4 graph-compositional discovery acceptance\n\n"
        f"Decision: **{acceptance_decision}**\n\n"
        f"Scientific terminal state: `{outcome}`.\n\n"
        f"All rollout references are: **{ROLLOUT_LABEL}**.\n",
        encoding="utf-8",
    )
    _json(
        result_dir / "lineage.json",
        _lineage(
            run_id, "acceptance", parent_results=[controlled.name, residual.name, adversarial.name]
        ),
    )
    _write_hashes(result_dir)
    if status != "PASS":
        raise RuntimeError("v4 acceptance has not reached an allowed terminal state")


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
        raise ValueError(f"unsupported M0-AFSD v4 stage: {stage}")
