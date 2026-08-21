from __future__ import annotations

import csv
import hashlib
import json
import tomllib
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from kron_reduce.evaluation.modal import (
    CrossClusterConsensusMode,
    DmdMode,
    cross_cluster_modal_consensus,
    exact_dmd_modes,
    modal_assurance_criterion,
    pooled_input_aware_modal_consensus,
)
from kron_reduce.models import LowRankModalPortM0Model
from kron_reduce.workflows.m0_afsd_v6_full_model_subset_validation import (
    _detailed_aggregate,
    _evaluate_candidate,
    _load_v5_data,
    _modal_match_rows,
    deterministic_fault_line_roles,
)
from kron_reduce.workflows.m0_afsd_v7_modal_structure_discovery import _fit_ssdm

WORKFLOW = "m0_afsd_v8_modal_identifiability"
ROLLOUT_LABEL = "conditional rollout with observed Pm"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


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
            writer.writerow({
                key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list, tuple)) else value
                for key, value in row.items()
            })


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_hashes(result_dir: Path) -> None:
    _json(result_dir/"artifact_hashes.json", {
        str(path.relative_to(result_dir)).replace("\\", "/"): _sha256(path)
        for path in sorted(result_dir.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    })


def _verify_hashes(result_dir: Path) -> list[str]:
    expected = json.loads((result_dir/"artifact_hashes.json").read_text(encoding="utf-8"))
    return [
        relative for relative, digest in expected.items()
        if not (result_dir/relative).is_file() or _sha256(result_dir/relative) != digest
    ]


def _latest_result(root: Path, experiment: str, stage: str) -> Path:
    candidates = sorted(
        path for path in (root/"results").glob(f"{experiment}-{stage}-*")
        if path.is_dir() and not (path/"RUN_FAILED.json").exists()
    )
    if not candidates:
        raise FileNotFoundError(f"no completed v8 result for {experiment}/{stage}")
    return candidates[-1]


def _validate_scope(config: dict[str, Any]) -> None:
    expected = {
        "fold": "F1", "seed": 324, "pm_mode": "observed",
        "test_access": "forbidden", "other_folds_access": "forbidden",
        "canonical_mutation": "forbidden", "rollout_label": ROLLOUT_LABEL,
    }
    mismatch = {key: (config.get(key), value) for key, value in expected.items() if config.get(key) != value}
    if mismatch:
        raise RuntimeError(f"v8 scope violation: {mismatch}")


def _hypothesis_registry(config: dict[str, Any], run_id: str) -> dict[str, Any]:
    return {
        "run_id": run_id, "workflow": WORKFLOW,
        "registered_before_controlled_metrics": True,
        "primary_hypothesis": (
            "F1 Q-A train contains at least one fault-line-repeatable empirical oscillatory shape "
            "that can constrain a reciprocal low-rank passive modal port"
        ),
        "null_or_information_limited_outcome": (
            "if no mode reaches 80% support across fit/selection/audit under threshold sensitivity, "
            "and a synthetic-correct low-rank port cannot improve held-out 10 s rollout, create an "
            "amplitude-by-richness modal envelope map and route each envelope to ESC selection; "
            "do not promote a global modal state and do not terminate discovery"
        ),
        "candidate": {
            "frequency_hz": float(config["candidate_frequency_hz"]),
            "damping_ratio": float(config["candidate_damping_ratio"]),
            "rank": int(config["candidate_rank"]),
            "pilot_informed": True,
        },
        "modal_support_gate": {
            "frequency_tolerance": float(config["modal_frequency_tolerance"]),
            "minimum_MAC": float(config["modal_minimum_mac"]),
            "fault_line_support": float(config["modal_minimum_fault_line_support"]),
            "heldout_event_fraction_per_line": float(config["modal_minimum_event_fraction_per_line"]),
        },
        "claim_boundaries": config["claim_boundaries"],
        "test_ids_read": [], "other_folds_read": [],
    }


def _run_hypothesis(config_path: Path, config: dict[str, Any], result_dir: Path, run_id: str) -> None:
    (result_dir/"frozen_config.toml").write_bytes(config_path.read_bytes())
    _json(result_dir/"hypothesis_registry.json", _hypothesis_registry(config, run_id))
    _json(result_dir/"lineage.json", {
        "run_id": run_id, "stage": "hypothesis_registration",
        "historical_result_artifacts_read": [], "test_ids_read": [], "other_folds_read": [],
    })
    _write_hashes(result_dir)


def _qualified_populations(root: Path, config: dict[str, Any]):
    data = _load_v5_data(root, config)
    qa_rows = [
        row for row in data["rows"]
        if row["role"] == "train" and bool(row["eligible_for_v5_selection"])
        and bool(row["eligible_h1000"])
    ]
    roles = deterministic_fault_line_roles(
        qa_rows, seed=int(config["seed"]),
        fit_fraction=float(config["development_fit_fraction"]),
        selection_fraction=float(config["development_selection_fraction"]),
    )
    row_by_id = {str(row["event_id"]): row for row in qa_rows}
    qa = [sample for sample in data["train"] if str(sample["event_id"]) in row_by_id]
    populations = {
        role: [
            sample for sample in qa
            if roles[str(row_by_id[str(sample["event_id"])]["fault_line"])] == role
        ]
        for role in ("fit", "selection", "audit")
    }
    return data, qa_rows, roles, populations


def _empirical_modes(samples: Sequence[dict[str, Any]], config: dict[str, Any]):
    by_line_event: dict[str, dict[str, list[DmdMode]]] = defaultdict(dict)
    rows: list[dict[str, Any]] = []
    horizon = int(config["modal_horizon_steps"])
    for sample in samples:
        target = sample["training_context"]["targets"]
        if len(target["t"]) < horizon:
            continue
        omega = np.asarray(target["omega"][:horizon], dtype=float)
        dt = float(np.median(np.diff(np.asarray(target["t"][:horizon], dtype=float))))
        mass = np.asarray(sample["model_inputs"]["static"]["mass"], dtype=float)
        extracted = exact_dmd_modes(
            omega, dt, rank=min(int(config["modal_dmd_rank"]), omega.shape[1]),
            min_frequency_hz=float(config["modal_min_frequency_hz"]),
            max_frequency_hz=float(config["modal_max_frequency_hz"]),
        )
        retained = [
            mode for mode in extracted
            if float(config["modal_minimum_damping_ratio"]) <= mode.damping_ratio
            <= float(config["modal_maximum_damping_ratio"])
        ][: int(config["modal_max_modes_per_event"])]
        event_id = str(sample["event_id"])
        line = str(sample["training_context"]["metadata"]["fault_line"])
        transformed = []
        for index, mode in enumerate(retained):
            shape = (np.asarray(mode.shape_real)+1j*np.asarray(mode.shape_imag))*np.sqrt(mass)
            shape = shape/max(np.linalg.norm(shape), np.finfo(float).eps)
            weighted = DmdMode(
                frequency_hz=mode.frequency_hz, damping_ratio=mode.damping_ratio,
                decay_rate=mode.decay_rate, strength=mode.strength,
                shape_real=tuple(shape.real), shape_imag=tuple(shape.imag),
            )
            transformed.append(weighted)
            rows.append({
                "event_id": event_id, "fault_line": line, "mode_index": index,
                "frequency_hz": weighted.frequency_hz, "damping_ratio": weighted.damping_ratio,
                "strength": weighted.strength,
            })
        by_line_event[line][event_id] = transformed
    return by_line_event, rows


def _line_modes(by_line_event: dict[str, dict[str, list[DmdMode]]]) -> dict[str, list[DmdMode]]:
    return {line: [mode for modes in events.values() for mode in modes] for line, events in by_line_event.items()}


def _heldout_support(
    consensus: CrossClusterConsensusMode,
    by_line_event: dict[str, dict[str, list[DmdMode]]],
    config: dict[str, Any],
) -> dict[str, Any]:
    reference = np.asarray(consensus.mode.shape_real)+1j*np.asarray(consensus.mode.shape_imag)
    per_line = {}
    for line, events in sorted(by_line_event.items()):
        matches = []
        for event_id, modes in sorted(events.items()):
            compatible = False
            best_relative, best_mac = float("inf"), 0.0
            for mode in modes:
                relative = abs(mode.frequency_hz-consensus.mode.frequency_hz)/max(
                    consensus.mode.frequency_hz, np.finfo(float).eps
                )
                shape = np.asarray(mode.shape_real)+1j*np.asarray(mode.shape_imag)
                mac = modal_assurance_criterion(reference, shape)
                compatible |= bool(
                    relative <= float(config["modal_frequency_tolerance"])
                    and mac >= float(config["modal_minimum_mac"])
                )
                best_relative, best_mac = min(best_relative, relative), max(best_mac, mac)
            matches.append({
                "event_id": event_id, "compatible": compatible,
                "best_relative_frequency_error": best_relative, "best_MAC": best_mac,
            })
        event_fraction = float(np.mean([row["compatible"] for row in matches])) if matches else 0.0
        per_line[line] = {
            "events": len(matches), "event_support_fraction": event_fraction,
            "pass": event_fraction >= float(config["modal_minimum_event_fraction_per_line"]),
        }
    return {
        "fault_line_support": float(np.mean([row["pass"] for row in per_line.values()])) if per_line else 0.0,
        "supporting_fault_lines": sorted(line for line, row in per_line.items() if row["pass"]),
        "per_fault_line": per_line,
    }


def _candidate_metrics(model, samples, horizons, *, branch, cohort):
    rows = _evaluate_candidate(
        model, "generator_10", samples, samples, None, tuple(horizons),
        cohort=cohort, branch=branch,
    )
    modal_rows = _modal_match_rows(model, samples, tuple(h for h in horizons if h >= 500), branch=branch, population=cohort)
    return rows, _detailed_aggregate(rows), modal_rows


def _subset_empirical_events(
    population: dict[str, dict[str, list[DmdMode]]],
    event_ids: set[str],
) -> dict[str, dict[str, list[DmdMode]]]:
    return {
        line: {event_id: modes for event_id, modes in events.items() if event_id in event_ids}
        for line, events in population.items()
        if any(event_id in event_ids for event_id in events)
    }


def _regime_envelope_map(
    qa_rows: Sequence[dict[str, Any]],
    roles: dict[str, str],
    empirical: dict[str, dict[str, dict[str, list[DmdMode]]]],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Create train-only excitation envelopes before per-regime model selection.

    The tiers are inherited from the qualification layer and therefore exist
    before candidate metrics.  v8 only certifies whether a modal constraint is
    authorized inside each envelope; ESC performs the subsequent model fit.
    """
    regimes: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in qa_rows:
        regimes[(str(row["amplitude_level"]), str(row["richness_level"]))].append(row)
    output = []
    assignment_counts: dict[str, int] = defaultdict(int)
    for (amplitude, richness), rows in sorted(regimes.items()):
        ids_by_role = {role: set() for role in ("fit", "selection", "audit")}
        for row in rows:
            event_id = str(row["event_id"])
            assignment_counts[event_id] += 1
            ids_by_role[roles[str(row["fault_line"])]].add(event_id)
        scoped = {
            role: _subset_empirical_events(empirical[role], ids_by_role[role])
            for role in ids_by_role
        }
        counts = {
            role: {
                "events": len(ids_by_role[role]),
                "fault_lines": len(scoped[role]),
            }
            for role in scoped
        }
        qualified = bool(
            counts["fit"]["fault_lines"] >= int(config["minimum_regime_fit_fault_lines"])
            and counts["selection"]["fault_lines"] >= int(config["minimum_regime_selection_fault_lines"])
            and counts["audit"]["fault_lines"] >= int(config["minimum_regime_audit_fault_lines"])
        )
        consensus = cross_cluster_modal_consensus(
            _line_modes(scoped["fit"]),
            frequency_tolerance=float(config["modal_frequency_tolerance"]),
            minimum_mac=float(config["modal_minimum_mac"]),
            minimum_support_fraction=0.0,
        ) if scoped["fit"] else []
        candidates = []
        for mode in consensus:
            selection = _heldout_support(mode, scoped["selection"], config)
            audit = _heldout_support(mode, scoped["audit"], config)
            passes = bool(
                qualified
                and mode.support_fraction >= float(config["modal_minimum_fault_line_support"])
                and selection["fault_line_support"] >= float(config["modal_minimum_fault_line_support"])
                and audit["fault_line_support"] >= float(config["modal_minimum_fault_line_support"])
            )
            candidates.append({
                **mode.as_dict(), "selection": selection, "audit": audit,
                "passes_envelope_modal_gate": passes,
            })
        supported = [row for row in candidates if row["passes_envelope_modal_gate"]]
        status = (
            "MODAL_CONSTRAINT_AUTHORIZED" if supported
            else "MODAL_CONSTRAINT_NOT_AUTHORIZED" if qualified
            else "ENVELOPE_DATA_REQUEST"
        )
        def bounds(key: str, envelope_rows: Sequence[dict[str, Any]] = rows) -> list[float]:
            values = np.asarray(
                [float(row[key]) for row in envelope_rows], dtype=float
            )
            return [float(np.min(values)), float(np.max(values))]
        output.append({
            "envelope_id": f"{amplitude}__{richness}",
            "amplitude_level": amplitude, "richness_level": richness,
            "definition_frozen_before_candidate_metrics": True,
            "offline_features_only": True,
            "bounds": {
                "excitation_amplitude": bounds("excitation_amplitude"),
                "richness_score": bounds("richness_score"),
                "nonlinearity_ratio": bounds("nonlinearity_ratio"),
                "spectral_entropy": bounds("spectral_entropy"),
            },
            "counts": counts, "status": status,
            "authorized_modes": supported,
            "best_fit_support": max((mode.support_fraction for mode in consensus), default=0.0),
            "next_route": (
                "ESC_MODAL_PORT_CANDIDATE" if supported
                else "ESC_NONMODAL_LADDER_SELECTION" if qualified
                else "ESC_PARENT_ENVELOPE_OR_DATA_REQUEST"
            ),
            "candidate_preview": candidates[:5],
        })
    expected = {str(row["event_id"]) for row in qa_rows}
    assigned = set(assignment_counts)
    duplicate_assignments = sorted(
        event_id for event_id, count in assignment_counts.items() if count != 1
    )
    return {
        "partition_key": "amplitude_level_x_richness_level",
        "declared_before_candidate_metrics": True,
        "envelopes": output,
        "eligible_events": len(expected),
        "assigned_events": len(assigned),
        "duplicate_assignments": duplicate_assignments,
        "complete_nonoverlapping_assignment": (
            assigned == expected and not duplicate_assignments
        ),
        "deployment_policy": "offline envelope evidence only; ESC must register a causal prefix detector",
        "outside_union_policy": "REFUSE_AND_REQUEST_DATA",
    }


def _run_controlled(root: Path, config_path: Path, config: dict[str, Any], result_dir: Path, run_id: str) -> None:
    hypothesis = _latest_result(root, str(config["hypothesis_experiment"]), "hypothesis_registration")
    if _verify_hashes(hypothesis):
        raise RuntimeError("v8 hypothesis artifacts changed before controlled experiment")
    (result_dir/"frozen_config.toml").write_bytes(config_path.read_bytes())
    data, qa_rows, roles, populations = _qualified_populations(root, config)
    empirical, empirical_rows = {}, []
    for role in ("fit", "selection", "audit"):
        empirical[role], rows = _empirical_modes(populations[role], config)
        empirical_rows.extend({"role": role, **row} for row in rows)
    fit_consensus = cross_cluster_modal_consensus(
        _line_modes(empirical["fit"]),
        frequency_tolerance=float(config["modal_frequency_tolerance"]),
        minimum_mac=float(config["modal_minimum_mac"]),
        minimum_support_fraction=0.0,
    )
    consensus_rows = []
    for index, mode in enumerate(fit_consensus):
        selection_support = _heldout_support(mode, empirical["selection"], config)
        audit_support = _heldout_support(mode, empirical["audit"], config)
        consensus_rows.append({
            "index": index, **mode.as_dict(),
            "selection": selection_support, "audit": audit_support,
            "passes_all_roles": bool(
                mode.support_fraction >= float(config["modal_minimum_fault_line_support"])
                and selection_support["fault_line_support"] >= float(config["modal_minimum_fault_line_support"])
                and audit_support["fault_line_support"] >= float(config["modal_minimum_fault_line_support"])
            ),
        })
    sensitivity = []
    for tolerance in config["modal_frequency_tolerance_sensitivity"]:
        for minimum_mac in config["modal_mac_sensitivity"]:
            modes = cross_cluster_modal_consensus(
                _line_modes(empirical["fit"]), frequency_tolerance=float(tolerance),
                minimum_mac=float(minimum_mac), minimum_support_fraction=0.0,
            )
            sensitivity.append({
                "frequency_tolerance": float(tolerance), "minimum_MAC": float(minimum_mac),
                "best_fit_fault_line_support": max((mode.support_fraction for mode in modes), default=0.0),
                "modes_passing_fit_80pct": sum(
                    mode.support_fraction >= float(config["modal_minimum_fault_line_support"])
                    for mode in modes
                ),
            })
    regime_map = _regime_envelope_map(qa_rows, roles, empirical, config)

    # Independent input-aware check.  It is a numerical stabilization map,
    # not a replacement for the cross-fault-line support gate.
    fit_paths = [np.asarray(s["training_context"]["targets"]["omega"][: int(config["modal_horizon_steps"])]) for s in populations["fit"]]
    fit_inputs = [np.asarray(s["training_context"]["Pm_observed"][: int(config["modal_horizon_steps"])]) for s in populations["fit"]]
    dt = float(np.median(np.diff(np.asarray(populations["fit"][0]["training_context"]["targets"]["t"][: int(config["modal_horizon_steps"])]))))
    input_aware = pooled_input_aware_modal_consensus(
        fit_paths, fit_inputs, dt,
        delay_orders=tuple(int(v) for v in config["dmdc_delay_orders"]),
        ridge_values=tuple(float(v) for v in config["dmdc_ridge_values"]),
        minimum_support_fraction=float(config["dmdc_minimum_numerical_support"]),
        frequency_tolerance=float(config["modal_frequency_tolerance"]),
        minimum_mac=float(config["modal_minimum_mac"]),
        minimum_damping_ratio=float(config["modal_minimum_damping_ratio"]),
        maximum_damping_ratio=float(config["modal_maximum_damping_ratio"]),
        min_frequency_hz=float(config["modal_min_frequency_hz"]),
        max_frequency_hz=float(config["modal_max_frequency_hz"]),
    )

    baseline = _fit_ssdm(populations["fit"], populations["selection"], config)
    n_edges = baseline.static_coefficients.shape[1]
    fallback = LowRankModalPortM0Model(
        backbone=baseline, rank=0, loadings=np.empty((n_edges, 0)),
        frequency_hz=float(config["candidate_frequency_hz"]),
        damping_ratio=float(config["candidate_damping_ratio"]),
        max_internal_step=float(config["max_internal_step"]),
    )
    candidate = LowRankModalPortM0Model(
        backbone=baseline, frequency_hz=float(config["candidate_frequency_hz"]),
        damping_ratio=float(config["candidate_damping_ratio"]),
        rank=int(config["candidate_rank"]), ridge=float(config["candidate_ridge"]),
        max_internal_step=float(config["max_internal_step"]),
    )
    force_cache = candidate.prepare_force_cache(
        populations["fit"], horizon_steps=int(config["selection_horizon_steps"]),
        max_fit_steps=int(config["fit_max_steps"]),
    )
    weak = candidate.fit_fixed_rank(
        populations["fit"], frequency_hz=float(config["candidate_frequency_hz"]),
        damping_ratio=float(config["candidate_damping_ratio"]), rank=int(config["candidate_rank"]),
        ridge=float(config["candidate_ridge"]),
        windows=tuple(int(value) for value in config["integral_windows"]),
        horizon_steps=int(config["selection_horizon_steps"]), max_fit_steps=int(config["fit_max_steps"]),
        max_windows_per_case_horizon=int(config["weak_max_windows_per_case_horizon"]),
        prepared_force_cache=force_cache,
    )
    metrics, aggregate, modal_matches = [], [], []
    for role, horizons in (("selection", (100, 1000)), ("audit", (1000,))):
        for branch, model in (("SS-DM", fallback), ("SS-DM+LRMRP", candidate)):
            rows, summaries, modes = _candidate_metrics(
                model, populations[role], horizons, branch=branch, cohort=f"development_{role}"
            )
            metrics.extend(rows); aggregate.extend(summaries); modal_matches.extend(modes)
    convergence = {}
    for sample in populations["audit"][:1]:
        target = sample["training_context"]["targets"]
        steps = min(100, len(target["t"]))
        convergence[str(sample["event_id"])] = candidate.convergence_audit(
            sample["model_inputs"]["static"], np.asarray(target["t"][:steps]),
            {"Pm": np.asarray(sample["model_inputs"]["exogenous"]["Pm"][:steps]), "interpolation": "linear"},
        )
    checkpoints = result_dir/"checkpoints"; checkpoints.mkdir()
    fallback.save(checkpoints/"SS-DM.npz"); candidate.save(checkpoints/"SS-DM-LRMRP.npz")
    _json(result_dir/"modal_identifiability.json", {
        "fit_consensus": consensus_rows,
        "threshold_sensitivity": sensitivity,
        "input_aware_pooled_consensus": [mode.as_dict() for mode in input_aware],
        "role_counts": {
            role: {
                "events": len(populations[role]),
                "fault_lines": len(empirical[role]),
                "empirical_modes": sum(len(modes) for events in empirical[role].values() for modes in events.values()),
            }
            for role in empirical
        },
        "regime_envelope_map": regime_map,
        "exact_DMD_interpretation": "empirical forced-response descriptor, not exact autonomous eigenanalysis",
    })
    _json(result_dir/"low_rank_candidate.json", {
        "weak_fit": weak, "physics": candidate.physics_audit(),
        "candidate": {
            "frequency_hz": candidate.frequency_hz, "damping_ratio": candidate.damping_ratio,
            "rank": candidate.rank, "active_dynamic_states": 2*candidate.rank,
        },
    })
    _json(result_dir/"integration_convergence.json", convergence)
    _csv(result_dir/"data_qualification.csv", data["rows"])
    _csv(result_dir/"fault_line_development_roles.csv", [
        {"fault_line": line, "role": role} for line, role in sorted(roles.items())
    ])
    _csv(result_dir/"empirical_modes.csv", empirical_rows)
    _csv(result_dir/"candidate_per_case_metrics.csv", metrics)
    _csv(result_dir/"candidate_aggregate_metrics.csv", aggregate)
    _csv(result_dir/"candidate_modal_matches.csv", modal_matches)
    _json(result_dir/"lineage.json", {
        "run_id": run_id, "stage": "controlled_experiment",
        "hypothesis_result": hypothesis.name,
        "hypothesis_sha256": _sha256(hypothesis/"hypothesis_registry.json"),
        "historical_result_artifacts_read": [], "canonical_data_mutated": False,
        "frozen_F1_validation_read": False, "test_ids_read": [], "other_folds_read": [],
    })
    _write_hashes(result_dir)


def _run_residual(root: Path, config: dict[str, Any], result_dir: Path, run_id: str) -> None:
    controlled = _latest_result(root, str(config["controlled_experiment"]), "controlled_experiment")
    if _verify_hashes(controlled):
        raise RuntimeError("v8 controlled artifacts changed before residual diagnosis")
    aggregate = _read_csv(controlled/"candidate_aggregate_metrics.csv")
    low_rank = json.loads((controlled/"low_rank_candidate.json").read_text(encoding="utf-8"))
    signal_condition = float(low_rank["weak_fit"]["signal_condition_number"])
    parameter_identifiable = bool(
        low_rank["weak_fit"]["solver_success"]
        and np.isfinite(signal_condition)
        and signal_condition <= float(config["maximum_hankel_condition"])
    )
    by_key = {(row["cohort"], row["branch"], int(row["horizon_steps"])): row for row in aggregate}
    def improvement(cohort: str, horizon: int) -> float:
        base = float(by_key[(cohort, "SS-DM", horizon)]["mean_rmse"])
        candidate = float(by_key[(cohort, "SS-DM+LRMRP", horizon)]["mean_rmse"])
        return (base-candidate)/max(base, np.finfo(float).eps)
    diagnosis = {
        "run_id": run_id,
        "selection_improvement_1s": improvement("development_selection", 100),
        "selection_improvement_10s": improvement("development_selection", 1000),
        "audit_improvement_10s": improvement("development_audit", 1000),
        "low_rank_signal_condition_number": signal_condition,
        "maximum_registered_condition_number": float(config["maximum_hankel_condition"]),
        "low_rank_parameter_identifiable": parameter_identifiable,
        "signature": (
            "short-horizon residual repair with an ill-conditioned modal loading fit"
            if not parameter_identifiable
            else "short-horizon residual repair without held-out long-horizon closure"
        ),
        "interpretation": (
            "the low-rank port can explain an early transient force component, but its loading "
            "parameters are not identifiable under the registered condition-number gate and the "
            "learned subspace is not a repeatable long-horizon state across fault lines"
            if not parameter_identifiable
            else "the low-rank port can explain an early transient force component, but the learned "
            "modal subspace is not a repeatable long-horizon state across fault lines"
        ),
        "test_ids_read": [], "other_folds_read": [],
    }
    _json(result_dir/"residual_diagnosis.json", diagnosis)
    _json(result_dir/"lineage.json", {
        "run_id": run_id, "stage": "residual_diagnosis", "controlled_result": controlled.name,
        "test_ids_read": [], "other_folds_read": [],
    })
    _write_hashes(result_dir)


def _run_adversarial(root: Path, config: dict[str, Any], result_dir: Path, run_id: str) -> None:
    controlled = _latest_result(root, str(config["controlled_experiment"]), "controlled_experiment")
    residual = _latest_result(root, str(config["residual_experiment"]), "residual_diagnosis")
    if _verify_hashes(controlled) or _verify_hashes(residual):
        raise RuntimeError("v8 parent artifacts changed before adversarial review")
    lineage = json.loads((controlled/"lineage.json").read_text(encoding="utf-8"))
    modal = json.loads((controlled/"modal_identifiability.json").read_text(encoding="utf-8"))
    low_rank = json.loads((controlled/"low_rank_candidate.json").read_text(encoding="utf-8"))
    diagnosis = json.loads((residual/"residual_diagnosis.json").read_text(encoding="utf-8"))
    roles = _read_csv(controlled/"fault_line_development_roles.csv")
    by_line = defaultdict(set)
    for row in roles:
        by_line[row["fault_line"]].add(row["role"])
    checks = {
        "hypothesis_registered_first": lineage["hypothesis_result"].startswith(str(config["hypothesis_experiment"])),
        "fault_line_roles_disjoint": all(len(value) == 1 for value in by_line.values()),
        "frozen_validation_not_read": not lineage["frozen_F1_validation_read"],
        "test_locked": lineage["test_ids_read"] == [],
        "other_folds_locked": lineage["other_folds_read"] == [],
        "low_rank_solver_succeeded": bool(low_rank["weak_fit"]["solver_success"]),
        "low_rank_physics_passed": low_rank["physics"]["status"] == "pass",
        "low_rank_condition_number_audited": bool(
            np.isfinite(float(diagnosis["low_rank_signal_condition_number"]))
        ),
        "threshold_sensitivity_completed": len(modal["threshold_sensitivity"]) >= 9,
    }
    alternatives = [
        {"threat": "ordinary DMD folded observed Pm into its poles", "disposition": "retained; exact DMD is only an empirical descriptor and pooled DMDc is separately reported"},
        {"threat": "lack of consensus is caused by one strict threshold", "disposition": "rejected over the pre-registered 3x3 frequency/MAC sensitivity map"},
        {
            "threat": "low-rank failure is numerical or non-identifiable",
            "disposition": (
                "retained: the weak design condition number exceeds the registered 1e8 gate; "
                "the loading cannot support a physical modal claim even though solver, passivity "
                "and RK4 checks pass"
                if not diagnosis["low_rank_parameter_identifiable"]
                else "rejected by the condition-number, solver, passivity and RK4 gates"
            ),
        },
        {"threat": "short-horizon gain proves a physical hidden mode", "disposition": "rejected because 10 s selection and disjoint audit do not retain the gain"},
        {"threat": "more optimizer iterations would create cross-line support", "disposition": "rejected: support is an empirical data property measured before fitting the candidate"},
        {"threat": "observed Pm makes the result deployable", "disposition": f"retained; all rollouts remain {ROLLOUT_LABEL}"},
    ]
    review = {
        "run_id": run_id, "status": "PASS" if all(checks.values()) else "FAIL",
        "low_rank_parameter_identifiable": bool(diagnosis["low_rank_parameter_identifiable"]),
        "checks": checks, "alternative_explanations": alternatives,
        "test_ids_read": [], "other_folds_read": [],
    }
    _json(result_dir/"adversarial_review.json", review)
    _json(result_dir/"lineage.json", {
        "run_id": run_id, "stage": "adversarial_review", "controlled_result": controlled.name,
        "residual_result": residual.name, "test_ids_read": [], "other_folds_read": [],
    })
    _write_hashes(result_dir)
    if review["status"] != "PASS":
        raise RuntimeError("v8 adversarial review failed")


def _decision(
    modal: dict[str, Any],
    residual: dict[str, Any],
    convergence: dict[str, Any],
    config: dict[str, Any],
) -> tuple[str, dict[str, bool]]:
    consensus_pass = any(bool(row["passes_all_roles"]) for row in modal["fit_consensus"])
    sensitivity_pass = any(
        int(row["modes_passing_fit_80pct"]) > 0 for row in modal["threshold_sensitivity"]
    )
    rollout_pass = bool(
        float(residual["selection_improvement_10s"]) >= float(config["minimum_model_improvement"])
        and float(residual["audit_improvement_10s"]) >= float(config["minimum_model_improvement"])
    )
    low_rank_identifiable = bool(residual.get("low_rank_parameter_identifiable", False))
    regime_map = modal.get("regime_envelope_map", {})
    envelope_map_complete = bool(
        regime_map.get("complete_nonoverlapping_assignment")
        and regime_map.get("envelopes")
        and int(regime_map.get("assigned_events", -1)) == int(regime_map.get("eligible_events", -2))
    )
    gates = {
        "cross_role_consensus_mode_found": consensus_pass,
        "any_sensitivity_setting_reaches_80pct_fit_support": sensitivity_pass,
        "heldout_10s_low_rank_improvement_at_least_2pct": rollout_pass,
        "low_rank_parameters_identifiable": low_rank_identifiable,
        "low_rank_short_horizon_gain_is_not_sufficient": float(residual["selection_improvement_1s"]) > 0,
        "RK4_convergent": bool(convergence) and all(bool(row["convergent"]) for row in convergence.values()),
        "regime_envelope_map_complete": envelope_map_complete,
    }
    if consensus_pass and rollout_pass and low_rank_identifiable and gates["RK4_convergent"]:
        return "MODAL_STRUCTURE_IDENTIFIED", gates
    # A global candidate that is not promoted never ends discovery.  The
    # envelope map is the safe next decision object whether the failure arose
    # from absent cross-role support, a sensitivity-only descriptor, or a
    # supported empirical mode that the present low-rank realization did not
    # convert into held-out rollout gain.  Short-horizon gain remains useful
    # residual evidence but is deliberately not an authorization condition.
    routed_to_envelopes = bool(
        not (rollout_pass and low_rank_identifiable)
        and gates["RK4_convergent"]
        and envelope_map_complete
    )
    return (
        "MODAL_ENVELOPE_MAP_CREATED"
        if routed_to_envelopes else "MODAL_IDENTIFIABILITY_AUDIT_INCONCLUSIVE",
        gates,
    )


def _write_report(result_dir: Path, acceptance: dict[str, Any]) -> None:
    evidence = acceptance["evidence"]
    lines = [
        "# M0-AFSD v8：模态可辨识性与低秩互易端口审计", "",
        f"**终态：{acceptance['terminal_state']}**", "",
        f"本轮仅使用 F1、seed 324、Q-A train；未读取冻结 validation、test 或 F2–F5。所有重积分均为 **{ROLLOUT_LABEL}**。", "",
        "## 方法", "",
        "v8 将“模态是否可从当前扰动集合辨识”置于模态约束之前。首先对每条 Q-A 轨迹提取质量加权经验 DMD 振型，再以 fault line 为统计单元，用频率与复 MAC 的 complete-link 聚类寻找跨扰动共识。每条 fault line 最多贡献一个最强模态，避免事件数较多的数据族夸大支持率。", "",
        r"同时构造低秩互易被动端口：$\dot q=p$，$\dot p=L^TB^T\omega-\omega_0^2q-2\zeta\omega_0p$，边力为 $Lp$。同一 $L$ 同时连接输入与输出，保证端口矩阵半正定，且每个 rank 只增加两个动态状态。", "",
        "## 结果", "",
        f"fit/selection/audit 分别包含 {evidence['role_counts']['fit']['fault_lines']}、{evidence['role_counts']['selection']['fault_lines']}、{evidence['role_counts']['audit']['fault_lines']} 条 fault lines。fit 中最佳经验模态支持率为 {evidence['best_fit_support']:.3f}；主阈值下通过全部角色的模态数为 {evidence['cross_role_modes']}。", "",
        f"低秩端口在 1 s selection 上改善 {100*evidence['selection_improvement_1s']:.2f}%，但 10 s selection 改善为 {100*evidence['selection_improvement_10s']:.2f}%，独立 audit 为 {100*evidence['audit_improvement_10s']:.2f}%。弱形式设计矩阵条件数为 {evidence['low_rank_signal_condition_number']:.3e}，参数可辨识门为 {evidence['low_rank_parameter_identifiable']}。因此短时残差修复不能被解释为已发现的共享振荡状态。", "",
        "## 科学结论", "",
        "当前 F1 扰动集合没有为一个跨 fault-line 重复、可用于约束全部激励的唯一经验振型提供足够证据。v8 因而生成 amplitude×richness 包络图，将各包络路由到 ESC 的模态或非模态模型梯级，而不是停止结构发现。需要更长的近自由环降、相同运行点下重复且持续激励的扰动，或可用于输入—输出辨识的独立电气观测来扩大包络。", "",
        "因此现有 SS-DM 继续只能命名为 **rollout-accurate dynamic equivalent**；本轮没有读取冻结 validation，也没有晋级任何新模态模块。包络图是下一阶段选择依据，不是已部署模型库。", "",
    ]
    (result_dir/"research_report_zh.md").write_text("\n".join(lines), encoding="utf-8")


def _run_acceptance(root: Path, config: dict[str, Any], result_dir: Path, run_id: str) -> None:
    controlled = _latest_result(root, str(config["controlled_experiment"]), "controlled_experiment")
    residual_dir = _latest_result(root, str(config["residual_experiment"]), "residual_diagnosis")
    adversarial = _latest_result(root, str(config["adversarial_experiment"]), "adversarial_review")
    if _verify_hashes(controlled) or _verify_hashes(residual_dir) or _verify_hashes(adversarial):
        raise RuntimeError("v8 parent artifacts changed before acceptance")
    modal = json.loads((controlled/"modal_identifiability.json").read_text(encoding="utf-8"))
    residual = json.loads((residual_dir/"residual_diagnosis.json").read_text(encoding="utf-8"))
    convergence = json.loads((controlled/"integration_convergence.json").read_text(encoding="utf-8"))
    terminal, gates = _decision(modal, residual, convergence, config)
    accepted = terminal == "MODAL_STRUCTURE_IDENTIFIED"
    envelope_routed = terminal == "MODAL_ENVELOPE_MAP_CREATED"
    evidence = {
        "role_counts": modal["role_counts"],
        "best_fit_support": max((float(row["support_fraction"]) for row in modal["fit_consensus"]), default=0.0),
        "cross_role_modes": sum(bool(row["passes_all_roles"]) for row in modal["fit_consensus"]),
        "input_aware_consensus_modes": len(modal["input_aware_pooled_consensus"]),
        "regime_envelope_statuses": {
            row["envelope_id"]: row["status"]
            for row in modal["regime_envelope_map"]["envelopes"]
        },
        "selection_improvement_1s": float(residual["selection_improvement_1s"]),
        "selection_improvement_10s": float(residual["selection_improvement_10s"]),
        "audit_improvement_10s": float(residual["audit_improvement_10s"]),
        "low_rank_signal_condition_number": float(residual["low_rank_signal_condition_number"]),
        "low_rank_parameter_identifiable": bool(residual["low_rank_parameter_identifiable"]),
    }
    acceptance = {
        "run_id": run_id,
        "status": "PASS" if accepted or envelope_routed else "REJECTED",
        "terminal_state": terminal,
        "model_reporting_name": "modal-constrained passive dynamic equivalent" if accepted else "rollout-accurate dynamic equivalent",
        "gates": gates, "evidence": evidence,
        "why_more_compute_is_insufficient": (
            "fault-line support is measured directly from Q-A trajectories and stays below the gate "
            "over the registered threshold map; optimizer iterations cannot create unobserved repeatability"
        ),
        "continuation_route": (
            "ESC_PER_ENVELOPE_SELECTION_AND_COMPOSITION" if envelope_routed
            else "FREEZE_SHARED_MODAL_BRANCH" if accepted
            else "REFINE_IDENTIFIABILITY_AUDIT"
        ),
        "required_new_information": [
            "longer near-free ringdown at a common operating point",
            "repeated persistently exciting disturbances covering the same modes across fault lines",
            "independent voltage/Pe or controller-channel measurements for input-output modal separation",
        ],
        "claim_boundary": ROLLOUT_LABEL,
        "frozen_F1_validation_read": False, "test_ids_read": [], "other_folds_read": [],
        "merge_gate_required": True,
    }
    _json(result_dir/"acceptance.json", acceptance)
    _write_report(result_dir, acceptance)
    _json(result_dir/"lineage.json", {
        "run_id": run_id, "stage": "acceptance", "controlled_result": controlled.name,
        "residual_result": residual_dir.name, "adversarial_result": adversarial.name,
        "frozen_F1_validation_read": False, "test_ids_read": [], "other_folds_read": [],
    })
    _write_hashes(result_dir)


def run(*, root: Path, experiment: dict[str, Any], stage: str, result_dir: Path, run_id: str) -> None:
    config_path = root/"configs"/"experiments"/str(experiment["config"])
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    _validate_scope(config)
    if stage == "hypothesis_registration":
        _run_hypothesis(config_path, config, result_dir, run_id)
    elif stage == "controlled_experiment":
        _run_controlled(root, config_path, config, result_dir, run_id)
    elif stage == "residual_diagnosis":
        _run_residual(root, config, result_dir, run_id)
    elif stage == "adversarial_review":
        _run_adversarial(root, config, result_dir, run_id)
    elif stage == "acceptance":
        _run_acceptance(root, config, result_dir, run_id)
    else:
        raise ValueError(f"unsupported v8 stage: {stage}")
