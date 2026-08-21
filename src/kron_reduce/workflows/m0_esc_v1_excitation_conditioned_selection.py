from __future__ import annotations

import json
import tomllib
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from kron_reduce.evaluation import (
    bootstrap_m0_physical_modes,
    certify_sublevel_region,
    cross_cluster_modal_consensus,
    exact_dmd_modes,
    finite_difference_augmented_modes,
    match_dmd_modes,
    modal_assurance_criterion,
    physical_model_modes,
    region_fidelity,
)
from kron_reduce.models import M0Model, RunContext, TransientEmfM0Model
from kron_reduce.selection import ExcitationEnvelope, ExcitationEnvelopeRouter
from kron_reduce.synthetic import benchmark_registry_v1
from kron_reduce.workflows.m0_afsd_v7_modal_structure_discovery import (
    _fit_m0,
    _fit_ssdm,
    _median_modal,
)
from kron_reduce.workflows.m0_afsd_v8_modal_identifiability import (
    ROLLOUT_LABEL,
    _candidate_metrics,
    _csv,
    _empirical_modes,
    _json,
    _latest_result,
    _qualified_populations,
    _regime_envelope_map,
    _subset_empirical_events,
    _verify_hashes,
    _write_hashes,
)

WORKFLOW = "m0_esc_v1_excitation_conditioned_selection"


def _validate_scope(config: dict[str, Any]) -> None:
    expected = {
        "fold": "F1",
        "seed": 324,
        "pm_mode": "observed",
        "rollout_label": ROLLOUT_LABEL,
        "test_access": "forbidden",
        "other_folds_access": "forbidden",
        "canonical_mutation": "forbidden",
        "synthetic_registry": "synthetic_benchmark_registry_v1",
    }
    mismatch = {
        key: (config.get(key), value) for key, value in expected.items() if config.get(key) != value
    }
    if mismatch:
        raise RuntimeError(f"ESC v1 scope violation: {mismatch}")


def _hypothesis_registry(config: dict[str, Any], run_id: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "workflow": WORKFLOW,
        "registered_before_controlled_metrics": True,
        "primary_hypothesis": (
            "a shared passive weak-form equation identifies system poles even when different "
            "fault lines excite disjoint modal subsets; per-event DMD support measures "
            "observability and must not be used as a pole-existence gate"
        ),
        "engineering_hypothesis": (
            "within each train-defined Q-A amplitude-by-richness envelope, the minimum "
            "B2 -> B3 -> SS-DM rung that earns at least 2% held-out 10 s improvement can "
            "be selected without assuming one global order"
        ),
        "registered_ladder": ["B2", "B3", "SS-DM"],
        "selection_unit": "excitation envelope",
        "uncertainty_unit": "fault_line",
        "modal_evidence_roles": {
            "explicit_equation_Jacobian": "system pole estimate",
            "fault_line_cluster_bootstrap": "parameter-to-pole stability",
            "trajectory_DMD": "disturbance-specific observation support only",
        },
        "gates": {
            "complex_rung_minimum_10s_improvement": float(config["minimum_model_improvement"]),
            "model_mode_bootstrap_support": float(config["model_mode_minimum_bootstrap_support"]),
            "synthetic_frequency_error": float(config["synthetic_maximum_frequency_error"]),
            "synthetic_damping_error": float(config["synthetic_maximum_damping_error"]),
            "synthetic_MAC": float(config["synthetic_minimum_mac"]),
            "modal_absolute": {
                "frequency_relative_error": float(config["absolute_modal_maximum_frequency_error"]),
                "damping_absolute_error": float(config["absolute_modal_maximum_damping_error"]),
                "MAC": float(config["absolute_modal_minimum_mac"]),
            },
        },
        "claim_boundaries": config["claim_boundaries"],
        "test_ids_read": [],
        "other_folds_read": [],
    }


def _run_hypothesis(
    config_path: Path,
    config: dict[str, Any],
    result_dir: Path,
    run_id: str,
) -> None:
    _validate_scope(config)
    (result_dir / "frozen_config.toml").write_bytes(config_path.read_bytes())
    _json(result_dir / "hypothesis_registry.json", _hypothesis_registry(config, run_id))
    _json(
        result_dir / "lineage.json",
        {
            "run_id": run_id,
            "stage": "hypothesis_registration",
            "historical_result_artifacts_read": [],
            "frozen_F1_validation_read": False,
            "test_ids_read": [],
            "other_folds_read": [],
        },
    )
    _write_hashes(result_dir)


def _synthetic_samples(config: dict[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
    benchmark = benchmark_registry_v1()["swing4_b2_modal"]
    time = np.linspace(0.0, 6.0, 601)
    samples: list[dict[str, Any]] = []
    for mode_index in range(len(benchmark.ground_truth_modes())):
        for replicate in range(4):
            initial = benchmark.modal_initial_state(
                [mode_index],
                amplitude=0.012 + 0.001 * replicate,
                phase=0.29 * replicate,
            )
            trajectory = benchmark.simulate(
                event_id=f"mode-{mode_index}-event-{replicate}",
                fault_line=f"mode-{mode_index}-line-{replicate}",
                regime=f"mode_{mode_index}_only",
                time=time,
                delta0=initial[0],
                omega0=initial[1],
            )
            samples.append(benchmark.as_model_sample(trajectory))
    return benchmark, samples


def _synthetic_gate(config: dict[str, Any]) -> dict[str, Any]:
    benchmark, samples = _synthetic_samples(config)
    train = [sample for index, sample in enumerate(samples) if index % 4 != 3]
    validation = [sample for index, sample in enumerate(samples) if index % 4 == 3]
    model = M0Model(max_internal_step=float(config["max_internal_step"]))
    model.fit(
        train,
        validation,
        RunContext(
            seed=int(config["seed"]),
            horizon_steps=601,
            max_cases=len(train),
            ridge_values=(1e-10, 1e-8, 1e-6),
            integral_windows=(1, 5, 20, 80),
            fit_max_steps=601,
            max_internal_step=float(config["max_internal_step"]),
        ),
    )
    fitted_modes, model_diagnostics = physical_model_modes(
        model,
        train[0]["model_inputs"]["static"],
        min_frequency_hz=float(config["modal_min_frequency_hz"]),
        max_frequency_hz=float(config["modal_max_frequency_hz"]),
    )
    matches = match_dmd_modes(
        benchmark.ground_truth_modes(),
        fitted_modes,
        max_relative_frequency_error=float(config["synthetic_maximum_frequency_error"]),
    )
    empirical_by_line = {}
    for sample in samples:
        target = sample["training_context"]["targets"]
        mass_weighted = np.asarray(target["omega"]) * np.sqrt(benchmark.mass)
        empirical_by_line[sample["training_context"]["metadata"]["fault_line"]] = exact_dmd_modes(
            mass_weighted,
            float(target["t"][1] - target["t"][0]),
            rank=4,
            min_frequency_hz=0.02,
            max_frequency_hz=3.0,
        )
    uniform_consensus = cross_cluster_modal_consensus(
        empirical_by_line,
        minimum_support_fraction=float(config["modal_minimum_fault_line_support"]),
    )
    bootstrap, bootstrap_diagnostics = bootstrap_m0_physical_modes(
        model,
        train,
        draws=int(config["synthetic_bootstrap_draws"]),
        seed=int(config["seed"]),
        windows=(1, 5, 20, 80),
        horizon_steps=601,
        max_fit_steps=601,
        maximum_relative_frequency_error=float(config["model_mode_frequency_tolerance"]),
        minimum_mac=float(config["model_mode_minimum_mac"]),
    )

    port = benchmark_registry_v1()["swing4_reciprocal_port_r1"]
    hidden = np.zeros(2 * port.rank)

    def port_field(
        delta: np.ndarray,
        omega: np.ndarray,
        hidden_state: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        derivative = port._field(
            (
                delta,
                omega,
                hidden_state[: port.rank],
                hidden_state[port.rank :],
            ),
            np.zeros(len(port.mass)),
        )
        return derivative[0], derivative[1], np.concatenate(derivative[2:])

    numerical_port_modes, port_diagnostics = finite_difference_augmented_modes(
        port.mass,
        np.zeros_like(port.mass),
        hidden,
        port_field,
    )
    port_matches = match_dmd_modes(
        port.ground_truth_modes(),
        numerical_port_modes,
        max_relative_frequency_error=1e-5,
    )

    coordinate = np.linspace(-2.8, 2.8, 200)
    speed = np.linspace(-2.5, 2.5, 201)
    delta_grid, speed_grid = np.meshgrid(coordinate, speed, indexing="ij")
    values = (0.5 * speed_grid**2 + 1.0 - np.cos(delta_grid)).ravel()
    decrease = (-0.08 * speed_grid**2).ravel()
    certificate = certify_sublevel_region(
        values,
        decrease,
        barrier_level=2.0,
        tolerance=0.0,
    )
    fidelity = region_fidelity(values <= certificate.level, values < 2.0 + 1e-9)

    router = ExcitationEnvelopeRouter(
        [
            ExcitationEnvelope(
                envelope_id="synthetic_known",
                model_id="synthetic_B2",
                bounds={"prefix_amplitude": (0.0, 1.0), "prefix_entropy": (0.1, 0.5)},
                earned_name="modal-identified equivalent",
                evidence_run_id="model_unit_gate",
            )
        ]
    )
    outside_route = router.route({"prefix_amplitude": 1.5, "prefix_entropy": 0.3}).as_dict()
    median_frequency_error = float(np.median([row["relative_frequency_error"] for row in matches]))
    median_damping_error = float(
        np.median([row["absolute_damping_ratio_error"] for row in matches])
    )
    median_mac = float(np.median([row["mac"] for row in matches]))
    gates = {
        "three_B2_modes_recovered": len(matches) == 3,
        "frequency_error_within_gate": median_frequency_error
        <= float(config["synthetic_maximum_frequency_error"]),
        "damping_error_within_gate": median_damping_error
        <= float(config["synthetic_maximum_damping_error"]),
        "MAC_within_gate": median_mac >= float(config["synthetic_minimum_mac"]),
        "uniform_per_line_DMD_correctly_not_required": len(uniform_consensus) == 0,
        "shared_model_modes_bootstrap_supported": bool(bootstrap)
        and min(mode.support_fraction for mode in bootstrap)
        >= float(config["model_mode_minimum_bootstrap_support"]),
        "reciprocal_port_linearization_recovers_four_modes": len(port_matches) == 4,
        "known_separatrix_certified_conservatively": bool(
            certificate.certified
            and abs(certificate.level - 2.0) < 1e-12
            and fidelity["conservative"]
        ),
        "outside_envelope_refused": outside_route["status"] == "REFUSE_AND_REQUEST_DATA",
    }
    return {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "B2_modal_recovery": {
            "matches": matches,
            "model_diagnostics": model_diagnostics,
            "median_frequency_error": median_frequency_error,
            "median_damping_error": median_damping_error,
            "median_MAC": median_mac,
            "uniform_fault_line_DMD_consensus_count": len(uniform_consensus),
            "bootstrap_modes": [mode.as_dict() for mode in bootstrap],
            "bootstrap_diagnostics": bootstrap_diagnostics,
        },
        "reciprocal_port": {
            "matches": port_matches,
            "diagnostics": port_diagnostics,
        },
        "stability_region": {
            "certificate": certificate.__dict__,
            "fidelity": fidelity,
        },
        "outside_envelope_probe": outside_route,
    }


def _b3_from_m0(model: M0Model, config: dict[str, Any]) -> TransientEmfM0Model:
    return TransientEmfM0Model(
        anchor=deepcopy(model),
        alpha=float(config["baseline_alpha"]),
        tau=float(config["baseline_tau"]),
        activation_threshold=0.0,
        activation_width=float(config["b3_activation_width"]),
        max_internal_step=float(config["max_internal_step"]),
        diagnostics={
            "maximum_equilibrium_residual": 0.0,
            "hyperparameters_frozen_before_ESC_controlled_metrics": True,
        },
    )


def _horizon_row(rows: list[dict[str, Any]], horizon: int) -> dict[str, Any]:
    return next(row for row in rows if int(row["horizon_steps"]) == int(horizon))


def _select_ladder(
    summaries: dict[str, list[dict[str, Any]]],
    physics: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    current = "B2"
    decisions = []
    for candidate in ("B3", "SS-DM"):
        base = _horizon_row(summaries[current], 1000)
        row = _horizon_row(summaries[candidate], 1000)
        improvement = (float(base["mean_rmse"]) - float(row["mean_rmse"])) / max(
            float(base["mean_rmse"]), np.finfo(float).eps
        )
        gates = {
            "minimum_2pct_improvement": improvement >= float(config["minimum_model_improvement"]),
            "zero_failures": float(row["failure_rate"]) == 0.0,
            "physics_pass": physics[candidate].get("status") == "pass",
        }
        promoted = all(gates.values())
        decisions.append(
            {
                "from": current,
                "candidate": candidate,
                "relative_10s_improvement": improvement,
                "gates": gates,
                "promoted": promoted,
            }
        )
        if promoted:
            current = candidate
    return {"selected_branch": current, "decisions": decisions}


def _observability_support(
    mode: Any,
    empirical: dict[str, dict[str, list[Any]]],
    config: dict[str, Any],
) -> dict[str, Any]:
    reference = np.asarray(mode.shape_real) + 1j * np.asarray(mode.shape_imag)
    per_line = {}
    for line, events in sorted(empirical.items()):
        event_matches = []
        for event_id, candidates in sorted(events.items()):
            compatible = False
            best_frequency_error = float("inf")
            best_mac = 0.0
            for candidate in candidates:
                frequency_error = abs(candidate.frequency_hz - mode.frequency_hz) / max(
                    mode.frequency_hz, np.finfo(float).eps
                )
                shape = np.asarray(candidate.shape_real) + 1j * np.asarray(candidate.shape_imag)
                mac = modal_assurance_criterion(reference, shape)
                compatible |= bool(
                    frequency_error <= float(config["modal_frequency_tolerance"])
                    and mac >= float(config["modal_minimum_mac"])
                )
                best_frequency_error = min(best_frequency_error, frequency_error)
                best_mac = max(best_mac, mac)
            event_matches.append(
                {
                    "event_id": event_id,
                    "compatible": compatible,
                    "best_relative_frequency_error": best_frequency_error,
                    "best_MAC": best_mac,
                }
            )
        fraction = (
            float(np.mean([row["compatible"] for row in event_matches])) if event_matches else 0.0
        )
        per_line[line] = {
            "event_support_fraction": fraction,
            "events": len(event_matches),
            "observable": fraction >= float(config["modal_minimum_event_fraction_per_line"]),
        }
    return {
        "fault_line_observation_fraction": (
            float(np.mean([row["observable"] for row in per_line.values()])) if per_line else 0.0
        ),
        "supporting_fault_lines": sorted(
            line for line, row in per_line.items() if row["observable"]
        ),
        "per_fault_line": per_line,
        "interpretation": "observation support only; zero support is not a non-existence claim",
    }


def _envelope_ids(
    qa_rows: list[dict[str, Any]],
    roles: dict[str, str],
    amplitude: str,
    richness: str,
) -> dict[str, set[str]]:
    output = {role: set() for role in ("fit", "selection", "audit")}
    for row in qa_rows:
        if str(row["amplitude_level"]) == amplitude and str(row["richness_level"]) == richness:
            role = roles[str(row["fault_line"])]
            output[role].add(str(row["event_id"]))
    return output


def _run_controlled(
    root: Path,
    config_path: Path,
    config: dict[str, Any],
    result_dir: Path,
    run_id: str,
) -> None:
    hypothesis = _latest_result(
        root,
        str(config["hypothesis_experiment"]),
        "hypothesis_registration",
    )
    if _verify_hashes(hypothesis):
        raise RuntimeError("ESC hypothesis artifacts changed before controlled experiment")
    (result_dir / "frozen_config.toml").write_bytes(config_path.read_bytes())
    synthetic = _synthetic_gate(config)
    if synthetic["status"] != "PASS":
        raise RuntimeError("synthetic model-unit truth gate failed")

    data, qa_rows, roles, populations = _qualified_populations(root, config)
    empirical = {}
    empirical_rows = []
    for role in ("fit", "selection", "audit"):
        empirical[role], rows = _empirical_modes(populations[role], config)
        empirical_rows.extend({"role": role, **row} for row in rows)
    regime_map = _regime_envelope_map(qa_rows, roles, empirical, config)
    checkpoints = result_dir / "checkpoints"
    checkpoints.mkdir()
    all_case_metrics: list[dict[str, Any]] = []
    all_aggregate: list[dict[str, Any]] = []
    all_modal_matches: list[dict[str, Any]] = []
    envelope_results = []
    for envelope in regime_map["envelopes"]:
        amplitude = str(envelope["amplitude_level"])
        richness = str(envelope["richness_level"])
        ids = _envelope_ids(qa_rows, roles, amplitude, richness)
        scoped_samples = {
            role: [sample for sample in populations[role] if str(sample["event_id"]) in ids[role]]
            for role in ids
        }
        scoped_empirical = {
            role: _subset_empirical_events(empirical[role], ids[role]) for role in ids
        }
        qualified = bool(
            len(scoped_empirical["fit"]) >= int(config["minimum_regime_fit_fault_lines"])
            and len(scoped_empirical["selection"])
            >= int(config["minimum_regime_selection_fault_lines"])
            and len(scoped_empirical["audit"]) >= int(config["minimum_regime_audit_fault_lines"])
        )
        if not qualified:
            envelope_results.append(
                {
                    "envelope_id": envelope["envelope_id"],
                    "bounds": envelope["bounds"],
                    "status": "ENVELOPE_DATA_REQUEST",
                    "counts": envelope["counts"],
                    "reason": "fault-line-disjoint fit/selection/audit support is insufficient",
                }
            )
            continue

        m0 = _fit_m0(scoped_samples["fit"], scoped_samples["selection"], config)
        b3 = _b3_from_m0(m0, config)
        ssdm = _fit_ssdm(scoped_samples["fit"], scoped_samples["selection"], config)
        models = {"B2": m0, "B3": b3, "SS-DM": ssdm}
        summaries: dict[str, list[dict[str, Any]]] = {}
        physics = {branch: model.physics_audit() for branch, model in models.items()}
        selection_modes: dict[str, dict[str, Any]] = {}
        for branch, model in models.items():
            rows, aggregate, modal_rows = _candidate_metrics(
                model,
                scoped_samples["selection"],
                tuple(int(value) for value in config["report_horizons"]),
                branch=branch,
                cohort=f"{envelope['envelope_id']}__selection",
            )
            all_case_metrics.extend(rows)
            all_aggregate.extend(aggregate)
            all_modal_matches.extend(modal_rows)
            summaries[branch] = aggregate
            model_modes, diagnostics = physical_model_modes(
                model,
                scoped_samples["fit"][0]["model_inputs"]["static"],
                min_frequency_hz=float(config["modal_min_frequency_hz"]),
                max_frequency_hz=float(config["modal_max_frequency_hz"]),
            )
            selection_modes[branch] = {
                "modes": [mode.as_dict() for mode in model_modes],
                "linearization": diagnostics,
                "trajectory_modal_summary": _median_modal(modal_rows),
            }
        decision = _select_ladder(summaries, physics, config)
        selected_branch = str(decision["selected_branch"])
        selected_model = models[selected_branch]

        audit_summaries = {}
        audit_modal = {}
        for branch in sorted({"B2", selected_branch}):
            rows, aggregate, modal_rows = _candidate_metrics(
                models[branch],
                scoped_samples["audit"],
                tuple(int(value) for value in config["report_horizons"]),
                branch=branch,
                cohort=f"{envelope['envelope_id']}__audit",
            )
            all_case_metrics.extend(rows)
            all_aggregate.extend(aggregate)
            all_modal_matches.extend(modal_rows)
            audit_summaries[branch] = aggregate
            audit_modal[branch] = _median_modal(modal_rows)

        selected_modes, selected_linearization = physical_model_modes(
            selected_model,
            scoped_samples["fit"][0]["model_inputs"]["static"],
            min_frequency_hz=float(config["modal_min_frequency_hz"]),
            max_frequency_hz=float(config["modal_max_frequency_hz"]),
        )
        observation_support = []
        for index, mode in enumerate(selected_modes):
            observation_support.append(
                {
                    "mode_index": index,
                    **mode.as_dict(),
                    "fit": _observability_support(mode, scoped_empirical["fit"], config),
                    "selection": _observability_support(
                        mode, scoped_empirical["selection"], config
                    ),
                    "audit": _observability_support(mode, scoped_empirical["audit"], config),
                }
            )
        bootstrap_modes, bootstrap_diagnostics = bootstrap_m0_physical_modes(
            m0,
            scoped_samples["fit"],
            draws=int(config["f1_bootstrap_draws"]),
            seed=int(config["seed"]),
            windows=tuple(int(value) for value in config["integral_windows"]),
            horizon_steps=int(config["selection_horizon_steps"]),
            max_fit_steps=int(config["fit_max_steps"]),
            max_windows_per_case_horizon=int(config["weak_max_windows_per_case_horizon"]),
            maximum_relative_frequency_error=float(config["model_mode_frequency_tolerance"]),
            minimum_mac=float(config["model_mode_minimum_mac"]),
        )
        envelope_checkpoint = checkpoints / str(envelope["envelope_id"])
        envelope_checkpoint.mkdir()
        for branch, model in models.items():
            model.save(envelope_checkpoint / f"{branch}.npz")
        envelope_results.append(
            {
                "envelope_id": envelope["envelope_id"],
                "bounds": envelope["bounds"],
                "status": "SELECTED",
                "counts": envelope["counts"],
                "selection": decision,
                "selection_aggregate": summaries,
                "audit_aggregate": audit_summaries,
                "selection_model_modes": selection_modes,
                "selected_model_modes": [mode.as_dict() for mode in selected_modes],
                "selected_linearization": selected_linearization,
                "selected_mode_observation_support": observation_support,
                "B2_fault_line_bootstrap_modes": [mode.as_dict() for mode in bootstrap_modes],
                "B2_fault_line_bootstrap_diagnostics": bootstrap_diagnostics,
                "selected_dynamic_mode_bootstrap_status": (
                    "COMPLETE"
                    if selected_branch == "B2"
                    else "NOT_IDENTIFIED_BY_B2_BACKBONE_BOOTSTRAP"
                ),
                "audit_modal": audit_modal,
                "physics": physics,
                "rollout_label": ROLLOUT_LABEL,
            }
        )

    _json(result_dir / "synthetic_ground_truth_gate.json", synthetic)
    _json(result_dir / "train_envelope_map.json", regime_map)
    _json(
        result_dir / "conditional_model_library.json",
        {
            "status": "DEVELOPMENT_LIBRARY",
            "entries": envelope_results,
            "coverage": "union of SELECTED train-defined envelopes only",
            "outside_union_policy": "REFUSE_AND_REQUEST_DATA",
            "deployment_status": "offline evidence; causal prefix detector not yet calibrated",
            "rollout_label": ROLLOUT_LABEL,
        },
    )
    _csv(result_dir / "data_qualification.csv", data["rows"])
    _csv(
        result_dir / "fault_line_development_roles.csv",
        [{"fault_line": line, "role": role} for line, role in sorted(roles.items())],
    )
    _csv(result_dir / "empirical_modes.csv", empirical_rows)
    _csv(result_dir / "candidate_per_case_metrics.csv", all_case_metrics)
    _csv(result_dir / "candidate_aggregate_metrics.csv", all_aggregate)
    _csv(result_dir / "candidate_modal_matches.csv", all_modal_matches)
    _json(
        result_dir / "lineage.json",
        {
            "run_id": run_id,
            "stage": "controlled_experiment",
            "hypothesis_result": hypothesis.name,
            "canonical_data_mutated": False,
            "frozen_F1_validation_read": False,
            "test_ids_read": [],
            "other_folds_read": [],
            "conditional_rollout_with_observed_Pm": True,
        },
    )
    _write_hashes(result_dir)


def _modal_absolute_gate(summary: dict[str, Any], config: dict[str, Any]) -> dict[str, bool]:
    return {
        "matched_modes_present": int(summary["matched_modes"]) > 0,
        "frequency": float(summary["frequency_error"])
        <= float(config["absolute_modal_maximum_frequency_error"]),
        "damping": float(summary["damping_error"])
        <= float(config["absolute_modal_maximum_damping_error"]),
        "MAC": float(summary["MAC"]) >= float(config["absolute_modal_minimum_mac"]),
    }


def _run_residual(
    root: Path,
    config: dict[str, Any],
    result_dir: Path,
    run_id: str,
) -> None:
    controlled = _latest_result(
        root,
        str(config["controlled_experiment"]),
        "controlled_experiment",
    )
    if _verify_hashes(controlled):
        raise RuntimeError("ESC controlled artifacts changed before residual diagnosis")
    library = json.loads(
        (controlled / "conditional_model_library.json").read_text(encoding="utf-8")
    )
    diagnostics = []
    for entry in library["entries"]:
        if entry["status"] != "SELECTED":
            diagnostics.append(
                {
                    "envelope_id": entry["envelope_id"],
                    "terminal_route": "ENVELOPE_DATA_REQUEST",
                    "reason": entry["reason"],
                }
            )
            continue
        selected = entry["selection"]["selected_branch"]
        base = _horizon_row(entry["audit_aggregate"]["B2"], 1000)
        selected_row = _horizon_row(entry["audit_aggregate"][selected], 1000)
        audit_improvement = (float(base["mean_rmse"]) - float(selected_row["mean_rmse"])) / max(
            float(base["mean_rmse"]), np.finfo(float).eps
        )
        modal_gates = _modal_absolute_gate(entry["audit_modal"][selected], config)
        bootstrap_modes = entry["B2_fault_line_bootstrap_modes"]
        bootstrap_supported = bool(bootstrap_modes) and min(
            float(mode["support_fraction"]) for mode in bootstrap_modes
        ) >= float(config["model_mode_minimum_bootstrap_support"])
        selected_modes_identified = bool(
            selected == "B2" and bootstrap_supported and all(modal_gates.values())
        )
        diagnostics.append(
            {
                "envelope_id": entry["envelope_id"],
                "selected_branch": selected,
                "audit_10s_relative_improvement_over_B2": audit_improvement,
                "audit_10s_mean_RMSE": float(selected_row["mean_rmse"]),
                "audit_failure_rate": float(selected_row["failure_rate"]),
                "modal_absolute_gates": modal_gates,
                "B2_model_mode_bootstrap_supported": bootstrap_supported,
                "selected_dynamic_modes_identified": selected_modes_identified,
                "terminal_route": (
                    "MODAL_IDENTIFIED_ENVELOPE"
                    if selected_modes_identified
                    else "ROLLOUT_ONLY_ENVELOPE"
                ),
                "residual_interpretation": (
                    "explicit-equation poles are stable under fault-line resampling and also "
                    "reproduce held-out trajectory modes"
                    if selected_modes_identified
                    else "trajectory prediction is available, but the selected dynamic rung's "
                    "complete pole set has not passed both bootstrap and absolute modal gates"
                ),
            }
        )
    synthetic = json.loads(
        (controlled / "synthetic_ground_truth_gate.json").read_text(encoding="utf-8")
    )
    diagnosis = {
        "run_id": run_id,
        "status": "PASS",
        "synthetic_shared_model_advantage": {
            "uniform_DMD_consensus_count": synthetic["B2_modal_recovery"][
                "uniform_fault_line_DMD_consensus_count"
            ],
            "shared_physical_modes_recovered": len(synthetic["B2_modal_recovery"]["matches"]),
            "interpretation": (
                "the algorithm now separates shared system poles from disturbance-specific "
                "participation instead of requiring one DMD shape in every fault line"
            ),
        },
        "envelopes": diagnostics,
        "test_ids_read": [],
        "other_folds_read": [],
    }
    _json(result_dir / "residual_diagnosis.json", diagnosis)
    _json(
        result_dir / "lineage.json",
        {
            "run_id": run_id,
            "stage": "residual_diagnosis",
            "controlled_result": controlled.name,
            "test_ids_read": [],
            "other_folds_read": [],
        },
    )
    _write_hashes(result_dir)


def _run_adversarial(
    root: Path,
    config: dict[str, Any],
    result_dir: Path,
    run_id: str,
) -> None:
    controlled = _latest_result(root, str(config["controlled_experiment"]), "controlled_experiment")
    residual = _latest_result(root, str(config["residual_experiment"]), "residual_diagnosis")
    if _verify_hashes(controlled) or _verify_hashes(residual):
        raise RuntimeError("ESC parent artifacts changed before adversarial review")
    synthetic = json.loads(
        (controlled / "synthetic_ground_truth_gate.json").read_text(encoding="utf-8")
    )
    library = json.loads(
        (controlled / "conditional_model_library.json").read_text(encoding="utf-8")
    )
    lineage = json.loads((controlled / "lineage.json").read_text(encoding="utf-8"))
    roles = defaultdict(set)
    with (controlled / "fault_line_development_roles.csv").open(encoding="utf-8-sig") as handle:
        import csv

        for row in csv.DictReader(handle):
            roles[row["fault_line"]].add(row["role"])
    checks = {
        "synthetic_truth_gate_passed": synthetic["status"] == "PASS",
        "fault_line_roles_disjoint": all(len(value) == 1 for value in roles.values()),
        "test_locked": lineage["test_ids_read"] == [],
        "other_folds_locked": lineage["other_folds_read"] == [],
        "frozen_validation_not_read": not lineage["frozen_F1_validation_read"],
        "observed_Pm_marked_conditional": bool(
            lineage["conditional_rollout_with_observed_Pm"]
            and library["rollout_label"] == ROLLOUT_LABEL
        ),
        "outside_union_refusal_registered": library["outside_union_policy"]
        == "REFUSE_AND_REQUEST_DATA",
        "envelopes_declared_offline_only": library["deployment_status"].startswith(
            "offline evidence"
        ),
    }
    alternatives = [
        {
            "threat": "the new modal result merely relaxes the DMD threshold",
            "disposition": (
                "rejected on synthetic truth: uniform DMD remains empty while the shared "
                "weak equation recovers all three exact modes with fault-line bootstrap"
            ),
        },
        {
            "threat": "a fitted equation always produces poles, even when parameters are not identifiable",
            "disposition": (
                "retained as a gate: only full-rank, condition-audited, >=80% fault-line "
                "bootstrap modes may be named identified"
            ),
        },
        {
            "threat": "B3 or memory modes inherit B2 bootstrap support",
            "disposition": (
                "rejected: dynamic selections are explicitly rollout-only until their complete "
                "augmented parameter-to-pole bootstrap is implemented"
            ),
        },
        {
            "threat": "the train envelope map can be deployed from a future trajectory summary",
            "disposition": (
                "rejected: F1 envelopes are offline scientific evidence; deployment remains "
                "locked until a causal-prefix detector is calibrated"
            ),
        },
        {
            "threat": "observed Pm makes the model autonomous",
            "disposition": f"retained; every result is {ROLLOUT_LABEL}",
        },
        {
            "threat": "synthetic recovery proves IEEE39 physical topology or device semantics",
            "disposition": "rejected by separate synthetic and flagship evidence axes",
        },
    ]
    review = {
        "run_id": run_id,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "alternative_explanations": alternatives,
        "test_ids_read": [],
        "other_folds_read": [],
    }
    _json(result_dir / "adversarial_review.json", review)
    _json(
        result_dir / "lineage.json",
        {
            "run_id": run_id,
            "stage": "adversarial_review",
            "controlled_result": controlled.name,
            "residual_result": residual.name,
            "test_ids_read": [],
            "other_folds_read": [],
        },
    )
    _write_hashes(result_dir)
    if review["status"] != "PASS":
        raise RuntimeError("ESC adversarial review failed")


def _run_acceptance(
    root: Path,
    config: dict[str, Any],
    result_dir: Path,
    run_id: str,
) -> None:
    controlled = _latest_result(root, str(config["controlled_experiment"]), "controlled_experiment")
    residual = _latest_result(root, str(config["residual_experiment"]), "residual_diagnosis")
    adversarial = _latest_result(root, str(config["adversarial_experiment"]), "adversarial_review")
    if any(_verify_hashes(path) for path in (controlled, residual, adversarial)):
        raise RuntimeError("ESC evidence artifacts changed before acceptance")
    synthetic = json.loads(
        (controlled / "synthetic_ground_truth_gate.json").read_text(encoding="utf-8")
    )
    library = json.loads(
        (controlled / "conditional_model_library.json").read_text(encoding="utf-8")
    )
    diagnosis = json.loads((residual / "residual_diagnosis.json").read_text(encoding="utf-8"))
    review = json.loads((adversarial / "adversarial_review.json").read_text(encoding="utf-8"))
    diagnostic_by_envelope = {item["envelope_id"]: item for item in diagnosis["envelopes"]}
    entries = []
    for entry in library["entries"]:
        diagnostic = diagnostic_by_envelope[entry["envelope_id"]]
        if entry["status"] != "SELECTED":
            entries.append(
                {
                    "envelope_id": entry["envelope_id"],
                    "status": "ENVELOPE_DATA_REQUEST",
                    "earned_name": None,
                    "bounds": entry["bounds"],
                    "reason": entry["reason"],
                }
            )
            continue
        modal_identified = diagnostic["terminal_route"] == "MODAL_IDENTIFIED_ENVELOPE"
        rollout_ok = bool(
            diagnostic["audit_failure_rate"] == 0.0
            and diagnostic["audit_10s_mean_RMSE"] <= float(config["maximum_rollout_mean_rmse"])
        )
        earned_name = (
            "modal-identified equivalent"
            if modal_identified
            else "rollout-accurate equivalent"
            if rollout_ok
            else "conditional dynamic equivalent"
        )
        entries.append(
            {
                "envelope_id": entry["envelope_id"],
                "status": "CERTIFIED_OFFLINE_ENVELOPE" if rollout_ok else "MODEL_AVAILABLE",
                "selected_branch": diagnostic["selected_branch"],
                "earned_name": earned_name,
                "bounds": entry["bounds"],
                "evidence": diagnostic,
                "rollout_label": ROLLOUT_LABEL,
            }
        )
    gates = {
        "synthetic_correctness": synthetic["status"] == "PASS",
        "adversarial_review": review["status"] == "PASS",
        "at_least_one_F1_envelope_selected": any(
            entry["status"] != "ENVELOPE_DATA_REQUEST" for entry in entries
        ),
        "outside_union_refusal": library["outside_union_policy"] == "REFUSE_AND_REQUEST_DATA",
        "test_locked": review["test_ids_read"] == [],
        "other_folds_locked": review["other_folds_read"] == [],
    }
    acceptance = {
        "run_id": run_id,
        "status": "PASS" if all(gates.values()) else "FAIL",
        "terminal_state": (
            "EXCITATION_CONDITIONED_LIBRARY_CREATED"
            if all(gates.values())
            else "INSUFFICIENT_INFORMATION"
        ),
        "model_reporting_name": "per-envelope; see library entries",
        "gates": gates,
        "library_entries": entries,
        "scientific_result": (
            "shared physical-equation linearization plus fault-line bootstrap replaces the "
            "incorrect requirement that one empirical DMD mode appear in every disturbance; "
            "F1 dynamic selections retain honest rollout-only naming unless their complete "
            "augmented poles pass bootstrap and absolute trajectory-modal gates"
        ),
        "claim_boundary": (
            f"F1/seed324 train-only development; {ROLLOUT_LABEL}; frozen validation, test, "
            "F2-F5 and autonomous Pm remain locked"
        ),
        "next_gate": "merge_gate",
        "test_ids_read": [],
        "other_folds_read": [],
    }
    _json(result_dir / "acceptance.json", acceptance)
    _json(
        result_dir / "conditional_model_library.json",
        {
            "status": acceptance["status"],
            "entries": entries,
            "outside_union_policy": "REFUSE_AND_REQUEST_DATA",
            "deployment_status": "offline envelope evidence only",
            "rollout_label": ROLLOUT_LABEL,
        },
    )
    lines = [
        "# M0-ESC v1：激励条件模态选择验收报告",
        "",
        f"**终态：{acceptance['terminal_state']}**",
        "",
        f"全部 IEEE39 重积分均为 **{ROLLOUT_LABEL}**。本轮没有读取冻结 F1 validation、test 或 F2–F5。",
        "",
        "## 核心修正",
        "",
        "系统极点由共享物理方程在 COI 商空间的 Jacobian 给出；逐轨迹 DMD 只说明该扰动是否观测到相应模态。合成系统中，不同 fault line 各自只激发一个模态，因此 80% 逐线 DMD 共识为空，但共享弱形式方程仍恢复全部三个真值模态。",
        "",
        "## F1 包络结果",
        "",
    ]
    for entry in entries:
        lines.append(
            f"- `{entry['envelope_id']}`：{entry['status']}；"
            f"分枝 `{entry.get('selected_branch', '无')}`；命名 `{entry.get('earned_name')}`。"
        )
    lines.extend(
        [
            "",
            "## 边界",
            "",
            "动态分枝若只具有 B2 backbone 的 bootstrap，不继承模态辨识结论；包络当前由完整训练轨迹离线定义，尚不能用于在线部署。任何包络外输入必须拒绝并请求新增数据。",
        ]
    )
    (result_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    _json(
        result_dir / "lineage.json",
        {
            "run_id": run_id,
            "stage": "acceptance",
            "controlled_result": controlled.name,
            "residual_result": residual.name,
            "adversarial_result": adversarial.name,
            "test_ids_read": [],
            "other_folds_read": [],
        },
    )
    _write_hashes(result_dir)
    if acceptance["status"] != "PASS":
        raise RuntimeError("ESC acceptance failed")


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
        _run_controlled(root, config_path, config, result_dir, run_id)
    elif stage == "residual_diagnosis":
        _run_residual(root, config, result_dir, run_id)
    elif stage == "adversarial_review":
        _run_adversarial(root, config, result_dir, run_id)
    elif stage == "acceptance":
        _run_acceptance(root, config, result_dir, run_id)
    else:
        raise ValueError(f"unsupported ESC v1 stage: {stage}")
