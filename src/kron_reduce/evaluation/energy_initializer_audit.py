"""One registered comparison of saved synthetic inputs and three energy radii.

Importing this module neither reads source arrays nor evaluates a candidate.
The dedicated workflow owns authorization and the shared one-use reservation.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from functools import partial
from importlib.metadata import version

import numpy as np

from kron_reduce.data.energy_saved_inputs import SavedEnergyInputs
from kron_reduce.evaluation.constant_c_observability import (
    Operations,
    check_path,
    errors,
    exact_path,
    model_parameters,
    rk_path,
    save_record,
    write_json,
)
from kron_reduce.evaluation.energy_constrained_initializer import (
    EnergyBallNumericalError,
    solve_energy_ball,
)
from kron_reduce.models.nonlinear_latent import NonlinearPassiveClosure
from project_harness.evidence import sha256_file
from project_harness.git_state import git_head

CASES = (
    "L100_clean", "L100_alt", "L100_weakdir", "S100_clean", "S100_alt",
    "W100_clean", "W100_weakdir", "K100_mismatch", "U_hidden_null",
)
RADII = ("zero", "oracle_radius", "fixed_radius")
TOLERANCES = {
    "coi": 1e-10, "energy_identity": 1e-10, "refinement": 1e-8,
    "hidden_clean": 1e-3, "hidden_noisy": 1e-2,
    "future_clean": 1e-3, "future_noisy": 1e-2, "noninferiority": 1e-8,
}
PARAMETERS = (
    "mass", "edge_stiffness", "damping", "coupling", "latent_skew",
    "latent_dissipation", "backbone",
)


def _norm(value):
    return math.hypot(*np.asarray(value).ravel())


def _ratio(numerator, denominator):
    return None if denominator == 0 else float(numerator / denominator)


def _fine(paths):
    return paths[-1] if isinstance(paths, list) else paths


def _context(case_id, record, fixtures, sine_record):
    sine = case_id.startswith("S")
    if case_id == "U_hidden_null":
        history, future = record["first"][10:21], record["first"][20:]
        observed = history[:, :20]
        baseline = None
        oracle = {"history": history, "reset": future[0], "future": future}
        family, power = "null", record["power"]
        transition = record["bundle"]["transition"]
    else:
        observed, history, future = (
            record["observed"], record["truth_history"], record["truth_future"])
        baseline, oracle = record["paths"]["integral_ls"], record["paths"]["true_z_reference"]
        family = ("sine" if sine else "weak" if case_id.startswith("W") else
                  "mismatch" if case_id.startswith("K") else "main")
        power = sine_record["power"] if sine else fixtures["powers"][family]
        transition = None if sine else fixtures["bundles"][family]["transition"]
    if observed.shape != (11, 20) or history.shape != (11, 22) or future.shape != (801, 22):
        raise ValueError("saved case does not match the fixed 100 ms / 8 s shapes")
    integral = record["integral"]
    if integral["svd"]["rank"] != (1 if family == "null" else 2):
        raise ValueError("saved source rank differs from its registered case")
    return {
        "case_id": case_id, "family": family, "sine": sine,
        "observed": observed, "truth_history": history, "truth_future": future,
        "baseline_paths": baseline, "oracle_paths": oracle,
        "baseline_refinement": record["metrics"]["integral_ls"]["refinement_max"] if sine else None,
        "matrix": integral["matrix"], "right": integral["right4"],
        "svd": integral["svd"], "unconstrained": integral["z4"],
        "radius_oracle": _norm(history[0, 20:]), "power": power,
        "transition": transition,
    }


def solve_candidate(context, radius_id, ops):
    """One logged call; retain solver diagnostics before journal failure handling."""
    radius = {"zero": 0., "oracle_radius": context["radius_oracle"],
              "fixed_radius": 1.}[radius_id]

    def solve():
        try:
            result = solve_energy_ball(
                context["matrix"], context["right"], context["svd"], radius,
                unconstrained=context["unconstrained"])
        except EnergyBallNumericalError as error:
            ops.partial["solver_failure"] = error.diagnostics
            raise
        result["radius_id"] = radius_id
        return result

    return ops.call("probe", solve, tag=f"energy_ball_solver:{radius_id}")


def _run_pair(context, model, z, radius_id, ops, tolerances, step=None):
    initial = np.r_[context["observed"][0], z]
    power = context["power"]
    if step is None:
        history = exact_path(context["transition"], initial, 10, ops)
    else:
        history = rk_path(model, power, initial, 10, step, ops,
                          role="history", method=radius_id)
    history_checks = check_path(model, power, history, ops, tolerances)
    reset = history[-1].copy()
    reset[:20] = context["observed"][-1]
    if step is None:
        future = exact_path(context["transition"], reset, 800, ops)
    else:
        future = rk_path(model, power, reset, 800, step, ops,
                         role="future", method=radius_id)
    future_checks = check_path(model, power, future, ops, tolerances)
    reset_energy = ops.call("probe", lambda: [
        model.energy(history[-1], np.zeros(10)), model.energy(reset, np.zeros(10)),
    ], tag=f"reset_energy_pair:{radius_id}:{step}")
    return {
        "step": step, "history": history, "reset": reset, "future": future,
        "history_checks": history_checks, "future_checks": future_checks,
        "reset_energy": reset_energy,
        "reset_energy_change": reset_energy[1] - reset_energy[0],
    }


def build_candidate_paths(context, model, solution, radius_id, ops, tolerances):
    """Inactive paths are references; unresolved states never enter propagation."""
    bundle = {"paths": None, "reused_unconstrained": False, "refinement_max": None,
              "new_linear_candidates": 0, "new_sine_candidates": 0}
    if solution["status"] == "unresolved":
        return bundle
    if solution["status"] == "inactive":
        if context["baseline_paths"] is None:
            raise ValueError("inactive candidate has no registered baseline paths")
        if not np.array_equal(solution["z"], context["unconstrained"]):
            raise ValueError("inactive candidate changed its saved unconstrained state")
        bundle.update(paths=context["baseline_paths"], reused_unconstrained=True,
                      refinement_max=context["baseline_refinement"])
        return bundle
    if solution["status"] not in {"active", "zero_baseline"}:
        raise ValueError("unregistered solver branch")
    if not context["sine"]:
        bundle.update(paths=_run_pair(context, model, solution["z"], radius_id, ops,
                                      tolerances), new_linear_candidates=1)
        return bundle
    paths = [_run_pair(context, model, solution["z"], radius_id, ops, tolerances, step)
             for step in (.00125, .000625)]
    refinement = ops.call("probe", lambda: max(
        float(np.max(np.abs(paths[0][key] - paths[1][key])))
        for key in ("history", "reset", "future")), tag=f"sine_refinement:{radius_id}")
    if refinement > tolerances["refinement"]:
        raise FloatingPointError("candidate sine refinement gate failed")
    bundle.update(paths=paths, refinement_max=refinement, new_sine_candidates=1)
    return bundle


def _metrics(context, solution, bundle, truth_errors, oracle_errors, baseline_errors, tolerances):
    paths = bundle["paths"]
    fine = _fine(paths)
    for path in paths if isinstance(paths, list) else [paths]:
        if not (np.array_equal(path["history"][0, :20], context["observed"][0])
                and np.array_equal(path["history"][0, 20:], solution["z"])
                and np.array_equal(path["reset"][:20], context["observed"][-1])
                and np.array_equal(path["reset"][20:], path["history"][-1, 20:])
                and np.array_equal(path["future"][0], path["reset"])):
            raise FloatingPointError("candidate causal initialization/reset identity failed")
    if context["sine"] and (bundle["refinement_max"] is None
                             or bundle["refinement_max"] > tolerances["refinement"]):
        raise FloatingPointError("missing or invalid sine refinement evidence")
    z_h, z_t = fine["history"][0, 20:], fine["reset"][20:]
    true_h, true_t = context["truth_history"][0, 20:], context["truth_future"][0, 20:]
    h_h, h_t = .5*float(z_h @ z_h), .5*float(z_t @ z_t)
    true_h_energy, true_t_energy = .5*float(true_h @ true_h), .5*float(true_t @ true_t)
    baseline = _fine(context["baseline_paths"]) if context["baseline_paths"] is not None else None
    baseline_h = .5*float(baseline["history"][0, 20:] @ baseline["history"][0, 20:]) if baseline else None
    baseline_t = .5*float(baseline["reset"][20:] @ baseline["reset"][20:]) if baseline else None
    noisy = context["case_id"] in {"L100_alt", "L100_weakdir", "W100_weakdir", "S100_alt"}
    state_tol = tolerances["hidden_noisy" if noisy else "hidden_clean"]
    future_tol = tolerances["future_noisy" if noisy else "future_clean"]
    initial_error, origin_error = _norm(z_h-true_h), _norm(z_t-true_t)
    deltas = (None if baseline_errors is None else {
        horizon: truth_errors[horizon]["visible_max_abs"]-baseline_errors[horizon]["visible_max_abs"]
        for horizon in ("3", "8")})
    return {
        "hidden_initial_error": initial_error, "hidden_origin_error": origin_error,
        "hidden_initial_energy": h_h, "hidden_origin_energy": h_t,
        "hidden_energy_growth": h_t-h_h, "hidden_energy_origin_over_initial": _ratio(h_t, h_h),
        "hidden_initial_energy_error": h_h-true_h_energy,
        "hidden_origin_energy_error": h_t-true_t_energy,
        "initial_energy_over_truth": _ratio(h_h, true_h_energy),
        "origin_energy_over_truth": _ratio(h_t, true_t_energy),
        "initial_energy_over_unconstrained": None if baseline_h is None else _ratio(h_h, baseline_h),
        "origin_energy_over_unconstrained": None if baseline_t is None else _ratio(h_t, baseline_t),
        "initial_energy_delta_unconstrained": None if baseline_h is None else h_h-baseline_h,
        "origin_energy_delta_unconstrained": None if baseline_t is None else h_t-baseline_t,
        "history_observed_visible_mse_11points": float(
            np.mean((fine["history"][:, :20]-context["observed"])**2)),
        "weighted_residual": context["matrix"] @ z_h-context["right"],
        "truth_feasible": _norm(true_h) <= solution["radius"]*(1+1e-12),
        "state_recovered": initial_error <= state_tol,
        "future_3s": truth_errors["3"]["visible_max_abs"] <= future_tol,
        "future_8s": truth_errors["8"]["visible_max_abs"] <= future_tol,
        "state_tolerance": state_tol, "future_tolerance": future_tol,
        "future_errors": truth_errors, "future_relative_to_true_z_reference": oracle_errors,
        "future_max_error_delta_unconstrained": deltas,
        "future_numerically_noninferior": (None if deltas is None else {
            horizon: value <= tolerances["noninferiority"] for horizon, value in deltas.items()}),
        "normalization": "fixed_unit_scales_no_data_normalizer",
        "reference_scales": {"q_rad": 1., "v_rad_per_s": 1.},
        "raw_rms_units": {"q_rms": "rad", "v_rms": "rad/s"},
        "refinement_max": bundle["refinement_max"],
    }


def execute(*, plan, root, result_dir):
    """Called only after the new workflow spends its own registered authority."""
    if tuple(plan["cases"]) != CASES or plan["tolerances"] != TOLERANCES:
        raise ValueError("execution plan differs from the frozen cases or tolerances")
    manifest_bytes = (root / plan["saved_sources_path"]).read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != plan["saved_sources_sha256"]:
        raise ValueError("saved source manifest differs from its bound digest")
    manifest = json.loads(manifest_bytes)
    ops = Operations(result_dir / "journal", plan)
    results, read_receipt = {}, None
    new_linear = new_sine = solver_evaluations = solver_updates = 0
    write_json(result_dir / "execution_freeze.json", {
        "plan": plan, "source_commit": git_head(root), "python": sys.version,
        "executable": sys.executable,
        "packages": {name: version(name) for name in ("numpy", "scipy", "torch", "filelock", "h5py")},
        "real_data_reads": 0, "fit_attempts": 0, "optimizer_updates": 0,
    })
    try:
        with ops.scope("source-inputs"):
            reader = SavedEnergyInputs(root, manifest, ops)
            reader.verify()
            records = {name: reader.load(name) for name in ("linear-fixtures", "sine-truth", *CASES)}
            read_receipt = reader.report()
            write_json(result_dir / "source_read_receipt.json", read_receipt)
        fixtures, sine_record = records["linear-fixtures"], records["sine-truth"]
        with ops.scope("setup"):
            parameters = {family: fixtures["parameters"][family] for family in ("main", "weak", "mismatch")}
            parameters.update(sine=sine_record["parameters"], null=records["U_hidden_null"]["parameters"])
            models = {family: ops.call("probe", lambda p=p: NonlinearPassiveClosure(
                **{key: p[key] for key in PARAMETERS}), tag=f"model_construction:{family}")
                for family, p in parameters.items()}
            save_record(result_dir, "models", {family: model_parameters(model) for family, model in models.items()})
        for case_id in CASES:
            with ops.scope(case_id):
                context = ops.call("probe", lambda case_id=case_id: _context(
                    case_id, records[case_id], fixtures, sine_record), tag="case_input_summary")
                baseline_errors = (None if context["baseline_paths"] is None else errors(
                    _fine(context["baseline_paths"])["future"], context["truth_future"], ops))
                candidates = {}
                for radius_id in RADII:
                    solution = solve_candidate(context, radius_id, ops)
                    solver_evaluations += solution["secular_evaluations"]
                    solver_updates += solution["iterations"]
                    if solver_evaluations > 1280 or solver_updates > 1024:
                        raise RuntimeError("global scalar solver budget exceeded")
                    bundle = build_candidate_paths(context, models[context["family"]], solution,
                                                   radius_id, ops, plan["tolerances"])
                    new_linear += bundle["new_linear_candidates"]
                    new_sine += bundle["new_sine_candidates"]
                    metrics = None
                    if bundle["paths"] is not None:
                        future = _fine(bundle["paths"])["future"]
                        truth_errors = errors(future, context["truth_future"], ops)
                        oracle_errors = errors(future, _fine(context["oracle_paths"])["future"], ops)
                        metrics = ops.call("probe", partial(_metrics,
                            context, solution, bundle, truth_errors, oracle_errors,
                            baseline_errors, plan["tolerances"]), tag=f"candidate_metrics:{radius_id}")
                    candidates[radius_id] = {"solver": solution, **bundle, "metrics": metrics}
                record = {"inputs": context, "baseline_errors": baseline_errors, "candidates": candidates,
                          "source_record": case_id, "source_manifest_sha256": plan["saved_sources_sha256"]}
                save_record(result_dir, case_id, record)
                results[case_id] = {key: {
                    "status": value["solver"]["status"], "reused_unconstrained": value["reused_unconstrained"],
                    "metrics": None if value["metrics"] is None else {
                        k: v for k, v in value["metrics"].items() if k != "weighted_residual"},
                } for key, value in candidates.items()}
        required = {
            "probe": 338+new_linear+3*new_sine, "rollouts": 2*new_linear+4*new_sine,
            "physical_rhs": 6*new_linear+12*new_sine, "integration_rhs": 77760*new_sine,
        }
        if not 7 <= new_linear <= 19 or not 2 <= new_sine <= 6:
            raise RuntimeError("conditional candidate branches are incomplete")
        if any(ops.counts[key] != value or ops.returns[key] != value for key, value in required.items()):
            raise RuntimeError("actual operations differ from the exact conditional call tree")
        if any(count for key, count in ops.counts.items() if key not in required):
            raise RuntimeError("unregistered numerical operation category")
        if (len(ops.clock_checks) != 4*new_sine
                or 4*sum(row["rk4_updates"] for row in ops.clock_checks) != required["integration_rhs"]):
            raise RuntimeError("sine path clock population differs from RHS count")
        if len(ops.journal.closed_scopes) != 11:
            raise RuntimeError("the fixed scope roster did not close completely")
        summary = {
            "process_status": "PASS", "case_count": 9, "candidate_count": 27,
            "scope_count": 11, "cases": results,
            "new_linear_candidates": new_linear, "new_sine_candidates": new_sine,
            "operator_attempted": dict(ops.counts), "operator_returned": dict(ops.returns),
            "conditional_required_counts": required, "rk4_clock_checks": ops.clock_checks,
            "solver_evaluations": solver_evaluations, "solver_updates": solver_updates,
            "nested_psd_eigvalsh_calls": 3*(5+4*new_sine),
            "pure_reset_energy_calls": 2*new_linear+4*new_sine,
            "saved_transition_actions": 810*new_linear,
            "real_data_reads": 0, "fit_attempts": 0, "optimizer_updates": 0,
            "promotion_decision": "KEEP_BASELINE",
            "claim_boundary": "fixed saved synthetic cases only; no predetermined scientific acceptance",
        }
        write_json(result_dir / "summary.json", summary)
        write_json(result_dir / "stage_evidence.json", {
            "status": "PASS", "process_status": "PASS", "case_count": 9, "candidate_count": 27,
            "source_commit": git_head(root), "data_policy": "saved_synthetic_only",
            "promotion_decision": "KEEP_BASELINE", "claim_boundary": summary["claim_boundary"],
        })
    except BaseException as error:
        write_json(result_dir / "failure.json", {
            "process_status": "FAIL", "error_type": type(error).__name__, "message": str(error),
            "completed_cases": list(results), "operator_attempted": dict(ops.counts),
            "operator_returned": dict(ops.returns), "retry_allowed": False,
        })
        raise
    finally:
        ops.journal.close()
    write_json(result_dir / "artifact_hashes.json", {
        path.relative_to(result_dir).as_posix(): sha256_file(path)
        for path in sorted(result_dir.rglob("*")) if path.is_file()
    })
    return summary
