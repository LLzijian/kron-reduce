"""Fixed-radius confirmation on newly specified linear synthetic observations."""
from __future__ import annotations

import sys
from functools import partial
from importlib.metadata import version

import numpy as np

from kron_reduce.evaluation.constant_c_observability import (
    Operations,
    check_path,
    decompose,
    errors,
    exact_path,
    integral_system,
    linear_bundle,
    make_model,
    model_parameters,
    sampled_operator,
    save_record,
    write_json,
)
from kron_reduce.evaluation.energy_initializer_audit import (
    _run_pair,
    build_candidate_paths,
    solve_candidate,
)
from kron_reduce.physics.coordinates import coi_project
from project_harness.evidence import sha256_file
from project_harness.git_state import git_head

CASES = ("CL_clean", "CL_periodic", "CL_ramp", "CW_clean", "CW_periodic", "CW_ramp")
METHODS = ("integral_ls", "zero", "true_z_reference", "fixed_radius")
TOLERANCES = {
    "coi": 1e-10, "energy_identity": 1e-10, "exponential": 1e-10,
    "hidden_clean": 1e-3, "hidden_noisy": 1e-2,
    "future_clean": 1e-3, "future_noisy": 1e-2, "noninferiority": 1e-8,
}


def observation(truth, mass, mode):
    """Frozen time tables and COI directions, independent of any singular vector."""
    if truth.shape != (11, 22):
        raise ValueError("confirmation history must contain eleven full states")
    q_direction = coi_project(np.eye(10)[0] + np.eye(10)[3], mass)
    v_direction = coi_project(np.eye(10)[1] - np.eye(10)[4], mass)
    q_direction /= np.linalg.norm(q_direction)
    v_direction /= np.linalg.norm(v_direction)
    noise = np.zeros((11, 20))
    if mode == "periodic":
        a = np.array([1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1])
        b = np.array([0, 1, 1, 0, -1, -1, 0, 1, 1, 0, -1])
        noise[:, :10] = 2e-5 * a[:, None] * q_direction
        noise[:, 10:] = 7e-5 * b[:, None] * v_direction
    elif mode == "ramp":
        noise[:, :10] = 2e-5 * v_direction
        noise[:, 10:] = 7e-5 * ((np.arange(11)-5)/5)[:, None] * q_direction
    elif mode != "clean":
        raise ValueError("unregistered observation mode")
    return {"observed": truth[:, :20] + noise, "noise": noise,
            "q_direction": q_direction, "v_direction": v_direction}


def endpoint_check(bundle, initial, truth, tolerance):
    endpoint = (bundle["endpoints"][800]
                @ (bundle["endpoints"][10] @ np.r_[initial, 1.]))[:22]
    error = float(np.max(np.abs(endpoint-truth[-1])))
    if error > tolerance * max(1., float(np.linalg.norm(endpoint))):
        raise FloatingPointError("confirmation truth endpoint differs from direct transition")
    return {"direct_endpoint": endpoint, "maximum_error": error}


def metrics(context, z, path, truth_errors, oracle_errors, baseline_errors, tolerances):
    observed, history, reset, future = (
        context["observed"], path["history"], path["reset"], path["future"])
    if not (np.array_equal(history[0], np.r_[observed[0], z])
            and np.array_equal(reset[:20], observed[-1])
            and np.array_equal(reset[20:], history[-1, 20:])
            and np.array_equal(future[0], reset)):
        raise FloatingPointError("confirmation causal initialization/reset identity failed")
    true_h, true_t = context["truth_history"][0, 20:], context["truth_future"][0, 20:]
    z_t = reset[20:]
    h_h, h_t = .5*float(z @ z), .5*float(z_t @ z_t)
    e_h, e_t = float(np.linalg.norm(z-true_h)), float(np.linalg.norm(z_t-true_t))
    suffix = "clean" if context["noise_mode"] == "clean" else "noisy"
    state_tol, future_tol = tolerances[f"hidden_{suffix}"], tolerances[f"future_{suffix}"]
    deltas = {h: truth_errors[h]["visible_max_abs"]-baseline_errors[h]["visible_max_abs"]
              for h in ("3", "8")}
    return {
        "hidden_initial_error": e_h, "hidden_origin_error": e_t,
        "hidden_initial_energy": h_h, "hidden_origin_energy": h_t,
        "hidden_energy_growth": h_t-h_h,
        "hidden_energy_origin_over_initial": h_t/h_h if h_h else None,
        "hidden_initial_energy_error": h_h-.5*float(true_h @ true_h),
        "hidden_origin_energy_error": h_t-.5*float(true_t @ true_t),
        "history_observed_visible_mse_11points": float(np.mean((history[:, :20]-observed)**2)),
        "weighted_residual": context["matrix"] @ z-context["right"],
        "state_recovered": e_h <= state_tol, "origin_state_recovered": e_t <= state_tol,
        "state_tolerance": state_tol, "future_tolerance": future_tol,
        "future_3s": truth_errors["3"]["visible_max_abs"] <= future_tol,
        "future_8s": truth_errors["8"]["visible_max_abs"] <= future_tol,
        "future_errors": truth_errors, "future_relative_to_true_z_reference": oracle_errors,
        "future_max_error_delta_unconstrained": deltas,
        "future_numerically_noninferior": {h: d <= tolerances["noninferiority"] for h, d in deltas.items()},
        "reset_energy_change": path["reset_energy_change"],
        "reference_scales": {"q_rad": 1., "v_rad_per_s": 1.},
        "normalization": "fixed_unit_scales_no_data_normalizer",
    }


def execute(*, plan, root, result_dir):
    """Execute once, after the dedicated workflow consumes its own authority."""
    if (tuple(c["id"] for c in plan["cases"]) != CASES
            or plan["tolerances"] != TOLERANCES or plan["radius"] != 1.):
        raise ValueError("confirmation plan differs from frozen cases, radius or tolerances")
    ops = Operations(result_dir / "journal", plan)
    groups, results = {}, {}
    active_count = evaluations = updates = 0
    write_json(result_dir / "execution_freeze.json", {
        "plan": plan, "source_commit": git_head(root), "python": sys.version,
        "executable": sys.executable,
        "packages": {n: version(n) for n in ("numpy", "scipy", "torch", "filelock", "h5py")},
        "source_array_reads": 0, "real_data_reads": 0, "fit_attempts": 0, "optimizer_updates": 0,
    })
    try:
        for name in ("main", "weak"):
            with ops.scope(f"setup-{name}"):
                model, power, initial = make_model(plan["base"], ops, weak=name == "weak")
                bundle = linear_bundle(model, power, ops, TOLERANCES["exponential"])
                truth = exact_path(bundle["transition"], initial, 810, ops)
                physical = check_path(model, power, truth, ops, TOLERANCES)
                endpoint = ops.call("probe", partial(endpoint_check,
                    bundle, initial, truth, TOLERANCES["exponential"]), tag="truth_endpoint")
                operator = sampled_operator(bundle["transition"], 10, ops)
                spectrum = decompose(operator, ops, natural=1.)
                groups[name] = {"model": model, "power": power, "bundle": bundle, "truth": truth}
                save_record(result_dir, f"setup-{name}", {
                    "parameters": model_parameters(model), "power": power, "initial": initial,
                    "bundle": bundle, "truth": truth, "physical_checks": physical,
                    "endpoint_check": endpoint, "sampled_operator": operator,
                    "sampled_svd": spectrum,
                })
        for spec in plan["cases"]:
            case_id = spec["id"]
            with ops.scope(case_id):
                group = groups[spec["group"]]
                model, power, truth = group["model"], group["power"], group["truth"]
                obs = ops.call("probe", partial(observation,
                    truth[:11], model.mass, spec["noise"]), tag="fixed_observation_noise")
                observed = obs["observed"]
                integral = integral_system(model, observed[:, :10], observed[:, 10:], power, ops)
                if (integral["svd"]["rank"] != 2 or integral["pc_svd"]["rank"] != 2
                        or integral["z4"] is None):
                    ops.partial["rank_unresolved"] = {"status": "unresolved", "integral": integral,
                                                      "case_id": case_id, "candidate_propagated": False}
                    raise FloatingPointError("confirmation integral rank gate failed")
                context = {"case_id": case_id, "noise_mode": spec["noise"], "sine": False,
                           "observed": observed, "truth_history": truth[:11], "truth_future": truth[10:],
                           "matrix": integral["matrix"], "right": integral["right4"],
                           "svd": integral["svd"], "unconstrained": integral["z4"],
                           "radius_oracle": float(np.linalg.norm(truth[0, 20:])),
                           "power": power, "transition": group["bundle"]["transition"],
                           "baseline_refinement": None}
                starts = {"integral_ls": integral["z4"], "zero": np.zeros(2),
                          "true_z_reference": truth[0, 20:].copy()}
                paths = {method: _run_pair(context, model, z, method, ops, TOLERANCES)
                         for method, z in starts.items()}
                context["baseline_paths"] = paths["integral_ls"]
                solution = solve_candidate(context, "fixed_radius", ops)
                evaluations += solution["secular_evaluations"]
                updates += solution["iterations"]
                if evaluations > 480 or updates > 384:
                    raise RuntimeError("confirmation scalar budget exceeded")
                fixed = build_candidate_paths(context, model, solution, "fixed_radius", ops, TOLERANCES)
                if fixed["paths"] is None:
                    raise FloatingPointError("full-rank confirmation candidate unresolved")
                active_count += int(solution["status"] == "active")
                starts["fixed_radius"], paths["fixed_radius"] = solution["z"], fixed["paths"]
                truth_errors = {method: errors(path["future"], context["truth_future"], ops)
                                for method, path in paths.items()}
                method_metrics = {}
                for method in METHODS:
                    oracle_errors = errors(paths[method]["future"], paths["true_z_reference"]["future"], ops)
                    method_metrics[method] = ops.call("probe", partial(metrics,
                        context, starts[method], paths[method], truth_errors[method], oracle_errors,
                        truth_errors["integral_ls"], TOLERANCES), tag=f"method_metrics:{method}")
                save_record(result_dir, case_id, {
                    "inputs": context, "observation": obs, "integral": integral,
                    "fixed_solver": solution, "fixed_reused_unconstrained": fixed["reused_unconstrained"],
                    "starts": starts, "paths": paths, "metrics": method_metrics,
                })
                results[case_id] = {"fixed_status": solution["status"],
                    "fixed_reused_unconstrained": fixed["reused_unconstrained"],
                    "metrics": {m: {k: v for k, v in row.items() if k != "weighted_residual"}
                                for m, row in method_metrics.items()}}
        required = {"probe": 110+active_count, "rollouts": 38+2*active_count,
                    "physical_rhs": 120+6*active_count, "expm": 20, "svd": 14, "ls": 12, "force": 360}
        if dict(ops.counts) != required or dict(ops.returns) != required:
            raise RuntimeError("confirmation operations differ from the exact conditional call tree")
        if len(ops.journal.closed_scopes) != 8 or ops.clock_checks:
            raise RuntimeError("confirmation scope or integration-clock roster mismatch")
        fixed_rows = {case: row["metrics"]["fixed_radius"] for case, row in results.items()}
        absolute = all(row["future_3s"] and row["future_8s"] for row in fixed_rows.values())
        noninferior = all(all(row["future_numerically_noninferior"].values()) for row in fixed_rows.values())
        weak_active = any(results[case]["fixed_status"] == "active" for case in ("CW_periodic", "CW_ramp"))
        summary = {"process_status": "PASS", "case_count": 6, "method_count": 24,
            "scope_count": 8, "cases": results, "active_fixed_count": active_count,
            "operator_attempted": dict(ops.counts), "operator_returned": dict(ops.returns),
            "solver_evaluations": evaluations, "solver_updates": updates,
            "nested_psd_eigvalsh_calls": 6, "pure_reset_energy_calls": 36+2*active_count,
            "path_transition_actions": 16200+810*active_count, "truth_endpoint_actions": 4,
            "confirmation_gates": {"fixed_all_absolute": absolute, "fixed_all_noninferior": noninferior,
                "weak_noisy_constraint_exercised": weak_active,
                "confirmation_criteria_met": absolute and noninferior and weak_active},
            "source_array_reads": 0, "real_data_reads": 0, "fit_attempts": 0, "optimizer_updates": 0,
            "promotion_decision": "KEEP_BASELINE",
            "claim_boundary": "New parameters and deterministic noises in an exposed matched linear family; no external confirmation or real-data claim."}
        write_json(result_dir / "summary.json", summary)
        write_json(result_dir / "stage_evidence.json", {"status": "PASS", "process_status": "PASS",
            "source_commit": git_head(root), "data_policy": "analytic_synthetic_only",
            "case_count": 6, "method_count": 24, "promotion_decision": "KEEP_BASELINE",
            "claim_boundary": summary["claim_boundary"]})
    except BaseException as error:
        write_json(result_dir / "failure.json", {"process_status": "FAIL", "error_type": type(error).__name__,
            "message": str(error), "completed_cases": list(results),
            "operator_attempted": dict(ops.counts), "operator_returned": dict(ops.returns), "retry_allowed": False})
        raise
    finally:
        ops.journal.close()
    write_json(result_dir / "artifact_hashes.json", {
        p.relative_to(result_dir).as_posix(): sha256_file(p)
        for p in sorted(result_dir.rglob("*")) if p.is_file()})
    return summary
