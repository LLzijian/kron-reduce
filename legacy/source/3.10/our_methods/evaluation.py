"""Evaluation helpers for M0, M2A, and M2A-LYAP."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from support.common import PM_MODE, binary_label, binary_metrics, trajectory_metrics
from support.data_protocol import variant_sequences

from evaluate_8cluster_dataonly_engineering_ieee39 import load_model
from diag_evaluate_external_cct_v3 import classify_external_stability

from our_methods.m0 import fit_m0, rollout_m0
from our_methods.m2a import prepare_m2a_rollout, rollout_m2a_prepared, train_or_load_m2a
from our_methods.m2a_lyap import prepare_m2a_lyap_rollout, rollout_m2a_lyap_prepared, train_or_load_m2a_lyap


def _cleaning_fields(row: dict[str, Any], seq: dict[str, Any]) -> dict[str, Any]:
    diag = row.get("_clean_diag")
    if not isinstance(diag, dict):
        diag = seq.get("_clean_diag", {})
    if not isinstance(diag, dict):
        diag = {}
    return {
        "cleaned_has_flatline": int(bool(diag.get("has_flatline", False))),
        "cleaned_valid_steps": diag.get("valid_steps", ""),
        "cleaned_total_steps": diag.get("total_steps", ""),
        "cleaned_divergence_time": diag.get("divergence_time", ""),
        "cleaned_flatline_fraction": diag.get("flatline_fraction", ""),
    }


def _prepare_rollout(method: str, params: np.ndarray | None):
    if method == "M0":
        return None
    if method == "M2A":
        assert params is not None
        return prepare_m2a_rollout(params)
    if method == "M2A-LYAP":
        assert params is not None
        return prepare_m2a_lyap_rollout(params)
    raise ValueError(method)


def _rollout_one(method: str, bm, prepared, seq: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if method == "M0":
        return rollout_m0(bm, seq)
    if method == "M2A":
        return rollout_m2a_prepared(prepared, bm, seq)
    if method == "M2A-LYAP":
        return rollout_m2a_lyap_prepared(prepared, bm, seq)
    raise ValueError(method)


def evaluate_rollout_method(
    method: str,
    protocol: str,
    split_id: str,
    seed: int,
    bm,
    params: np.ndarray | None,
    eval_rows: list[dict[str, Any]],
    eval_seqs: list[dict[str, Any]],
    eval_kind: str,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    per_case: list[dict[str, Any]] = []
    prepared = _prepare_rollout(method, params)
    for row, seq in zip(eval_rows, eval_seqs):
        true_label = row.get("label", seq.get("true_label", "stable" if protocol == "A" and eval_kind == "trajectory" else ""))
        try:
            dp, wp = _rollout_one(method, bm, prepared, seq)
            pred_label, reason, angle_margin, max_w, tail_growth, first_crossing, diverged = classify_external_stability(dp, wp)
            tm = trajectory_metrics(seq["delta"], seq["omega"], dp, wp)
            failed = False
            error = ""
        except Exception as exc:
            pred_label, reason = "unstable", f"rollout_error:{exc}"
            angle_margin, max_w, tail_growth, first_crossing, diverged = 10.0, 10.0, 10.0, 0.0, True
            tm = {
                "delta_r2": float("nan"),
                "omega_r2": float("nan"),
                "state_r2": float("nan"),
                "delta_rmse": float("nan"),
                "omega_rmse": float("nan"),
                "state_rmse": float("nan"),
            }
            failed = True
            error = str(exc)[:240]
        per_case.append({
            "protocol": protocol,
            "split_id": split_id,
            "method": method,
            "family": "physics_equation_rollout",
            "variant": "clustered",
            "window_len": "1",
            "seed": seed,
            "pm_mode": PM_MODE,
            "eval_kind": eval_kind,
            "scenario_id": row.get("scenario_id", seq.get("scenario_id", "")),
            "fault_line": row.get("fault_line", ""),
            "fault_duration": row.get("fault_duration", ""),
            "true_label": true_label,
            "pred_label": pred_label,
            "pred_reason": reason,
            "angle_margin": angle_margin,
            "max_frequency_deviation": max_w,
            "tail_growth": tail_growth,
            "first_crossing_index": first_crossing,
            "diverged": int(bool(diverged)),
            "failed": int(failed),
            "error": error,
            **_cleaning_fields(row, seq),
            **tm,
        })

    binary = [r for r in per_case if binary_label(str(r.get("true_label"))) is not None]
    y_true = [binary_label(str(r["true_label"])) for r in binary]
    y_pred = [1 if r["pred_label"] == "unstable" else 0 for r in binary]
    failed_count = sum(1 for r in per_case if r["failed"])
    stability = {
        "protocol": protocol,
        "split_id": split_id,
        "method": method,
        "family": "physics_equation_rollout",
        "variant": "clustered",
        "window_len": "1",
        "seed": seed,
        "status": "degraded" if failed_count else "ok",
        "pm_mode": PM_MODE,
        "non_binary_total": len(per_case) - len(binary),
        "rollout_failed_cases": failed_count,
        "rollout_failure_rate": failed_count / max(len(per_case), 1),
        **binary_metrics(y_true, y_pred),
    }

    ok = [r for r in per_case if not r["failed"] and math.isfinite(float(r["state_r2"]))]
    dynamics = {
        "protocol": protocol,
        "split_id": split_id,
        "method": method,
        "family": "physics_equation_rollout",
        "variant": "clustered",
        "window_len": "1",
        "seed": seed,
        "status": "ok" if ok else "failed",
        "pm_mode": PM_MODE,
        "rollout_r2": float(np.mean([float(r["state_r2"]) for r in ok])) if ok else float("nan"),
        "worst_case_r2": float(np.min([float(r["state_r2"]) for r in ok])) if ok else float("nan"),
        "mean_delta_r2": float(np.mean([float(r["delta_r2"]) for r in ok])) if ok else float("nan"),
        "mean_omega_r2": float(np.mean([float(r["omega_r2"]) for r in ok])) if ok else float("nan"),
        "rmse": float(np.mean([float(r["state_rmse"]) for r in ok])) if ok else float("nan"),
        "failure_rate": 1.0 - len(ok) / max(len(per_case), 1),
        "rollout_failed_cases": failed_count,
        "rollout_failure_rate": failed_count / max(len(per_case), 1),
        "binary_total": len(binary),
        "non_binary_total": len(per_case) - len(binary),
    }
    return stability, dynamics, per_case


def run_our_methods(
    protocol: str,
    split_id: str,
    train: list[dict[str, Any]],
    val: list[dict[str, Any]],
    internal_test: list[dict[str, Any]],
    stability_rows: list[dict[str, Any]],
    stability_seqs: list[dict[str, Any]],
    seeds: list[int],
    out_dir: Path,
    epochs_m2a: int,
    epochs_lyap: int,
    resume: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    model_root = out_dir / "our_methods" / f"protocol_{protocol}" / split_id
    bm, m0_path = fit_m0(train, model_root / "M0", resume=resume)
    area_internal_test = variant_sequences(internal_test, "clustered")
    area_stability = variant_sequences(stability_seqs, "clustered")

    stability_out: list[dict[str, Any]] = []
    dynamics_out: list[dict[str, Any]] = []
    full_grid: list[dict[str, Any]] = []
    per_case: list[dict[str, Any]] = []

    m0_seed = 0
    print(f"[{protocol}/{split_id}] M0 rollout evaluation", flush=True)
    stab, dyn, pc = evaluate_rollout_method("M0", protocol, split_id, m0_seed, bm, None, stability_rows, area_stability, "stability")
    stability_out.append(stab)
    full_grid.append({**stab, "grid_kind": "stability"})
    per_case.extend(pc)
    traj_rows = [{"scenario_id": seq.get("scenario_id", f"test{i}"), "label": "stable"} for i, seq in enumerate(area_internal_test)]
    _stab_unused, dyn_traj, pc_traj = evaluate_rollout_method("M0", protocol, split_id, m0_seed, bm, None, traj_rows, area_internal_test, "trajectory")
    dynamics_out.append(dyn_traj)
    full_grid.append({**dyn_traj, "grid_kind": "trajectory"})
    per_case.extend(pc_traj)

    for seed in seeds:
        print(f"[{protocol}/{split_id}] M2A seed={seed}", flush=True)
        m2a = train_or_load_m2a(train, val, internal_test, seed, model_root / "M2A", m0_path, epochs_m2a, resume)
        stab, dyn, pc = evaluate_rollout_method("M2A", protocol, split_id, seed, bm, m2a["params_c4"], stability_rows, area_stability, "stability")
        stability_out.append(stab)
        full_grid.append({**stab, "grid_kind": "stability"})
        per_case.extend(pc)
        _stab_unused, dyn_traj, pc_traj = evaluate_rollout_method("M2A", protocol, split_id, seed, bm, m2a["params_c4"], traj_rows, area_internal_test, "trajectory")
        dynamics_out.append(dyn_traj)
        full_grid.append({**dyn_traj, "grid_kind": "trajectory"})
        per_case.extend(pc_traj)

        print(f"[{protocol}/{split_id}] M2A-LYAP seed={seed}", flush=True)
        ly = train_or_load_m2a_lyap(train, val, internal_test, seed, model_root / "M2A-LYAP", m0_path, epochs_lyap, resume)
        stab, dyn, pc = evaluate_rollout_method("M2A-LYAP", protocol, split_id, seed, bm, ly["params_c4"], stability_rows, area_stability, "stability")
        stability_out.append(stab)
        full_grid.append({**stab, "grid_kind": "stability"})
        per_case.extend(pc)
        _stab_unused, dyn_traj, pc_traj = evaluate_rollout_method("M2A-LYAP", protocol, split_id, seed, bm, ly["params_c4"], traj_rows, area_internal_test, "trajectory")
        dynamics_out.append(dyn_traj)
        full_grid.append({**dyn_traj, "grid_kind": "trajectory"})
        per_case.extend(pc_traj)

    return stability_out, dynamics_out, full_grid, per_case
