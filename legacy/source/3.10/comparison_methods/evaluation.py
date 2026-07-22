"""Unified evaluation for external comparison methods."""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

from support.common import PM_MODE, binary_label, binary_metrics, trajectory_metrics
from support.data_protocol import variant_sequences

from diag_evaluate_external_cct_v3 import classify_external_stability

from comparison_methods.neural_classifiers import run_classifier
from comparison_methods.neural_forecast import fit_lstm_rollout, rollout_lstm, rollout_lstm_many
from comparison_methods.operator_methods import fit_operator, rollout_operator
from comparison_methods.sindyc import fit_score_sindyc, fit_sindyc, rollout_sindyc


RolloutFn = Callable[[dict[str, Any]], tuple[np.ndarray, np.ndarray]]


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


def _summarize_per_case(
    protocol: str,
    split_id: str,
    method: str,
    family: str,
    variant: str,
    window_len: int | str,
    seed: int,
    rows: list[dict[str, Any]],
    seqs: list[dict[str, Any]],
    rollout_fn: RolloutFn,
    eval_kind: str,
    fit_meta: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    per_case: list[dict[str, Any]] = []
    for row, seq in zip(rows, seqs):
        true_label = row.get("label", seq.get("true_label", "stable" if eval_kind == "trajectory" else ""))
        try:
            dp, wp = rollout_fn(seq)
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
            "family": family,
            "variant": variant,
            "window_len": window_len,
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
        "family": family,
        "variant": variant,
        "window_len": window_len,
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
        "family": family,
        "variant": variant,
        "window_len": window_len,
        "seed": seed,
        "status": "ok" if ok else "failed",
        "pm_mode": PM_MODE,
        "rollout_r2": float(np.mean([float(r["state_r2"]) for r in ok])) if ok else float("nan"),
        "forecast_r2": float(np.mean([float(r["state_r2"]) for r in ok])) if ok else float("nan"),
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
    if fit_meta:
        dynamics.update(fit_meta)
        stability.update({k: v for k, v in fit_meta.items() if k in {"fit_r2", "fit_rmse", "active_terms"}})
    return stability, dynamics, per_case


def _summarize_precomputed_cases(
    protocol: str,
    split_id: str,
    method: str,
    family: str,
    variant: str,
    window_len: int | str,
    seed: int,
    rows: list[dict[str, Any]],
    seqs: list[dict[str, Any]],
    predictions: list[tuple[np.ndarray, np.ndarray] | Exception],
    eval_kind: str,
    fit_meta: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    pred_by_id = {id(seq): pred for seq, pred in zip(seqs, predictions)}

    def rollout_fn(seq: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        pred = pred_by_id[id(seq)]
        if isinstance(pred, Exception):
            raise pred
        return pred

    return _summarize_per_case(
        protocol, split_id, method, family, variant, window_len, seed,
        rows, seqs, rollout_fn, eval_kind, fit_meta,
    )


def _not_applicable_classifier_rows(protocol: str) -> list[dict[str, Any]]:
    rows = []
    for method in ["CNN", "LSTM", "CNN+LSTM", "Transformer"]:
        rows.append({
            "protocol": protocol,
            "split_id": "A",
            "method": method,
            "family": "direct_classifier",
            "variant": "N/A",
            "window_len": "N/A",
            "seed": "",
            "status": "not_applicable",
            "pm_mode": PM_MODE,
            "error": "Protocol A highorder train split contains stable trajectories only; supervised direct classifiers are evaluated in Protocol B.",
        })
    return rows


def run_rollout_comparisons(
    protocol: str,
    split_id: str,
    train: list[dict[str, Any]],
    trajectory_rows: list[dict[str, Any]],
    trajectory_seqs: list[dict[str, Any]],
    stability_rows: list[dict[str, Any]],
    stability_seqs: list[dict[str, Any]],
    seeds: list[int],
    windows: list[int],
    quick: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    stability_out: list[dict[str, Any]] = []
    dynamics_out: list[dict[str, Any]] = []
    full_grid: list[dict[str, Any]] = []
    per_case: list[dict[str, Any]] = []

    for variant in ["clustered", "unclustered"]:
        tr_v = variant_sequences(train, variant)
        traj_v = variant_sequences(trajectory_seqs, variant)
        stab_v = variant_sequences(stability_seqs, variant)
        for seed in seeds:
            print(f"[{protocol}/{split_id}] SINDYc variant={variant} seed={seed}", flush=True)
            try:
                sindy = fit_sindyc(tr_v, seed=seed, sample_cap=1800 if quick else 6000)
                fit_meta = fit_score_sindyc(sindy, tr_v[: min(len(tr_v), 8)])
                stab, dyn, pc = _summarize_per_case(
                    protocol, split_id, "SINDYc", "sparse_identification", variant, "1",
                    seed, stability_rows, stab_v, lambda s, m=sindy: rollout_sindyc(m, s), "stability", fit_meta,
                )
                stability_out.append(stab)
                full_grid.append({**stab, "grid_kind": "stability"})
                per_case.extend(pc)
                _stab_unused, dyn_traj, pc_traj = _summarize_per_case(
                    protocol, split_id, "SINDYc", "sparse_identification", variant, "1",
                    seed, trajectory_rows, traj_v, lambda s, m=sindy: rollout_sindyc(m, s), "trajectory", fit_meta,
                )
                dynamics_out.append(dyn_traj)
                full_grid.append({**dyn_traj, "grid_kind": "trajectory"})
                per_case.extend(pc_traj)
            except Exception as exc:
                row = {
                    "protocol": protocol, "split_id": split_id, "method": "SINDYc",
                    "family": "sparse_identification", "variant": variant, "window_len": "1",
                    "seed": seed, "status": "failed", "pm_mode": PM_MODE,
                    "rollout_failed_cases": len(stab_v), "rollout_failure_rate": 1.0,
                    "error": str(exc)[:240],
                }
                stability_out.append(row)
                full_grid.append({**row, "grid_kind": "stability"})
                dynamics_out.append(row)
                full_grid.append({**row, "grid_kind": "trajectory"})

            for window_len in windows:
                for method in ["DMD", "HODMD", "EDMD-Koopman"]:
                    print(f"[{protocol}/{split_id}] {method} variant={variant} seed={seed} window={window_len}", flush=True)
                    try:
                        op = fit_operator(tr_v, method=method, window_len=window_len, seed=seed, sample_cap=1800 if quick else 8000)
                        stab, dyn, pc = _summarize_per_case(
                            protocol, split_id, method, "operator_forecast", variant, window_len,
                            seed, stability_rows, stab_v, lambda s, m=op, w=window_len: rollout_operator(m, s, w), "stability",
                        )
                        stability_out.append(stab)
                        full_grid.append({**stab, "grid_kind": "stability"})
                        per_case.extend(pc)
                        _stab_unused, dyn_traj, pc_traj = _summarize_per_case(
                            protocol, split_id, method, "operator_forecast", variant, window_len,
                            seed, trajectory_rows, traj_v, lambda s, m=op, w=window_len: rollout_operator(m, s, w), "trajectory",
                        )
                        dynamics_out.append(dyn_traj)
                        full_grid.append({**dyn_traj, "grid_kind": "trajectory"})
                        per_case.extend(pc_traj)
                    except Exception as exc:
                        row = {
                            "protocol": protocol, "split_id": split_id, "method": method,
                            "family": "operator_forecast", "variant": variant, "window_len": window_len,
                            "seed": seed, "status": "failed", "pm_mode": PM_MODE,
                            "rollout_failed_cases": len(stab_v), "rollout_failure_rate": 1.0,
                            "error": str(exc)[:240],
                        }
                        stability_out.append(row)
                        full_grid.append({**row, "grid_kind": "stability"})
                        dynamics_out.append(row)
                        full_grid.append({**row, "grid_kind": "trajectory"})
                print(f"[{protocol}/{split_id}] LSTM-rollout variant={variant} seed={seed} window={window_len}", flush=True)
                try:
                    bundle = fit_lstm_rollout(tr_v, seed=seed, window_len=window_len, quick=quick)
                    stab_predictions = rollout_lstm_many(bundle, stab_v, window_len)
                    traj_predictions = rollout_lstm_many(bundle, traj_v, window_len)
                    stab, dyn, pc = _summarize_precomputed_cases(
                        protocol, split_id, "LSTM-rollout", "blackbox_rollout", variant, window_len,
                        seed, stability_rows, stab_v, stab_predictions, "stability",
                    )
                    stability_out.append(stab)
                    full_grid.append({**stab, "grid_kind": "stability"})
                    per_case.extend(pc)
                    _stab_unused, dyn_traj, pc_traj = _summarize_precomputed_cases(
                        protocol, split_id, "LSTM-rollout", "blackbox_rollout", variant, window_len,
                        seed, trajectory_rows, traj_v, traj_predictions, "trajectory",
                    )
                    dynamics_out.append(dyn_traj)
                    full_grid.append({**dyn_traj, "grid_kind": "trajectory"})
                    per_case.extend(pc_traj)
                except Exception as exc:
                    row = {
                        "protocol": protocol, "split_id": split_id, "method": "LSTM-rollout",
                        "family": "blackbox_rollout", "variant": variant, "window_len": window_len,
                        "seed": seed, "status": "failed", "pm_mode": PM_MODE,
                        "rollout_failed_cases": len(stab_v), "rollout_failure_rate": 1.0,
                        "error": str(exc)[:240],
                    }
                    stability_out.append(row)
                    full_grid.append({**row, "grid_kind": "stability"})
                    dynamics_out.append(row)
                    full_grid.append({**row, "grid_kind": "trajectory"})
    if protocol == "A":
        na_rows = _not_applicable_classifier_rows(protocol)
        stability_out.extend(na_rows)
        full_grid.extend({**row, "grid_kind": "stability"} for row in na_rows)
    return stability_out, dynamics_out, full_grid, per_case


def run_protocol_b_classifiers(
    split_id: str,
    train_rows: list[dict[str, Any]],
    train_seqs: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    test_seqs: list[dict[str, Any]],
    seeds: list[int],
    windows: list[int],
    quick: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    stability_out: list[dict[str, Any]] = []
    full_grid: list[dict[str, Any]] = []
    per_case: list[dict[str, Any]] = []
    for variant in ["clustered", "unclustered"]:
        tr_v = variant_sequences(train_seqs, variant)
        te_v = variant_sequences(test_seqs, variant)
        for seed in seeds:
            for window_len in windows:
                for method in ["CNN", "LSTM", "CNN+LSTM", "Transformer"]:
                    print(f"[B/{split_id}] {method} classifier variant={variant} seed={seed} window={window_len}", flush=True)
                    row, pc = run_classifier(
                        method, train_rows, tr_v, test_rows, te_v,
                        protocol="B", split_id=split_id, variant=variant,
                        seed=seed, window_len=window_len, quick=quick,
                    )
                    stability_out.append(row)
                    full_grid.append({**row, "grid_kind": "stability"})
                    per_case.extend(pc)
    return stability_out, full_grid, per_case
