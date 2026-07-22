"""Task 3.11: Protocol B M2A/M2A-LYAP accuracy improvement.

This harness is intentionally isolated from the 3.10 final comparison code.  It
reuses the 3.10 data protocol and training wrappers, but writes all artifacts to
a new 3.11 result directory.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


def find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "AGENTS.md").exists() and (parent / "discovery").exists():
            return parent
    raise RuntimeError(f"Could not locate project root from {here}")


PROJECT_ROOT = find_project_root()
TASK_DIR = Path(__file__).resolve().parent
THREE_TEN_DIR = TASK_DIR.with_name("hist_310_final_paper_comparison")
KRON_DIR = PROJECT_ROOT / "discovery" / "3.2-kron"
for path in (str(THREE_TEN_DIR), str(PROJECT_ROOT / "discovery"), str(KRON_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from graph_sindy_data import OMEGA_S  # noqa: E402
from support.common import (  # noqa: E402
    PM_MODE,
    SEEDS_FULL,
    SEEDS_SMOKE,
    binary_label,
    binary_metrics,
    code_fingerprint,
    dataset_fingerprint,
    read_csv,
    read_json,
    sequence_signature,
    sha256_file,
    trajectory_metrics,
    write_csv,
    write_json,
)
from support.data_protocol import (  # noqa: E402
    PROTOCOL_B_SPLITS,
    assert_protocol_b_disjoint,
    dataset_audit,
    protocol_b_datasets,
    variant_sequences,
)
from our_methods.evaluation import evaluate_rollout_method  # noqa: E402
from our_methods.m0 import fit_m0  # noqa: E402
from our_methods.m2a import train_or_load_m2a  # noqa: E402
from our_methods.m2a_lyap import train_or_load_m2a_lyap  # noqa: E402
from diag_evaluate_external_cct_v3 import classify_external_stability  # noqa: E402
from evaluate_8cluster_dataonly_engineering_ieee39 import load_model  # noqa: E402

try:  # Optional; only needed for the CLS fine-tuning branch.
    import torch
    import torch.nn.functional as F

    import hist_run_two_tier_batch as two_tier
    import hist_run_384_m2a_lyapunov as lyap_old
    from coherent_aggregation import aggregate_sequences
    from graph_sindy_data import M_PRIOR
    from hist_models_forward_residual_v3 import make_mask, split_theta, swing_acc_torch
    from run_8cluster_forward_residual_sindy_ieee39 import feature_names
    from run_forward_residual_graphsindy_v324_ieee39 import baseline_tensors, to_torch_seq
except Exception:  # pragma: no cover
    torch = None
    F = None


RESULT_DIR = PROJECT_ROOT / "results/03_kron_graph_sindy/02_cluster/3.11-protocol_b_m2a_acc"
QUICK_RESULT_DIR = RESULT_DIR.with_name(f"{RESULT_DIR.name}_quick_smoke")
REPORT_PATH = PROJECT_ROOT / "reports/hist_3.11-Protocol_B_M2A_LYAP_Accuracy.md"
QUICK_REPORT_PATH = PROJECT_ROOT / "reports/hist_3.11-Protocol_B_M2A_LYAP_Accuracy_quick_smoke.md"
RESULT_310_DIR = PROJECT_ROOT / "results/03_kron_graph_sindy/02_cluster/3.10-final_paper_comparison_protocol_ab"

ANGLE_THR_PI = [1.0, 1.1, 1.25, 1.5, 2.0, 3.0]
FREQ_THR_PU = [0.05, 0.075, 0.10, 0.15, 0.20]
TAIL_THR = [2.0, 3.0, 5.0, 8.0, float("inf")]
M0_ANCHORS = ["all", "stable_near", "stable_only"]
PASSH_HORIZONS = [50, 100, 200, 400]
PASSH_ALPHAS = [0.0, 0.5, 1.0]
PASSH_GATE_PI = 0.95
PASSH_DAMP = 0.0


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return val if math.isfinite(val) else None
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return str(obj)


def _anchor_train(rows: list[dict[str, Any]], seqs: list[dict[str, Any]], anchor: str) -> list[dict[str, Any]]:
    if anchor == "all":
        return seqs
    if anchor == "stable_near":
        keep = {"stable", "near_critical"}
    elif anchor == "stable_only":
        keep = {"stable"}
    else:
        raise ValueError(anchor)
    return [seq for row, seq in zip(rows, seqs) if row.get("label") in keep]


def _is_unstable_by_threshold(row: dict[str, Any], angle_thr_pi: float, freq_thr_pu: float, tail_thr: float) -> bool:
    if int(float(row.get("failed", 0) or 0)):
        return True
    if int(float(row.get("diverged", 0) or 0)):
        return True
    angle_margin = float(row.get("angle_margin", 0.0) or 0.0)
    max_w = float(row.get("max_frequency_deviation", 0.0) or 0.0)
    tail_growth = float(row.get("tail_growth", 0.0) or 0.0)
    if angle_margin > angle_thr_pi - 1.0:
        return True
    if max_w > freq_thr_pu * OMEGA_S:
        return True
    if math.isfinite(tail_thr) and tail_growth > tail_thr:
        return True
    return False


def _metrics_for_threshold(cases: list[dict[str, Any]], angle_thr_pi: float, freq_thr_pu: float, tail_thr: float) -> dict[str, Any]:
    binary = [r for r in cases if binary_label(str(r.get("true_label"))) is not None]
    y_true = [int(binary_label(str(r["true_label"]))) for r in binary]
    y_pred = [
        1 if _is_unstable_by_threshold(r, angle_thr_pi, freq_thr_pu, tail_thr) else 0
        for r in binary
    ]
    return binary_metrics(y_true, y_pred)


def _line_balanced_threshold_metrics(cases: list[dict[str, Any]], angle_thr_pi: float, freq_thr_pu: float, tail_thr: float) -> dict[str, Any]:
    pooled = _metrics_for_threshold(cases, angle_thr_pi, freq_thr_pu, tail_thr)
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in cases:
        if binary_label(str(row.get("true_label"))) is None:
            continue
        groups.setdefault(str(row.get("fault_line", "")), []).append(row)
    line_metrics = [
        _metrics_for_threshold(rows, angle_thr_pi, freq_thr_pu, tail_thr)
        for rows in groups.values()
        if rows
    ]
    if not line_metrics:
        return {
            **pooled,
            "calib_line_mean_Accuracy": 0.0,
            "calib_line_min_Accuracy": 0.0,
            "calib_line_mean_BalancedAccuracy": 0.0,
            "calib_line_min_BalancedAccuracy": 0.0,
            "calib_line_max_FSR": 1.0,
            "calib_line_max_FUR": 1.0,
            "calib_n_lines": 0,
        }
    return {
        **pooled,
        "calib_line_mean_Accuracy": float(statistics.fmean(float(m["Accuracy"]) for m in line_metrics)),
        "calib_line_min_Accuracy": float(min(float(m["Accuracy"]) for m in line_metrics)),
        "calib_line_mean_BalancedAccuracy": float(statistics.fmean(float(m["BalancedAccuracy"]) for m in line_metrics)),
        "calib_line_min_BalancedAccuracy": float(min(float(m["BalancedAccuracy"]) for m in line_metrics)),
        "calib_line_max_FSR": float(max(float(m["FSR"]) for m in line_metrics)),
        "calib_line_max_FUR": float(max(float(m["FUR"]) for m in line_metrics)),
        "calib_n_lines": len(line_metrics),
    }


def _calibrate_thresholds(
    protocol: str,
    split_id: str,
    method: str,
    seed: int,
    m0_anchor: str,
    val_cases: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    grid_rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for angle_thr in ANGLE_THR_PI:
        for freq_thr in FREQ_THR_PU:
            for tail_thr in TAIL_THR:
                metrics = _metrics_for_threshold(val_cases, angle_thr, freq_thr, tail_thr)
                row = {
                    "protocol": protocol,
                    "split_id": split_id,
                    "method": method,
                    "seed": seed,
                    "m0_anchor": m0_anchor,
                    "calibrator_type": "threshold_grid",
                    "angle_thr_pi": angle_thr,
                    "freq_thr_pu": freq_thr,
                    "tail_thr": "inf" if not math.isfinite(tail_thr) else tail_thr,
                    "val_Accuracy": metrics["Accuracy"],
                    "val_BalancedAccuracy": metrics["BalancedAccuracy"],
                    "val_FSR": metrics["FSR"],
                    "val_FUR": metrics["FUR"],
                    "val_tp": metrics["tp"],
                    "val_fn": metrics["fn"],
                    "val_fp": metrics["fp"],
                    "val_tn": metrics["tn"],
                    "binary_total": metrics["binary_total"],
                }
                grid_rows.append(row)
                score = (
                    metrics["Accuracy"],
                    metrics["BalancedAccuracy"],
                    -metrics["FSR"],
                    -angle_thr,
                    -freq_thr,
                    -(tail_thr if math.isfinite(tail_thr) else 1e9),
                )
                if best is None or score > best["_score"]:
                    best = {**row, "_score": score}
    assert best is not None
    best.pop("_score", None)
    return best, grid_rows


def _apply_calibrator(
    cases: list[dict[str, Any]],
    calib: dict[str, Any],
    method: str,
    seed: int,
    m0_anchor: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    angle_thr = float(calib["angle_thr_pi"])
    freq_thr = float(calib["freq_thr_pu"])
    raw_tail = calib["tail_thr"]
    tail_thr = float("inf") if str(raw_tail) == "inf" else float(raw_tail)
    calibrator_type = str(calib.get("calibrator_type", "threshold_grid"))
    out_cases: list[dict[str, Any]] = []
    meta_keys = (
        "rollout_horizon",
        "passivity_gate_pi",
        "passivity_damp",
        "residual_alpha",
        "calibration_scope",
        "calibration_objective",
        "calib_line_mean_Accuracy",
        "calib_line_min_Accuracy",
        "calib_line_mean_BalancedAccuracy",
        "calib_line_min_BalancedAccuracy",
        "calib_line_max_FSR",
        "calib_line_max_FUR",
        "calib_n_lines",
    )
    calib_meta = {
        k: calib[k]
        for k in meta_keys
        if k in calib
    }
    for row in cases:
        pred = "unstable" if _is_unstable_by_threshold(row, angle_thr, freq_thr, tail_thr) else "stable"
        out_cases.append({
            **row,
            "method": method,
            "seed": seed,
            "m0_anchor": m0_anchor,
            "calibrator_type": calibrator_type,
            "angle_thr_pi": angle_thr,
            "freq_thr_pu": freq_thr,
            "tail_thr": "inf" if not math.isfinite(tail_thr) else tail_thr,
            "pred_label": pred,
            **calib_meta,
        })
    binary = [r for r in out_cases if binary_label(str(r.get("true_label"))) is not None]
    metrics = binary_metrics(
        [int(binary_label(str(r["true_label"]))) for r in binary],
        [1 if r["pred_label"] == "unstable" else 0 for r in binary],
    )
    failed = sum(1 for r in out_cases if int(float(r.get("failed", 0) or 0)))
    summary = {
        "protocol": "B",
        "split_id": out_cases[0].get("split_id", "") if out_cases else "",
        "method": method,
        "family": "physics_equation_rollout_calibrated",
        "variant": "clustered",
        "window_len": "1",
        "seed": seed,
        "status": "degraded" if failed else "ok",
        "pm_mode": PM_MODE,
        "m0_anchor": m0_anchor,
        "calibrator_type": calibrator_type,
        "angle_thr_pi": angle_thr,
        "freq_thr_pu": freq_thr,
        "tail_thr": "inf" if not math.isfinite(tail_thr) else tail_thr,
        "val_Accuracy": calib["val_Accuracy"],
        "val_BalancedAccuracy": calib["val_BalancedAccuracy"],
        "val_FSR": calib["val_FSR"],
        "val_FUR": calib["val_FUR"],
        **calib_meta,
        "rollout_failed_cases": failed,
        "rollout_failure_rate": failed / max(len(out_cases), 1),
        "non_binary_total": len(out_cases) - len(binary),
        **metrics,
    }
    return summary, out_cases


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metric_names = ["FSR", "FUR", "Accuracy", "BalancedAccuracy", "rollout_failure_rate"]
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") not in {"ok", "degraded"}:
            continue
        key = (
            row.get("protocol", ""),
            row.get("method", ""),
            row.get("family", ""),
            row.get("variant", ""),
            row.get("window_len", ""),
            row.get("m0_anchor", ""),
            row.get("calibrator_type", ""),
        )
        groups.setdefault(key, []).append(row)

    out = []
    for key, vals in sorted(groups.items()):
        rec = {
            "protocol": key[0],
            "method": key[1],
            "family": key[2],
            "variant": key[3],
            "window_len": key[4],
            "m0_anchor": key[5],
            "calibrator_type": key[6],
            "n_runs": len(vals),
        }
        for metric in metric_names:
            nums = []
            for row in vals:
                try:
                    nums.append(float(row.get(metric, "")))
                except Exception:
                    pass
            if nums:
                rec[f"mean_{metric}"] = float(statistics.fmean(nums))
                rec[f"std_{metric}"] = float(statistics.pstdev(nums)) if len(nums) > 1 else 0.0
                rec[f"min_{metric}"] = float(min(nums))
                rec[f"max_{metric}"] = float(max(nums))
        out.append(rec)
    return out


def _baseline_rows_from_310() -> list[dict[str, Any]]:
    path = PROJECT_ROOT / "results/03_kron_graph_sindy/02_cluster/3.10-final_paper_comparison_protocol_ab/protocol_b_stability_summary.csv"
    if not path.exists():
        return []
    rows = []
    for row in read_csv(path):
        if row.get("method") in {"M0", "M2A", "M2A-LYAP"}:
            rows.append(row)
    return rows


def _write_report(out_dir: Path, report_path: Path, mode: str) -> None:
    summary = read_csv(out_dir / "protocol_b_stability_summary.csv")
    full = read_csv(out_dir / "protocol_b_full_grid_results.csv")
    baseline = read_csv(out_dir / "baseline_310_protocol_b_summary.csv")
    audit = read_csv(out_dir / "audit_findings_table.csv")
    data = dataset_audit(quick=(mode == "quick-smoke"))

    def table(rows: list[dict[str, Any]], cols: list[str], limit: int | None = None) -> list[str]:
        rows_use = rows if limit is None else rows[:limit]
        lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
        for row in rows_use:
            vals = []
            for col in cols:
                val = row.get(col, "")
                if val in (None, ""):
                    vals.append("-")
                else:
                    try:
                        vals.append(f"{float(val):.4g}")
                    except Exception:
                        vals.append(str(val))
            lines.append("| " + " | ".join(vals) + " |")
        if limit is not None and len(rows) > limit:
            lines.append("| ... | " + f"{len(rows) - limit} more rows in CSV |" + "|".join("-" for _ in cols[2:]) + "|")
        return lines

    best = sorted(summary, key=lambda r: float(r.get("mean_Accuracy") or -1), reverse=True)
    lines = [
        "---",
        'title: "3.11 Protocol B M2A / LYAP Accuracy"',
        f'date: "{time.strftime("%Y-%m-%d")}"',
        "status: experiment",
        "---",
        "",
        "# 3.11 Protocol B M2A / LYAP Accuracy",
        "",
        f"Run mode: `{mode}`.",
        f"Result directory: `{out_dir}`.",
        "",
        "## Protocol",
        "",
        f"- Dense dataset: `{data['dense_path']}` with counts {data['dense_counts']}.",
        "- Splits: fixed Protocol B fault-line group splits from 3.10.",
        "- Metric: binary `stable` / `unstable` Accuracy. `near_critical` is excluded from binary metrics.",
        "- Goal: M2A and M2A-LYAP family mean Accuracy over splits/seeds above 0.90.",
        "- Leakage guard: generator-written result summaries such as `case_results.csv` maximum angle/frequency fields are forbidden as model inputs.",
        "- PASSH-R calibration uses train+val labels only, scored by fault-line mean Accuracy, then fault-line mean Balanced Accuracy, with long-horizon tie-breaking.",
        "",
        "## 3.10 Baseline",
        "",
        *table(baseline, ["method", "family", "variant", "window_len", "mean_Accuracy", "mean_BalancedAccuracy", "mean_FSR", "mean_FUR", "n_runs"], limit=20),
        "",
        "## 3.11 Summary",
        "",
        *table(best, ["method", "m0_anchor", "calibrator_type", "n_runs", "mean_Accuracy", "std_Accuracy", "min_Accuracy", "mean_BalancedAccuracy", "mean_FSR", "mean_FUR"], limit=60),
        "",
        "## Per Split / Seed Grid",
        "",
        *table(full, ["split_id", "method", "m0_anchor", "seed", "status", "Accuracy", "BalancedAccuracy", "FSR", "FUR", "val_Accuracy", "calib_line_mean_Accuracy", "calib_line_mean_BalancedAccuracy", "rollout_horizon", "residual_alpha", "passivity_gate_pi", "angle_thr_pi", "freq_thr_pu", "tail_thr"], limit=120),
        "",
        "## Audit",
        "",
        *table(audit, ["severity", "check", "message"]),
        "",
        "## Output Files",
        "",
        "- `run_config.json`",
        "- `baseline_310_protocol_b_summary.csv`",
        "- `protocol_b_full_grid_results.csv`",
        "- `protocol_b_stability_summary.csv`",
        "- `per_case_predictions.csv`",
        "- `calibrator_grid_results.csv`",
        "- `m0_anchor_ablation.csv`",
        "- `audit_findings_table.csv`",
        "- `invalidated_leakage_rows.csv` when old leakage rows are encountered",
        "- `artifact_hashes.csv`",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _audit(out_dir: Path, quick: bool, run_cls: bool) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    try:
        assert_protocol_b_disjoint()
    except Exception as exc:
        findings.append({"severity": "ERROR", "check": "protocol_b_disjoint", "message": str(exc)})
    full = read_csv(out_dir / "protocol_b_full_grid_results.csv")
    summary = read_csv(out_dir / "protocol_b_stability_summary.csv")
    splits = {"B1"} if quick else set(PROTOCOL_B_SPLITS)
    got_splits = {r.get("split_id") for r in full}
    missing = sorted(splits - got_splits)
    if missing:
        findings.append({"severity": "ERROR", "check": "split_coverage", "message": f"Missing splits: {missing}"})
    leaked = [
        r for r in [*full, *summary]
        if r.get("family") == "observed_summary_head" or "ObsHead" in str(r.get("method", ""))
    ]
    if leaked:
        findings.append({
            "severity": "ERROR",
            "check": "leakage_observed_summary",
            "message": f"Found {len(leaked)} observed-summary rows. These use generator-written post-hoc summaries and are invalid.",
        })
    invalidated = read_csv(out_dir / "invalidated_leakage_rows.csv")
    if invalidated:
        findings.append({
            "severity": "ERROR",
            "check": "invalidated_leakage_rows",
            "message": f"Quarantined {len(invalidated)} rows from old observed-summary outputs. They are excluded from valid summaries.",
        })
    config_path = out_dir / "run_config.json"
    config = read_json(config_path) if config_path.exists() else {}
    passh_only = bool(config.get("passh_only", False))
    target_methods = set() if passh_only else {"M2A-BCal", "M2A-LYAP-BCal"}
    if bool(config.get("run_passh_val", config.get("run_passh", False))):
        target_methods |= {"M2A-PASSH", "M2A-LYAP-PASSH"}
    if bool(config.get("run_passh_robust", False)):
        target_methods |= {"M2A-PASSH-R", "M2A-LYAP-PASSH-R"}
    if run_cls and not passh_only:
        target_methods |= {"M2A-CLS", "M2A-LYAP-CLS"}
    got_methods = {r.get("method") for r in full}
    missing_methods = sorted(target_methods - got_methods)
    if missing_methods:
        findings.append({"severity": "WARNING", "check": "method_coverage", "message": f"Missing methods: {missing_methods}"})
    valid_summary = [r for r in summary if "ObsHead" not in str(r.get("method", ""))]
    family_targets = {
        "M2A": [r for r in valid_summary if str(r.get("method", "")).startswith("M2A-") and not str(r.get("method", "")).startswith("M2A-LYAP")],
        "M2A-LYAP": [r for r in valid_summary if str(r.get("method", "")).startswith("M2A-LYAP-")],
    }
    for family, rows in family_targets.items():
        best = max((float(r.get("mean_Accuracy") or 0.0) for r in rows), default=0.0)
        if best < 0.90:
            findings.append({
                "severity": "WARNING",
                "check": "target_accuracy",
                "message": f"{family} best mean Accuracy={best:.4f} below 0.90",
            })
    if not findings:
        findings.append({"severity": "OK", "check": "audit", "message": "No blocking findings."})
    return findings


def _artifact_hashes(out_dir: Path, report_path: Path) -> list[dict[str, Any]]:
    paths = [
        out_dir / "run_config.json",
        out_dir / "protocol_b_full_grid_results.csv",
        out_dir / "protocol_b_stability_summary.csv",
        out_dir / "per_case_predictions.csv",
        out_dir / "calibrator_grid_results.csv",
        out_dir / "audit_findings_table.csv",
        report_path,
    ]
    rows = []
    for path in paths:
        rows.append({
            "path": str(path.relative_to(PROJECT_ROOT)) if path.exists() else str(path),
            "exists": int(path.exists()),
            "sha256": sha256_file(path) if path.exists() and path.is_file() else "",
            "bytes": path.stat().st_size if path.exists() and path.is_file() else 0,
        })
    return rows


def _is_leakage_row(row: dict[str, Any]) -> bool:
    return row.get("family") == "observed_summary_head" or "ObsHead" in str(row.get("method", ""))


def _write_outputs(out_dir: Path, report_path: Path, mode: str, full_rows: list[dict[str, Any]], per_case: list[dict[str, Any]], grid_rows: list[dict[str, Any]], quick: bool, run_cls: bool) -> None:
    invalid_rows = [r for r in [*full_rows, *per_case, *grid_rows] if _is_leakage_row(r)]
    if invalid_rows:
        write_csv(out_dir / "invalidated_leakage_rows.csv", invalid_rows)
    else:
        write_csv(out_dir / "invalidated_leakage_rows.csv", [], fieldnames=["protocol", "method", "family", "message"])
    full_rows = [r for r in full_rows if not _is_leakage_row(r)]
    per_case = [r for r in per_case if not _is_leakage_row(r)]
    grid_rows = [r for r in grid_rows if not _is_leakage_row(r)]
    write_csv(out_dir / "protocol_b_full_grid_results.csv", full_rows)
    summary = _aggregate(full_rows)
    write_csv(out_dir / "protocol_b_stability_summary.csv", summary)
    write_csv(out_dir / "per_case_predictions.csv", per_case)
    write_csv(out_dir / "calibrator_grid_results.csv", grid_rows)
    write_csv(out_dir / "m0_anchor_ablation.csv", [r for r in summary if r.get("m0_anchor")])
    write_csv(out_dir / "baseline_310_protocol_b_summary.csv", _baseline_rows_from_310())
    findings = _audit(out_dir, quick=quick, run_cls=run_cls)
    write_csv(out_dir / "audit_findings_table.csv", findings)
    write_json(out_dir / "audit_findings.json", findings)
    _write_report(out_dir, report_path, mode)
    write_csv(out_dir / "artifact_hashes.csv", _artifact_hashes(out_dir, report_path))


def _run_rollout_cases(method: str, split_id: str, seed: int, bm, params: np.ndarray | None, rows: list[dict[str, Any]], seqs: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    area = variant_sequences(seqs, "clustered")
    _summary, _dyn, cases = evaluate_rollout_method(method, "B", split_id, seed, bm, params, rows, area, kind)
    return cases


def _project_positive_power(residual: np.ndarray, omega: np.ndarray, delta: np.ndarray, M: np.ndarray, gate_pi: float, damp: float) -> np.ndarray:
    if float(np.max(delta) - np.min(delta)) >= gate_pi * math.pi:
        return residual
    power = float(np.sum(M * omega * residual))
    norm = float(np.sum(M * omega ** 2)) + 1e-12
    target = -float(damp) * norm
    if power <= target:
        return residual
    return residual - ((power - target) / norm) * omega


def _rollout_m2a_passh(params: np.ndarray, bm, seq: dict[str, Any], horizon: int, gate_pi: float = PASSH_GATE_PI, damp: float = PASSH_DAMP, residual_alpha: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    from our_methods.m2a import prepare_m2a_rollout

    prepared = prepare_m2a_rollout(params)
    B = prepared["B"]
    node_theta = prepared["node_theta"]
    edge_theta = prepared["edge_theta"]
    t = np.asarray(seq["t"], dtype=np.float64)
    n = min(len(t), int(horizon))
    da = np.asarray(seq["delta"], dtype=np.float64)
    wa = np.asarray(seq["omega"], dtype=np.float64)
    dpm = np.asarray(seq["tm"], dtype=np.float64) - np.asarray(seq["eq"]["Pm0"], dtype=np.float64)
    s0 = np.asarray(seq["eq"]["sin_eta0"], dtype=np.float64)
    c0 = np.asarray(seq["eq"]["cos_eta0"], dtype=np.float64)
    dt_actual = float(np.mean(np.diff(t)))
    ddpm = np.gradient(dpm, dt_actual, axis=0, edge_order=2)
    M = np.asarray(bm.M, dtype=np.float64)
    dp = np.zeros((n, da.shape[1]), dtype=np.float64)
    wp = np.zeros((n, wa.shape[1]), dtype=np.float64)
    dp[0], wp[0] = da[0], wa[0]
    for k in range(n - 1):
        eta = dp[k] @ B
        dsin = np.sin(eta) - s0
        dcos = np.cos(eta) - c0
        graph_sin = (dsin * bm.k_s) @ B.T
        base_acc = (two_tier.OMEGA_S * dpm[k] - bm.D * wp[k] - two_tier.OMEGA_S * graph_sin) / bm.M
        wc = np.clip(wp[k], -50, 50)
        node_feat = np.column_stack([wc, np.abs(wc) * wc, wc ** 2, dpm[k], dpm[k] * wc, ddpm[k]])
        domega_e = wc @ B
        edge_feat = np.column_stack([
            dsin,
            dcos,
            domega_e,
            np.clip(domega_e * dsin, -100, 100),
            np.clip(domega_e * dcos, -100, 100),
        ])
        residual = residual_alpha * (np.sum(node_feat * node_theta, axis=-1) + np.sum(edge_feat * edge_theta, axis=-1) @ B.T)
        residual = _project_positive_power(residual, wp[k], dp[k], M, gate_pi, damp)
        dw = base_acc + residual
        dt_v = float(t[k + 1] - t[k])
        wp[k + 1] = wp[k] + dt_v * dw
        dp[k + 1] = dp[k] + 0.5 * dt_v * (wp[k] + wp[k + 1])
        if not np.all(np.isfinite(dp[k + 1])) or not np.all(np.isfinite(wp[k + 1])):
            raise RuntimeError("non-finite")
        if np.max(np.abs(dp[k + 1])) > 1e6 or np.max(np.abs(wp[k + 1])) > 1e6:
            raise RuntimeError("diverged")
    return dp, wp


def _rollout_lyap_passh(params: np.ndarray, bm, seq: dict[str, Any], horizon: int, gate_pi: float = PASSH_GATE_PI, damp: float = PASSH_DAMP, residual_alpha: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    from our_methods.m2a_lyap import prepare_m2a_lyap_rollout

    prepared = prepare_m2a_lyap_rollout(params)
    B = prepared["B"]
    node_theta = prepared["node_theta"]
    edge_theta = prepared["edge_theta"]
    t = np.asarray(seq["t"], dtype=np.float64)
    n = min(len(t), int(horizon))
    da = np.asarray(seq["delta"], dtype=np.float64)
    wa = np.asarray(seq["omega"], dtype=np.float64)
    dpm = np.asarray(seq["tm"], dtype=np.float64) - np.asarray(seq["eq"]["Pm0"], dtype=np.float64)
    s0 = np.asarray(seq["eq"]["sin_eta0"], dtype=np.float64)
    c0 = np.asarray(seq["eq"]["cos_eta0"], dtype=np.float64)
    dt_actual = float(np.mean(np.diff(t)))
    ddpm = np.gradient(dpm, dt_actual, axis=0, edge_order=2)
    M = np.asarray(bm.M, dtype=np.float64)
    dp = np.zeros((n, da.shape[1]), dtype=np.float64)
    wp = np.zeros((n, wa.shape[1]), dtype=np.float64)
    dp[0], wp[0] = da[0], wa[0]
    for k in range(n - 1):
        eta = dp[k] @ B
        es = np.sin(eta) - s0
        graph_sin = (es * bm.k_s) @ B.T
        base_acc = (lyap_old.OMEGA_S * dpm[k] - bm.D * wp[k] - lyap_old.OMEGA_S * graph_sin) / bm.M
        wc = np.clip(wp[k], -50, 50)
        node_feat = np.column_stack([wc, np.abs(wc) * wc, wc ** 3, dpm[k], dpm[k] * wc, ddpm[k]])
        domega_e = wc @ B
        edge_feat = lyap_old.build_edge_features_np(eta, domega_e, s0, c0)
        r_node = np.sum(node_feat * node_theta, axis=-1)
        edge_force = np.sum(edge_feat * edge_theta, axis=-1) @ B.T
        residual = residual_alpha * (r_node + edge_force / bm.M)
        residual = residual - np.sum(M * residual) / np.sum(M)
        residual = _project_positive_power(residual, wp[k], dp[k], M, gate_pi, damp)
        dw = base_acc + residual
        dt_v = float(t[k + 1] - t[k])
        wp[k + 1] = wp[k] + dt_v * dw
        dp[k + 1] = dp[k] + 0.5 * dt_v * (wp[k] + wp[k + 1])
        if not np.all(np.isfinite(dp[k + 1])) or not np.all(np.isfinite(wp[k + 1])):
            raise RuntimeError("non-finite")
        if np.max(np.abs(dp[k + 1])) > 1e6 or np.max(np.abs(wp[k + 1])) > 1e6:
            raise RuntimeError("diverged")
    return dp, wp


def _run_passh_cases(
    base_method: str,
    out_method: str,
    split_id: str,
    seed: int,
    bm,
    params: np.ndarray,
    rows: list[dict[str, Any]],
    seqs: list[dict[str, Any]],
    eval_kind: str,
    horizon: int,
    residual_alpha: float,
) -> list[dict[str, Any]]:
    area = variant_sequences(seqs, "clustered")
    per_case: list[dict[str, Any]] = []
    for row, seq in zip(rows, area):
        true_label = row.get("label", seq.get("true_label", ""))
        try:
            if base_method == "M2A":
                dp, wp = _rollout_m2a_passh(params, bm, seq, horizon, residual_alpha=residual_alpha)
            elif base_method == "M2A-LYAP":
                dp, wp = _rollout_lyap_passh(params, bm, seq, horizon, residual_alpha=residual_alpha)
            else:
                raise ValueError(base_method)
            pred_label, reason, angle_margin, max_w, tail_growth, first_crossing, diverged = classify_external_stability(dp, wp)
            tm = trajectory_metrics(seq["delta"][: len(dp)], seq["omega"][: len(wp)], dp, wp)
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
            "protocol": "B",
            "split_id": split_id,
            "method": out_method,
            "family": "physics_equation_rollout_passivity_horizon",
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
            "rollout_horizon": horizon,
            "residual_alpha": residual_alpha,
            "passivity_gate_pi": PASSH_GATE_PI,
            "passivity_damp": PASSH_DAMP,
            **tm,
        })
    return per_case


def _calibrate_passh(
    base_method: str,
    out_method: str,
    split_id: str,
    seed: int,
    bm,
    params: np.ndarray,
    rows: list[dict[str, Any]],
    seqs: list[dict[str, Any]],
    m0_anchor: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    all_grid: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for horizon in PASSH_HORIZONS:
        for alpha in PASSH_ALPHAS:
            val_cases = _run_passh_cases(base_method, out_method, split_id, seed, bm, params, rows, seqs, "validation", horizon, alpha)
            calib, grid = _calibrate_thresholds("B", split_id, out_method, seed, m0_anchor, val_cases)
            for row in grid:
                row.update({
                    "rollout_horizon": horizon,
                    "residual_alpha": alpha,
                    "passivity_gate_pi": PASSH_GATE_PI,
                    "passivity_damp": PASSH_DAMP,
                })
            all_grid.extend(grid)
            score = (calib["val_Accuracy"], calib["val_BalancedAccuracy"], -calib["val_FSR"], -calib["val_FUR"], alpha, -horizon)
            if best is None or score > best["_score"]:
                best = {
                    **calib,
                    "rollout_horizon": horizon,
                    "residual_alpha": alpha,
                    "passivity_gate_pi": PASSH_GATE_PI,
                    "passivity_damp": PASSH_DAMP,
                    "_score": score,
                }
    assert best is not None
    best.pop("_score", None)
    return best, all_grid


def _calibrate_passh_line_balanced(
    base_method: str,
    out_method: str,
    split_id: str,
    seed: int,
    bm,
    params: np.ndarray,
    rows: list[dict[str, Any]],
    seqs: list[dict[str, Any]],
    m0_anchor: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    all_grid: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for horizon in PASSH_HORIZONS:
        for alpha in PASSH_ALPHAS:
            calib_cases = _run_passh_cases(
                base_method, out_method, split_id, seed, bm, params,
                rows, seqs, "train_validation", horizon, alpha,
            )
            for angle_thr in ANGLE_THR_PI:
                for freq_thr in FREQ_THR_PU:
                    for tail_thr in TAIL_THR:
                        metrics = _line_balanced_threshold_metrics(calib_cases, angle_thr, freq_thr, tail_thr)
                        row = {
                            "protocol": "B",
                            "split_id": split_id,
                            "method": out_method,
                            "seed": seed,
                            "m0_anchor": m0_anchor,
                            "calibrator_type": "line_balanced_train_val_threshold_grid",
                            "calibration_scope": "train_val",
                            "calibration_objective": "line_mean_accuracy_then_balanced_accuracy_long_horizon",
                            "angle_thr_pi": angle_thr,
                            "freq_thr_pu": freq_thr,
                            "tail_thr": "inf" if not math.isfinite(tail_thr) else tail_thr,
                            "rollout_horizon": horizon,
                            "residual_alpha": alpha,
                            "passivity_gate_pi": PASSH_GATE_PI,
                            "passivity_damp": PASSH_DAMP,
                            "val_Accuracy": metrics["Accuracy"],
                            "val_BalancedAccuracy": metrics["BalancedAccuracy"],
                            "val_FSR": metrics["FSR"],
                            "val_FUR": metrics["FUR"],
                            "val_tp": metrics["tp"],
                            "val_fn": metrics["fn"],
                            "val_fp": metrics["fp"],
                            "val_tn": metrics["tn"],
                            "binary_total": metrics["binary_total"],
                            "calib_line_mean_Accuracy": metrics["calib_line_mean_Accuracy"],
                            "calib_line_min_Accuracy": metrics["calib_line_min_Accuracy"],
                            "calib_line_mean_BalancedAccuracy": metrics["calib_line_mean_BalancedAccuracy"],
                            "calib_line_min_BalancedAccuracy": metrics["calib_line_min_BalancedAccuracy"],
                            "calib_line_max_FSR": metrics["calib_line_max_FSR"],
                            "calib_line_max_FUR": metrics["calib_line_max_FUR"],
                            "calib_n_lines": metrics["calib_n_lines"],
                        }
                        all_grid.append(row)
                        score = (
                            metrics["calib_line_mean_Accuracy"],
                            metrics["calib_line_mean_BalancedAccuracy"],
                            -metrics["calib_line_max_FUR"],
                            -metrics["calib_line_max_FSR"],
                            metrics["Accuracy"],
                            alpha,
                            horizon,
                            -angle_thr,
                            -freq_thr,
                            -(tail_thr if math.isfinite(tail_thr) else 1e9),
                        )
                        if best is None or score > best["_score"]:
                            best = {**row, "_score": score}
    assert best is not None
    best.pop("_score", None)
    return best, all_grid


def _torch_rollout_m2a(params, seq, base, horizon: int):
    theta_eff = params * _torch_rollout_m2a.mask
    node_theta, edge_theta = split_theta(theta_eff, two_tier.N_NODES, two_tier.N_EDGES)
    B = base["B"]
    delta = seq.delta[0].clone()
    omega = seq.omega[0].clone()
    pd, pw = [], []
    for k in range(horizon):
        dw = swing_acc_torch(delta, omega, seq.dpm[k], seq, base)
        eta = delta @ B
        dsin = torch.sin(eta) - seq.sin0
        dcos = torch.cos(eta) - seq.cos0
        domega_e = omega @ B
        node_feat = torch.stack([
            omega,
            torch.abs(omega) * omega,
            omega ** 2,
            seq.dpm[k],
            seq.dpm[k] * omega,
            seq.ddpm[k],
        ], dim=-1)
        edge_feat = torch.stack([dsin, dcos, domega_e, domega_e * dsin, domega_e * dcos], dim=-1)
        residual = torch.sum(node_feat * node_theta, dim=-1) + torch.sum(edge_feat * edge_theta, dim=-1) @ B.T
        dw = dw + residual
        dt_v = float(seq.t[k + 1] - seq.t[k])
        omega_next = omega + dt_v * dw
        delta_next = delta + 0.5 * dt_v * (omega + omega_next)
        delta, omega = delta_next, omega_next
        pd.append(delta)
        pw.append(omega)
    return torch.stack(pd), torch.stack(pw)


def _torch_rollout_lyap(params, seq, base, horizon: int):
    M = base["M"]
    delta = seq.delta[0].clone()
    omega = seq.omega[0].clone()
    pd, pw = [], []
    for k in range(horizon):
        dw = swing_acc_torch(delta, omega, seq.dpm[k], seq, base)
        dr, _, _ = lyap_old.m2a_lyap_residual_torch(params, delta, omega, seq.dpm[k], seq.ddpm[k], base)
        dr = lyap_old.coi_project(dr, M)
        dw = dw + dr
        dt_v = float(seq.t[k + 1] - seq.t[k])
        omega_next = omega + dt_v * dw
        delta_next = delta + 0.5 * dt_v * (omega + omega_next)
        delta, omega = delta_next, omega_next
        pd.append(delta)
        pw.append(omega)
    return torch.stack(pd), torch.stack(pw)


def _cls_finetune(
    method: str,
    init_params: np.ndarray,
    bm,
    train_rows: list[dict[str, Any]],
    train_seqs: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    val_seqs: list[dict[str, Any]],
    model_dir: Path,
    seed: int,
    epochs: int,
    resume: bool,
) -> dict[str, Any]:
    if torch is None:
        raise RuntimeError("torch unavailable")
    model_dir.mkdir(parents=True, exist_ok=True)
    result_path = model_dir / f"seed{seed}_cls_result.json"
    params_path = model_dir / f"seed{seed}_cls_params.npz"
    cache_signature = {
        "method": method,
        "seed": seed,
        "epochs": epochs,
        "train_signature": sequence_signature(train_seqs),
        "val_signature": sequence_signature(val_seqs),
        "init_sha1": hashlib.sha1(np.asarray(init_params, dtype=np.float64).tobytes()).hexdigest(),
        "code_fingerprint": code_fingerprint(),
    }
    if resume and result_path.exists() and params_path.exists():
        meta = read_json(result_path)
        if meta.get("cache_signature") == cache_signature:
            return {**meta, "params_c4": np.load(params_path)["params_c4"]}

    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = "cpu"
    base = baseline_tensors(bm, device)
    train_area = aggregate_sequences(train_seqs, two_tier.ACTIVE_GROUPS, M_PRIOR)
    val_area = aggregate_sequences(val_seqs, two_tier.ACTIVE_GROUPS, M_PRIOR)
    train_items = [
        (row, to_torch_seq(seq, device=device))
        for row, seq in zip(train_rows, train_area)
        if binary_label(row.get("label")) is not None
    ]
    val_items = [
        (row, to_torch_seq(seq, device=device))
        for row, seq in zip(val_rows, val_area)
        if binary_label(row.get("label")) is not None
    ]
    if len(train_items) < 4:
        raise RuntimeError("insufficient binary train cases for CLS fine-tune")

    if method == "M2A":
        fnames = feature_names(two_tier.N_NODES, False)
        _torch_rollout_m2a.mask = torch.tensor(make_mask(fnames, ["full"]), dtype=torch.float64, device=device)
        rollout_fn = _torch_rollout_m2a
    elif method == "M2A-LYAP":
        rollout_fn = _torch_rollout_lyap
    else:
        raise ValueError(method)

    params = torch.nn.Parameter(torch.tensor(np.asarray(init_params), dtype=torch.float64, device=device))
    opt = torch.optim.AdamW([params], lr=5e-4, weight_decay=1e-5)
    t0 = time.time()
    best_score = -float("inf")
    best_params = params.detach().cpu().numpy().copy()
    divs = 0
    for ep in range(epochs):
        random.shuffle(train_items)
        opt.zero_grad(set_to_none=True)
        loss = torch.tensor(0.0, dtype=torch.float64, device=device)
        used = 0
        for row, seq in train_items[: min(8, len(train_items))]:
            horizon = min(200, len(seq.t) - 2)
            if horizon < 20:
                continue
            try:
                pd, pw = rollout_fn(params, seq, base, horizon)
            except Exception:
                divs += 1
                continue
            target_d = seq.delta[1:horizon + 1]
            target_w = seq.omega[1:horizon + 1]
            traj_loss = torch.mean((pw - target_w) ** 2) + 0.03 * torch.mean((pd - target_d) ** 2)
            sep_ratio = torch.max(torch.max(pd, dim=1).values - torch.min(pd, dim=1).values) / math.pi
            freq_pu = torch.max(torch.abs(pw)) / OMEGA_S
            tail_len = max(5, horizon // 10)
            tail = torch.mean(torch.max(torch.abs(pw[-tail_len:]), dim=1).values)
            head = torch.mean(torch.max(torch.abs(pw[:tail_len]), dim=1).values)
            tail_ratio = tail / torch.clamp(head, min=1e-9)
            logit = 6.0 * (sep_ratio - 1.0) + 8.0 * (freq_pu - 0.05) + 0.5 * (tail_ratio - 2.0)
            y = torch.tensor(float(binary_label(row.get("label"))), dtype=torch.float64, device=device)
            cls_loss = F.binary_cross_entropy_with_logits(logit, y)
            loss = loss + traj_loss + 0.2 * cls_loss
            if method == "M2A-LYAP":
                loss = loss + 1e-3 * torch.abs(params).sum()
            used += 1
        if used == 0:
            continue
        loss = loss / used
        if not torch.isfinite(loss):
            divs += 1
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_([params], 5.0)
        opt.step()
        if ep % 10 == 0 or ep == epochs - 1:
            with torch.no_grad():
                correct = total = 0
                for row, seq in val_items[: min(16, len(val_items))]:
                    horizon = min(200, len(seq.t) - 2)
                    if horizon < 20:
                        continue
                    try:
                        pd, pw = rollout_fn(params, seq, base, horizon)
                        sep_ratio = torch.max(torch.max(pd, dim=1).values - torch.min(pd, dim=1).values) / math.pi
                        freq_pu = torch.max(torch.abs(pw)) / OMEGA_S
                        pred = 1 if float(sep_ratio) > 1.0 or float(freq_pu) > 0.05 else 0
                    except Exception:
                        pred = 1
                    true = int(binary_label(row.get("label")))
                    correct += int(pred == true)
                    total += 1
                score = correct / max(total, 1)
                if score > best_score:
                    best_score = score
                    best_params = params.detach().cpu().numpy().copy()

    meta = {
        "method": f"{method}-CLS",
        "base_method": method,
        "seed": seed,
        "epochs": epochs,
        "best_val_proxy_accuracy": best_score,
        "divergences": divs,
        "time_sec": time.time() - t0,
        "cache_signature": cache_signature,
    }
    np.savez(params_path, params_c4=best_params)
    write_json(result_path, meta)
    return {**meta, "params_c4": best_params}


def _stage_config(
    mode: str,
    quick: bool,
    seeds: list[int],
    epochs_m2a: int,
    epochs_lyap: int,
    epochs_cls: int,
    run_cls: bool,
    run_passh_val: bool,
    run_passh_robust: bool,
    passh_only: bool,
    passh_robust_only: bool,
    reuse_310_models: bool,
) -> dict[str, Any]:
    return {
        "task": "3.11 Protocol B M2A/M2A-LYAP accuracy improvement",
        "mode": mode,
        "quick": quick,
        "seeds": seeds,
        "epochs_m2a": epochs_m2a,
        "epochs_m2a_lyap": epochs_lyap,
        "epochs_cls": epochs_cls,
        "run_cls": run_cls,
        "run_passh": run_passh_val or run_passh_robust,
        "run_passh_val": run_passh_val,
        "run_passh_robust": run_passh_robust,
        "passh_only": passh_only,
        "passh_robust_only": passh_robust_only,
        "reuse_310_models": reuse_310_models,
        "reuse_310_dir": str(RESULT_310_DIR.relative_to(PROJECT_ROOT)),
        "m0_anchors": M0_ANCHORS,
        "passh_horizons": PASSH_HORIZONS,
        "passh_residual_alphas": PASSH_ALPHAS,
        "passh_gate_pi": PASSH_GATE_PI,
        "passh_damp": PASSH_DAMP,
        "passh_robust_objective": "line_mean_accuracy_then_balanced_accuracy_long_horizon",
        "angle_thr_pi": ANGLE_THR_PI,
        "freq_thr_pu": FREQ_THR_PU,
        "tail_thr": ["inf" if not math.isfinite(v) else v for v in TAIL_THR],
        "protocol_b_splits": PROTOCOL_B_SPLITS,
        "pm_mode": PM_MODE,
        "code_fingerprint": code_fingerprint(),
        "dataset_fingerprint": dataset_fingerprint(),
        "source_310_dir": str(THREE_TEN_DIR.relative_to(PROJECT_ROOT)),
    }


def run_experiments(args) -> None:
    quick = bool(args.quick_smoke)
    mode = "quick-smoke" if quick else "full"
    out_dir = QUICK_RESULT_DIR if quick else RESULT_DIR
    report_path = QUICK_REPORT_PATH if quick else REPORT_PATH
    seeds = SEEDS_SMOKE if quick else SEEDS_FULL
    epochs_m2a = args.quick_epochs_m2a if quick else args.epochs_m2a
    epochs_lyap = args.quick_epochs_lyap if quick else args.epochs_lyap
    epochs_cls = args.quick_epochs_cls if quick else args.epochs_cls
    run_cls = bool(torch is not None and not args.skip_cls and (args.run_cls or args.run_all))
    run_passh_val = bool(not args.skip_passh and not args.passh_robust_only)
    run_passh_robust = bool(not args.skip_passh and not args.skip_passh_robust)
    passh_only = bool(args.passh_only or args.passh_robust_only)
    passh_robust_only = bool(args.passh_robust_only)
    reuse_310_models = bool(args.reuse_310_models)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = _stage_config(
        mode, quick, seeds, epochs_m2a, epochs_lyap, epochs_cls,
        run_cls, run_passh_val, run_passh_robust, passh_only, passh_robust_only, reuse_310_models,
    )
    write_json(out_dir / "run_config.json", config)

    full_rows: list[dict[str, Any]] = []
    per_case_rows: list[dict[str, Any]] = []
    calibrator_rows: list[dict[str, Any]] = []

    for dataset in protocol_b_datasets(quick=quick):
        split_id = dataset["split_id"]
        print(f"=== 3.11 Protocol B {split_id} ===", flush=True)
        if reuse_310_models:
            model_root = RESULT_310_DIR / "our_methods" / "protocol_B" / split_id
            bm = load_model(model_root / "M0" / "model.json")
            for seed in seeds:
                for base_method, val_method, robust_method in (
                    ("M2A", "M2A-PASSH", "M2A-PASSH-R"),
                    ("M2A-LYAP", "M2A-LYAP-PASSH", "M2A-LYAP-PASSH-R"),
                ):
                    params_path = model_root / base_method / f"seed{seed}_params.npz"
                    if not params_path.exists():
                        raise FileNotFoundError(params_path)
                    params = np.load(params_path)["params_c4"]
                    if run_passh_val:
                        calib, grid = _calibrate_passh(
                            base_method, val_method, split_id, seed, bm, params,
                            dataset["val_rows"], dataset["val"], "3.10_all",
                        )
                        calibrator_rows.extend(grid)
                        test_cases = _run_passh_cases(
                            base_method, val_method, split_id, seed, bm, params,
                            dataset["test_rows"], dataset["test"], "stability",
                            int(calib["rollout_horizon"]), float(calib["residual_alpha"]),
                        )
                        summary, cases = _apply_calibrator(test_cases, calib, val_method, seed, "3.10_all")
                        full_rows.append(summary)
                        per_case_rows.extend(cases)
                    if run_passh_robust:
                        calib, grid = _calibrate_passh_line_balanced(
                            base_method, robust_method, split_id, seed, bm, params,
                            dataset["train_rows"] + dataset["val_rows"],
                            dataset["train"] + dataset["val"],
                            "3.10_all",
                        )
                        calibrator_rows.extend(grid)
                        test_cases = _run_passh_cases(
                            base_method, robust_method, split_id, seed, bm, params,
                            dataset["test_rows"], dataset["test"], "stability",
                            int(calib["rollout_horizon"]), float(calib["residual_alpha"]),
                        )
                        summary, cases = _apply_calibrator(test_cases, calib, robust_method, seed, "3.10_all")
                        full_rows.append(summary)
                        per_case_rows.extend(cases)
            _write_outputs(out_dir, report_path, mode, full_rows, per_case_rows, calibrator_rows, quick, run_cls)
            continue
        for anchor in M0_ANCHORS:
            anchor_train = _anchor_train(dataset["train_rows"], dataset["train"], anchor)
            if len(anchor_train) < 4:
                print(f"[{split_id}/{anchor}] skip M0 anchor with {len(anchor_train)} cases", flush=True)
                continue
            model_root = out_dir / "our_methods" / f"protocol_B" / split_id / f"M0_{anchor}"
            bm, m0_path = fit_m0(anchor_train, model_root / "M0", resume=args.resume)

            val_m0_cases = _run_rollout_cases("M0", split_id, 0, bm, None, dataset["val_rows"], dataset["val"], "validation")
            test_m0_cases = _run_rollout_cases("M0", split_id, 0, bm, None, dataset["test_rows"], dataset["test"], "stability")
            calib, grid = _calibrate_thresholds("B", split_id, "M0-BCal", 0, anchor, val_m0_cases)
            calibrator_rows.extend(grid)
            summary, cases = _apply_calibrator(test_m0_cases, calib, "M0-BCal", 0, anchor)
            full_rows.append(summary)
            per_case_rows.extend(cases)

            for seed in seeds:
                print(f"[{split_id}/{anchor}] M2A seed={seed}", flush=True)
                m2a = train_or_load_m2a(dataset["train"], dataset["val"], dataset["test"], seed, model_root / "M2A", m0_path, epochs_m2a, args.resume)
                val_cases = _run_rollout_cases("M2A", split_id, seed, bm, m2a["params_c4"], dataset["val_rows"], dataset["val"], "validation")
                test_cases = _run_rollout_cases("M2A", split_id, seed, bm, m2a["params_c4"], dataset["test_rows"], dataset["test"], "stability")
                calib, grid = _calibrate_thresholds("B", split_id, "M2A-BCal", seed, anchor, val_cases)
                calibrator_rows.extend(grid)
                summary, cases = _apply_calibrator(test_cases, calib, "M2A-BCal", seed, anchor)
                full_rows.append(summary)
                per_case_rows.extend(cases)
                if run_passh_val:
                    calib, grid = _calibrate_passh(
                        "M2A", "M2A-PASSH", split_id, seed, bm, m2a["params_c4"],
                        dataset["val_rows"], dataset["val"], anchor,
                    )
                    calibrator_rows.extend(grid)
                    test_cases = _run_passh_cases(
                        "M2A", "M2A-PASSH", split_id, seed, bm, m2a["params_c4"],
                        dataset["test_rows"], dataset["test"], "stability", int(calib["rollout_horizon"]), float(calib["residual_alpha"]),
                    )
                    summary, cases = _apply_calibrator(test_cases, calib, "M2A-PASSH", seed, anchor)
                    full_rows.append(summary)
                    per_case_rows.extend(cases)
                if run_passh_robust:
                    calib, grid = _calibrate_passh_line_balanced(
                        "M2A", "M2A-PASSH-R", split_id, seed, bm, m2a["params_c4"],
                        dataset["train_rows"] + dataset["val_rows"],
                        dataset["train"] + dataset["val"],
                        anchor,
                    )
                    calibrator_rows.extend(grid)
                    test_cases = _run_passh_cases(
                        "M2A", "M2A-PASSH-R", split_id, seed, bm, m2a["params_c4"],
                        dataset["test_rows"], dataset["test"], "stability", int(calib["rollout_horizon"]), float(calib["residual_alpha"]),
                    )
                    summary, cases = _apply_calibrator(test_cases, calib, "M2A-PASSH-R", seed, anchor)
                    full_rows.append(summary)
                    per_case_rows.extend(cases)

                print(f"[{split_id}/{anchor}] M2A-LYAP seed={seed}", flush=True)
                ly = train_or_load_m2a_lyap(dataset["train"], dataset["val"], dataset["test"], seed, model_root / "M2A-LYAP", m0_path, epochs_lyap, args.resume)
                val_cases = _run_rollout_cases("M2A-LYAP", split_id, seed, bm, ly["params_c4"], dataset["val_rows"], dataset["val"], "validation")
                test_cases = _run_rollout_cases("M2A-LYAP", split_id, seed, bm, ly["params_c4"], dataset["test_rows"], dataset["test"], "stability")
                calib, grid = _calibrate_thresholds("B", split_id, "M2A-LYAP-BCal", seed, anchor, val_cases)
                calibrator_rows.extend(grid)
                summary, cases = _apply_calibrator(test_cases, calib, "M2A-LYAP-BCal", seed, anchor)
                full_rows.append(summary)
                per_case_rows.extend(cases)
                if run_passh_val:
                    calib, grid = _calibrate_passh(
                        "M2A-LYAP", "M2A-LYAP-PASSH", split_id, seed, bm, ly["params_c4"],
                        dataset["val_rows"], dataset["val"], anchor,
                    )
                    calibrator_rows.extend(grid)
                    test_cases = _run_passh_cases(
                        "M2A-LYAP", "M2A-LYAP-PASSH", split_id, seed, bm, ly["params_c4"],
                        dataset["test_rows"], dataset["test"], "stability", int(calib["rollout_horizon"]), float(calib["residual_alpha"]),
                    )
                    summary, cases = _apply_calibrator(test_cases, calib, "M2A-LYAP-PASSH", seed, anchor)
                    full_rows.append(summary)
                    per_case_rows.extend(cases)
                if run_passh_robust:
                    calib, grid = _calibrate_passh_line_balanced(
                        "M2A-LYAP", "M2A-LYAP-PASSH-R", split_id, seed, bm, ly["params_c4"],
                        dataset["train_rows"] + dataset["val_rows"],
                        dataset["train"] + dataset["val"],
                        anchor,
                    )
                    calibrator_rows.extend(grid)
                    test_cases = _run_passh_cases(
                        "M2A-LYAP", "M2A-LYAP-PASSH-R", split_id, seed, bm, ly["params_c4"],
                        dataset["test_rows"], dataset["test"], "stability", int(calib["rollout_horizon"]), float(calib["residual_alpha"]),
                    )
                    summary, cases = _apply_calibrator(test_cases, calib, "M2A-LYAP-PASSH-R", seed, anchor)
                    full_rows.append(summary)
                    per_case_rows.extend(cases)

                if run_cls:
                    print(f"[{split_id}/{anchor}] CLS fine-tune seed={seed}", flush=True)
                    cls_m2a = _cls_finetune("M2A", m2a["params_c4"], bm, dataset["train_rows"], dataset["train"], dataset["val_rows"], dataset["val"], model_root / "M2A-CLS", seed, epochs_cls, args.resume)
                    val_cases = _run_rollout_cases("M2A", split_id, seed, bm, cls_m2a["params_c4"], dataset["val_rows"], dataset["val"], "validation")
                    test_cases = _run_rollout_cases("M2A", split_id, seed, bm, cls_m2a["params_c4"], dataset["test_rows"], dataset["test"], "stability")
                    calib, grid = _calibrate_thresholds("B", split_id, "M2A-CLS", seed, anchor, val_cases)
                    calibrator_rows.extend(grid)
                    summary, cases = _apply_calibrator(test_cases, calib, "M2A-CLS", seed, anchor)
                    full_rows.append(summary)
                    per_case_rows.extend(cases)

                    cls_ly = _cls_finetune("M2A-LYAP", ly["params_c4"], bm, dataset["train_rows"], dataset["train"], dataset["val_rows"], dataset["val"], model_root / "M2A-LYAP-CLS", seed, epochs_cls, args.resume)
                    val_cases = _run_rollout_cases("M2A-LYAP", split_id, seed, bm, cls_ly["params_c4"], dataset["val_rows"], dataset["val"], "validation")
                    test_cases = _run_rollout_cases("M2A-LYAP", split_id, seed, bm, cls_ly["params_c4"], dataset["test_rows"], dataset["test"], "stability")
                    calib, grid = _calibrate_thresholds("B", split_id, "M2A-LYAP-CLS", seed, anchor, val_cases)
                    calibrator_rows.extend(grid)
                    summary, cases = _apply_calibrator(test_cases, calib, "M2A-LYAP-CLS", seed, anchor)
                    full_rows.append(summary)
                    per_case_rows.extend(cases)

            _write_outputs(out_dir, report_path, mode, full_rows, per_case_rows, calibrator_rows, quick, run_cls)

    _write_outputs(out_dir, report_path, mode, full_rows, per_case_rows, calibrator_rows, quick, run_cls)
    print(f"3.11 complete: results={out_dir} report={report_path}", flush=True)


def report_only(args) -> None:
    quick = bool(args.quick_report)
    mode = "quick-smoke" if quick else "full"
    out_dir = QUICK_RESULT_DIR if quick else RESULT_DIR
    report_path = QUICK_REPORT_PATH if quick else REPORT_PATH
    if not (out_dir / "protocol_b_full_grid_results.csv").exists():
        raise FileNotFoundError(f"Missing existing outputs in {out_dir}")
    full_rows = read_csv(out_dir / "protocol_b_full_grid_results.csv")
    per_case_rows = read_csv(out_dir / "per_case_predictions.csv")
    calibrator_rows = read_csv(out_dir / "calibrator_grid_results.csv")
    config_path = out_dir / "run_config.json"
    run_cls = bool(read_json(config_path).get("run_cls", False)) if config_path.exists() else bool(args.run_cls)
    _write_outputs(out_dir, report_path, mode, full_rows, per_case_rows, calibrator_rows, quick, run_cls=run_cls)
    print(f"3.11 report regenerated: {report_path}", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Task 3.11 Protocol B M2A/M2A-LYAP accuracy improvement")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--quick-smoke", action="store_true")
    group.add_argument("--run-all", action="store_true")
    group.add_argument("--report-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-cls", action="store_true", help="Run classification-loss fine tuning during quick smoke.")
    parser.add_argument("--skip-cls", action="store_true", help="Skip classification-loss fine tuning during full runs.")
    parser.add_argument("--skip-passh", action="store_true", help="Skip passivity-projected finite-horizon rollout variants.")
    parser.add_argument("--skip-passh-robust", action="store_true", help="Skip train+val line-balanced PASSH-R calibration.")
    parser.add_argument("--passh-only", action="store_true", help="Run only PASSH variants.")
    parser.add_argument("--passh-robust-only", action="store_true", help="Run only PASSH-R variants when possible.")
    parser.add_argument("--reuse-310-models", action="store_true", help="Reuse trained Protocol B M0/M2A/M2A-LYAP parameters from the 3.10 result directory.")
    parser.add_argument("--quick-report", action="store_true", help="With --report-only, regenerate the quick-smoke report.")
    parser.add_argument("--epochs-m2a", type=int, default=1500)
    parser.add_argument("--epochs-lyap", type=int, default=500)
    parser.add_argument("--epochs-cls", type=int, default=120)
    parser.add_argument("--quick-epochs-m2a", type=int, default=80)
    parser.add_argument("--quick-epochs-lyap", type=int, default=60)
    parser.add_argument("--quick-epochs-cls", type=int, default=20)
    args = parser.parse_args(argv)
    if args.report_only:
        report_only(args)
    else:
        run_experiments(args)


if __name__ == "__main__":
    main()
