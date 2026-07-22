"""Adversarial audit checks for the final 3.10 Protocol A/B outputs."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from support.common import PM_MODE, binary_label, binary_metrics, read_csv, read_json
from support.data_protocol import PROTOCOL_B_SPLITS, dataset_audit


def _key(row: dict[str, Any]) -> tuple[str, str, str, str, str, str, str]:
    return (
        str(row.get("protocol", "")),
        str(row.get("split_id", "")),
        str(row.get("method", "")),
        str(row.get("family", "")),
        str(row.get("variant", "")),
        str(row.get("window_len", "")),
        str(row.get("seed", "")),
    )


def _find_group(per_case: list[dict[str, Any]], row: dict[str, Any]) -> list[dict[str, Any]]:
    key = _key(row)
    return [r for r in per_case if r.get("eval_kind") == "stability" and _key(r) == key]


def audit_outputs(out_dir: Path, report_path: Path, quick: bool) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    severity_for_full = "WARNING" if quick else "ERROR"
    required = [
        "protocol_a_full_grid_results.csv",
        "protocol_a_stability_summary.csv",
        "protocol_a_trajectory_summary.csv",
        "protocol_b_full_grid_results.csv",
        "protocol_b_stability_summary.csv",
        "protocol_b_window_sweep_summary.csv",
        "per_case_predictions.csv",
        "cleaning_diagnostics.csv",
        "cleaning_diagnostics_summary.csv",
        "method_capability_table.csv",
        "audit_findings.json",
        "artifact_hashes.csv",
    ]
    for name in required:
        path = out_dir / name
        if not path.exists() or path.stat().st_size == 0:
            findings.append({"severity": "ERROR", "check": "missing_artifact", "message": f"Missing {name}", "evidence": {}})

    run_config_path = out_dir / "run_config.json"
    if run_config_path.exists() and run_config_path.stat().st_size > 0:
        try:
            run_config = read_json(run_config_path)
            cleaning = run_config.get("data_cleaning", {})
            if cleaning.get("flatline_tail_truncation") is not True:
                findings.append({
                    "severity": "ERROR",
                    "check": "flatline_cleaning_config",
                    "message": "run_config.json does not declare Protocol A/B flatline-tail truncation.",
                    "evidence": cleaning,
                })
        except Exception as exc:
            findings.append({"severity": "ERROR", "check": "run_config_parse", "message": f"Cannot parse run_config.json: {exc}", "evidence": {}})

    data = dataset_audit(quick=quick)
    if data["dense_contains_v1_cases"] != data["v1_counts"]["total"]:
        findings.append({"severity": "ERROR", "check": "dense_v1_overlap", "message": "Dense should contain all v1 cases per data inventory.", "evidence": data})
    if data["v1_not_in_dense"] != 0:
        findings.append({"severity": "ERROR", "check": "dense_v1_overlap", "message": "Some v1 cases are not in dense.", "evidence": data})
    if data["near_critical_overlap_v1_dense"] <= 0:
        findings.append({"severity": "WARNING", "check": "near_critical_overlap", "message": "Expected near-critical overlap was not detected.", "evidence": data})
    if data["highorder_external_fault_line_overlap"]:
        findings.append({"severity": "ERROR", "check": "protocol_a_fault_line_overlap", "message": "Highorder and v1 fault lines overlap.", "evidence": data["highorder_external_fault_line_overlap"]})

    for split_id, spec in PROTOCOL_B_SPLITS.items():
        train, val, test = set(spec["train"]), set(spec["val"]), set(spec["test"])
        if train & val or train & test or val & test:
            findings.append({"severity": "ERROR", "check": "protocol_b_line_disjoint", "message": f"{split_id} has overlapping fault lines.", "evidence": spec})

    per_case = [dict(r) for r in read_csv(out_dir / "per_case_predictions.csv")]
    per_case_groups: dict[tuple[str, str, str, str, str, str, str], list[dict[str, Any]]] = {}
    for r in per_case:
        if r.get("eval_kind") == "stability":
            per_case_groups.setdefault(_key(r), []).append(r)
    stability_rows = [dict(r) for r in read_csv(out_dir / "protocol_a_full_grid_results.csv") + read_csv(out_dir / "protocol_b_full_grid_results.csv")]
    stability_rows = [r for r in stability_rows if r.get("grid_kind") == "stability" and r.get("status") in {"ok", "degraded"}]
    for row in stability_rows:
        group = per_case_groups.get(_key(row), [])
        if not group:
            findings.append({"severity": "ERROR", "check": "missing_per_case_group", "message": "No per-case rows for stability summary.", "evidence": row})
            continue
        binary = [r for r in group if binary_label(r.get("true_label")) is not None]
        yt = [binary_label(r.get("true_label")) for r in binary]
        yp = [1 if r.get("pred_label") == "unstable" else 0 for r in binary]
        metrics = binary_metrics(yt, yp)
        failed_count = sum(1 for r in group if str(r.get("failed", "0")) == "1")
        for metric in ["tp", "fn", "fp", "tn", "binary_total"]:
            got = int(float(row.get(metric, 0)))
            if got != metrics[metric]:
                findings.append({"severity": "ERROR", "check": "metric_recompute", "message": f"{metric} mismatch from per-case rows.", "evidence": {"summary": row, "expected": metrics}})
        got_failed = int(float(row.get("rollout_failed_cases", 0) or 0))
        if got_failed != failed_count:
            findings.append({"severity": "ERROR", "check": "rollout_failure_recompute", "message": "rollout_failed_cases mismatch from per-case rows.", "evidence": {"summary": row, "expected_failed": failed_count}})
        got_fail_rate = float(row.get("rollout_failure_rate", "0") or 0)
        exp_fail_rate = failed_count / max(len(group), 1)
        if not (math.isfinite(got_fail_rate) and abs(got_fail_rate - exp_fail_rate) <= 1e-9):
            findings.append({"severity": "ERROR", "check": "rollout_failure_recompute", "message": "rollout_failure_rate mismatch from per-case rows.", "evidence": {"summary": row, "expected_failure_rate": exp_fail_rate}})
        for metric in ["FSR", "FUR", "Accuracy", "BalancedAccuracy"]:
            got = float(row.get(metric, "nan"))
            exp = float(metrics[metric])
            if not (math.isfinite(got) and abs(got - exp) <= 1e-9):
                findings.append({"severity": "ERROR", "check": "metric_recompute", "message": f"{metric} mismatch from per-case rows.", "evidence": {"summary": row, "expected": metrics}})

        if row.get("protocol") == "A":
            if int(float(row.get("binary_total", 0))) != 130 or int(float(row.get("non_binary_total", 0))) != 13:
                findings.append({
                    "severity": severity_for_full,
                    "check": "protocol_a_external_coverage",
                    "message": "Protocol A stability row should cover 130 binary + 13 near-critical v1 cases.",
                    "evidence": row,
                })
        if row.get("pm_mode") != PM_MODE:
            findings.append({"severity": "ERROR", "check": "pm_mode", "message": "Summary row has wrong pm_mode.", "evidence": row})

    for r in per_case:
        if r.get("pm_mode") not in ("", PM_MODE):
            findings.append({"severity": "ERROR", "check": "pm_mode_per_case", "message": "Per-case row has wrong pm_mode.", "evidence": r})

    # Expected grid coverage checks.
    a_full = [dict(r) for r in read_csv(out_dir / "protocol_a_full_grid_results.csv")]
    b_full = [dict(r) for r in read_csv(out_dir / "protocol_b_full_grid_results.csv")]
    seeds = {
        r.get("seed")
        for r in a_full + b_full
        if r.get("method") != "M0" and str(r.get("seed", "")) not in ("", "0")
    }
    windows = {str(r.get("window_len")) for r in a_full + b_full if str(r.get("window_len", "")).isdigit()}
    if not quick and seeds != {"324", "325", "326", "327", "328"}:
        findings.append({"severity": "ERROR", "check": "seed_coverage", "message": "Full run does not cover all required seeds.", "evidence": sorted(seeds)})
    if not quick and not {"20", "40", "100", "200"}.issubset(windows):
        findings.append({"severity": "ERROR", "check": "window_coverage", "message": "Full run does not cover all required windows.", "evidence": sorted(windows)})
    if not quick:
        b_splits = {r.get("split_id") for r in b_full}
        if set(PROTOCOL_B_SPLITS) - b_splits:
            findings.append({"severity": "ERROR", "check": "protocol_b_split_coverage", "message": "Missing Protocol B splits.", "evidence": sorted(set(PROTOCOL_B_SPLITS) - b_splits)})

    for path in [report_path, out_dir / "run_config.json"]:
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="ignore")
            legacy_patterns = [
                (".".join(["3", "11", "2"]), re.compile(r"(?<![\d.])3\.11\.2(?![\d.])")),
                (".".join(["3", "11"]), re.compile(r"(?<![\d.])3\.11(?![\d.])")),
                ("run_31" + "1", re.compile(r"run_31" + "1")),
            ]
            legacy = [label for label, pattern in legacy_patterns if pattern.search(text)]
            if legacy:
                findings.append({"severity": "ERROR", "check": "legacy_reference", "message": "Final report/config references legacy result markers.", "evidence": {"path": str(path), "markers": legacy}})
            old_pm_marker = "Pm0" + "/" + ("Z" + "O" + "H")
            if old_pm_marker in text:
                findings.append({"severity": "ERROR", "check": "old_pm_contract_reference", "message": "Final report/config should not use the old constant-mechanical-power contract as main result source.", "evidence": {"path": str(path)}})

    if quick:
        findings.append({"severity": "WARNING", "check": "quick_smoke_scope", "message": "Quick smoke verifies the harness; it is not a paper-number run.", "evidence": {}})
    return findings
