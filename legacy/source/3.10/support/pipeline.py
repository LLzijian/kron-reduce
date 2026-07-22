"""Final 3.10 Protocol A/B paper-comparison pipeline."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any

from support.audit import audit_outputs
from support.common import (
    PM_MODE,
    QUICK_REPORT_PATH,
    QUICK_RESULT_DIR,
    REPORT_PATH,
    RESULT_DIR,
    SEEDS_FULL,
    SEEDS_SMOKE,
    WINDOWS_FULL,
    WINDOWS_SMOKE,
    aggregate_runs,
    code_fingerprint,
    dataset_fingerprint,
    read_csv,
    read_json,
    sha256_file,
    write_csv,
    write_json,
)
from support.data_protocol import PROTOCOL_B_SPLITS, dataset_audit, protocol_a_dataset, protocol_b_datasets
from support.data_protocol import cleaning_summary_rows
from support.reporting import write_capability_csv, write_report

from comparison_methods.evaluation import run_protocol_b_classifiers, run_rollout_comparisons
from our_methods.evaluation import run_our_methods


ACCEPTED_SUMMARY_STATUSES = {"ok", "degraded"}


def _status_ok(rows: list[dict[str, Any]], grid_kind: str) -> list[dict[str, Any]]:
    return [r for r in rows if r.get("grid_kind") == grid_kind and r.get("status") in ACCEPTED_SUMMARY_STATUSES]


def _aggregate_stability(rows: list[dict[str, Any]], protocol: str) -> list[dict[str, Any]]:
    group_keys = ["protocol", "method", "family", "variant", "window_len"]
    if protocol == "B":
        group_keys = ["protocol", "method", "family", "variant", "window_len"]
    return aggregate_runs(
        _status_ok(rows, "stability"),
        group_keys,
        ["FSR", "FUR", "Accuracy", "BalancedAccuracy", "rollout_failure_rate"],
    )


def _aggregate_trajectory(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return aggregate_runs(
        _status_ok(rows, "trajectory"),
        ["protocol", "method", "family", "variant", "window_len"],
        ["rollout_r2", "mean_omega_r2", "rmse", "failure_rate"],
    )


def _aggregate_b_window(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    direct = [r for r in _status_ok(rows, "stability") if r.get("family") == "direct_classifier"]
    return aggregate_runs(
        direct,
        ["protocol", "method", "family", "variant", "window_len"],
        ["FSR", "FUR", "Accuracy", "BalancedAccuracy"],
    )


def _audit_table(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not findings:
        return [{"severity": "OK", "check": "all", "message": "No audit findings."}]
    return [
        {
            "severity": f.get("severity", ""),
            "check": f.get("check", ""),
            "message": f.get("message", ""),
            "evidence": str(f.get("evidence", ""))[:500],
        }
        for f in findings
    ]


def _grid_key(row: dict[str, Any]) -> tuple[str, str, str, str, str, str, str]:
    return (
        str(row.get("protocol", "")),
        str(row.get("split_id", "")),
        str(row.get("method", "")),
        str(row.get("family", "")),
        str(row.get("variant", "")),
        str(row.get("window_len", "")),
        str(row.get("seed", "")),
    )


def _backfill_rollout_failure_stats(full_grid: list[dict[str, Any]], per_case: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, str, str, str], list[dict[str, Any]]] = {}
    for row in per_case:
        if row.get("eval_kind") == "stability":
            groups.setdefault(_grid_key(row), []).append(row)
    out = []
    for row in full_grid:
        rec = dict(row)
        if rec.get("grid_kind") == "stability" and rec.get("status") in {"ok", "degraded"}:
            group = groups.get(_grid_key(rec), [])
            failed = sum(1 for r in group if str(r.get("failed", "0")) == "1")
            if group:
                rec["rollout_failed_cases"] = failed
                rec["rollout_failure_rate"] = failed / max(len(group), 1)
                if failed and rec.get("status") == "ok":
                    rec["status"] = "degraded"
        out.append(rec)
    return out


def write_artifact_hashes(out_dir: Path, report_path: Path) -> None:
    paths = [
        Path(__file__),
        out_dir / "run_config.json",
        out_dir / "cleaning_diagnostics.csv",
        out_dir / "cleaning_diagnostics_summary.csv",
        out_dir / "protocol_a_full_grid_results.csv",
        out_dir / "protocol_a_stability_summary.csv",
        out_dir / "protocol_a_trajectory_summary.csv",
        out_dir / "protocol_b_full_grid_results.csv",
        out_dir / "protocol_b_stability_summary.csv",
        out_dir / "protocol_b_window_sweep_summary.csv",
        out_dir / "per_case_predictions.csv",
        out_dir / "method_capability_table.csv",
        out_dir / "audit_findings.json",
        out_dir / "audit_findings_table.csv",
        report_path,
    ]
    rows = []
    for path in paths:
        if path.exists() and path.is_file():
            rows.append({"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    write_csv(out_dir / "artifact_hashes.csv", rows, ["path", "sha256", "bytes"])


def _write_cleaning_outputs(out_dir: Path, quick: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    diagnostics: list[dict[str, Any]] = []
    pa = protocol_a_dataset(quick=quick)
    diagnostics.extend(pa.get("cleaning_diagnostics", []))
    for pb in protocol_b_datasets(quick=quick):
        diagnostics.extend(pb.get("cleaning_diagnostics", []))
    summary = cleaning_summary_rows(diagnostics)
    write_csv(out_dir / "cleaning_diagnostics.csv", diagnostics)
    write_csv(out_dir / "cleaning_diagnostics_summary.csv", summary)
    return diagnostics, summary


def _write_all_outputs(
    out_dir: Path,
    a_full: list[dict[str, Any]],
    b_full: list[dict[str, Any]],
    per_case: list[dict[str, Any]],
) -> None:
    a_full = _backfill_rollout_failure_stats(a_full, per_case)
    b_full = _backfill_rollout_failure_stats(b_full, per_case)
    write_csv(out_dir / "protocol_a_full_grid_results.csv", a_full)
    write_csv(out_dir / "protocol_a_stability_summary.csv", _aggregate_stability(a_full, "A"))
    write_csv(out_dir / "protocol_a_trajectory_summary.csv", _aggregate_trajectory(a_full))
    write_csv(out_dir / "protocol_b_full_grid_results.csv", b_full)
    write_csv(out_dir / "protocol_b_stability_summary.csv", _aggregate_stability(b_full, "B"))
    write_csv(out_dir / "protocol_b_window_sweep_summary.csv", _aggregate_b_window(b_full))
    write_csv(out_dir / "per_case_predictions.csv", per_case)
    write_capability_csv(out_dir)


def _load_existing_outputs(out_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    return (
        [dict(r) for r in read_csv(out_dir / "protocol_a_full_grid_results.csv")],
        [dict(r) for r in read_csv(out_dir / "protocol_b_full_grid_results.csv")],
        [dict(r) for r in read_csv(out_dir / "per_case_predictions.csv")],
    )


def _stage_config(
    quick: bool,
    seeds: list[int],
    windows: list[int],
    epochs_m2a: int,
    epochs_lyap: int,
) -> dict[str, Any]:
    return {
        "quick_smoke": quick,
        "seeds": seeds,
        "windows": windows,
        "epochs_m2a": epochs_m2a,
        "epochs_m2a_lyap": epochs_lyap,
        "pm_mode": PM_MODE,
        "protocol_b_splits": PROTOCOL_B_SPLITS,
        "code_fingerprint": code_fingerprint(),
        "dataset_fingerprint": dataset_fingerprint(),
    }


def _stage_path(out_dir: Path, protocol: str, split_id: str) -> Path:
    return out_dir / "stage_cache" / f"protocol_{protocol.lower()}_{split_id}.json"


def _load_stage(
    out_dir: Path,
    protocol: str,
    split_id: str,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | None:
    path = _stage_path(out_dir, protocol, split_id)
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        payload = read_json(path)
    except Exception:
        return None
    if payload.get("config") != config:
        return None
    print(f"[stage-cache] Protocol {protocol} {split_id}", flush=True)
    return list(payload.get("full_grid", [])), list(payload.get("per_case", []))


def _write_stage(
    out_dir: Path,
    protocol: str,
    split_id: str,
    config: dict[str, Any],
    full_grid: list[dict[str, Any]],
    per_case: list[dict[str, Any]],
) -> None:
    write_json(_stage_path(out_dir, protocol, split_id), {
        "protocol": protocol,
        "split_id": split_id,
        "config": config,
        "full_grid": full_grid,
        "per_case": per_case,
    })


def run_experiments(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    quick = bool(args.quick_smoke)
    out_dir = QUICK_RESULT_DIR if quick else RESULT_DIR
    seeds = SEEDS_SMOKE if quick else SEEDS_FULL
    windows = WINDOWS_SMOKE if quick else WINDOWS_FULL
    epochs_m2a = args.quick_epochs_m2a if quick else args.epochs_m2a
    epochs_lyap = args.quick_epochs_lyap if quick else args.epochs_lyap
    config = _stage_config(quick, seeds, windows, epochs_m2a, epochs_lyap)

    a_full: list[dict[str, Any]] = []
    b_full: list[dict[str, Any]] = []
    per_case: list[dict[str, Any]] = []

    print("=== Protocol A: Blind External Generalization ===", flush=True)
    cached = _load_stage(out_dir, "A", "A", config) if args.resume else None
    if cached is not None:
        full, pc = cached
        a_full.extend(full)
        per_case.extend(pc)
    else:
        pa = protocol_a_dataset(quick=quick)
        a_traj_rows = [{"scenario_id": seq.get("scenario_id", f"test{i}"), "label": "stable"} for i, seq in enumerate(pa["test"])]
        a_stage_full: list[dict[str, Any]] = []
        a_stage_pc: list[dict[str, Any]] = []
        st, dyn, full, pc = run_our_methods(
            "A", "A", pa["train"], pa["val"], pa["test"], pa["external_rows"], pa["external"],
            seeds, out_dir, epochs_m2a, epochs_lyap, resume=args.resume,
        )
        a_stage_full.extend(full)
        a_stage_pc.extend(pc)
        st, dyn, full, pc = run_rollout_comparisons(
            "A", "A", pa["train"], a_traj_rows, pa["test"], pa["external_rows"], pa["external"],
            seeds, windows, quick=quick,
        )
        a_stage_full.extend(full)
        a_stage_pc.extend(pc)
        _write_stage(out_dir, "A", "A", config, a_stage_full, a_stage_pc)
        a_full.extend(a_stage_full)
        per_case.extend(a_stage_pc)
    if not args.eval_only:
        _write_all_outputs(out_dir, a_full, b_full, per_case)

    print("=== Protocol B: Dense Boundary Fault-Line Group Splits ===", flush=True)
    for pb in protocol_b_datasets(quick=quick):
        split_id = pb["split_id"]
        print(f"--- {split_id} ---", flush=True)
        cached = _load_stage(out_dir, "B", split_id, config) if args.resume else None
        if cached is not None:
            full, pc = cached
            b_full.extend(full)
            per_case.extend(pc)
            if not args.eval_only:
                _write_all_outputs(out_dir, a_full, b_full, per_case)
            continue
        b_stage_full: list[dict[str, Any]] = []
        b_stage_pc: list[dict[str, Any]] = []
        st, dyn, full, pc = run_our_methods(
            "B", split_id, pb["train"], pb["val"], pb["test"], pb["test_rows"], pb["test"],
            seeds, out_dir, epochs_m2a, epochs_lyap, resume=args.resume,
        )
        b_stage_full.extend(full)
        b_stage_pc.extend(pc)
        st, dyn, full, pc = run_rollout_comparisons(
            "B", split_id, pb["train"], pb["test_rows"], pb["test"], pb["test_rows"], pb["test"],
            seeds, windows, quick=quick,
        )
        b_stage_full.extend(full)
        b_stage_pc.extend(pc)
        st, full, pc = run_protocol_b_classifiers(
            split_id, pb["train_rows"], pb["train"], pb["test_rows"], pb["test"],
            seeds, windows, quick=quick,
        )
        b_stage_full.extend(full)
        b_stage_pc.extend(pc)
        _write_stage(out_dir, "B", split_id, config, b_stage_full, b_stage_pc)
        b_full.extend(b_stage_full)
        per_case.extend(b_stage_pc)
        _write_all_outputs(out_dir, a_full, b_full, per_case)

    return a_full, b_full, per_case


def run_pipeline(args: argparse.Namespace) -> None:
    quick = bool(args.quick_smoke)
    mode = "quick-smoke" if quick else ("report-only" if args.report_only else "full")
    out_dir = QUICK_RESULT_DIR if quick else RESULT_DIR
    report_path = QUICK_REPORT_PATH if quick else REPORT_PATH
    if out_dir.exists() and any(out_dir.iterdir()) and not args.resume and not args.report_only:
        raise FileExistsError(f"Refusing to write into non-empty output directory without --resume: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = SEEDS_SMOKE if quick else SEEDS_FULL
    windows = WINDOWS_SMOKE if quick else WINDOWS_FULL
    _cleaning_diagnostics, cleaning_summary = _write_cleaning_outputs(out_dir, quick)
    run_config = {
        "task": "3.10 final paper comparison Protocol A/B",
        "mode": mode,
        "quick_smoke": quick,
        "seeds": seeds,
        "windows": windows,
        "pm_mode": PM_MODE,
        "epochs_m2a": args.quick_epochs_m2a if quick else args.epochs_m2a,
        "epochs_m2a_lyap": args.quick_epochs_lyap if quick else args.epochs_lyap,
        "epochs_lstm_rollout": 8 if quick else 30,
        "epochs_direct_classifiers": 8 if quick else 30,
        "protocol_a": "highorder stable train/val/test plus untouched v1 external stability test",
        "protocol_b": "dense boundary fault-line group split; v1 is not used as a separate test set",
        "data_cleaning": {
            "flatline_tail_truncation": True,
            "default_scope": "Protocol A external v1 and Protocol B dense train/val/test; internal stable sets audited too",
            "cleaning_utility": "discovery/3.2-kron/diag_clean_unstable_trajectories.py",
            "diagnostics_files": ["cleaning_diagnostics.csv", "cleaning_diagnostics_summary.csv"],
            "model_input_policy": "cleaning diagnostics are audit-only and must not be used as model features",
        },
        "cleaning_summary": cleaning_summary,
        "protocol_b_splits": PROTOCOL_B_SPLITS,
        "code_fingerprint": code_fingerprint(),
        "dataset_fingerprint": dataset_fingerprint(),
        "dataset_audit": dataset_audit(quick=quick),
    }
    if args.report_only:
        existing_config = read_json(out_dir / "run_config.json") if (out_dir / "run_config.json").exists() else {}
        mode = str(existing_config.get("mode", mode))
    else:
        write_json(out_dir / "run_config.json", run_config)

    if args.report_only:
        a_full, b_full, per_case = _load_existing_outputs(out_dir)
        _write_all_outputs(out_dir, a_full, b_full, per_case)
    else:
        # run_experiments writes outputs after Protocol A and every Protocol B split.
        # Avoid a redundant final rewrite of the large per-case table.
        a_full, b_full, per_case = run_experiments(args)
        if args.eval_only:
            _write_all_outputs(out_dir, a_full, b_full, per_case)

    # Placeholders make the first report/audit pass self-contained.
    write_json(out_dir / "audit_findings.json", [])
    write_csv(out_dir / "audit_findings_table.csv", _audit_table([]))
    write_artifact_hashes(out_dir, report_path)
    write_report(out_dir, report_path, mode)

    findings = audit_outputs(out_dir, report_path, quick=quick)
    write_json(out_dir / "audit_findings.json", findings)
    write_csv(out_dir / "audit_findings_table.csv", _audit_table(findings))
    write_report(out_dir, report_path, mode)
    write_artifact_hashes(out_dir, report_path)

    errors = [f for f in findings if f.get("severity") == "ERROR"]
    warnings = [f for f in findings if f.get("severity") == "WARNING"]
    print(f"3.10 Protocol A/B comparison complete: {len(errors)} ERROR, {len(warnings)} WARNING")
    print(f"Results: {out_dir}")
    print(f"Report: {report_path}")
    if errors:
        raise SystemExit(2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final 3.10 paper comparison with Protocol A/B")
    parser.add_argument("--quick-smoke", action="store_true", help="Run one seed, two windows, and small balanced subsets.")
    parser.add_argument("--run-all", action="store_true", help="Run the full Protocol A/B grid.")
    parser.add_argument("--eval-only", action="store_true", help="Alias for run-all with resume semantics for this harness.")
    parser.add_argument("--report-only", action="store_true", help="Regenerate summaries, audit, hashes, and report from existing CSVs.")
    parser.add_argument("--resume", action="store_true", help="Reuse cached our-method checkpoints and allow writing into existing result directory.")
    parser.add_argument("--epochs-m2a", type=int, default=1500)
    parser.add_argument("--epochs-lyap", type=int, default=500)
    parser.add_argument("--quick-epochs-m2a", type=int, default=30)
    parser.add_argument("--quick-epochs-lyap", type=int, default=30)
    args = parser.parse_args()
    if not any([args.quick_smoke, args.run_all, args.eval_only, args.report_only]):
        parser.error("Choose one of --quick-smoke, --run-all, --eval-only, --report-only")
    if args.eval_only:
        args.run_all = True
        args.resume = True
    return args


def main() -> None:
    run_pipeline(parse_args())


if __name__ == "__main__":
    main()
