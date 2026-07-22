"""Report and table generation for final 3.10 comparison."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from support.common import PM_MODE, fmt_num, mean_std_text, read_csv, write_csv
from support.data_protocol import PROTOCOL_B_SPLITS, dataset_audit


def build_method_capability_table() -> list[dict[str, Any]]:
    return [
        {
            "method": "M0",
            "reference": "This work baseline; report 3.5",
            "family": "physics_equation_rollout",
            "equation_available": "yes",
            "physical_constraints": "swing backbone; nonnegative D/K after fit",
            "input_form": "clustered 7-area only",
            "required_input_window": "1 point",
            "pm_mode": PM_MODE,
            "rollout_capable": "yes",
            "classifier_capable": "rollout-derived",
        },
        {
            "method": "M2A",
            "reference": "This work residual Graph-SINDy",
            "family": "physics_equation_rollout",
            "equation_available": "yes",
            "physical_constraints": "M0 backbone plus sparse residual library",
            "input_form": "clustered 7-area only",
            "required_input_window": "1 point",
            "pm_mode": PM_MODE,
            "rollout_capable": "yes",
            "classifier_capable": "rollout-derived",
        },
        {
            "method": "M2A-LYAP",
            "reference": "This work Lyapunov-constrained residual; report 3.8.4",
            "family": "physics_equation_rollout",
            "equation_available": "yes",
            "physical_constraints": "M-weighted energy/Lyapunov channel",
            "input_form": "clustered 7-area only",
            "required_input_window": "1 point",
            "pm_mode": PM_MODE,
            "rollout_capable": "yes",
            "classifier_capable": "rollout-derived",
        },
        {
            "method": "SINDYc",
            "reference": "Brunton et al. 2016",
            "family": "sparse_identification",
            "equation_available": "identified surrogate",
            "physical_constraints": "sparsity only",
            "input_form": "clustered and unclustered",
            "required_input_window": "1 point",
            "pm_mode": PM_MODE,
            "rollout_capable": "yes",
            "classifier_capable": "rollout-derived",
        },
        {
            "method": "DMD",
            "reference": "Schmid 2010",
            "family": "operator_forecast",
            "equation_available": "linear operator",
            "physical_constraints": "none",
            "input_form": "clustered and unclustered",
            "required_input_window": "20/40/100/200",
            "pm_mode": PM_MODE,
            "rollout_capable": "forecast",
            "classifier_capable": "rollout-derived",
        },
        {
            "method": "HODMD",
            "reference": "Higher-order DMD literature",
            "family": "operator_forecast",
            "equation_available": "delay linear operator",
            "physical_constraints": "none",
            "input_form": "clustered and unclustered",
            "required_input_window": "20/40/100/200",
            "pm_mode": PM_MODE,
            "rollout_capable": "forecast",
            "classifier_capable": "rollout-derived",
        },
        {
            "method": "EDMD-Koopman",
            "reference": "Williams et al. 2015",
            "family": "operator_forecast",
            "equation_available": "lifted operator",
            "physical_constraints": "dictionary only",
            "input_form": "clustered and unclustered",
            "required_input_window": "20/40/100/200",
            "pm_mode": PM_MODE,
            "rollout_capable": "forecast",
            "classifier_capable": "rollout-derived",
        },
        {
            "method": "LSTM-rollout",
            "reference": "Sequence forecasting baseline",
            "family": "blackbox_rollout",
            "equation_available": "no",
            "physical_constraints": "none",
            "input_form": "clustered and unclustered",
            "required_input_window": "20/40/100/200",
            "pm_mode": PM_MODE,
            "rollout_capable": "forecast",
            "classifier_capable": "rollout-derived",
        },
        {
            "method": "CNN / LSTM / CNN+LSTM / Transformer",
            "reference": "Supervised TSA baselines",
            "family": "direct_classifier",
            "equation_available": "no",
            "physical_constraints": "none",
            "input_form": "clustered and unclustered",
            "required_input_window": "20/40/100/200",
            "pm_mode": PM_MODE,
            "rollout_capable": "no",
            "classifier_capable": "Protocol B only",
        },
    ]


def _table(rows: list[dict[str, Any]], columns: list[str], limit: int | None = None) -> list[str]:
    use_rows = rows if limit is None else rows[:limit]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in use_rows:
        vals = []
        for col in columns:
            val = row.get(col, "")
            if isinstance(val, float):
                vals.append(fmt_num(val))
            else:
                vals.append(str(val) if val not in (None, "") else "-")
        lines.append("| " + " | ".join(vals) + " |")
    if limit is not None and len(rows) > limit:
        lines.append(f"| ... | {len(rows) - limit} more rows in CSV | " + " | ".join("-" for _ in columns[2:]) + " |")
    return lines


def _summary_table(rows: list[dict[str, Any]], metrics: list[str]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        rec = {k: row.get(k, "") for k in ["protocol", "split_id", "method", "family", "variant", "window_len", "n_runs"]}
        for metric in metrics:
            rec[metric] = mean_std_text(row, metric)
        out.append(rec)
    return out


def write_report(out_dir: Path, report_path: Path, mode: str) -> None:
    capability = read_csv(out_dir / "method_capability_table.csv")
    a_stab = read_csv(out_dir / "protocol_a_stability_summary.csv")
    a_traj = read_csv(out_dir / "protocol_a_trajectory_summary.csv")
    b_stab = read_csv(out_dir / "protocol_b_stability_summary.csv")
    b_window = read_csv(out_dir / "protocol_b_window_sweep_summary.csv")
    a_full = read_csv(out_dir / "protocol_a_full_grid_results.csv")
    b_full = read_csv(out_dir / "protocol_b_full_grid_results.csv")
    audit = read_csv(out_dir / "audit_findings_table.csv")
    cleaning_summary = read_csv(out_dir / "cleaning_diagnostics_summary.csv")
    data = dataset_audit(quick=(mode == "quick-smoke"))
    v1_flat = data.get("v1_flatline_audit", {})
    dense_flat = data.get("dense_flatline_audit", {})

    lines: list[str] = [
        "---",
        'title: "3.10 Final Paper Comparison"',
        f'date: "{time.strftime("%Y-%m-%d")}"',
        "status: final-candidate",
        "---",
        "",
        "# 3.10 Final Paper Comparison",
        "",
        f"Run mode: `{mode}`.",
        f"Result directory: `{out_dir}`.",
        "",
        "This report is the replacement final comparison experiment for 3.10. It uses two clean protocols: Protocol A for blind external generalization and Protocol B for dense boundary adaptation with fault-line group splits.",
        "",
        "All rollout and controlled-operator methods receive `P_m(t)` as an observed exogenous input. The controlled input is represented in the centered model as `dpm(t)=P_m(t)-P_m0`, the same convention used by the M0/M2A/M2A-LYAP equations.",
        "",
        "## 1. Experiment Protocol",
        "",
        f"- Highorder stable dataset: `{data['highorder_path']}` with train/val/test = {data['highorder_counts']}.",
        f"- Protocol A external set: `{data['v1_path']}` with counts {data['v1_counts']}; only stable/unstable enter TP/FN/FP/TN.",
        f"- Protocol B dense set: `{data['dense_path']}` with counts {data['dense_counts']}. Dense contains {data['dense_contains_v1_cases']} v1 cases and has {data['dense_extra_cases']} additional boundary cases, so v1 is not reused as a held-out test in Protocol B.",
        f"- ANDES flatline audit: v1 has {v1_flat.get('flatlined_cases', 0)}/{v1_flat.get('total_cases', 0)} flatlined cases; dense has {dense_flat.get('flatlined_cases', 0)}/{dense_flat.get('total_cases', 0)} flatlined cases.",
        "- Flatline tails are truncated before training and rollout/forecast scoring. Cleaning diagnostics are audit-only and are not model inputs.",
        f"- Highorder and v1 fault-line overlap: `{data['highorder_external_fault_line_overlap']}`.",
        "",
        "Protocol B fixed splits:",
        "",
        "| Split | Train fault lines | Val fault line | Test fault lines |",
        "|---|---|---|---|",
    ]
    for split_id, spec in PROTOCOL_B_SPLITS.items():
        lines.append(f"| {split_id} | {', '.join(spec['train'])} | {', '.join(spec['val'])} | {', '.join(spec['test'])} |")

    lines.extend([
        "",
        "## 2. Method Capability And Input Fairness",
        "",
        *_table(capability, ["method", "reference", "family", "input_form", "required_input_window", "pm_mode", "rollout_capable", "classifier_capable"]),
        "",
        "Flatline cleaning diagnostics by protocol split:",
        "",
        *_table(cleaning_summary, ["protocol", "split_id", "data_split", "label", "total", "flatlined", "mean_flatline_fraction", "max_flatline_fraction"], limit=20),
        "",
        "## 3. Protocol A Complete Stability Results",
        "",
        "Protocol A trains on highorder stable train/val and tests stability on untouched v1 external cases. Supervised direct classifiers are marked not applicable here.",
        "",
        *_table(_summary_table(a_stab, ["FSR", "FUR", "Accuracy", "BalancedAccuracy", "rollout_failure_rate"]), ["method", "family", "variant", "window_len", "n_runs", "FSR", "FUR", "Accuracy", "BalancedAccuracy", "rollout_failure_rate"]),
        "",
        "Protocol A per-seed/per-variant stability grid:",
        "",
        *_table([r for r in a_full if r.get("grid_kind") == "stability"], ["method", "family", "variant", "window_len", "seed", "status", "tp", "fn", "fp", "tn", "FSR", "FUR", "Accuracy", "BalancedAccuracy", "rollout_failure_rate"]),
        "",
        "## 4. Protocol A Complete Trajectory Results",
        "",
        "These rows use the highorder internal test split for trajectory rollout/forecast fidelity.",
        "",
        *_table(_summary_table(a_traj, ["rollout_r2", "mean_omega_r2", "rmse", "failure_rate"]), ["method", "family", "variant", "window_len", "n_runs", "rollout_r2", "mean_omega_r2", "rmse", "failure_rate"]),
        "",
        "Protocol A per-seed/per-variant trajectory grid:",
        "",
        *_table([r for r in a_full if r.get("grid_kind") == "trajectory"], ["method", "family", "variant", "window_len", "seed", "status", "rollout_r2", "mean_omega_r2", "worst_case_r2", "rmse", "failure_rate"]),
        "",
        "## 5. Protocol B Complete Stability Results",
        "",
        "Protocol B trains/validates/tests on dense using fixed held-out fault-line group splits. All direct classifiers are evaluated only here.",
        "",
        *_table(_summary_table(b_stab, ["FSR", "FUR", "Accuracy", "BalancedAccuracy", "rollout_failure_rate"]), ["method", "family", "variant", "window_len", "n_runs", "FSR", "FUR", "Accuracy", "BalancedAccuracy", "rollout_failure_rate"]),
        "",
        "Protocol B per-split/per-seed stability grid:",
        "",
        *_table([r for r in b_full if r.get("grid_kind") == "stability"], ["split_id", "method", "family", "variant", "window_len", "seed", "status", "tp", "fn", "fp", "tn", "FSR", "FUR", "Accuracy", "BalancedAccuracy", "rollout_failure_rate"]),
        "",
        "## 6. Protocol B Neural Window Sweep",
        "",
        "This table lists every neural direct-classifier window and input form, aggregated over seeds and fault-line splits.",
        "",
        *_table(_summary_table(b_window, ["FSR", "FUR", "Accuracy", "BalancedAccuracy"]), ["method", "family", "variant", "window_len", "n_runs", "FSR", "FUR", "Accuracy", "BalancedAccuracy"]),
        "",
        "## 7. Protocol B Trajectory / Forecast Grid",
        "",
        "Rollout-capable methods also produce held-out dense trajectory forecasts. Direct classifiers have no trajectory row.",
        "",
        *_table([r for r in b_full if r.get("grid_kind") == "trajectory"], ["split_id", "method", "family", "variant", "window_len", "seed", "status", "rollout_r2", "mean_omega_r2", "worst_case_r2", "rmse", "failure_rate"]),
        "",
        "## 8. Adversarial Audit",
        "",
        *_table(audit, ["severity", "check", "message"]),
        "",
        "## 9. Paper-Ready Conclusions And Limits",
        "",
        "- Protocol A is the clean blind external generalization claim: no dense/v1 boundary cases are used in training.",
        "- Protocol B is an adaptation and early-warning claim: all methods see the same dense train/val fault lines and are tested only on held-out fault lines.",
        "- M0/M2A/M2A-LYAP are 7-area clustered physics methods; comparison methods are reported in both clustered and unclustered forms.",
        "- The direct neural classifiers are not a fair Protocol A comparison because the highorder training set has no unstable labels; they are therefore Protocol B methods.",
        "",
        "## Output Files",
        "",
        "- `protocol_a_full_grid_results.csv`",
        "- `protocol_a_stability_summary.csv`",
        "- `protocol_a_trajectory_summary.csv`",
        "- `protocol_b_full_grid_results.csv`",
        "- `protocol_b_stability_summary.csv`",
        "- `protocol_b_window_sweep_summary.csv`",
        "- `per_case_predictions.csv`",
        "- `cleaning_diagnostics.csv`",
        "- `cleaning_diagnostics_summary.csv`",
        "- `method_capability_table.csv`",
        "- `audit_findings.json` and `artifact_hashes.csv`",
    ])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_capability_csv(out_dir: Path) -> None:
    write_csv(out_dir / "method_capability_table.csv", build_method_capability_table())
