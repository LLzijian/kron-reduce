from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state(root: Path) -> dict:
    return json.loads(
        (root / ".project_state" / "loop_state.json").read_text(encoding="utf-8")
    )


def _result_for_stage(root: Path, stage: str) -> Path:
    for entry in reversed(_state(root)["history"]):
        if entry.get("to") != stage:
            continue
        evidence = entry.get("evidence", {})
        result_name = evidence.get("result_dir") if isinstance(evidence, dict) else None
        if result_name:
            path = root / "results" / result_name
            if path.is_dir():
                return path
    raise RuntimeError(f"no registered immutable result for prerequisite stage {stage}")


def _verify_artifacts(result_dir: Path) -> None:
    manifest_path = result_dir / "artifact_hashes.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mismatches = [
        relative
        for relative, expected in manifest.items()
        if not (result_dir / relative).is_file()
        or _sha(result_dir / relative) != expected
    ]
    if mismatches:
        raise RuntimeError(f"immutable artifact verification failed: {mismatches[:5]}")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_artifacts(result_dir: Path) -> None:
    artifacts = {
        path.relative_to(result_dir).as_posix(): _sha(path)
        for path in result_dir.rglob("*")
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    (result_dir / "artifact_hashes.json").write_text(
        json.dumps(artifacts, indent=2), encoding="utf-8"
    )


def _summarize(root: Path, result_dir: Path, run_id: str) -> None:
    full = _result_for_stage(root, "full_run")
    _verify_artifacts(full)
    aggregates = _read_csv(full / "aggregate_metrics.csv")
    statuses = _read_csv(full / "model_status.csv")
    summary = {
        "run_id": run_id,
        "evidence_status": "full",
        "source_full_run": full.name,
        "source_full_run_artifact_manifest_sha256": _sha(full / "artifact_hashes.json"),
        "completed_model_fits": sum(row["status"] == "completed" for row in statuses),
        "failed_model_fits": sum(row["status"] == "failed" for row in statuses),
        "aggregate_rows": len(aggregates),
        "claim_policy": "trajectory_primary_stability_exploratory",
    }
    (result_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    shutil.copy2(full / "aggregate_metrics.csv", result_dir / "aggregate_metrics.csv")


def _adversarial_audit(root: Path, result_dir: Path, run_id: str) -> None:
    full = _result_for_stage(root, "full_run")
    summary = _result_for_stage(root, "summarize")
    _verify_artifacts(full)
    _verify_artifacts(summary)
    run_config = json.loads((full / "run_config.json").read_text(encoding="utf-8"))
    run_audit = json.loads((full / "audit.json").read_text(encoding="utf-8"))
    energy = _read_csv(full / "energy_gate.csv")
    allowed_names = {"M2A-LYAP", "M2A-storage", "M2A-energy-regularized"}
    checks = {
        "selection_marker_exists": (full / "selection_complete.json").is_file(),
        "test_unlocked_only_after_selection": run_config.get("test_access")
        == "unlocked_after_selection_complete",
        "failed_models_not_zero_filled": run_audit.get("zero_metrics_used_for_failures") is False,
        "stability_declared_exploratory": run_audit.get("stability_is_exploratory") is True,
        "energy_names_gate_controlled": all(
            row.get("reported_name") in allowed_names for row in energy
        ),
        "source_artifacts_verified": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"adversarial audit failed: {checks}")
    audit = {
        "run_id": run_id,
        "status": "PASS",
        "audit_type": "automated_adversarial_contract_audit",
        "human_scientific_review_still_required": True,
        "source_full_run": full.name,
        "checks": checks,
    }
    (result_dir / "adversarial_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )


def _paper_tables(root: Path, result_dir: Path, run_id: str) -> None:
    full = _result_for_stage(root, "full_run")
    audit = _result_for_stage(root, "adversarial_audit")
    _verify_artifacts(full)
    _verify_artifacts(audit)
    rows = _read_csv(full / "aggregate_metrics.csv")
    main = [row for row in rows if row["analysis"] == "main_zoh"]
    sensitivity = [
        row for row in rows if row["analysis"] == "conditional_observed_pm"
    ]

    def table(title: str, entries: list[dict[str, str]]) -> str:
        lines = [
            "\\begin{table*}[t]",
            "\\centering",
            f"\\caption{{{title}}}",
            "\\begin{tabular}{llrrrr}",
            "\\toprule",
            "Method & Horizon & Mean RMSE & Median RMSE & Pooled $R^2$ & Failure rate \\\\",
            "\\midrule",
        ]
        for row in entries:
            method = row["method"].replace("_", "\\_")
            lines.append(
                f"{method} & {row['horizon_steps']} & "
                f"{float(row['mean_rmse']):.5g} & {float(row['median_rmse']):.5g} & "
                f"{float(row['pooled_r2']):.5g} & {float(row['failure_rate']):.3f} \\\\"
            )
        lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}"])
        return "\n".join(lines)

    generated = "\n\n".join(
        (
            "% MACHINE GENERATED. evidence_status=full",
            table("Causal ZOH closed-loop rollout.", main),
            table("Conditional observed-$P_m(t)$ sensitivity.", sensitivity),
        )
    )
    table_path = result_dir / "main_tables.tex"
    table_path.write_text(generated, encoding="utf-8")
    paper_generated = root / "paper" / "generated" / "main_tables.tex"
    paper_generated.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(table_path, paper_generated)
    (result_dir / "table_manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "evidence_status": "full",
                "source_full_run": full.name,
                "source_aggregate_sha256": _sha(full / "aggregate_metrics.csv"),
                "generated_table_sha256": _sha(table_path),
                "manual_numbers": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _manuscript(root: Path, result_dir: Path, run_id: str) -> None:
    tables = _result_for_stage(root, "paper_tables")
    _verify_artifacts(tables)
    required_human_fields = [
        "authors_and_affiliations",
        "funding",
        "code_license",
        "data_redistribution_permission",
    ]
    manifest = {
        "run_id": run_id,
        "status": "DRAFT_EVIDENCE_ASSEMBLED",
        "paper_source": "paper/main.tex",
        "machine_table_source": tables.name,
        "machine_table_sha256": _sha(tables / "main_tables.tex"),
        "publication_blockers": required_human_fields,
        "publication_gate_must_remain_locked": True,
    }
    (result_dir / "manuscript_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    shutil.copy2(root / "paper" / "main.tex", result_dir / "main.tex")


def run(*, root: Path, experiment: dict, stage: str, result_dir: Path, run_id: str) -> None:
    actions = {
        "summarize": _summarize,
        "adversarial_audit": _adversarial_audit,
        "paper_tables": _paper_tables,
        "manuscript": _manuscript,
    }
    if stage not in actions:
        raise RuntimeError(f"unsupported postprocess stage: {stage}")
    actions[stage](root, result_dir, run_id)
    _write_artifacts(result_dir)
