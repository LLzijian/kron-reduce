from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path


WORKFLOW = "paper_v2_integral"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state(root: Path) -> dict:
    return json.loads(
        (root / ".project_state" / f"{WORKFLOW}_state.json").read_text(encoding="utf-8")
    )


def _result_for_stage(root: Path, stage: str) -> Path:
    for entry in reversed(_state(root)["history"]):
        if entry.get("to") != stage:
            continue
        evidence = entry.get("evidence", {})
        name = evidence.get("result_dir") if isinstance(evidence, dict) else None
        if name and (root / "results" / name).is_dir():
            return root / "results" / name
    raise RuntimeError(f"no immutable {WORKFLOW} result for stage {stage}")


def _verify(result_dir: Path) -> None:
    manifest = json.loads((result_dir / "artifact_hashes.json").read_text(encoding="utf-8"))
    mismatches = [
        name for name, digest in manifest.items()
        if not (result_dir / name).is_file() or _sha(result_dir / name) != digest
    ]
    if mismatches:
        raise RuntimeError(f"artifact verification failed: {mismatches[:5]}")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_manifest(result_dir: Path) -> None:
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
    _verify(full)
    acceptance = json.loads((full / "round_acceptance.json").read_text(encoding="utf-8"))
    selections = json.loads((full / "final_model_selection.json").read_text(encoding="utf-8"))
    aggregates = _read_csv(full / "aggregate_metrics.csv")
    summary = {
        "run_id": run_id,
        "workflow": WORKFLOW,
        "source_full_run": full.name,
        "source_artifact_manifest_sha256": _sha(full / "artifact_hashes.json"),
        "decision": acceptance["decision"],
        "acceptance": acceptance,
        "final_source_method_counts": dict(Counter(row["source_method"] for row in selections)),
        "aggregate_rows": len(aggregates),
        "next_improvement_requires_operator_review": True,
    }
    (result_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for name in (
        "aggregate_metrics.csv", "round_acceptance.json", "final_model_selection.json",
        "final_stability_metrics.csv", "deterministic_repair_gate.json",
        "governor_ablation_gate.json", "energy_gate.csv",
    ):
        shutil.copy2(full / name, result_dir / name)


def _final_report(root: Path, result_dir: Path, run_id: str) -> None:
    full = _result_for_stage(root, "full_run")
    summary_dir = _result_for_stage(root, "summarize")
    _verify(full); _verify(summary_dir)
    acceptance = json.loads((full / "round_acceptance.json").read_text(encoding="utf-8"))
    selections = json.loads((full / "final_model_selection.json").read_text(encoding="utf-8"))
    aggregates = _read_csv(full / "aggregate_metrics.csv")
    final_rows = [
        row for row in aggregates
        if row["method"] == "FINAL" and row["analysis"] == "main_zoh"
    ]
    final_rows.sort(key=lambda row: int(row["horizon_steps"]))
    source_counts = Counter(row["source_method"] for row in selections)
    table = [
        "| Horizon | mean RMSE | median RMSE | pooled omega R2 | failure rate |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in final_rows:
        table.append(
            f"| {row['horizon_steps']} | {float(row['mean_rmse']):.6g} | "
            f"{float(row['median_rmse']):.6g} | {float(row['pooled_omega_r2']):.6g} | "
            f"{float(row['failure_rate']):.4f} |"
        )
    selection_text = ", ".join(f"{name}: {count}" for name, count in sorted(source_counts.items()))
    report = f"""# Integral-v2 第1轮完整实验报告

**Run ID:** `{run_id}`  
**Source full run:** `{full.name}`  
**Decision:** **{acceptance['decision']}**

## 方法冻结

最终模型在读取 test 前，按每个 fold/seed 的 validation loss 从预注册物理候选中冻结。
选择频数：{selection_text}。主输入为 `Pm_clear` ZOH；稳定性只由最终预测轨迹推导。

## 完整结果

{chr(10).join(table)}

H=100 pooled omega R2 为 **{acceptance['omega_r2']:.6g}**，严格目标为
`> {acceptance['omega_r2_strict_target']}`，门结果为
**{'PASS' if acceptance['omega_r2_gate_passed'] else 'FAIL'}**。

最终预测轨迹的失稳 FSR 为 **{acceptance['instability_FSR']:.6g}**，严格目标为
`< {acceptance['instability_FSR_strict_target']}`，门结果为
**{'PASS' if acceptance['instability_FSR_gate_passed'] else 'FAIL'}**。

闭环 failure rate 为 **{acceptance['failure_rate']:.6g}**。

## 接受/不接受判定

本轮判定为 **{acceptance['decision']}**。只有 omega R2、FSR 和零失败门全部通过时
才接受本轮方法。该判定不触发下一轮改进；新的方案必须先由用户评估批准。

## 证据边界

- test 仅在 selection marker 和最终模型映射写入后读取；
- `Pe`、电压、标签和未来状态未进入动力学拟合或选择；
- 稳定性结论来自 `FINAL` 的完整预测轨迹，而非真实未来输入；
- 结果目录及其哈希清单保持不可变。
"""
    path = result_dir / "round1_method_results_report.md"
    path.write_text(report, encoding="utf-8")


def run(*, root: Path, experiment: dict, stage: str, result_dir: Path, run_id: str) -> None:
    if experiment.get("workflow") != WORKFLOW:
        raise RuntimeError("integral-v2 postprocess requires the integral-v2 workflow")
    if stage == "summarize":
        _summarize(root, result_dir, run_id)
    elif stage == "final_report":
        _final_report(root, result_dir, run_id)
    else:
        raise RuntimeError(f"unsupported integral-v2 postprocess stage: {stage}")
    _write_manifest(result_dir)
