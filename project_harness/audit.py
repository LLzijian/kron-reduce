from __future__ import annotations

import ast
import json
from pathlib import Path

from .manifest import load_manifest

REQUIRED = (
    "AGENTS.md",
    "README.md",
    "pyproject.toml",
    "configs/project_manifest.json",
    "docs/architecture.md",
    "docs/scientific_contract.md",
    "docs/reports/README.md",
    "reports/generated",
    "src/kron_reduce/data",
    "src/kron_reduce/models",
    "project_harness",
    "tests",
    "paper",
)

AFSD_EVIDENCE_CHAIN = (
    "hypothesis_registration",
    "controlled_experiment",
    "residual_diagnosis",
    "adversarial_review",
    "acceptance",
)


def audit_artifact_layout(root: Path) -> list[dict]:
    """Enforce the boundary between evidence, generated output, and prose reports."""
    findings = []
    reports = root / "reports"
    if reports.is_dir():
        misplaced = sorted(
            path.name
            for path in reports.iterdir()
            if path.is_file() and path.name != "README.md"
        )
        for name in misplaced:
            findings.append(
                {
                    "severity": "ERROR",
                    "check": "artifact_layout",
                    "message": (
                        f"manual or generated report at reports/{name}; "
                        "use docs/reports for curated prose or reports/generated for generated output"
                    ),
                }
            )
    return findings


def _workflow_state_path(root: Path, workflow_name: str) -> Path:
    filename = "loop_state.json" if workflow_name == "paper_v1" else f"{workflow_name}_state.json"
    return root / ".project_state" / filename


def audit_workflow_registry(root: Path) -> list[dict]:
    """Validate workflow definitions and any durable state-machine histories."""
    findings = []
    workflow_dir = root / "configs" / "workflows"
    if not workflow_dir.is_dir():
        return [
            {
                "severity": "ERROR",
                "check": "workflow_registry",
                "message": "missing configs/workflows",
            }
        ]

    for path in sorted(workflow_dir.glob("*.json")):
        try:
            workflow = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            findings.append(
                {
                    "severity": "ERROR",
                    "check": "workflow_registry",
                    "message": f"cannot read {path.relative_to(root)}: {exc}",
                }
            )
            continue

        name = workflow.get("workflow")
        stages = workflow.get("stages")
        approvals = set(workflow.get("approval_required", []))
        if name != path.stem:
            findings.append(
                {
                    "severity": "ERROR",
                    "check": "workflow_identity",
                    "message": f"workflow identity mismatch in {path.relative_to(root)}",
                }
            )
        if not isinstance(stages, list) or not stages or len(stages) != len(set(stages)):
            findings.append(
                {
                    "severity": "ERROR",
                    "check": "workflow_stages",
                    "message": f"stages must be a non-empty unique list in {path.relative_to(root)}",
                }
            )
            continue
        if workflow.get("initial_stage") != stages[0]:
            findings.append(
                {
                    "severity": "ERROR",
                    "check": "workflow_initial_stage",
                    "message": f"initial stage must equal the first stage in {path.relative_to(root)}",
                }
            )
        unknown_approvals = approvals.difference(stages)
        if unknown_approvals:
            findings.append(
                {
                    "severity": "ERROR",
                    "check": "workflow_approval",
                    "message": f"approval gates are not stages in {path.relative_to(root)}",
                }
            )

        if isinstance(name, str) and name.startswith("m0_afsd_"):
            if stages[-1] != "merge_gate" or "merge_gate" not in approvals:
                findings.append(
                    {
                        "severity": "ERROR",
                        "check": "afsd_merge_gate",
                        "message": f"{name} must terminate at an approval-controlled merge_gate",
                    }
                )
            if "hypothesis_registration" in stages:
                indices = [stages.index(stage) for stage in AFSD_EVIDENCE_CHAIN if stage in stages]
                if len(indices) != len(AFSD_EVIDENCE_CHAIN) or indices != sorted(indices):
                    findings.append(
                        {
                            "severity": "ERROR",
                            "check": "afsd_evidence_order",
                            "message": f"{name} does not preserve the registered scientific loop order",
                        }
                    )

        state_path = _workflow_state_path(root, path.stem)
        if not state_path.is_file():
            continue
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            findings.append(
                {
                    "severity": "ERROR",
                    "check": "workflow_state",
                    "message": f"cannot read {state_path.relative_to(root)}: {exc}",
                }
            )
            continue
        if state.get("workflow") != name or state.get("current_stage") not in stages:
            findings.append(
                {
                    "severity": "ERROR",
                    "check": "workflow_state",
                    "message": f"invalid identity or current stage in {state_path.relative_to(root)}",
                }
            )
            continue

        expected_from = stages[0]
        approved = set(state.get("approvals", {}))
        if not approved.issubset(approvals):
            findings.append(
                {
                    "severity": "ERROR",
                    "check": "workflow_state_approval",
                    "message": f"unregistered approval recorded in {state_path.relative_to(root)}",
                }
            )
        for event in state.get("history", []):
            if event.get("action") != "advance":
                continue
            source = event.get("from")
            target = event.get("to")
            source_index = stages.index(source) if source in stages else -1
            target_index = stages.index(target) if target in stages else -1
            evidence = event.get("evidence", {})
            if (
                source != expected_from
                or target_index != source_index + 1
                or evidence.get("status") != "PASS"
            ):
                findings.append(
                    {
                        "severity": "ERROR",
                        "check": "workflow_state_history",
                        "message": f"non-contiguous or unverified transition in {state_path.relative_to(root)}",
                    }
                )
                break
            expected_from = target
        if state.get("current_stage") != expected_from:
            findings.append(
                {
                    "severity": "ERROR",
                    "check": "workflow_state_history",
                    "message": f"current stage does not match transition history in {state_path.relative_to(root)}",
                }
            )
        current = state.get("current_stage")
        if current in approvals and current not in approved:
            findings.append(
                {
                    "severity": "ERROR",
                    "check": "workflow_state_approval",
                    "message": f"approval-controlled stage lacks approval in {state_path.relative_to(root)}",
                }
            )
    return findings


def audit_project(root: Path) -> dict:
    findings = []
    for relative in REQUIRED:
        if not (root / relative).exists():
            findings.append({"severity": "ERROR", "check": "required_path", "message": f"missing {relative}"})
    manifest = load_manifest(root)
    if manifest["default_dataset"] not in manifest["datasets"]:
        findings.append({"severity": "ERROR", "check": "manifest", "message": "default dataset is not registered"})
    for path in (root / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "E:\\03-nn_Transfer_Func\\swing_equation" in text or "E:/03-nn_Transfer_Func/swing_equation" in text:
            findings.append({"severity": "ERROR", "check": "absolute_legacy_path", "message": str(path.relative_to(root))})
        tree = ast.parse(text, filename=str(path))
        for node in ast.walk(tree):
            module = None
            if isinstance(node, ast.ImportFrom): module = node.module
            elif isinstance(node, ast.Import):
                for name in node.names:
                    if name.name.startswith("legacy"):
                        findings.append({"severity": "ERROR", "check": "legacy_import", "message": str(path.relative_to(root))})
            if module and (module.startswith("legacy") or "hist_3" in module):
                findings.append({"severity": "ERROR", "check": "legacy_import", "message": str(path.relative_to(root))})
    findings.extend(audit_artifact_layout(root))
    findings.extend(audit_workflow_registry(root))
    status = "PASS" if not any(row["severity"] == "ERROR" for row in findings) else "FAIL"
    return {"status": status, "errors": sum(row["severity"] == "ERROR" for row in findings), "findings": findings}
