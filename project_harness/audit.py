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

# Workflow families that must terminate at an approval-controlled merge_gate
# and preserve the ordered scientific evidence chain.
STRUCTURED_LOOP_PREFIXES = ("m0_afsd_", "m0_esc_")

STABILITY_CERTIFIED_MARKER = "stability-certified"


def audit_stability_certificate_names(root: Path) -> list[dict]:
    """A run claiming a stability-certified name must carry a passing Level 6 certificate."""
    findings = []
    results = root / "results"
    if not results.is_dir():
        return findings
    for path in sorted(results.glob("*/acceptance.json")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if STABILITY_CERTIFIED_MARKER not in text:
            continue
        certificate = path.parent / "stability_certificate.json"
        passed = False
        if certificate.is_file():
            try:
                passed = json.loads(certificate.read_text(encoding="utf-8")).get("status") == "pass"
            except (json.JSONDecodeError, OSError):
                passed = False
        if not passed:
            findings.append(
                {
                    "severity": "ERROR",
                    "check": "stability_certificate",
                    "message": (
                        f"{path.parent.name} claims a stability-certified name without a "
                        "passing stability_certificate.json"
                    ),
                }
            )
    return findings


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


def _transition_targets(workflow: dict, stage: str, entry_evidence: dict) -> set[str]:
    transitions = workflow.get("transitions")
    stages = workflow["stages"]
    if not transitions:
        index = stages.index(stage)
        return {stages[index + 1]} if index + 1 < len(stages) else set()
    spec = transitions.get(stage)
    if spec is None:
        return set()
    if isinstance(spec, str):
        return {spec}
    if not isinstance(spec, dict):
        return set()
    verdict = entry_evidence.get("scientific_verdict")
    if verdict in spec:
        return {spec[verdict]}
    return set(spec.values())


def _is_empty_evidence_value(value: object) -> bool:
    return (
        value is None
        or (isinstance(value, str) and not value.strip())
        or (isinstance(value, (dict, list, tuple, set)) and not value)
    )


def _stage_evidence_error(workflow: dict, stage: str, evidence: object) -> str | None:
    if not isinstance(evidence, dict) or evidence.get("status") != "PASS":
        return "evidence is not a passing JSON object"
    contract = workflow.get("stage_evidence_contracts", {}).get(stage, {})
    required = contract.get("required_fields", [])
    missing = [field for field in required if field not in evidence]
    if missing:
        return f"missing required evidence fields {missing}"
    allowed_values = contract.get("allowed_values", {})
    for field, allowed in allowed_values.items():
        if evidence.get(field) not in allowed:
            return f"field {field!r} must be one of {allowed}"
    empty = [
        field
        for field in contract.get("non_empty_fields", [])
        if _is_empty_evidence_value(evidence.get(field))
    ]
    if empty:
        return f"empty required evidence fields {empty}"
    conditional_missing = []
    for discriminator, cases in contract.get("conditional_required_fields", {}).items():
        conditional_missing.extend(
            field
            for field in cases.get(evidence.get(discriminator), [])
            if field not in evidence or _is_empty_evidence_value(evidence.get(field))
        )
    if conditional_missing:
        return f"missing conditional evidence fields {conditional_missing}"
    verdict = evidence.get("scientific_verdict")
    if verdict is not None and verdict not in {"ACCEPT", "REJECT", "RETRY"}:
        return f"invalid scientific_verdict {verdict!r}"
    promotion = evidence.get("promotion_decision")
    if promotion is not None and promotion not in {"PROMOTE", "KEEP_BASELINE"}:
        return f"invalid promotion_decision {promotion!r}"
    accepted = workflow.get("accepted_scientific_verdicts")
    if accepted and verdict is not None and verdict not in accepted:
        return "master-stage evidence does not carry an accepted scientific verdict"
    return None


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

        transitions = workflow.get("transitions")
        terminal = set(workflow.get("terminal_stages", []))
        if transitions is not None:
            if not isinstance(transitions, dict):
                findings.append(
                    {
                        "severity": "ERROR",
                        "check": "workflow_transitions",
                        "message": f"transitions must be an object in {path.relative_to(root)}",
                    }
                )
            else:
                unknown_sources = set(transitions).difference(stages)
                unknown_terminal = terminal.difference(stages)
                targets: set[str] = set()
                invalid_specs = []
                for source, spec in transitions.items():
                    if isinstance(spec, str):
                        targets.add(spec)
                    elif isinstance(spec, dict) and spec:
                        if not set(spec).issubset({"ACCEPT", "REJECT", "RETRY"}):
                            invalid_specs.append(source)
                        targets.update(value for value in spec.values() if isinstance(value, str))
                        if len(targets) < 1 or any(not isinstance(value, str) for value in spec.values()):
                            invalid_specs.append(source)
                    else:
                        invalid_specs.append(source)
                unknown_targets = targets.difference(stages)
                missing_sources = set(stages).difference(terminal).difference(transitions)
                if unknown_sources or unknown_targets or unknown_terminal or invalid_specs or missing_sources:
                    findings.append(
                        {
                            "severity": "ERROR",
                            "check": "workflow_transitions",
                            "message": (
                                f"invalid transition graph in {path.relative_to(root)}: "
                                f"unknown_sources={sorted(unknown_sources)}, "
                                f"unknown_targets={sorted(unknown_targets)}, "
                                f"unknown_terminal={sorted(unknown_terminal)}, "
                                f"invalid_specs={sorted(set(invalid_specs))}, "
                                f"missing_sources={sorted(missing_sources)}"
                            ),
                        }
                    )
                if terminal.intersection(transitions):
                    findings.append(
                        {
                            "severity": "ERROR",
                            "check": "workflow_transitions",
                            "message": f"terminal stages have outgoing edges in {path.relative_to(root)}",
                        }
                    )

        contracts = workflow.get("stage_evidence_contracts", {})
        invalid_contracts = []
        if not isinstance(contracts, dict):
            invalid_contracts.append("<root>")
        else:
            for stage, contract in contracts.items():
                required = contract.get("required_fields") if isinstance(contract, dict) else None
                if stage not in stages or not isinstance(required, list) or not all(
                    isinstance(field, str) for field in required
                ):
                    invalid_contracts.append(stage)
                    continue
                allowed_values = contract.get("allowed_values", {})
                non_empty_fields = contract.get("non_empty_fields", [])
                conditional = contract.get("conditional_required_fields", {})
                valid_allowed = isinstance(allowed_values, dict) and all(
                    isinstance(field, str) and isinstance(values, list) and values
                    for field, values in allowed_values.items()
                )
                valid_non_empty = isinstance(non_empty_fields, list) and all(
                    isinstance(field, str) for field in non_empty_fields
                )
                valid_conditional = isinstance(conditional, dict) and all(
                    isinstance(field, str)
                    and isinstance(cases, dict)
                    and all(
                        isinstance(value, str)
                        and isinstance(fields, list)
                        and all(isinstance(required_field, str) for required_field in fields)
                        for value, fields in cases.items()
                    )
                    for field, cases in conditional.items()
                )
                if not (valid_allowed and valid_non_empty and valid_conditional):
                    invalid_contracts.append(stage)
        if invalid_contracts:
            findings.append(
                {
                    "severity": "ERROR",
                    "check": "workflow_evidence_contract",
                    "message": f"invalid stage evidence contracts in {path.relative_to(root)}: {invalid_contracts}",
                }
            )

        if isinstance(name, str) and name.startswith(STRUCTURED_LOOP_PREFIXES):
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
        entry_evidence: dict = {}
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
            evidence = event.get("evidence", {})
            allowed_targets = (
                _transition_targets(workflow, source, entry_evidence)
                if source in stages
                else set()
            )
            evidence_error = _stage_evidence_error(workflow, target, evidence)
            if (
                source != expected_from
                or target not in allowed_targets
                or evidence_error is not None
            ):
                findings.append(
                    {
                        "severity": "ERROR",
                        "check": "workflow_state_history",
                        "message": (
                            f"non-contiguous or unverified transition in {state_path.relative_to(root)}; "
                            f"allowed={sorted(allowed_targets)}, evidence_error={evidence_error}"
                        ),
                    }
                )
                break
            expected_from = target
            entry_evidence = evidence
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
    findings.extend(audit_stability_certificate_names(root))
    status = "PASS" if not any(row["severity"] == "ERROR" for row in findings) else "FAIL"
    return {"status": status, "errors": sum(row["severity"] == "ERROR" for row in findings), "findings": findings}
