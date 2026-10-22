from __future__ import annotations

import ast
import json
from pathlib import Path

from .evidence import (
    SHA256_RE,
    EvidenceContext,
    EvidenceValidationError,
    _safe_relative,
    sha256_file,
    validate_stage_evidence,
)
from .manifest import load_manifest
from .policy import audit_research_policy

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

STABILITY_CERTIFIED_MARKER = "stability-certified equivalent"
LEVEL6_SCHEMA = "kron_reduce.level6_certificate.v1"


def _protected_level6_claim(acceptance: dict) -> bool:
    names = [str(acceptance.get(key, "")).strip().casefold() for key in ("model_reporting_name", "reported_name", "model_name")]
    return (
        any("stability-certified" in name for name in names)
        or acceptance.get("level_6_claimed") is True
    )


def _binding(record: dict, *names: str) -> dict | None:
    for name in names:
        value = record.get(name)
        if isinstance(value, dict):
            return value
    return None


def _verify_bound_file(run_dir: Path, binding: object, label: str) -> Path:
    if not isinstance(binding, dict):
        raise EvidenceValidationError("LEVEL6_BINDING", f"{label} binding is missing")
    path_value = binding.get("path")
    digest = binding.get("sha256", binding.get("sha"))
    if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest.lower()):
        raise EvidenceValidationError("LEVEL6_BINDING_SHA", f"{label} binding needs a SHA256")
    path = _safe_relative(run_dir, path_value)
    if path.is_symlink() or not path.is_file():
        raise EvidenceValidationError("LEVEL6_BINDING_PATH", f"{label} path is missing or unsafe")
    if sha256_file(path) != digest.lower():
        raise EvidenceValidationError("LEVEL6_BINDING_SHA", f"{label} SHA256 does not match")
    return path


def _group_audit_error(path: Path) -> None:
    try:
        audit = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceValidationError("LEVEL6_GROUP_AUDIT", "independent group audit is unreadable") from exc
    if not isinstance(audit, dict) or audit.get("group_unit", audit.get("unit")) != "fault_line_x_operating_point":
        raise EvidenceValidationError("LEVEL6_GROUP_UNIT", "independent group unit must be fault_line_x_operating_point")
    if audit.get("status") != "PASS":
        raise EvidenceValidationError("LEVEL6_GROUP_STATUS", "independent group audit status must be PASS")
    count = audit.get("independent_unstable_groups", audit.get("unique_group_count"))
    duplicates = audit.get("duplicate_groups", audit.get("duplicate_group_count"))
    if "independent_unstable_groups" in audit and "unique_group_count" in audit and audit["independent_unstable_groups"] != audit["unique_group_count"]:
        raise EvidenceValidationError("LEVEL6_GROUP_COUNT", "group count fields disagree")
    if "duplicate_groups" in audit and "duplicate_group_count" in audit and audit["duplicate_groups"] != audit["duplicate_group_count"]:
        raise EvidenceValidationError("LEVEL6_GROUP_DUPLICATES", "duplicate count fields disagree")
    if not isinstance(count, int) or isinstance(count, bool) or count < 80:
        raise EvidenceValidationError("LEVEL6_GROUP_COUNT", "independent group audit requires at least 80 groups")
    if not isinstance(duplicates, int) or isinstance(duplicates, bool) or duplicates != 0:
        raise EvidenceValidationError("LEVEL6_GROUP_DUPLICATES", "independent group audit requires zero duplicates")


def _certificate_from_metrics(certificate: dict):
    """Rebuild Level 6 gates; never accept a certificate's reported status."""
    from kron_reduce.evaluation.stability_region import StabilityCertificate

    metrics = certificate.get("metrics", certificate.get("stability_metrics"))
    if not isinstance(metrics, dict):
        metrics = certificate
    fields = (
        "region_conservative", "stable_envelope_coverage",
        "region_symmetric_difference_ratio", "normalized_margin_mae",
        "margin_rank_correlation", "independent_unstable_groups", "fsr_95_upper",
        "false_stable_certificates", "physics_gates_passed", "numerical_gates_passed",
    )
    missing = [field for field in fields if field not in metrics]
    if missing:
        raise EvidenceValidationError("LEVEL6_METRICS", f"certificate metrics are missing {missing}")
    try:
        values = {field: metrics[field] for field in fields}
        values["claim_boundary"] = certificate.get("claim_boundary", metrics.get("claim_boundary"))
        values["sample_based"] = certificate.get("sample_based")
        return StabilityCertificate(**values)
    except (TypeError, ValueError) as exc:
        raise EvidenceValidationError("LEVEL6_METRICS", f"certificate metrics are invalid: {exc}") from exc


def audit_stability_certificate_names(root: Path) -> list[dict]:
    """Reject protected names unless their independently bound Level 6 evidence passes."""
    findings = []
    results = root / "results"
    if not results.is_dir():
        return findings
    for path in sorted(results.glob("*/acceptance.json")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        try:
            acceptance = json.loads(text)
        except json.JSONDecodeError:
            continue
        if not isinstance(acceptance, dict) or not _protected_level6_claim(acceptance):
            continue
        try:
            binding = _binding(acceptance, "level6_certificate", "level_6_certificate")
            if binding is None and "certificate_path" in acceptance:
                binding = {"path": acceptance.get("certificate_path"), "sha256": acceptance.get("certificate_sha256")}
            certificate_path = _verify_bound_file(path.parent, binding, "Level 6 certificate")
            certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
            if not isinstance(certificate, dict) or certificate.get("schema") != LEVEL6_SCHEMA:
                raise EvidenceValidationError("LEVEL6_SCHEMA", f"certificate schema must be {LEVEL6_SCHEMA}")
            if certificate.get("run_id") != path.parent.name:
                raise EvidenceValidationError("LEVEL6_RUN_ID", "certificate run_id must equal its run directory")
            if not isinstance(certificate.get("workflow", ""), str) or not certificate.get("workflow", ""):
                raise EvidenceValidationError("LEVEL6_WORKFLOW", "certificate workflow must be nonempty")
            if not isinstance(certificate.get("sample_based"), bool):
                raise EvidenceValidationError("LEVEL6_SAMPLE_BASED", "certificate sample_based must be boolean")
            if certificate.get("integration_free") is not True:
                raise EvidenceValidationError("LEVEL6_INTEGRATION_FREE", "certificate integration_free must be true")
            if not isinstance(certificate.get("claim_boundary"), str) or not certificate["claim_boundary"].strip():
                raise EvidenceValidationError("LEVEL6_CLAIM_BOUNDARY", "certificate claim_boundary must be nonempty")
            bindings = certificate.get("bindings", certificate.get("artifacts", certificate))
            if not isinstance(bindings, dict):
                raise EvidenceValidationError("LEVEL6_BINDING", "certificate bindings must be an object")
            _verify_bound_file(path.parent, _binding(bindings, "model_artifact", "model_manifest"), "model artifact")
            datasets = bindings.get("dataset_evidence")
            if isinstance(datasets, list):
                roles = {entry.get("role"): entry for entry in datasets if isinstance(entry, dict)}
                required_roles = {"synthetic_ground_truth", "ieee39_boundary_audit"}
                if len(datasets) != 2 or len(roles) != 2 or set(roles) != required_roles or any(not entry.get("dataset_id") for entry in roles.values()) or roles["synthetic_ground_truth"]["dataset_id"] == roles["ieee39_boundary_audit"]["dataset_id"]:
                    raise EvidenceValidationError("LEVEL6_DATASET_ROLES", "dataset evidence needs two distinct registered roles and IDs")
                for role, entry in roles.items():
                    _verify_bound_file(path.parent, {"path": entry.get("manifest_path"), "sha256": entry.get("manifest_sha256")}, role)
                datasets = None
            if not isinstance(datasets, (list, dict)):
                raise EvidenceValidationError("LEVEL6_BINDING", "dataset_evidence must contain both dataset roles")
            if datasets is not None:
                _verify_bound_file(path.parent, _binding(datasets, "synthetic", "synthetic_manifest"), "synthetic data manifest")
                _verify_bound_file(path.parent, _binding(datasets, "ieee39", "ieee39_manifest"), "IEEE39 data manifest")
            audit_path = _verify_bound_file(path.parent, _binding(bindings, "independent_group_audit", "independent_groups_audit"), "independent group audit")
            _group_audit_error(audit_path)
            reconstructed = _certificate_from_metrics(certificate)
            group_count = json.loads(audit_path.read_text(encoding="utf-8")).get("independent_unstable_groups", json.loads(audit_path.read_text(encoding="utf-8")).get("unique_group_count"))
            if reconstructed.independent_unstable_groups != group_count:
                raise EvidenceValidationError("LEVEL6_GROUP_CROSSCHECK", "certificate and group audit counts differ")
            if not reconstructed.level_6_eligible:
                raise EvidenceValidationError("LEVEL6_GATES", "recomputed Level 6 certificate gates do not all pass")
        except (EvidenceValidationError, OSError, json.JSONDecodeError, ImportError) as exc:
            findings.append(
                {
                    "severity": "ERROR",
                    "check": "stability_certificate",
                    "message": (
                        f"{path.parent.name} has an invalid protected Level 6 claim: {exc}"
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


def _stage_evidence_error(
    root: Path, workflow: dict, stage: str, evidence: object, predecessor: str | None, state: dict | None = None
) -> str | None:
    try:
        validate_stage_evidence(
            EvidenceContext(root=root, workflow=workflow, target=stage, state=state, predecessor=predecessor),
            evidence,
        )
    except EvidenceValidationError as exc:
        return str(exc)
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
        for event_index, event in enumerate(state.get("history", [])):
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
            prefix_state = {**state, "history": state.get("history", [])[:event_index]}
            evidence_error = _stage_evidence_error(root, workflow, target, evidence, source, prefix_state)
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
    findings.extend(audit_research_policy(root, manifest))
    findings.extend(audit_artifact_layout(root))
    findings.extend(audit_workflow_registry(root))
    findings.extend(audit_stability_certificate_names(root))
    status = "PASS" if not any(row["severity"] == "ERROR" for row in findings) else "FAIL"
    return {"status": status, "errors": sum(row["severity"] == "ERROR" for row in findings), "findings": findings}
