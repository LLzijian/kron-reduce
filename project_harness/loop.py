from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from .audit import audit_project
from .data_check import verify_data
from .git_state import assert_clean, git_head

SCIENTIFIC_VERDICTS = {"ACCEPT", "REJECT", "RETRY"}
PROMOTION_DECISIONS = {"PROMOTE", "KEEP_BASELINE"}


def _is_empty_evidence_value(value: object) -> bool:
    return value is None or (isinstance(value, str) and not value.strip()) or (
        isinstance(value, (dict, list, tuple, set)) and not value
    )


def _workflow(root: Path, workflow_name: str = "paper_v1") -> dict:
    path = root / "configs" / "workflows" / f"{workflow_name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"unregistered workflow: {workflow_name}")
    workflow = json.loads(path.read_text(encoding="utf-8"))
    if workflow.get("workflow") != workflow_name:
        raise ValueError(f"workflow identity mismatch in {path}")
    return workflow


def _state_path(root: Path, workflow_name: str = "paper_v1") -> Path:
    # Keep the original path for paper_v1 so its completed history remains
    # immutable and readable after named workflows are introduced.
    filename = "loop_state.json" if workflow_name == "paper_v1" else f"{workflow_name}_state.json"
    path = root / ".project_state" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_state(root: Path, workflow_name: str = "paper_v1") -> dict:
    path = _state_path(root, workflow_name)
    if path.exists(): return json.loads(path.read_text(encoding="utf-8"))
    workflow = _workflow(root, workflow_name)
    return {"workflow": workflow["workflow"], "current_stage": workflow["initial_stage"], "approvals": {}, "history": []}


def save_state(root: Path, state: dict, workflow_name: str = "paper_v1") -> None:
    path = _state_path(root, workflow_name); temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8"); temporary.replace(path)


def is_approved(root: Path, gate: str, workflow_name: str = "paper_v1") -> bool:
    return bool(load_state(root, workflow_name).get("approvals", {}).get(gate))


def approve(root: Path, gate: str, workflow_name: str = "paper_v1") -> dict:
    workflow = _workflow(root, workflow_name)
    if gate not in workflow["approval_required"]:
        raise ValueError(f"gate is not approval-controlled: {gate}")
    state = load_state(root, workflow_name)
    state.setdefault("approvals", {})[gate] = {
        "approved_at": datetime.now(UTC).isoformat(),
        "actor": "explicit_operator_action",
    }
    state["history"].append(
        {"action": "approve", "gate": gate, "at": datetime.now(UTC).isoformat()}
    )
    save_state(root, state, workflow_name); return state


def _entry_evidence(state: dict, stage: str) -> dict:
    """Return the evidence that advanced the workflow into ``stage``."""
    for event in reversed(state.get("history", [])):
        if event.get("action") == "advance" and event.get("to") == stage:
            evidence = event.get("evidence", {})
            return evidence if isinstance(evidence, dict) else {}
    return {}


def _transition_target(
    workflow: dict, state: dict, outcome: str | None = None
) -> str | None:
    current = state["current_stage"]
    terminal = set(workflow.get("terminal_stages", []))
    if current in terminal:
        return None
    transitions = workflow.get("transitions")
    if not transitions:
        index = workflow["stages"].index(current)
        return workflow["stages"][index + 1] if index + 1 < len(workflow["stages"]) else None
    spec = transitions.get(current)
    if spec is None:
        return None
    if isinstance(spec, str):
        return spec
    if not isinstance(spec, dict):
        raise TypeError(f"invalid transition specification for {current!r}")
    selected = outcome or _entry_evidence(state, current).get("scientific_verdict")
    if selected not in spec:
        raise RuntimeError(
            f"stage {current!r} requires one of the registered outcomes: "
            f"{sorted(spec)}"
        )
    return spec[selected]


def next_stage(
    root: Path, workflow_name: str = "paper_v1", outcome: str | None = None
) -> str | None:
    workflow, state = _workflow(root, workflow_name), load_state(root, workflow_name)
    return _transition_target(workflow, state, outcome)


def _validate_stage_evidence(workflow: dict, target: str, evidence: object) -> None:
    if not isinstance(evidence, dict):
        raise TypeError("stage evidence must be a JSON object")
    if evidence.get("status") != "PASS":
        raise RuntimeError(f"gate {target} failed")
    contract = workflow.get("stage_evidence_contracts", {}).get(target, {})
    required = contract.get("required_fields", [])
    missing = [field for field in required if field not in evidence]
    if missing:
        raise RuntimeError(
            f"stage {target!r} evidence is missing required fields: {missing}"
        )
    allowed_values = contract.get("allowed_values", {})
    for field, allowed in allowed_values.items():
        if evidence.get(field) not in allowed:
            raise RuntimeError(
                f"stage {target!r} evidence field {field!r} must be one of {allowed}"
            )
    empty = [field for field in contract.get("non_empty_fields", []) if _is_empty_evidence_value(evidence.get(field))]
    if empty:
        raise RuntimeError(f"stage {target!r} evidence has empty required fields: {empty}")
    conditional_missing = []
    for discriminator, cases in contract.get("conditional_required_fields", {}).items():
        conditional_missing.extend(
            field
            for field in cases.get(evidence.get(discriminator), [])
            if field not in evidence or _is_empty_evidence_value(evidence.get(field))
        )
    if conditional_missing:
        raise RuntimeError(
            f"stage {target!r} evidence is missing conditional fields: {conditional_missing}"
        )
    process_status = evidence.get("process_status")
    if process_status is not None and process_status != "PASS":
        raise RuntimeError(f"stage {target!r} process_status must be PASS")
    verdict = evidence.get("scientific_verdict")
    if verdict is not None and verdict not in SCIENTIFIC_VERDICTS:
        raise RuntimeError(f"unregistered scientific_verdict: {verdict!r}")
    promotion = evidence.get("promotion_decision")
    if promotion is not None and promotion not in PROMOTION_DECISIONS:
        raise RuntimeError(f"unregistered promotion_decision: {promotion!r}")
    accepted = workflow.get("accepted_scientific_verdicts")
    if accepted and verdict is not None and verdict not in accepted:
        raise RuntimeError(
            f"master stage {target!r} requires an accepted scientific verdict"
        )


def record_stage(root: Path, target: str, evidence: object, workflow_name: str = "paper_v1") -> dict:
    """Advance exactly one workflow edge after durable evidence exists."""
    workflow = _workflow(root, workflow_name)
    state = load_state(root, workflow_name)
    current = state["current_stage"]
    expected = next_stage(root, workflow_name)
    if target != expected:
        raise RuntimeError(
            f"cannot record {target!r} from {current!r}; expected {expected!r}"
        )
    if target in workflow["approval_required"] and not is_approved(root, target, workflow_name):
        raise RuntimeError(f"stage requires explicit approval: {target}")
    _validate_stage_evidence(workflow, target, evidence)
    state["current_stage"] = target
    state["history"].append(
        {
            "action": "advance",
            "from": current,
            "to": target,
            "at": datetime.now(UTC).isoformat(),
            "evidence": evidence,
        }
    )
    save_state(root, state, workflow_name)
    return state


def run_one(root: Path, workflow_name: str = "paper_v1") -> dict:
    state = load_state(root, workflow_name); target = next_stage(root, workflow_name)
    if target is None: return state
    workflow = _workflow(root, workflow_name)
    if target in workflow["approval_required"] and not is_approved(root, target, workflow_name):
        raise RuntimeError(f"next stage requires explicit approval: {target}")
    if target == "data_verified": evidence = verify_data(root)
    elif target in {"contracts_verified", "model_unit_gate"}: evidence = audit_project(root)
    elif target in {"source_commit", "source_recommit", "source_finalize_commit"}:
        assert_clean(root)
        head = git_head(root)
        if head == "UNCOMMITTED":
            raise RuntimeError(f"{target} requires a real Git commit")
        evidence = {"status": "PASS", "git_commit": head}
    elif target in workflow["approval_required"]:
        evidence = {"status": "PASS", "explicit_approval": target}
    else:
        raise RuntimeError(f"stage {target} requires its registered experiment command; loop will not infer it")
    return record_stage(root, target, evidence, workflow_name)
