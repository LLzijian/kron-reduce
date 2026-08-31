from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from .audit import audit_project
from .data_check import verify_data
from .evidence import EvidenceContext, validate_stage_evidence
from .git_state import assert_clean, git_head


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
    validate_stage_evidence(
        EvidenceContext(root=root, workflow=workflow, target=target, state=state, predecessor=current),
        evidence,
    )
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
        if target == "source_commit" and "previous_retry_evidence" in workflow:
            evidence["previous_retry_evidence"] = workflow["previous_retry_evidence"]
    elif target in workflow["approval_required"]:
        evidence = {"status": "PASS", "explicit_approval": target}
    else:
        raise RuntimeError(f"stage {target} requires its registered experiment command; loop will not infer it")
    return record_stage(root, target, evidence, workflow_name)
