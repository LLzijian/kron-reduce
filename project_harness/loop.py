from __future__ import annotations

import json
import re
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from filelock import FileLock, Timeout

from .data_check import verify_data
from .evidence import EvidenceContext, validate_stage_evidence
from .git_state import assert_clean, git_head
from .manifest import load_manifest
from .policy import load_policy

SOURCE_STAGES = {"source_commit", "source_recommit", "source_finalize_commit"}
CHECK_STAGES = {"data_verified", "contracts_verified", "model_unit_gate"} | SOURCE_STAGES


def _workflow(root: Path, workflow_name: str = "paper_v1") -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", workflow_name):
        raise ValueError(f"invalid workflow name: {workflow_name!r}")
    path = root / "configs" / "workflows" / f"{workflow_name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"unregistered workflow: {workflow_name}")
    workflow = json.loads(path.read_text(encoding="utf-8"))
    if workflow.get("workflow") != workflow_name:
        raise ValueError(f"workflow identity mismatch in {path}")
    return workflow


def workflow_state_path(root: Path, workflow_name: str) -> Path:
    # Keep the original path for paper_v1 so its completed history remains
    # immutable and readable after named workflows are introduced.
    filename = "loop_state.json" if workflow_name == "paper_v1" else f"{workflow_name}_state.json"
    return root / ".project_state" / filename


def load_state(root: Path, workflow_name: str = "paper_v1") -> dict:
    return _load_state(root, _workflow(root, workflow_name))


def _load_state(root: Path, workflow: dict) -> dict:
    name = workflow["workflow"]
    path = workflow_state_path(root, name)
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("workflow") != name or state.get("current_stage") not in workflow["stages"]:
            raise ValueError(f"invalid workflow identity or stage in {path}")
        return state
    return {"workflow": name, "current_stage": workflow["initial_stage"], "approvals": {}, "history": []}


def _save_state(root: Path, state: dict) -> None:
    path = workflow_state_path(root, state["workflow"])
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temporary.replace(path)


@contextmanager
def workflow_lock(root: Path, workflow_name: str):
    lock_dir = root / ".project_state" / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock = FileLock(lock_dir / f"{workflow_name}.lock", timeout=0)
    try:
        lock.acquire()
    except Timeout as error:
        raise RuntimeError(f"WORKFLOW_BUSY: {workflow_name} already has an active operation") from error
    try:
        yield
    finally:
        lock.release()


def is_approved(root: Path, gate: str, workflow_name: str = "paper_v1") -> bool:
    return approval_recorded(load_state(root, workflow_name), gate)


def approval_recorded(state: dict, gate: str) -> bool:
    return bool(state.get("approvals", {}).get(gate)) and any(
        event.get("action") == "approve" and event.get("gate") == gate
        for event in state["history"]
    )


def approve(root: Path, gate: str, workflow_name: str = "paper_v1") -> dict:
    workflow = _workflow(root, workflow_name)
    if gate not in workflow["approval_required"]:
        raise ValueError(f"gate is not approval-controlled: {gate}")
    with workflow_lock(root, workflow_name):
        state = _load_state(root, workflow)
        at = datetime.now(UTC).isoformat()
        state.setdefault("approvals", {})[gate] = {"approved_at": at, "actor": "explicit_operator_action"}
        state["history"].append({"action": "approve", "gate": gate, "at": at})
        _save_state(root, state)
        return state


def _entry_evidence(state: dict, stage: str) -> dict:
    """Return the evidence that advanced the workflow into ``stage``."""
    for event in reversed(state.get("history", [])):
        if event.get("action") == "advance" and event.get("to") == stage:
            evidence = event.get("evidence", {})
            return evidence if isinstance(evidence, dict) else {}
    return {}


def transition_target(
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
    workflow = _workflow(root, workflow_name)
    return transition_target(workflow, _load_state(root, workflow), outcome)


def plan(root: Path, workflow_name: str = "paper_v1") -> dict:
    """Inspect one workflow edge using only local state and registered metadata."""
    workflow = _workflow(root, workflow_name)
    state = _load_state(root, workflow)
    target = transition_target(workflow, state)
    pending_gate = (
        target if target in workflow["approval_required"] and not approval_recorded(state, target) else None
    )
    evidence = _entry_evidence(state, state["current_stage"])
    last_result = {
        key: evidence[key]
        for key in ("result_dir", "scientific_verdict", "promotion_decision")
        if key in evidence
    }
    authorizations = load_policy(root).get("development_evaluations", {})
    experiments = []
    for name, entry in load_manifest(root)["experiments"].items():
        stage = entry.get("stage")
        if target is None or entry.get("workflow", "paper_v1") != workflow_name or stage not in {None, target}:
            continue
        authorization_id = entry.get("development_authorization_id")
        authorization = authorizations.get(authorization_id, {})
        gate = entry.get("required_gate")
        experiments.append({
            "name": name,
            "stage": stage,
            "execution_mode": entry.get("execution_mode", "generic"),
            "module": entry["module"],
            "pending_approval": gate if gate and not approval_recorded(state, gate) else None,
            "authorization_id": authorization_id,
            "authorization_status": authorization.get("status"),
            "completed_run_id": authorization.get("completed_run_id"),
        })
    built_in_check = (
        target in CHECK_STAGES
        and not (target == "model_unit_gate" and workflow.get("evidence_contract_version") == 2)
        and not any(row["stage"] == target for row in experiments)
    )
    return {
        "workflow": workflow_name,
        "current_stage": state["current_stage"],
        "next_stage": target,
        "pending_approval": pending_gate,
        "built_in_check": built_in_check,
        "evidence_contract": workflow.get("stage_evidence_contracts", {}).get(target),
        "registered_experiments": [row for row in experiments if row["stage"] is not None],
        "experiments_without_stage": [row["name"] for row in experiments if row["stage"] is None],
        "last_result": last_result or None,
    }


def check_stage(workflow: dict, state: dict, target: str) -> None:
    expected = transition_target(workflow, state)
    if target is None or target != expected:
        raise RuntimeError(
            f"cannot record {target!r} from {state['current_stage']!r}; expected {expected!r}"
        )
    if target in workflow["approval_required"] and not approval_recorded(state, target):
        raise RuntimeError(f"stage requires explicit approval: {target}")


def record_stage(root: Path, target: str, evidence: object, workflow_name: str = "paper_v1") -> dict:
    """Advance exactly one workflow edge after durable evidence exists."""
    workflow = _workflow(root, workflow_name)
    with workflow_lock(root, workflow_name):
        return _record_stage(root, workflow, _load_state(root, workflow), target, evidence)


def _record_stage(root: Path, workflow: dict, state: dict, target: str, evidence: object) -> dict:
    """Validate and record one transition with the workflow lock held."""
    check_stage(workflow, state, target)
    current = state["current_stage"]
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
    _save_state(root, state)
    return state


def run_one(root: Path, workflow_name: str = "paper_v1") -> dict:
    from .audit import audit_project

    workflow = _workflow(root, workflow_name)
    with workflow_lock(root, workflow_name):
        state = _load_state(root, workflow)
        target = transition_target(workflow, state)
        if target is None:
            return state
        check_stage(workflow, state, target)
        if target == "data_verified":
            evidence = verify_data(root)
        elif target == "contracts_verified" or (
            target == "model_unit_gate" and workflow.get("evidence_contract_version") != 2
        ):
            evidence = audit_project(root)
        elif target in SOURCE_STAGES:
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
            raise RuntimeError(f"stage {target} requires registered experiment or operator evidence")
        return _record_stage(root, workflow, state, target, evidence)
