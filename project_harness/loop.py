from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .audit import audit_project
from .data_check import verify_data
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
    state.setdefault("approvals", {})[gate] = {"approved_at": datetime.now(timezone.utc).isoformat(), "actor": "explicit_operator_action"}
    state["history"].append({"action": "approve", "gate": gate, "at": datetime.now(timezone.utc).isoformat()})
    save_state(root, state, workflow_name); return state


def next_stage(root: Path, workflow_name: str = "paper_v1") -> str | None:
    workflow, state = _workflow(root, workflow_name), load_state(root, workflow_name)
    index = workflow["stages"].index(state["current_stage"])
    return workflow["stages"][index + 1] if index + 1 < len(workflow["stages"]) else None


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
    state["current_stage"] = target
    state["history"].append(
        {
            "action": "advance",
            "from": current,
            "to": target,
            "at": datetime.now(timezone.utc).isoformat(),
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
    if evidence["status"] != "PASS": raise RuntimeError(f"gate {target} failed")
    return record_stage(root, target, evidence, workflow_name)
