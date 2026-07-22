from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .audit import audit_project
from .data_check import verify_data
from .git_state import assert_clean, git_head


def _workflow(root: Path) -> dict:
    return json.loads((root / "configs" / "workflows" / "paper_v1.json").read_text(encoding="utf-8"))


def _state_path(root: Path) -> Path:
    path = root / ".project_state" / "loop_state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_state(root: Path) -> dict:
    path = _state_path(root)
    if path.exists(): return json.loads(path.read_text(encoding="utf-8"))
    workflow = _workflow(root)
    return {"workflow": workflow["workflow"], "current_stage": workflow["initial_stage"], "approvals": {}, "history": []}


def save_state(root: Path, state: dict) -> None:
    path = _state_path(root); temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8"); temporary.replace(path)


def is_approved(root: Path, gate: str) -> bool:
    return bool(load_state(root).get("approvals", {}).get(gate))


def approve(root: Path, gate: str) -> dict:
    workflow = _workflow(root)
    if gate not in workflow["approval_required"]:
        raise ValueError(f"gate is not approval-controlled: {gate}")
    state = load_state(root)
    state.setdefault("approvals", {})[gate] = {"approved_at": datetime.now(timezone.utc).isoformat(), "actor": "explicit_operator_action"}
    state["history"].append({"action": "approve", "gate": gate, "at": datetime.now(timezone.utc).isoformat()})
    save_state(root, state); return state


def next_stage(root: Path) -> str | None:
    workflow, state = _workflow(root), load_state(root)
    index = workflow["stages"].index(state["current_stage"])
    return workflow["stages"][index + 1] if index + 1 < len(workflow["stages"]) else None


def record_stage(root: Path, target: str, evidence: object) -> dict:
    """Advance exactly one workflow edge after durable evidence exists."""
    workflow = _workflow(root)
    state = load_state(root)
    current = state["current_stage"]
    expected = next_stage(root)
    if target != expected:
        raise RuntimeError(
            f"cannot record {target!r} from {current!r}; expected {expected!r}"
        )
    if target in workflow["approval_required"] and not is_approved(root, target):
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
    save_state(root, state)
    return state


def run_one(root: Path) -> dict:
    state = load_state(root); target = next_stage(root)
    if target is None: return state
    if target in _workflow(root)["approval_required"] and not is_approved(root, target):
        raise RuntimeError(f"next stage requires explicit approval: {target}")
    if target == "data_verified": evidence = verify_data(root)
    elif target in {"contracts_verified", "model_unit_gate"}: evidence = audit_project(root)
    elif target == "source_commit":
        assert_clean(root)
        head = git_head(root)
        if head == "UNCOMMITTED":
            raise RuntimeError("source_commit requires a real Git commit")
        evidence = {"status": "PASS", "git_commit": head}
    elif target in _workflow(root)["approval_required"]:
        evidence = {"status": "PASS", "explicit_approval": target}
    else:
        raise RuntimeError(f"stage {target} requires its registered experiment command; loop will not infer it")
    if evidence["status"] != "PASS": raise RuntimeError(f"gate {target} failed")
    return record_stage(root, target, evidence)
