"""Register the joint V4 holdout after freeze, then close both scopes after completion."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile

from kron_reduce.workflows.noise_robustness_esrom_v4 import (
    CONFIG,
    SEALED_FREEZE,
    WORKFLOW,
    _phase,
    _plans,
)
from project_harness.evidence import _json_object_no_duplicates, _verify_real_git_commit
from project_harness.git_state import verify_bound_file
from project_harness.loop import approval_recorded, load_state, workflow_lock
from project_harness.manifest import project_root

MANIFEST = "configs/project_manifest.json"
POLICY = "configs/research_policy.json"
SCOPES = {
    "test": ("official_test_evaluations", "test_authorization"),
    "blind": ("independent_confirmations", "blind_authorization"),
}
DECODER = json.JSONDecoder(object_pairs_hook=_json_object_no_duplicates)


def _json_at(path, sha256=None):
    if path.is_symlink() or not path.is_file():
        raise PermissionError(f"missing or unsafe receipt: {path}")
    data = path.read_bytes()
    if sha256 is not None and hashlib.sha256(data).hexdigest() != sha256:
        raise PermissionError(f"receipt SHA-256 mismatch: {path}")
    return DECODER.decode(data.decode("utf-8")), data


def _stage(root, state, stage, digest):
    events = [event for event in state["history"] if event.get("to") == stage]
    if len(events) != 1 or events[0].get("action") != "advance":
        raise PermissionError(f"expected a single completed {stage} stage")
    event = events[0]
    evidence = event["evidence"]
    artifact = evidence["artifact_manifest"]
    run_id = artifact["run_id"]
    if (evidence["status"] != "PASS" or evidence["plan_sha256"] != digest
            or artifact != evidence["result_artifact_manifest"] or run_id != evidence["result_dir"]
            or artifact["path"] != "artifact_hashes.json"
            or re.fullmatch(rf"{WORKFLOW}-{stage}-[A-Za-z0-9-]+", run_id) is None
            or re.fullmatch(r"[0-9a-f]{64}", artifact["sha256"]) is None):
        raise PermissionError(f"invalid immutable {stage} run binding")
    directory = root / "results" / run_id
    if directory.resolve(strict=True) != directory or (directory / "RUN_FAILED.json").exists():
        raise PermissionError(f"unsafe or failed {stage} run")
    hashes, _ = _json_at(directory / artifact["path"], artifact["sha256"])
    if any(re.fullmatch(r"[0-9a-f]{64}", sha) is None for sha in hashes.values()):
        raise PermissionError(f"invalid artifact hashes for {stage}")
    recorded, _ = _json_at(directory / "stage_evidence.json", hashes["stage_evidence.json"])
    runner_fields = {"status", "result_dir", "artifact_manifest", "result_artifact_manifest"}
    if recorded != {key: value for key, value in evidence.items() if key not in runner_fields}:
        raise PermissionError(f"workflow history differs from the {stage} receipt")
    return directory, event, hashes


def _freeze_proof(root, state, plan, digest):
    directory, event, hashes = _stage(root, state, "freeze", digest)
    evidence = event["evidence"]
    if _phase(root, "freeze", digest) != (directory, evidence):
        raise PermissionError("freeze differs from the registered phase")
    _verify_real_git_commit(root, evidence["source_commit"])
    path = directory / "freeze_receipt.json"
    sha = hashes[path.name]
    frozen, data = _json_at(path, sha)
    predecessor = event["from"]
    if predecessor not in ("validation", "adapt"):
        raise PermissionError("freeze requires accepted val evidence")
    val_dir, val_event, val_hashes = _stage(root, state, predecessor, digest)
    val = val_event["evidence"]
    selection, _ = _json_at(val_dir / "selection.json", val_hashes["selection.json"])
    if (evidence["frozen_model_manifest"] != {"path": path.relative_to(root).as_posix(), "sha256": sha}
            or frozen["source_commit"] != evidence["source_commit"] or frozen["plan_sha256"] != digest
            or frozen["validation_run"] != val["artifact_manifest"]
            or not (frozen["selection_closed"] is True and frozen["fitting_closed"] is True)
            or val["scientific_verdict"] != "ACCEPT" or val["val_cases"] != plan["counts"]["val"]
            or (predecessor == "validation"
                and (val["event_predictions"] != plan["limits"]["validation_predictions"]
                     or val["optimizer_updates"] != 0))
            or selection["passed"] is not True or selection["candidate"] != frozen["candidate"]
            or selection["optimizer_updates"] != frozen["optimizer_updates"]
            or val["optimizer_updates"] != frozen["optimizer_updates"]
            or any(receipt[key] != 0 for receipt in (evidence, val)
                   for key in ("test_trajectory_reads", "blind_container_opens"))):
        raise PermissionError("the freeze receipt does not bind a complete passing val without holdout access")
    return {"path": SEALED_FREEZE, "sha256": sha}, data


def _scope_documents(manifest, policy, plan, digest, binding):
    base = manifest["experiments"][WORKFLOW]
    split = plan["split_reference_sha256"]
    if (plan["counts"] != {"train": 346, "val": 107, "test": 76, "blind_v2": 94}
            or plan["limits"]["test_executions"] != 1 or plan["limits"]["blind_executions"] != 1
            or not split == base["split_reference_sha256"] == policy["official_split_reference_sha256"]
            or plan["input_contract"] != base["input_contract_id"]
            or re.fullmatch(rf"{WORKFLOW}_[0-9]{{8}}", plan["scope"]) is None):
        raise PermissionError("the joint holdout differs from the registered plan or policy")
    blind_dataset = "kiki_ieee39_blind_v2"
    joint = {
        "status": "active", "authority": "explicit_user_v4_joint_holdout_after_accepted_val",
        "max_executions": 1, "retry_allowed": False, "fit_attempts": 0,
        "gradients": 0, "optimizer_updates": 0,
        "plan_path": CONFIG, "plan_sha256": digest, "joint_scope": plan["scope"],
        "frozen_model_manifest": binding,
    }
    entries, scopes = {}, {}
    for suffix, (category, authorization) in SCOPES.items():
        name = f"{WORKFLOW}_{suffix}"
        key = f"{name}_{plan['scope'].rsplit('_', 1)[1]}"
        entry = base | {
            "stage": "blind", "required_gate": "blind", "reads_validation_performance": False,
            "reads_test_performance": suffix == "test", authorization: key,
        }
        scope = joint | {"experiment": name}
        if suffix == "test":
            scope |= {"test_population": base["test_population"], "split_reference_sha256": split,
                      "test_cases": plan["counts"]["test"]}
        else:
            entry.pop("test_population")
            entry.pop("split_reference_sha256")
            entry.pop("data_policy")
            entry |= {"dataset": blind_dataset, "training_split": None,
                      "execution_mode": "bounded_kiki_independent_cli_only"}
            scope |= {
                "dataset": blind_dataset, "allowed_roles": ["blind"], "test_cases": 0,
                "events": plan["counts"]["blind_v2"],
                "dataset_manifest_sha256": plan["blind_manifest_sha256"],
                "dataset_checksums_sha256": plan["blind_checksums_sha256"],
            }
        entries[name] = entry
        scopes[category] = (key, scope)
    return entries, scopes


def _edit_section(document, section, updates):
    """Keep every byte outside the appended or replaced registry entries."""
    anchor = f'\n  "{section}": {{'
    if document.count(anchor) != 1:
        raise ValueError(f"unexpected JSON layout for {section}")
    start = document.index(anchor) + len(anchor) - 1
    old, end = DECODER.raw_decode(document, start)
    edits = []
    appended = []
    for name, value in updates.items():
        if name in old and old[name] == value:
            continue
        row = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False).replace("\n", "\n    ")
        field = f'\n    "{name}": '
        if name in old:
            position = document.index(field, start, end) + len(field)
            previous, stop = DECODER.raw_decode(document, position)
            if previous != old[name]:
                raise ValueError(f"unexpected JSON layout for {name}")
            edits.append((position, stop, row))
        else:
            appended.append(field + row)
    if appended:
        position = len(document[:end - 1].rstrip())
        edits.append((position, position, ("," if old else "") + ",".join(appended)))
    expected = DECODER.decode(document)
    expected[section].update(updates)
    for start, end, replacement in sorted(edits, reverse=True):
        document = document[:start] + replacement + document[end:]
    if DECODER.decode(document) != expected:
        raise ValueError(f"unexpected change outside V4 entries in {section}")
    return document


def _atomic_write(path, data, *, replace):
    with NamedTemporaryFile(dir=path.parent, prefix=".v4-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            stream.close()
            temporary.unlink()
            raise
    try:
        if replace:
            temporary.replace(path)
        else:
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _reservation(root, scope):
    common = subprocess.check_output(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=root, text=True,
    ).strip()
    return Path(common) / "m4v2_m5_budgets" / f"{scope}.json"


def update(root, action):
    state = load_state(root, WORKFLOW)
    valid_stages = {"register": {"freeze"}, "consume": {"blind", "closed"}}[action]
    if state["current_stage"] not in valid_stages:
        raise PermissionError(f"cannot {action} the V4 holdout at {state['current_stage']!r}")
    plan, _, digest = _plans(root)
    binding, freeze_bytes = _freeze_proof(root, state, plan, digest)
    manifest_path, policy_path = root / MANIFEST, root / POLICY
    manifest, manifest_bytes = _json_at(manifest_path)
    policy, policy_bytes = _json_at(policy_path)
    entries, scopes = _scope_documents(manifest, policy, plan, digest, binding)
    marker = _reservation(root, plan["scope"])
    sealed = root / SEALED_FREEZE
    if sealed.resolve() != sealed:
        raise PermissionError("unsafe sealed freeze destination")
    if sealed.exists():
        _json_at(sealed, binding["sha256"])
    consumed = {}
    if action == "register":
        if os.path.lexists(marker):
            raise PermissionError("V4 joint holdout has already reserved its one execution")
    else:
        verify_bound_file(root, SEALED_FREEZE, binding["sha256"])
        _, event, hashes = _stage(root, state, "blind", digest)
        held_out = event["evidence"]
        artifact = held_out["artifact_manifest"]
        run_id = artifact["run_id"]
        reservation, _ = _json_at(marker)
        if (event["from"] != "freeze" or not approval_recorded(state, "blind")
                or held_out["frozen_model_manifest"] != binding or held_out["optimizer_updates"] != 0
                or held_out["test_cases"] != plan["counts"]["test"]
                or held_out["test_trajectory_reads"] != plan["counts"]["test"]
                or held_out["blind_cases"] != plan["counts"]["blind_v2"] or held_out["blind_container_opens"] != 1
                or held_out["event_predictions"] != plan["limits"]["holdout_predictions"]
                or hashes["noise_manifest.json"] != held_out["noise_manifest_sha256"]
                or reservation != {"run_id": run_id, "source_commit": held_out["source_commit"], "retry_allowed": False}):
            raise PermissionError("the joint holdout receipt does not match its reservation and full-roster budgets")
        consumed = {"status": "consumed", "run_id": run_id, "result_manifest_sha256": artifact["sha256"],
                    "consumed_at": event["at"]}

    for name, entry in entries.items():
        previous = manifest["experiments"].get(name)
        if ((name in manifest["experiments"] or action == "consume") and previous != entry):
            raise PermissionError(f"conflicting V4 experiment: {name}")
    new_manifest = _edit_section(manifest_bytes.decode("utf-8"), "experiments", entries).encode("utf-8")
    new_policy = policy_bytes.decode("utf-8")
    for category, (key, active) in scopes.items():
        updated = active | consumed
        records = policy[category]
        if any(name != key and (row.get("experiment") == active["experiment"]
                               or row.get("joint_scope") == plan["scope"])
               for name, row in records.items()):
            raise PermissionError(f"conflicting V4 authorization in {category}")
        if (key in records or action == "consume") and records.get(key) not in (active, updated):
            raise PermissionError(f"authorization has changed or has already expired: {key}")
        new_policy = _edit_section(new_policy, category, {key: updated})
    new_policy = new_policy.encode("utf-8")

    if action == "register" and not sealed.exists():
        sealed.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(sealed, freeze_bytes, replace=False)
    if manifest_path.read_bytes() != manifest_bytes or policy_path.read_bytes() != policy_bytes:
        raise PermissionError("registry files changed during the V4 update")
    if new_manifest != manifest_bytes:
        _atomic_write(manifest_path, new_manifest, replace=True)
    if new_policy != policy_bytes:
        _atomic_write(policy_path, new_policy, replace=True)
    return {"action": action, "frozen_model_manifest": binding, "authorizations": [key for key, _ in scopes.values()],
            **consumed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "consume"))
    args = parser.parse_args()
    root = project_root()
    with workflow_lock(root, WORKFLOW):
        result = update(root, args.action)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
