"""Shared, deliberately mechanical evidence validation for workflow gates."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .report_storage import locate_report

SCIENTIFIC_VERDICTS = {"ACCEPT", "REJECT", "RETRY"}
PROMOTION_DECISIONS = {"PROMOTE", "KEEP_BASELINE"}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
V2_FIELD_TYPES: dict[str, type | tuple[type, ...]] = {
    "status": str,
    "process_status": str,
    "result_dir": str,
    "candidate_run": str,
    "unit_test_count": int,
    "project_audit_status": str,
    "data_check_status": str,
    "model_contract_status": str,
    "leakage_contract_status": str,
    "level_6_claimed": bool,
    "source_snapshot_limitations": str,
    "artifact_manifest": dict,
    "result_artifact_manifest": dict,
    "git_commit": str,
    "previous_retry_evidence": str,
    "scientific_verdict": str,
    "promotion_decision": str,
    "rollback_target": str,
    "claim_boundary": str,
    "paper_usable_claims": list,
    "negative_evidence_run": str,
    "next_hypothesis": str,
    "branch_terminal": str,
    "acceptance_run": str,
    "report_path": str,
    "merge_commit": str,
}


class EvidenceValidationError(RuntimeError):
    """Stable machine-readable evidence failure.

    ``code`` is intentionally short and fixed so a workflow report can expose
    a useful reason without scraping an English exception message.
    """

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class EvidenceContext:
    root: Path
    workflow: dict[str, Any]
    target: str
    state: dict[str, Any] | None = None
    predecessor: str | None = None
    allow_archived_runs: bool = False


def is_empty(value: object) -> bool:
    return value is None or (isinstance(value, str) and not value.strip()) or (
        isinstance(value, (dict, list, tuple, set)) and not value
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_object_no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _safe_relative(root: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative or "//" in relative:
        raise EvidenceValidationError("EVIDENCE_PATH", "artifact path must be a nonempty POSIX relative path")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts or "." in candidate.parts or relative.startswith("./"):
        raise EvidenceValidationError("EVIDENCE_PATH", f"unsafe artifact path: {relative!r}")
    resolved_root, resolved = root.resolve(), (root / candidate).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise EvidenceValidationError("EVIDENCE_PATH", f"artifact path escapes directory: {relative!r}") from exc
    return root / candidate


def _artifact_directory_missing(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return True
    except OSError as exc:
        raise EvidenceValidationError("ARTIFACT_DIRECTORY", "cannot inspect artifact directory") from exc
    return False


def verify_artifact_directory(
    directory: Path,
    manifest: dict[str, str] | None = None,
    *,
    manifest_name: str = "artifact_hashes.json",
    expected_manifest_sha256: str | None = None,
    strict: bool = False,
) -> dict[str, str]:
    """Verify a result directory, including a strict all-files mode.

    Existing callers can still pass an explicit ``manifest`` and get the old
    listed-only behaviour.  ``strict=True`` loads the manifest when omitted
    and also rejects every unlisted file, symlink, failed-run marker, bad hash,
    and optionally a manifest whose own SHA differs from the registered one.
    """
    directory = Path(directory)
    if not directory.is_dir() or directory.is_symlink():
        raise EvidenceValidationError("ARTIFACT_DIRECTORY", "artifact directory is missing or a symlink")
    if (directory / "RUN_FAILED.json").exists():
        raise EvidenceValidationError("ARTIFACT_FAILED_RUN", "artifact directory records a failed run")
    if not isinstance(manifest_name, str) or manifest_name != Path(manifest_name).name:
        raise EvidenceValidationError("ARTIFACT_MANIFEST", "manifest name must be a safe filename")
    manifest_path = directory / manifest_name
    if manifest is None:
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise EvidenceValidationError("ARTIFACT_MANIFEST", "artifact hash manifest is missing or unsafe")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"), object_pairs_hook=_json_object_no_duplicates)
        except (OSError, ValueError) as exc:
            raise EvidenceValidationError("ARTIFACT_MANIFEST", "artifact hash manifest is unreadable") from exc
    if not isinstance(manifest, dict) or not manifest or not all(isinstance(key, str) and isinstance(value, str) for key, value in manifest.items()):
        raise EvidenceValidationError("ARTIFACT_MANIFEST", "artifact hash manifest must map paths to SHA256 strings")
    if expected_manifest_sha256 is not None:
        if not isinstance(expected_manifest_sha256, str) or not SHA256_RE.fullmatch(expected_manifest_sha256.lower()):
            raise EvidenceValidationError("ARTIFACT_MANIFEST_SHA", "expected manifest SHA256 is invalid")
        if not manifest_path.is_file() or sha256_file(manifest_path) != expected_manifest_sha256.lower():
            raise EvidenceValidationError("ARTIFACT_MANIFEST_SHA", "artifact manifest SHA256 does not match")
    registered: set[str] = set()
    for relative, expected in manifest.items():
        if not SHA256_RE.fullmatch(expected.lower()):
            raise EvidenceValidationError("ARTIFACT_HASH", f"invalid SHA256 for {relative!r}")
        path = _safe_relative(directory, relative)
        if path.is_symlink() or not path.is_file():
            raise EvidenceValidationError("ARTIFACT_MISSING", f"registered artifact is missing or unsafe: {relative}")
        if sha256_file(path) != expected.lower():
            raise EvidenceValidationError("ARTIFACT_HASH", f"SHA256 mismatch: {relative}")
        canonical = Path(relative).as_posix()
        if canonical in registered or canonical != relative:
            raise EvidenceValidationError("ARTIFACT_MANIFEST", f"noncanonical or duplicate artifact path: {relative}")
        registered.add(canonical)
    if strict:
        actual: set[str] = set()
        for path in directory.rglob("*"):
            if path.is_symlink():
                raise EvidenceValidationError("ARTIFACT_SYMLINK", f"symlink in artifact directory: {path.relative_to(directory)}")
            if path.is_file() and path != manifest_path:
                actual.add(path.relative_to(directory).as_posix())
        extra = sorted(actual.difference(registered))
        missing = sorted(registered.difference(actual))
        if extra or missing:
            raise EvidenceValidationError("ARTIFACT_INVENTORY", f"unregistered={extra}, missing={missing}")
    return dict(manifest)


def _contract_error(context: EvidenceContext, evidence: dict[str, Any]) -> None:
    contract = context.workflow.get("stage_evidence_contracts", {}).get(context.target, {})
    required = contract.get("required_fields", [])
    missing = [field for field in required if field not in evidence]
    if missing:
        raise EvidenceValidationError("EVIDENCE_REQUIRED", f"stage {context.target!r} evidence is missing required fields: {missing}")
    for field, allowed in contract.get("allowed_values", {}).items():
        if evidence.get(field) not in allowed:
            raise EvidenceValidationError("EVIDENCE_ALLOWED", f"stage {context.target!r} evidence field {field!r} must be one of {allowed}")
    empty = [field for field in contract.get("non_empty_fields", []) if is_empty(evidence.get(field))]
    if empty:
        raise EvidenceValidationError("EVIDENCE_EMPTY", f"stage {context.target!r} evidence has empty required fields: {empty}")
    conditional_missing = [
        field for discriminator, cases in contract.get("conditional_required_fields", {}).items()
        for field in cases.get(evidence.get(discriminator), [])
        if field not in evidence or is_empty(evidence.get(field))
    ]
    if conditional_missing:
        raise EvidenceValidationError("EVIDENCE_CONDITIONAL", f"stage {context.target!r} evidence is missing conditional fields: {conditional_missing}")
    types = contract.get("field_types", contract.get("types", {}))
    if types:
        names = {"string": str, "object": dict, "array": list, "boolean": bool, "number": (int, float), "integer": int}
        for field, expected in types.items():
            expected_type = names.get(expected) if isinstance(expected, str) else None
            if expected_type is None or field not in evidence or not isinstance(evidence[field], expected_type) or (expected in {"number", "integer"} and isinstance(evidence[field], bool)):
                raise EvidenceValidationError("EVIDENCE_TYPE", f"stage {context.target!r} evidence field {field!r} has invalid type")


def _verify_registered_artifact_manifest(context: EvidenceContext, evidence: dict[str, Any]) -> None:
    binding = evidence.get("artifact_manifest")
    if not isinstance(binding, dict):
        raise EvidenceValidationError("EVIDENCE_ARTIFACT_MANIFEST", "artifact_manifest must be an object")
    run_id, relative, digest = binding.get("run_id"), binding.get("path"), binding.get("sha256")
    if not isinstance(run_id, str) or not run_id or "/" in run_id or "\\" in run_id or run_id in {".", ".."}:
        raise EvidenceValidationError("EVIDENCE_ARTIFACT_RUN", "artifact_manifest run_id is unsafe")
    if relative != "artifact_hashes.json":
        raise EvidenceValidationError("EVIDENCE_ARTIFACT_PATH", "artifact_manifest path must be artifact_hashes.json")
    if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest.lower()):
        raise EvidenceValidationError("EVIDENCE_ARTIFACT_SHA", "artifact_manifest sha256 is invalid")
    directory = _safe_relative(context.root / "results", run_id)
    if context.target == "model_unit_gate" and evidence.get("result_dir") != run_id:
        raise EvidenceValidationError("EVIDENCE_ARTIFACT_RUN", "model_unit_gate result_dir must equal artifact_manifest run_id")
    archived = False
    if context.allow_archived_runs and _artifact_directory_missing(directory):
        from .archive_storage import verify_archived_run

        archived = verify_archived_run(context.root, run_id, digest.lower())
    if not archived:
        verify_artifact_directory(directory, expected_manifest_sha256=digest.lower(), strict=True)
    if context.target == "report" and evidence.get("acceptance_run") != run_id:
        raise EvidenceValidationError("EVIDENCE_ARTIFACT_RUN", "report acceptance_run must equal artifact_manifest run_id")


def _verify_real_git_commit(root: Path, commit: object) -> None:
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-fA-F]{40}(?:[0-9a-fA-F]{24})?", commit):
        raise EvidenceValidationError("EVIDENCE_COMMIT", "merge_commit must be a Git commit SHA")
    completed = subprocess.run(
        ["git", "-C", str(root), "cat-file", "-e", f"{commit}^{{commit}}"],
        check=False, capture_output=True, text=True,
    )
    if completed.returncode != 0:
        raise EvidenceValidationError("EVIDENCE_COMMIT", "merge_commit does not resolve to a Git commit")
    ancestor = subprocess.run(["git", "-C", str(root), "merge-base", "--is-ancestor", commit, "HEAD"], check=False, capture_output=True, text=True)
    if ancestor.returncode != 0:
        raise EvidenceValidationError("EVIDENCE_COMMIT_ANCESTRY", "merge_commit must be an ancestor of HEAD")


def merge_evidence_commit(evidence: object) -> str | None:
    """Return a merge-gate commit, accepting the pre-``merge_commit`` schema.

    ``git_commit`` was the original merge-gate field.  New evidence records
    ``merge_commit`` explicitly, but durable reports must still be able to
    bind an earlier, otherwise-valid gate entry.
    """
    if not isinstance(evidence, dict):
        return None
    for field in ("merge_commit", "git_commit"):
        commit = evidence.get(field)
        if isinstance(commit, str) and commit.strip():
            return commit
    return None


def _history_evidence(context: EvidenceContext, stage: str) -> dict[str, Any] | None:
    for event in reversed((context.state or {}).get("history", [])):
        if event.get("action") == "advance" and event.get("to") == stage and isinstance(event.get("evidence"), dict):
            return event["evidence"]
    return None


def _verify_report_bindings(context: EvidenceContext, evidence: dict[str, Any]) -> None:
    acceptance = _history_evidence(context, "acceptance")
    if not acceptance or evidence.get("acceptance_run") != acceptance.get("result_dir"):
        raise EvidenceValidationError("EVIDENCE_ACCEPTANCE_RUN", "report acceptance_run must bind the recorded acceptance result_dir")
    if evidence.get("artifact_manifest") != acceptance.get("artifact_manifest"):
        raise EvidenceValidationError("EVIDENCE_ACCEPTANCE_MANIFEST", "report artifact_manifest must equal acceptance history")
    acceptance_dir = context.root / "results" / str(evidence["acceptance_run"])
    if not (acceptance_dir / "acceptance.json").is_file():
        raise EvidenceValidationError("EVIDENCE_ACCEPTANCE_RUN", "bound acceptance run lacks acceptance.json")
    verify_artifact_directory(acceptance_dir, strict=True)
    logical_path = _safe_relative(context.root, evidence.get("report_path"))
    relative = logical_path.relative_to(context.root).as_posix()
    binding = evidence.get("report_artifact")
    expected_sha = binding.get("sha256") if isinstance(binding, dict) else None
    report_path = _safe_relative(context.root, locate_report(context.root, relative, expected_sha))
    if not relative.startswith(("docs/reports/", "reports/generated/")) or not report_path.is_file():
        raise EvidenceValidationError("EVIDENCE_REPORT_PATH", "report_path must exist under docs/reports or reports/generated")
    if not isinstance(binding, dict) or binding.get("path") != relative or expected_sha != sha256_file(report_path):
        raise EvidenceValidationError("EVIDENCE_REPORT_HASH", "report_artifact must bind report_path and SHA256")
    if evidence.get("branch_terminal") == "rollback_report":
        rollback = _history_evidence(context, "rollback_report")
        if not rollback or evidence.get("negative_evidence_run") != rollback.get("negative_evidence_run"):
            raise EvidenceValidationError("EVIDENCE_NEGATIVE_RUN", "report negative_evidence_run must bind rollback history")
        if evidence.get("rollback_target") != rollback.get("rollback_target") or evidence.get("rollback_target") != acceptance.get("rollback_target"):
            raise EvidenceValidationError("EVIDENCE_ROLLBACK_TARGET", "report rollback_target must bind acceptance and rollback history")
        negative_id = evidence["negative_evidence_run"]
        if not isinstance(negative_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", negative_id):
            raise EvidenceValidationError("EVIDENCE_NEGATIVE_RUN", "negative run identity is unsafe")
        negative_dir = context.root / "results" / negative_id
        verify_artifact_directory(negative_dir, strict=True)
    if evidence.get("branch_terminal") == "merge_gate":
        merge = _history_evidence(context, "merge_gate")
        merge_commit = merge_evidence_commit(merge)
        if evidence.get("merge_commit") != merge_commit:
            raise EvidenceValidationError("EVIDENCE_MERGE_COMMIT", "report merge_commit must equal merge_gate history")
    result_binding = evidence.get("result_artifact_manifest")
    if "result_dir" in evidence and (
        not isinstance(result_binding, dict) or result_binding.get("run_id") != evidence["result_dir"]
    ):
        raise EvidenceValidationError("EVIDENCE_ARTIFACT_RUN", "report result_dir must bind its own result manifest")
    if result_binding is not None:
        result_evidence = dict(evidence, artifact_manifest=result_binding, acceptance_run=result_binding.get("run_id") if isinstance(result_binding, dict) else None)
        _verify_registered_artifact_manifest(context, result_evidence)


def validate_stage_evidence(context: EvidenceContext, evidence: object) -> None:
    """Validate one transition identically for runtime and durable audit."""
    if not isinstance(evidence, dict):
        raise EvidenceValidationError("EVIDENCE_OBJECT", "stage evidence must be a JSON object")
    if evidence.get("status") != "PASS":
        raise EvidenceValidationError("EVIDENCE_STATUS", f"gate {context.target} failed")
    if "process_status" in evidence and evidence["process_status"] != "PASS":
        raise EvidenceValidationError("EVIDENCE_PROCESS_STATUS", f"stage {context.target!r} process_status must be PASS")
    _contract_error(context, evidence)
    if context.workflow.get("evidence_contract_version") == 2:
        for field, expected in V2_FIELD_TYPES.items():
            if field in evidence and (
                not isinstance(evidence[field], expected)
                or (expected is int and isinstance(evidence[field], bool))
            ):
                raise EvidenceValidationError(
                    "EVIDENCE_TYPE",
                    f"stage {context.target!r} evidence field {field!r} has invalid v2 type",
                )
        if "unit_test_count" in evidence and (not isinstance(evidence["unit_test_count"], int) or isinstance(evidence["unit_test_count"], bool) or evidence["unit_test_count"] <= 0):
            raise EvidenceValidationError("EVIDENCE_UNIT_TEST_COUNT", "unit_test_count must be a positive integer")
    verdict = evidence.get("scientific_verdict")
    if verdict is not None and verdict not in SCIENTIFIC_VERDICTS:
        raise EvidenceValidationError("EVIDENCE_VERDICT", f"unregistered scientific_verdict: {verdict!r}")
    promotion = evidence.get("promotion_decision")
    if promotion is not None and promotion not in PROMOTION_DECISIONS:
        raise EvidenceValidationError("EVIDENCE_PROMOTION", f"unregistered promotion_decision: {promotion!r}")
    accepted = context.workflow.get("accepted_scientific_verdicts")
    if accepted and verdict is not None and verdict not in accepted:
        raise EvidenceValidationError("EVIDENCE_ACCEPTANCE", f"master stage {context.target!r} requires an accepted scientific verdict")
    if context.target == "report" and context.predecessor is not None:
        terminal = evidence.get("branch_terminal")
        if terminal != context.predecessor:
            raise EvidenceValidationError("EVIDENCE_BRANCH_TERMINAL", f"report branch_terminal {terminal!r} does not match predecessor {context.predecessor!r}")
    if context.workflow.get("evidence_contract_version") == 2 and context.target in {"model_unit_gate", "report"}:
        _verify_registered_artifact_manifest(context, evidence)
    if context.workflow.get("evidence_contract_version") == 2 and context.target != "report" and "result_dir" in evidence:
        binding = evidence.get("artifact_manifest")
        if not isinstance(binding, dict) or binding.get("run_id") != evidence["result_dir"]:
            raise EvidenceValidationError("EVIDENCE_ARTIFACT_RUN", "v2 result_dir must bind its registered artifact manifest")
        _verify_registered_artifact_manifest(context, evidence)
    if context.workflow.get("evidence_contract_version") == 2 and context.target == "report":
        _verify_report_bindings(context, evidence)
    if context.workflow.get("evidence_contract_version") == 2 and context.target == "report" and "merge_commit" in evidence:
        _verify_real_git_commit(context.root, evidence["merge_commit"])
    if context.workflow.get("evidence_contract_version") == 2 and context.target == "source_commit":
        _verify_real_git_commit(context.root, evidence.get("git_commit"))
        prior = evidence.get("previous_retry_evidence")
        if prior != context.workflow.get("previous_retry_evidence") or not _safe_relative(context.root, prior).is_file():
            raise EvidenceValidationError("EVIDENCE_PREVIOUS_RETRY", "previous_retry_evidence must bind the registered file")
