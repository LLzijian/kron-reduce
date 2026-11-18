from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path, PurePosixPath, PureWindowsPath

ARCHIVE_OUTCOMES = ("completed", "rejected", "failed", "abandoned")


def verify_bound_file(root: Path, relative: str, expected_sha256: str) -> None:
    """Require raw working-file and HEAD-blob bytes to match a bound SHA256."""
    if (not isinstance(relative, str) or not relative or "\\" in relative
            or PureWindowsPath(relative).is_absolute() or PureWindowsPath(relative).drive):
        raise RuntimeError("BOUND_FILE_PATH_UNSAFE: expected a repository-relative POSIX path")
    relative_path = PurePosixPath(relative)
    if relative_path.is_absolute() or any(part in ("", ".", "..") for part in relative_path.parts):
        raise RuntimeError("BOUND_FILE_PATH_UNSAFE: expected a normalized repository-relative path")
    if not isinstance(expected_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise RuntimeError(f"BOUND_FILE_SHA_INVALID: {relative}")

    try:
        root_resolved = root.resolve(strict=True)
        target = root_resolved.joinpath(*relative_path.parts)
        if target.is_symlink() or (hasattr(target, "is_junction") and target.is_junction()):
            raise RuntimeError(f"BOUND_FILE_SYMLINK: {relative}")
        target_resolved = target.resolve(strict=True)
        try:
            target_resolved.relative_to(root_resolved)
        except ValueError as error:
            raise RuntimeError(f"BOUND_FILE_PATH_ESCAPES_ROOT: {relative}") from error
        if not target_resolved.is_file():
            raise RuntimeError(f"BOUND_FILE_NOT_REGULAR_FILE: {relative}")
        worktree_sha = hashlib.sha256(target_resolved.read_bytes()).hexdigest()
    except RuntimeError:
        raise
    except OSError as error:
        raise RuntimeError(f"BOUND_FILE_UNREADABLE: {relative}: {error}") from error
    if worktree_sha != expected_sha256:
        raise RuntimeError(f"BOUND_FILE_WORKTREE_MISMATCH: {relative}")

    try:
        result = subprocess.run(
            ["git", "show", f"HEAD:{relative_path.as_posix()}"],
            cwd=root_resolved, capture_output=True, check=False,
        )
    except OSError as error:
        raise RuntimeError(f"BOUND_FILE_GIT_UNAVAILABLE: {relative}: {error}") from error
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"BOUND_FILE_NOT_IN_HEAD: {relative}"
            + (f" ({detail})" if detail else "")
        )
    if hashlib.sha256(result.stdout).hexdigest() != expected_sha256:
        raise RuntimeError(f"BOUND_FILE_HEAD_MISMATCH: {relative}")


def git_head(root: Path) -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "UNCOMMITTED"


def status_lines(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=root, capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "not a Git repository")
    return [line for line in result.stdout.splitlines() if line.strip()]


def assert_clean(root: Path) -> None:
    dirty = status_lines(root)
    if dirty:
        raise RuntimeError("canonical/full runs require a clean Git baseline:\n" + "\n".join(dirty[:30]))


def _archive_git(root: Path, *args: str, message: str | None = None) -> str:
    result = subprocess.run(
        ["git", *args], cwd=root, input=message, capture_output=True,
        encoding="utf-8", errors="replace", check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"git {args[0]} failed")
    return result.stdout.strip()


def archive_attempt(
    root: Path, outcome: str, message: str, *, push: bool = False,
) -> dict[str, str | bool]:
    """Pin the final commit and outcome before an attempt branch is retired."""
    if outcome not in ARCHIVE_OUTCOMES:
        raise RuntimeError(f"unknown archive outcome: {outcome}")
    summary = message.strip()
    if not summary or "\0" in summary:
        raise RuntimeError("archive requires a nonempty summary without NUL bytes")

    branch_ref = _archive_git(root, "symbolic-ref", "--quiet", "HEAD")
    if not branch_ref.startswith("refs/heads/codex/"):
        raise RuntimeError("archive requires a checked-out codex/ attempt branch")
    dirty = status_lines(root)
    if dirty:
        raise RuntimeError("commit the attempt's changes before archiving:\n" + "\n".join(dirty[:30]))
    branch = branch_ref.removeprefix("refs/heads/")
    commit = _archive_git(root, "rev-parse", "--verify", f"{branch_ref}^{{commit}}")
    tag = f"attempt/{branch}/{commit}"
    tag_ref = f"refs/tags/{tag}"
    annotation = f"Branch: {branch}\nOutcome: {outcome}\n\n{summary}"
    existing = _archive_git(
        root, "for-each-ref", "--format=%(refname)%00%(objecttype)%00%(*objectname)%00%(contents)",
        tag_ref,
    )
    if existing:
        if existing != f"{tag_ref}\0tag\0{commit}\0{annotation}":
            raise RuntimeError(f"archive tag already exists with a different record: {tag}")
    else:
        _archive_git(
            root, "tag", "--annotate", "--no-sign", "--cleanup=verbatim", "-F", "-",
            "--", tag, commit, message=annotation + "\n",
        )
    if push:
        _archive_git(root, "push", "--no-follow-tags", "origin", f"{tag_ref}:{tag_ref}")
    return {"tag": tag, "branch": branch, "commit": commit, "outcome": outcome, "pushed": push}
