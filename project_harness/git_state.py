from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path, PurePosixPath, PureWindowsPath


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
    result = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "not a Git repository")
    return [line for line in result.stdout.splitlines() if line.strip()]


def assert_clean(root: Path) -> None:
    dirty = status_lines(root)
    if dirty:
        raise RuntimeError("canonical/full runs require a clean Git baseline:\n" + "\n".join(dirty[:30]))
