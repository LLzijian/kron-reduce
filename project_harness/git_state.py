from __future__ import annotations

import subprocess
from pathlib import Path


def git_head(root: Path) -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else "UNCOMMITTED"


def status_lines(root: Path) -> list[str]:
    result = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "not a Git repository")
    return [line for line in result.stdout.splitlines() if line.strip()]


def assert_clean(root: Path) -> None:
    dirty = status_lines(root)
    if dirty:
        raise RuntimeError("canonical/full runs require a clean Git baseline:\n" + "\n".join(dirty[:30]))
