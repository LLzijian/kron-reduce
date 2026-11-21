"""Locate migrated report views and the exact bytes of historical bindings."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath


def locate_report(root: Path, relative: str, expected_sha256: str | None = None) -> str:
    """Resolve one registered move without changing the caller's digest check."""
    if not relative.startswith("docs/"):
        return relative
    registry = root / "configs/report_relocations.json"
    if not registry.is_file():
        return relative
    locations = json.loads(registry.read_text(encoding="utf-8"))
    if not isinstance(locations, dict):
        raise RuntimeError("REPORT_RELOCATION: expected an object")
    if relative not in locations:
        return relative
    entry = locations[relative]
    if not isinstance(entry, dict):
        raise RuntimeError(f"REPORT_RELOCATION: invalid entry for {relative}")
    target = entry.get("path")
    if expected_sha256 is not None and expected_sha256 == entry.get("sha256"):
        target = entry.get("original_path", target)
    if target == relative:
        return relative
    if not isinstance(target, str) or not target.startswith("docs/reports/"):
        raise RuntimeError(f"REPORT_RELOCATION: invalid destination for {relative}")
    candidate = PurePosixPath(target)
    if (candidate.as_posix() != target or ".." in candidate.parts or "\\" in target
            or ":" in target or candidate.name != PurePosixPath(relative).name):
        raise RuntimeError(f"REPORT_RELOCATION: unsafe destination for {relative}")
    absolute = root.resolve().joinpath(*candidate.parts)
    if absolute.resolve() != absolute or absolute.is_symlink() or absolute.is_junction():
        raise RuntimeError(f"REPORT_RELOCATION: linked destination for {relative}")
    return target
