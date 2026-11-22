"""Resolve the current location of a registered local archive."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

from .evidence import _safe_relative


def archive_path(root: Path, relative: str) -> Path:
    """Locate a moved archive group without changing member names or digests."""
    original = _safe_relative(root, relative)
    parts = original.relative_to(root).parts
    if parts[:2] != ("results", "backup") or len(parts) < 3:
        return original

    registry = root / "configs/archive_relocations.json"
    if not registry.is_file():
        return original
    locations = json.loads(registry.read_text(encoding="utf-8"))
    if not isinstance(locations, dict):
        raise RuntimeError("ARCHIVE_RELOCATION: expected an object")
    group = "/".join(parts[:3])
    if group not in locations:
        return original

    destination = locations[group]
    if (not isinstance(destination, str) or not destination.startswith("results/backup/")
            or PurePosixPath(destination).as_posix() != destination):
        raise RuntimeError(f"ARCHIVE_RELOCATION: invalid destination for {group}")
    target = _safe_relative(root, "/".join((destination, *parts[3:])))
    if target.resolve() != root.resolve().joinpath(*target.relative_to(root).parts):
        raise RuntimeError(f"ARCHIVE_RELOCATION: linked destination for {group}")
    if not target.exists() and original.exists():
        return original
    return target
