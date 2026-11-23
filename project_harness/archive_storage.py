"""Resolve the current location of a registered local archive."""

from __future__ import annotations

import hashlib
import json
import stat
import zipfile
import zlib
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

from .evidence import (
    SHA256_RE,
    EvidenceValidationError,
    _artifact_directory_missing,
    _json_object_no_duplicates,
    _safe_relative,
)


def _relocations(root: Path) -> dict:
    registry = root / "configs/archive_relocations.json"
    if not registry.is_file():
        return {}
    locations = json.loads(registry.read_text(encoding="utf-8"), object_pairs_hook=_json_object_no_duplicates)
    if not isinstance(locations, dict):
        raise EvidenceValidationError("ARCHIVE_RELOCATION", "expected an object")
    return locations


def archive_path(root: Path, relative: str) -> Path:
    """Locate a moved archive group without changing member names or digests."""
    original = _safe_relative(root, relative)
    parts = original.relative_to(root).parts
    if parts[:2] != ("results", "backup") or len(parts) < 3:
        return original

    locations = _relocations(root)
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


def _member_path(name: str) -> PurePosixPath:
    if not isinstance(name, str):
        raise EvidenceValidationError("ARTIFACT_ARCHIVE_PATH", "archive path must be a string")
    path = PurePosixPath(name)
    if (not path.parts or path.is_absolute() or path.as_posix() != name or ".." in path.parts
            or any(char in name for char in ("\\", ":", "\0"))):
        raise EvidenceValidationError("ARTIFACT_ARCHIVE_PATH", f"unsafe archive path: {name!r}")
    return path


@contextmanager
def _run_archive(root: Path, run_id: str):
    if len(_member_path(run_id).parts) != 1:
        raise EvidenceValidationError("ARTIFACT_ARCHIVE_PATH", f"invalid run ID: {run_id!r}")
    try:
        destination = _relocations(root).get(f"results/{run_id}")
        if destination is None:
            yield None
            return
        if (not isinstance(destination, str) or not destination.startswith("results/backup/")
                or _member_path(destination).suffix != ".zip"):
            raise EvidenceValidationError("ARTIFACT_ARCHIVE_PATH", f"invalid ZIP destination for {run_id}")
        archive = archive_path(root, destination)
        if archive.resolve() != root.resolve() / archive.relative_to(root):
            raise EvidenceValidationError("ARTIFACT_ARCHIVE_PATH", f"linked ZIP destination for {run_id}")
        with zipfile.ZipFile(archive) as stream:
            members: dict[str, zipfile.ZipInfo] = {}
            seen: set[str] = set()
            for entry in stream.infolist():
                name = entry.filename.removesuffix("/")
                path = _member_path(name)
                mode = stat.S_IFMT(entry.external_attr >> 16)
                expected_mode = stat.S_IFDIR if entry.is_dir() else stat.S_IFREG
                if (entry.orig_filename != entry.filename or mode not in (0, expected_mode)
                        or name.casefold() in seen):
                    raise EvidenceValidationError("ARTIFACT_ARCHIVE_PATH", f"unsafe or duplicate ZIP entry: {name!r}")
                seen.add(name.casefold())
                if path.parts[0] == run_id and not entry.is_dir():
                    if len(path.parts) < 2:
                        raise EvidenceValidationError("ARTIFACT_ARCHIVE_PATH", f"ZIP entry needs a run directory: {name!r}")
                    members[path.relative_to(run_id).as_posix()] = entry
            yield stream, members
    except EvidenceValidationError:
        raise
    except (EOFError, OSError, ValueError, RuntimeError, zipfile.BadZipFile, zlib.error) as exc:
        raise EvidenceValidationError("ARTIFACT_ARCHIVE", f"cannot read archived run {run_id}: {exc}") from exc


def verify_archived_run(root: Path, run_id: str, expected_manifest_sha256: str) -> bool:
    """Replay the registered manifest over an archived run without extracting it."""
    with _run_archive(root, run_id) as archived:
        if archived is None:
            return False
        stream, members = archived
        if any(relative.split("/", 1)[0].casefold() == "run_failed.json" for relative in members):
            raise EvidenceValidationError("ARTIFACT_FAILED_RUN", "artifact archive records a failed run")
        manifest_entry = members.pop("artifact_hashes.json", None)
        if manifest_entry is None:
            raise EvidenceValidationError("ARTIFACT_MANIFEST", "archive lacks the artifact hash manifest")
        with stream.open(manifest_entry) as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != expected_manifest_sha256:
                raise EvidenceValidationError("ARTIFACT_MANIFEST_SHA", "artifact manifest SHA256 does not match")
        manifest = json.loads(stream.read(manifest_entry).decode("utf-8"), object_pairs_hook=_json_object_no_duplicates)
        if not isinstance(manifest, dict) or not manifest:
            raise EvidenceValidationError("ARTIFACT_MANIFEST", "artifact hash manifest must map paths to SHA256 strings")
        if members.keys() != manifest.keys():
            raise EvidenceValidationError("ARTIFACT_INVENTORY", "ZIP inventory differs from the artifact manifest")
        for relative, expected in manifest.items():
            if not isinstance(expected, str) or not SHA256_RE.fullmatch(expected.lower()):
                raise EvidenceValidationError("ARTIFACT_HASH", f"invalid SHA256 for {relative!r}")
            with stream.open(members[relative]) as handle:
                if hashlib.file_digest(handle, "sha256").hexdigest() != expected.lower():
                    raise EvidenceValidationError("ARTIFACT_HASH", f"SHA256 mismatch: {relative}")
    return True


def read_run_artifact(root: Path, run_id: str, relative: str, expected_sha256: str) -> bytes:
    """Read one hash-bound file from a run directory or its registered archive."""
    if len(_member_path(run_id).parts) != 1:
        raise EvidenceValidationError("ARTIFACT_ARCHIVE_PATH", f"invalid run ID: {run_id!r}")
    _member_path(relative)
    directory = root / "results" / run_id
    path = _safe_relative(directory, relative)
    if not isinstance(expected_sha256, str) or not SHA256_RE.fullmatch(expected_sha256.lower()):
        raise EvidenceValidationError("ARTIFACT_HASH", f"invalid SHA256 for {relative!r}")
    if not _artifact_directory_missing(directory):
        if (not path.is_file() or path.resolve() != root.resolve() / path.relative_to(root)):
            raise EvidenceValidationError("ARTIFACT_MISSING", f"registered artifact is missing or unsafe: {relative}")
        payload = path.read_bytes()
    else:
        with _run_archive(root, run_id) as archived:
            if archived is None or relative not in archived[1]:
                raise EvidenceValidationError("ARTIFACT_MISSING", f"registered artifact is missing: {relative}")
            stream, members = archived
            payload = stream.read(members[relative])
    if hashlib.sha256(payload).hexdigest() != expected_sha256.lower():
        raise EvidenceValidationError("ARTIFACT_HASH", f"SHA256 mismatch: {relative}")
    return payload
