from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from kron_reduce.workflows.unified_first_swing import _archive_asset
from project_harness.archive_storage import archive_path


def _relocations(root: Path, locations: dict[str, str]) -> None:
    path = root / "configs/archive_relocations.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(locations), encoding="utf-8")


def test_archive_path_maps_full_root_and_preserves_other_paths(tmp_path: Path) -> None:
    _relocations(tmp_path, {"results/backup/old_group": "results/backup/04_topic"})

    relative = "results/backup/old_group/run.zip"
    old = tmp_path / relative
    old.parent.mkdir(parents=True)
    old.write_bytes(b"legacy layout")
    assert archive_path(tmp_path, relative) == old
    current = tmp_path / "results/backup/04_topic/run.zip"
    current.parent.mkdir(parents=True)
    current.write_bytes(b"numbered layout")
    assert archive_path(tmp_path, relative) == current
    similar_group = "results/backup/old_group_other/run.zip"
    assert archive_path(tmp_path, similar_group) == tmp_path / similar_group
    assert archive_path(tmp_path, "results/diagnostics/run/model.npz") == (
        tmp_path / "results/diagnostics/run/model.npz"
    )


def test_archive_path_rejects_traversal_mapping(tmp_path: Path) -> None:
    _relocations(tmp_path, {"results/backup/old_group": "results/backup/../escape"})

    with pytest.raises(RuntimeError):
        archive_path(tmp_path, "results/backup/old_group/run.zip")


def test_archive_asset_reads_relocated_zip_member_and_checks_sha(tmp_path: Path) -> None:
    old_archive = "results/backup/old_group/model.zip"
    new_archive = "results/backup/04_topic/model.zip"
    _relocations(tmp_path, {"results/backup/old_group": "results/backup/04_topic"})
    archive = tmp_path / new_archive
    archive.parent.mkdir(parents=True)
    member = "diagnostics/frozen-run/checkpoint.npz"
    payload = b"frozen checkpoint"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr(member, payload)

    binding = {
        "path": "results/" + member,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    assert _archive_asset(tmp_path, binding, old_archive) == payload

    binding["sha256"] = hashlib.sha256(b"changed checkpoint").hexdigest()
    with pytest.raises(RuntimeError, match="frozen starting asset changed"):
        _archive_asset(tmp_path, binding, old_archive)
