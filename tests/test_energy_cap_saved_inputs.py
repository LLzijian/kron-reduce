from __future__ import annotations

import hashlib
import io
import json
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pytest

from kron_reduce.data.energy_cap_saved_inputs import EnergyCapSavedInputs


class ProbeOperations:
    def __init__(self):
        self.attempted = 0
        self.returned = 0
        self.tags = []

    def call(self, category, function, *, tag=""):
        assert category == "probe"
        self.attempted += 1
        self.tags.append(tag)
        value = function()
        self.returned += 1
        return value


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _archive_bytes(**arrays) -> bytes:
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    return stream.getvalue()


def _write_source(root: Path, relative: str, payload: bytes) -> None:
    path = root / Path(relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _manifest_for(root: Path, files: dict[str, dict]) -> dict:
    for relative, spec in files.items():
        assert (root / Path(relative)).is_file()
        assert spec["size_bytes"] == (root / Path(relative)).stat().st_size
    return {"sources": {"prior": {"run_id": "run-1"}}, "files": files}


def _json_source(root: Path, name="record.json", payload=None):
    relative = f"results/run-1/{name}"
    content = json.dumps({"fixture": "artificial", "value": 7}, separators=(",", ":")).encode()
    if payload is not None:
        content = payload
    _write_source(root, relative, content)
    return relative, {"kind": "json", "size_bytes": len(content), "sha256": _sha(content)}


def _npz_source(root: Path, arrays, members, name="record.npz"):
    relative = f"results/run-1/{name}"
    payload = _archive_bytes(**arrays)
    _write_source(root, relative, payload)
    return relative, {
        "kind": "npz", "size_bytes": len(payload), "sha256": _sha(payload),
        "members": members,
    }


def test_json_read_is_single_spent_probe_and_snapshot_records_it(tmp_path):
    relative, spec = _json_source(tmp_path)
    operations = ProbeOperations()
    reader = EnergyCapSavedInputs(tmp_path, _manifest_for(tmp_path, {relative: spec}), operations)

    value = reader.read_json(relative)

    assert value == {"fixture": "artificial", "value": 7}
    assert operations.attempted == operations.returned == 1
    assert reader.snapshot()["files_attempted"] == 1
    assert reader.snapshot()["verified_files"][0]["sha256"] == spec["sha256"]
    with pytest.raises(PermissionError, match="already been spent"):
        reader.read_json(relative)
    assert operations.attempted == 1


def test_npz_decodes_only_whitelisted_arrays_and_preserves_int64_offset(tmp_path, monkeypatch):
    offset = np.array([2**60 + 37, -(2**55)], dtype="<i8")
    members = {
        "offset": {"shape": [2], "dtype": "<i8",
                   "array_sha256": _sha(np.ascontiguousarray(offset).tobytes())},
    }
    relative, spec = _npz_source(
        tmp_path,
        {"offset": offset, "forbidden_object": np.array([object()], dtype=object)},
        members,
    )
    operations = ProbeOperations()
    reader = EnergyCapSavedInputs(tmp_path, _manifest_for(tmp_path, {relative: spec}), operations)
    monkeypatch.setattr(np, "load", lambda *_args, **_kwargs: pytest.fail("must not use np.load"))
    arrays = reader.read_arrays(relative)

    assert list(arrays) == ["offset"]
    assert arrays["offset"].dtype.str == "<i8"
    np.testing.assert_array_equal(arrays["offset"], offset)
    assert arrays["offset"].flags.writeable is False
    assert operations.attempted == operations.returned == 2
    assert reader.snapshot()["members_attempted"] == 1
    with pytest.raises(PermissionError, match="not a registered NPZ source"):
        reader.read_arrays("results/run-1/forbidden.npz")


def test_npz_preserves_fortran_order_and_checks_canonical_array_hash(tmp_path):
    array = np.asfortranarray(np.array([[1.0, 2.0], [3.0, 5.0]], dtype="<f8"))
    members = {"matrix": {
        "shape": [2, 2], "dtype": "<f8",
        "array_sha256": _sha(np.ascontiguousarray(array).tobytes()),
    }}
    relative, spec = _npz_source(tmp_path, {"matrix": array}, members)
    reader = EnergyCapSavedInputs(
        tmp_path, _manifest_for(tmp_path, {relative: spec}), ProbeOperations())

    loaded = reader.read_arrays(relative)["matrix"]

    np.testing.assert_array_equal(loaded, array)
    assert loaded.flags.writeable is False


def test_invalid_npy_header_is_rejected_before_payload_read(tmp_path, monkeypatch):
    npy = io.BytesIO()
    np.lib.format.write_array_header_1_0(npy, {
        "descr": np.dtype("<f8").str, "fortran_order": False, "shape": (3,),
    })
    npy.write(np.array([1.0, 2.0], dtype="<f8").tobytes())
    archive_payload = io.BytesIO()
    with zipfile.ZipFile(archive_payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("allowed.npy", npy.getvalue())
    payload = archive_payload.getvalue()
    relative = "results/run-1/bad-header.npz"
    _write_source(tmp_path, relative, payload)
    spec = {
        "kind": "npz", "size_bytes": len(payload), "sha256": _sha(payload),
        "members": {"allowed": {"shape": [2], "dtype": "<f8"}},
    }
    reads = []
    original_read = zipfile.ZipExtFile.read

    def recording_read(stream, size=-1):
        reads.append(size)
        return original_read(stream, size)

    monkeypatch.setattr(zipfile.ZipExtFile, "read", recording_read)
    reader = EnergyCapSavedInputs(
        tmp_path, _manifest_for(tmp_path, {relative: spec}), ProbeOperations())

    with pytest.raises(ValueError, match="before payload decode"):
        reader.read_arrays(relative)

    assert len(reads) == 3
    assert reads[:2] == [8, 2]
    assert reads[2] > 0
    assert 16 not in reads


def test_bad_file_hash_is_spent_and_cannot_be_retried(tmp_path):
    relative, spec = _json_source(tmp_path)
    path = tmp_path / Path(relative)
    path.write_bytes(b'{"fixture":"artificial","value":8}')
    operations = ProbeOperations()
    reader = EnergyCapSavedInputs(tmp_path, _manifest_for(tmp_path, {relative: spec}), operations)

    with pytest.raises(ValueError, match="frozen hash"):
        reader.read_json(relative)
    with pytest.raises(PermissionError, match="already been spent"):
        reader.read_json(relative)
    assert operations.attempted == 1 and operations.returned == 0


@pytest.mark.parametrize(
    ("wrong_spec", "message"),
    [
        ({"shape": [3], "dtype": "<f8"}, "shape, dtype or finiteness"),
        ({"shape": [2], "dtype": "<f8", "array_sha256": "0" * 64},
         "decoded bytes differ"),
    ],
)
def test_array_shape_and_decoded_hash_mismatches_fail_after_spending_read(tmp_path, wrong_spec, message):
    array = np.array([1.0, 2.0], dtype="<f8")
    relative, spec = _npz_source(tmp_path, {"allowed": array}, {"allowed": wrong_spec})
    operations = ProbeOperations()
    reader = EnergyCapSavedInputs(tmp_path, _manifest_for(tmp_path, {relative: spec}), operations)

    with pytest.raises(ValueError, match=message):
        reader.read_arrays(relative)
    with pytest.raises(PermissionError, match="already been spent"):
        reader.read_arrays(relative)
    assert operations.attempted == 2
    assert operations.returned == 1


def test_duplicate_zip_member_names_are_rejected_before_any_array_decode(tmp_path):
    member = io.BytesIO()
    np.lib.format.write_array(member, np.array([1.0], dtype="<f8"), allow_pickle=False)
    payload_stream = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(payload_stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("allowed.npy", member.getvalue())
            archive.writestr("allowed.npy", member.getvalue())
    payload = payload_stream.getvalue()
    relative = "results/run-1/duplicate.npz"
    _write_source(tmp_path, relative, payload)
    spec = {
        "kind": "npz", "size_bytes": len(payload), "sha256": _sha(payload),
        "members": {"allowed": {"shape": [1], "dtype": "<f8"}},
    }
    operations = ProbeOperations()
    reader = EnergyCapSavedInputs(tmp_path, _manifest_for(tmp_path, {relative: spec}), operations)

    with pytest.raises(ValueError, match="duplicate ZIP members"):
        reader.read_arrays(relative)
    assert operations.attempted == operations.returned == 1
    assert reader.snapshot()["members_attempted"] == 0


def test_resolved_path_escape_is_rejected_before_reading_bytes(tmp_path, monkeypatch):
    relative, spec = _json_source(tmp_path, name="escape.json")
    operations = ProbeOperations()
    reader = EnergyCapSavedInputs(tmp_path, _manifest_for(tmp_path, {relative: spec}), operations)
    expected = (tmp_path / Path(relative)).resolve()
    outside = (tmp_path.parent / "outside-energy-cap.json").resolve()
    original_resolve = Path.resolve

    def escape_resolve(path, strict=False):
        if path == expected:
            return outside
        return original_resolve(path, strict=strict)

    monkeypatch.setattr(Path, "resolve", escape_resolve)
    with pytest.raises(PermissionError, match="escapes its bound location"):
        reader.read_json(relative)
    assert operations.attempted == 1 and operations.returned == 0
