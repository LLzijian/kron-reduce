from __future__ import annotations

import io
import struct
import zipfile

import numpy as np
import pytest

from kron_reduce.data.durable_audit_ledger import DurableAccessLedger, DurableAuditJournalError
from kron_reduce.data.limited_npz_audit import (
    AccessLedger,
    AccessLedgerError,
    FileSpec,
    MemberSpec,
    NpzWhitelistReader,
    _validate_file_specs,
    fixed_file_specs,
)


def _save_npy(value) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(stream, np.asarray(value), allow_pickle=True)
    return stream.getvalue()


def _reader(tmp_path, path, members, *, zip_factory=zipfile.ZipFile, ledger_factory=None):
    file_spec = FileSpec("SYNTH/cache.npz", tuple(members))
    return NpzWhitelistReader(
        {file_spec.key: path}, files=(file_spec,),
        ledger_factory=ledger_factory,
        _allow_reduced_plan_for_tests=True,
        zip_factory=zip_factory,
    ), file_spec


def _ledger_factory(plan, *, logical_element_limit, metadata_by_key):
    assert set(metadata_by_key) == {row["key"] for row in plan}
    return AccessLedger(plan, logical_element_limit=logical_element_limit)


class _SpyArchive:
    def __init__(self, path, mode):
        self._archive = zipfile.ZipFile(path, mode)
        self.opened = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return self._archive.__exit__(*args)

    def infolist(self):
        return self._archive.infolist()

    def open(self, item, mode="r"):
        self.opened.append(item.filename if hasattr(item, "filename") else item)
        return self._archive.open(item, mode)


def test_fixed_protocol_inventory_totals_and_shapes():
    specs = fixed_file_specs()
    numeric, elements = _validate_file_specs(specs, allow_reduced=False)
    assert len(specs) == 10
    assert sum(len(file.members) for file in specs) == 42
    assert numeric == 39
    assert elements == 440_733
    assert sum(member.kind == "unicode_scalar" for f in specs for member in f.members) == 3


def test_only_registered_member_payload_is_opened_future_is_name_only(tmp_path):
    path = tmp_path / "one.npz"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("history.npy", _save_npy(np.arange(6, dtype=np.float64).reshape(2, 3)))
        archive.writestr("future.npy", b"this must never be decoded")
    spies = []

    def factory(path, mode):
        spy = _SpyArchive(path, mode)
        spies.append(spy)
        return spy

    reader, file_spec = _reader(tmp_path, path, [MemberSpec("history", (2, 3), "f8")],
                                zip_factory=factory, ledger_factory=_ledger_factory)
    output = reader.read_file(file_spec.key)
    assert np.array_equal(output["history"], np.arange(6, dtype=np.float64).reshape(2, 3))
    assert output["history"].flags.writeable is False
    assert spies[0].opened == ["history.npy"]
    assert reader.snapshot()["ledger"]["actual_counts"]["array"]["returned"] == 1


def test_bad_shape_fails_after_header_before_payload_selector(tmp_path):
    path = tmp_path / "badshape.npz"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("history.npy", _save_npy(np.arange(3, dtype=np.float64)))
    reader, file_spec = _reader(tmp_path, path, [MemberSpec("history", (2,), "f8")],
                                ledger_factory=_ledger_factory)
    with pytest.raises(ValueError, match="shape mismatch"):
        reader.read_file(file_spec.key)
    snapshot = reader.snapshot()["ledger"]
    assert snapshot["actual_counts"]["array"]["attempted"] == 0
    assert snapshot["actual_counts"]["countdescriptor"]["failed"] == 1


def test_object_and_unicode_over_limit_rejected_before_payload(tmp_path):
    object_path = tmp_path / "object.npz"
    with zipfile.ZipFile(object_path, "w") as archive:
        archive.writestr("metadata.npy", _save_npy(np.array({"unsafe": 1}, dtype=object)))
    reader, spec = _reader(tmp_path, object_path, [MemberSpec("metadata", (), "unicode_scalar")],
                           ledger_factory=_ledger_factory)
    with pytest.raises(ValueError, match="object or structured"):
        reader.read_file(spec.key)
    assert reader.snapshot()["ledger"]["actual_counts"]["attr"]["attempted"] == 0
    assert reader.snapshot()["ledger"]["actual_counts"]["countdescriptor"]["failed"] == 1

    # Avoid allocating a million-character fixture: a forged header dtype is sufficient.
    from kron_reduce.data.limited_npz_audit import NpzWhitelistReader as R
    with pytest.raises(ValueError, match="character limit"):
        R._validate_header(MemberSpec("metadata", (), "unicode_scalar"), (), np.dtype("<U1048577"))


def test_unicode_scalar_counts_characters_utf8_and_storage_bytes(tmp_path):
    path = tmp_path / "metadata.npz"
    value = "方案α"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("metadata.npy", _save_npy(np.asarray(value)))
    reader, spec = _reader(tmp_path, path, [MemberSpec("metadata", (), "unicode_scalar")],
                           ledger_factory=_ledger_factory)
    assert reader.read_file(spec.key)["metadata"] == value
    stats = reader.snapshot()["member_stats"][f"{spec.key}::metadata"]
    assert stats["characters"] == len(value)
    assert stats["utf8_bytes"] == len(value.encode("utf-8"))
    assert stats["storage_bytes"] > stats["utf8_bytes"]
    assert reader.snapshot()["ledger"]["actual_counts"]["attr"]["returned"] == 1


def test_unknown_selector_and_second_file_open_fail_closed(tmp_path):
    path = tmp_path / "valid.npz"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("history.npy", _save_npy(np.array([1.0], dtype=np.float64)))
    calls = []

    def factory(path, mode):
        calls.append((path, mode))
        return zipfile.ZipFile(path, mode)

    reader, _spec = _reader(tmp_path, path, [MemberSpec("history", (1,), "f8")], zip_factory=factory)
    with pytest.raises(AccessLedgerError, match="unknown NPZ"):
        reader.read_file("SYNTH/other.npz")
    assert calls == []

    reader2, spec2 = _reader(tmp_path, path, [MemberSpec("history", (1,), "f8")], zip_factory=factory)
    reader2.read_file(spec2.key)
    with pytest.raises(AccessLedgerError, match="already spent"):
        reader2.read_file(spec2.key)
    assert len(calls) == 1


def test_failed_payload_is_spent_and_cannot_be_retried(tmp_path, monkeypatch):
    path = tmp_path / "shortpayload.npz"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("history.npy", _save_npy(np.array([1.0, 2.0], dtype=np.float64)))
    reader, spec = _reader(tmp_path, path, [MemberSpec("history", (2,), "f8")],
                           ledger_factory=_ledger_factory)
    monkeypatch.setattr(reader, "_read_exact_payload", lambda stream, count: b"x")
    with pytest.raises(AccessLedgerError, match="invalid element count"):
        reader.read_file(spec.key)
    snapshot = reader.snapshot()["ledger"]
    assert snapshot["actual_counts"]["array"]["attempted"] == 1
    assert snapshot["actual_counts"]["array"]["failed"] == 1
    with pytest.raises(AccessLedgerError, match="fail-closed"):
        reader.read_file(spec.key)


def test_open_failure_spends_the_one_file_open_attempt(tmp_path):
    path = tmp_path / "unopenable.npz"
    path.write_bytes(b"not a zip")
    calls = []

    def broken_factory(path, mode):
        calls.append(1)
        raise OSError("synthetic open failure")

    reader, spec = _reader(tmp_path, path, [MemberSpec("history", (1,), "f8")],
                           zip_factory=broken_factory)
    with pytest.raises(OSError, match="synthetic open failure"):
        reader.read_file(spec.key)
    with pytest.raises(AccessLedgerError, match="fail-closed"):
        reader.read_file(spec.key)
    assert calls == [1]




def test_malformed_header_and_duplicate_zip_names_fail_before_payload(tmp_path):
    malformed = tmp_path / "malformed.npz"
    with zipfile.ZipFile(malformed, "w") as archive:
        archive.writestr("history.npy", b"not a valid NPY header")
    reader, spec = _reader(tmp_path, malformed, [MemberSpec("history", (1,), "f8")],
                           ledger_factory=_ledger_factory)
    with pytest.raises((ValueError, EOFError)):
        reader.read_file(spec.key)
    counts = reader.snapshot()["ledger"]["actual_counts"]
    assert counts["countdescriptor"]["failed"] == 1
    assert counts["array"]["attempted"] == 0

    duplicate = tmp_path / "duplicate.npz"
    with zipfile.ZipFile(duplicate, "w") as archive:
        archive.writestr("history.npy", _save_npy(np.array([1.0], dtype=np.float64)))
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("history.npy", _save_npy(np.array([2.0], dtype=np.float64)))
    reader2, spec2 = _reader(tmp_path, duplicate, [MemberSpec("history", (1,), "f8")],
                             ledger_factory=_ledger_factory)
    with pytest.raises(ValueError, match="duplicate ZIP"):
        reader2.read_file(spec2.key)
    assert reader2.snapshot()["ledger"]["actual_counts"]["countdescriptor"]["attempted"] == 0


def test_production_file_specs_must_match_fixed_inventory_exactly(tmp_path):
    specs = list(fixed_file_specs())
    first = specs[0]
    changed = FileSpec(first.key, (MemberSpec("history", (258, 11, 19), "f8"),) + first.members[1:])
    specs[0] = changed
    paths = {item.key: tmp_path / (str(index) + ".npz") for index, item in enumerate(specs)}
    with pytest.raises(ValueError, match="exactly equal"):
        NpzWhitelistReader(paths, files=specs)


def test_npy_version_three_is_rejected_inside_descriptor_before_payload(tmp_path):
    header = b"{'descr': '<f8', 'fortran_order': False, 'shape': (1,), }"
    prefix_len = 6 + 2 + 4
    padding = (64 - ((prefix_len + len(header) + 1) % 64)) % 64
    header += b" " * padding + b"\n"
    raw_npy = b"\x93NUMPY\x03\x00" + struct.pack("<I", len(header)) + header + b"\x00" * 8
    path = tmp_path / "v3.npz"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("history.npy", raw_npy)
    reader, spec = _reader(tmp_path, path, [MemberSpec("history", (1,), "f8")],
                           ledger_factory=_ledger_factory)
    with pytest.raises(ValueError, match="version 3"):
        reader.read_file(spec.key)
    counts = reader.snapshot()["ledger"]["actual_counts"]
    assert counts["countdescriptor"]["failed"] == 1
    assert counts["array"]["attempted"] == 0


def test_durable_journal_failure_prevents_member_header_and_payload_reads(tmp_path):
    path = tmp_path / "journalfail.npz"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("history.npy", _save_npy(np.array([1.0], dtype=np.float64)))
        archive.writestr("future.npy", b"never read")
    spies = []

    def zip_factory(path, mode):
        spy = _SpyArchive(path, mode)
        spies.append(spy)
        return spy

    ledgers = []

    def durable_factory(plan, *, logical_element_limit, metadata_by_key):
        ledger = DurableAccessLedger(
            plan,
            logical_element_limit=logical_element_limit,
            metadata_by_key=metadata_by_key,
            journal_path=tmp_path / "access.jsonl",
        )
        ledgers.append(ledger)
        return ledger

    reader, spec = _reader(tmp_path, path, [MemberSpec("history", (1,), "f8")],
                           zip_factory=zip_factory, ledger_factory=durable_factory)

    def fail_journal(_entry):
        raise OSError("synthetic journal failure")

    ledgers[0]._journal.append = fail_journal
    try:
        with pytest.raises(DurableAuditJournalError, match="journal attempted failed"):
            reader.read_file(spec.key)
        assert spies[0].opened == []
        snapshot = reader.snapshot()["ledger"]
        assert snapshot["actual_counts"]["countdescriptor"]["attempted"] == 1
        assert snapshot["actual_counts"]["array"]["attempted"] == 0
    finally:
        ledgers[0].close_journal()
