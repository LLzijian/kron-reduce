import hashlib
import json

import h5py

from project_harness.data_check import _verify_external_bundle


def make_registry(root):
    registry = root / "registry"
    source = root / "external/source"
    registry.mkdir()
    source.mkdir(parents=True)
    with h5py.File(source / "dataset.h5", "w") as handle:
        handle.create_group("events/one")
    digest = hashlib.sha256((source / "dataset.h5").read_bytes()).hexdigest()
    frozen = {"dataset_id": "fixture", "revision": "v1", "event_count": 1,
              "source_root_env": "FIXTURE_KIKI_SOURCE", "source_root_relative": "external",
              "test_status": "prior_exposure_detected_not_fresh",
              "shards": {"source": {"events": 1, "sha256": {"dataset.h5": digest}}}}
    manifest = registry / "manifest.json"
    manifest.write_text(json.dumps(frozen), encoding="utf-8")
    checksum = hashlib.sha256(manifest.read_bytes()).hexdigest()
    (registry / "checksums.sha256").write_text(f"{checksum}  manifest.json\n")
    return {"path": "registry", "manifest": "manifest.json",
            "checksum_manifest": "checksums.sha256", "expected_events": 1}, source


def test_registry_hashes_and_counts_without_loading_holdout_arrays(tmp_path):
    entry, _ = make_registry(tmp_path)
    result = _verify_external_bundle(tmp_path, entry)
    assert result["status"] == "PASS"
    assert result["event_count"] == 1
    assert result["holdout_payload_arrays_read"] == 0


def test_changed_hdf5_is_rejected_without_parsing_corrupt_bytes(tmp_path):
    entry, source = make_registry(tmp_path)
    (source / "dataset.h5").write_bytes(b"intentionally corrupted fixture")
    result = _verify_external_bundle(tmp_path, entry)
    assert result["status"] == "FAIL"
    assert any(not row["pass"] for row in result["findings"])
