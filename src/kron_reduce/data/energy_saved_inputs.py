"""One-pass reader for the frozen phase-one energy-initializer inputs."""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

SCHEMA = "constant_c_energy_saved_sources_v1"
STATUS = "DRAFT_PREPARATION_NO_EXECUTION_AUTHORITY"
RUN_ID = "constant_c_100ms_synthetic_v1-synthetic_audit-20261109T072255936025Z-fcbcd287-40178a64"
SOURCE_COMMIT = "fcbcd28708d38b2a54b93725cfae4b803a6bd630"
ARTIFACT_MANIFEST_SHA256 = "978e6ad9c1f77ace19f3be6e09b0c6834e11861d0e607dfccade69d8ab6fb2d1"
EXPECTED_FILE_COUNT = 25
EXPECTED_JSON_COUNT = 14
EXPECTED_NPZ_COUNT = 11
EXPECTED_MEMBER_COUNT = 189
EXPECTED_ARTIFACT_ENTRY_COUNT = 80
_CASE_IDS = (
    "L100_clean", "L100_alt", "L100_weakdir", "S100_clean", "S100_alt",
    "W100_clean", "W100_weakdir", "K100_mismatch", "U_hidden_null",
)
_LINEAR_CASE_IDS = (
    "L100_clean", "L100_alt", "L100_weakdir", "W100_clean", "W100_weakdir",
    "K100_mismatch",
)
_SINE_CASE_IDS = ("S100_clean", "S100_alt")
_META_JSON_STEMS = ("artifact_hashes", "execution_freeze", "summary")


class SavedEnergyInputsError(RuntimeError):
    """A saved-source contract failed; the reader will not retry the source."""


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _walk_array_descriptors(value: Any, pointer: str = ""):
    if isinstance(value, dict):
        if "array" in value and "shape" in value and "dtype" in value:
            yield pointer or "/", value
            return
        for key, child in value.items():
            yield from _walk_array_descriptors(child, f"{pointer}/{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_array_descriptors(child, f"{pointer}/{index}")


def _member_spec(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != {"member", "shape", "dtype"}:
        raise ValueError("invalid saved NPZ member descriptor")
    member, shape, dtype = raw["member"], raw["shape"], raw["dtype"]
    if (not isinstance(member, str) or not member or "." in PurePosixPath(member).name
            or PurePosixPath(member).name != member or "\\" in member
            or not isinstance(shape, list) or not shape
            or any(type(size) is not int or size <= 0 for size in shape)
            or dtype != "<f8"):
        raise ValueError("unsupported saved NPZ member descriptor")
    return {"member": member, "shape": tuple(shape), "dtype": dtype}


def _compile_members(manifest: Mapping[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    whitelist = manifest.get("npz_member_whitelist")
    expected_keys = {
        "linear-fixtures.npz", "sine-truth.npz", "linear_100ms_cases",
        "sine_cases", "U_hidden_null.npz",
    }
    if not isinstance(whitelist, dict) or set(whitelist) != expected_keys:
        raise ValueError("saved NPZ whitelist keys do not match the registered contract")
    per_stem: dict[str, dict[str, dict[str, Any]]] = {}

    def add(stem: str, members: list[Any]) -> None:
        target = per_stem.setdefault(stem, {})
        for raw in members:
            spec = _member_spec(raw)
            if spec["member"] in target:
                raise ValueError(f"duplicate saved member in {stem}")
            target[spec["member"]] = spec

    add("linear-fixtures", whitelist["linear-fixtures.npz"])
    add("sine-truth", whitelist["sine-truth.npz"])
    add("U_hidden_null", whitelist["U_hidden_null.npz"])
    for key, cases, base_stem in (
        ("linear_100ms_cases", _LINEAR_CASE_IDS, "L100_clean"),
        ("sine_cases", _SINE_CASE_IDS, "S100_clean"),
    ):
        group = whitelist[key]
        if (not isinstance(group, dict) or group.get("case_ids") != list(cases)
                or group.get("name_pattern") != "{case_id}__{suffix}"
                or not isinstance(group.get("member_template"), list)):
            raise ValueError(f"saved member template does not match {key}")
        template = [_member_spec(item) for item in group["member_template"]]
        prefix = base_stem + "__"
        if any(not item["member"].startswith(prefix) for item in template):
            raise ValueError(f"invalid member template prefix in {key}")
        for case in cases:
            expanded = [
                {"member": case + "__" + item["member"][len(prefix):],
                 "shape": list(item["shape"]), "dtype": item["dtype"]}
                for item in template
            ]
            add(case, expanded)

    ordered_stems = ("linear-fixtures", "sine-truth", *_LINEAR_CASE_IDS,
                     *_SINE_CASE_IDS, "U_hidden_null")
    if set(per_stem) != set(ordered_stems):
        raise ValueError("saved NPZ stem roster does not match the registered contract")
    if sum(len(members) for members in per_stem.values()) != EXPECTED_MEMBER_COUNT:
        raise ValueError("saved NPZ member count does not match the registered contract")
    return {stem: per_stem[stem] for stem in ordered_stems}


def _expected_file_names() -> tuple[str, ...]:
    stems = ("linear-fixtures", "sine-truth", *_CASE_IDS)
    return tuple(
        [f"{stem}.{suffix}" for stem in stems for suffix in ("json", "npz")]
        + [f"{stem}.json" for stem in _META_JSON_STEMS]
    )


class SavedEnergyInputs:
    """Verify and decode only the registered saved arrays, once per source.

    The caller loads the bound source-plan JSON as metadata and constructs this
    reader inside the `source-inputs` operation scope. `verify()` accounts for
    one read/hash operation per source file; `load(stem)` accounts for exactly
    one decode operation per registered member in that NPZ file.
    """

    def __init__(self, root: Path, manifest: Mapping[str, Any], ops: Any) -> None:
        if not isinstance(manifest, Mapping):
            raise TypeError("saved source manifest must be a mapping")
        if manifest.get("schema") != SCHEMA or manifest.get("status") != STATUS:
            raise ValueError("saved source manifest identity mismatch")
        source = manifest.get("phase1_source")
        if (not isinstance(source, Mapping) or source.get("run_id") != RUN_ID
                or source.get("source_commit") != SOURCE_COMMIT
                or source.get("artifact_manifest_path") != f"results/{RUN_ID}/artifact_hashes.json"
                or source.get("artifact_manifest_sha256") != ARTIFACT_MANIFEST_SHA256
                or source.get("artifact_manifest_entry_count") != EXPECTED_ARTIFACT_ENTRY_COUNT
                or source.get("all_artifact_manifest_entries_match_raw_file_sha256") is not True):
            raise ValueError("saved phase-one provenance mismatch")
        files = manifest.get("files")
        expected_paths = {f"results/{RUN_ID}/{name}" for name in _expected_file_names()}
        if (not isinstance(files, Mapping) or set(files) != expected_paths
                or len(files) != EXPECTED_FILE_COUNT
                or any(not _is_sha256(value) for value in files.values())):
            raise ValueError("saved source file allowlist mismatch")
        if files.get(f"results/{RUN_ID}/artifact_hashes.json") != ARTIFACT_MANIFEST_SHA256:
            raise ValueError("artifact manifest digest does not match frozen provenance")

        self.root = Path(root).resolve(strict=True)
        self.run_id = RUN_ID
        results_dir = (self.root / "results").resolve(strict=True)
        if results_dir.parent != self.root or results_dir.name != "results":
            raise ValueError("results directory escapes the repository root")
        self._run_dir = (results_dir / RUN_ID).resolve(strict=True)
        if self._run_dir.parent != results_dir or self._run_dir.name != RUN_ID:
            raise ValueError("phase-one result directory escapes the registered results root")
        self._ops = ops
        self._file_hashes = dict(files)
        self._members = _compile_members(manifest)
        self._files_in_order = tuple(files)
        self._npz_files = {
            f"results/{RUN_ID}/{stem}.npz": stem for stem in self._members
        }
        self._json_files = {
            f"results/{RUN_ID}/{stem}.json": stem
            for stem in (*("linear-fixtures", "sine-truth", *_CASE_IDS), *_META_JSON_STEMS)
        }
        self._raw_npz: dict[str, bytes] = {}
        self._json: dict[str, dict[str, Any]] = {}
        self._file_report: dict[str, dict[str, Any]] = {}
        self._loaded: dict[str, dict[str, Any]] = {}
        self._verified = False
        self._failed = False
        self._file_attempted = 0
        self._file_returned = 0
        self._file_failed = 0
        self._member_attempted = 0
        self._member_returned = 0
        self._member_failed = 0
        self._decoded_member_keys: set[str] = set()
        self._total_probe_calls = 0

    def _require_scope(self) -> None:
        if getattr(self._ops, "scope_id", None) != "source-inputs":
            raise SavedEnergyInputsError("saved sources must be read inside source-inputs scope")
        caps = getattr(self._ops, "caps", {})
        probe_cap = caps.get("source-inputs", {}).get("probe") if isinstance(caps, Mapping) else None
        if probe_cap != EXPECTED_FILE_COUNT + EXPECTED_MEMBER_COUNT:
            raise SavedEnergyInputsError("source-inputs scope must reserve exactly 214 probes")

    def _resolved_file(self, relative: str) -> Path:
        path = PurePosixPath(relative)
        if (path.is_absolute() or ".." in path.parts or path.parts[:2] != ("results", RUN_ID)
                or len(path.parts) != 3):
            raise SavedEnergyInputsError("saved source path escapes the fixed result directory")
        result = (self.root.joinpath(*path.parts)).resolve(strict=True)
        if result.parent != self._run_dir or result.name != path.name or not result.is_file():
            raise SavedEnergyInputsError("saved source path does not resolve to its registered file")
        return result

    def _validate_npz_names(self, relative: str, raw: bytes) -> dict[str, Any]:
        stem = self._npz_files[relative]
        expected = self._members[stem]
        with zipfile.ZipFile(io.BytesIO(raw), "r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise ValueError("duplicate NPZ archive names")
            info_by_name = {info.filename: info for info in infos}
            missing = [f"{name}.npy" for name in expected if f"{name}.npy" not in info_by_name]
            if missing:
                raise ValueError(f"registered NPZ members missing: {missing}")
            return {"archive_member_count": len(names), "allowed_member_count": len(expected)}

    def _validate_json_descriptors(self, relative: str, document: dict[str, Any]) -> int:
        stem = self._json_files[relative]
        if stem not in self._members:
            return 0
        expected = self._members[stem]
        found: dict[str, list[dict[str, Any]]] = {key: [] for key in expected}
        descriptor_count = 0
        for _pointer, descriptor in _walk_array_descriptors(document):
            descriptor_count += 1
            key = descriptor.get("array")
            if key in expected:
                found[key].append(descriptor)
        for key, spec in expected.items():
            descriptors = found[key]
            if len(descriptors) != 1:
                raise ValueError(f"JSON array descriptor count for {key} must equal one")
            descriptor = descriptors[0]
            if descriptor.get("shape") != list(spec["shape"]) or descriptor.get("dtype") != spec["dtype"]:
                raise ValueError(f"JSON array descriptor mismatch for {key}")
        return descriptor_count

    def verify(self) -> dict[str, Any]:
        """Read and hash each of the 25 allowlisted files exactly once."""
        self._require_scope()
        if self._failed:
            raise SavedEnergyInputsError("reader is closed after an earlier failure")
        if self._verified:
            return self._verification_report()
        try:
            for relative in self._files_in_order:
                path = self._resolved_file(relative)
                expected_sha = self._file_hashes[relative]
                self._file_attempted += 1
                self._total_probe_calls += 1

                def read_verify(path=path, relative=relative, expected_sha=expected_sha):
                    raw = path.read_bytes()
                    actual_sha = hashlib.sha256(raw).hexdigest()
                    if actual_sha != expected_sha:
                        raise ValueError(f"saved source SHA256 mismatch: {relative}")
                    metadata: dict[str, Any] = {
                        "path": relative, "sha256": actual_sha, "byte_count": len(raw),
                    }
                    if relative in self._npz_files:
                        metadata.update(self._validate_npz_names(relative, raw))
                        self._raw_npz[relative] = raw
                    elif relative in self._json_files:
                        document = json.loads(raw)
                        if not isinstance(document, dict):
                            raise ValueError(f"saved metadata JSON must be an object: {relative}")
                        descriptor_count = self._validate_json_descriptors(relative, document)
                        metadata["array_descriptor_count"] = descriptor_count
                        self._json[relative] = document
                    else:
                        raise SavedEnergyInputsError("unregistered source file")
                    return metadata

                try:
                    meta = self._ops.call("probe", read_verify, tag="saved_source_file_hash_and_metadata")
                except BaseException:
                    self._file_failed += 1
                    raise
                self._file_returned += 1
                self._file_report[relative] = meta

            manifest_rel = f"results/{RUN_ID}/artifact_hashes.json"
            artifact_manifest = self._json[manifest_rel]
            if len(artifact_manifest) != EXPECTED_ARTIFACT_ENTRY_COUNT:
                raise ValueError("saved artifact manifest entry count mismatch")
            for relative, expected_sha in self._file_hashes.items():
                if relative == manifest_rel:
                    continue
                member_path = relative.removeprefix(f"results/{RUN_ID}/")
                if artifact_manifest.get(member_path) != expected_sha:
                    raise ValueError(f"source plan and artifact manifest disagree: {relative}")
            self._verified = True
            return self._verification_report()
        except BaseException:
            self._failed = True
            raise

    def _decode_member(self, stem: str, member: str) -> np.ndarray:
        relative = f"results/{RUN_ID}/{stem}.npz"
        spec = self._members[stem][member]
        raw = self._raw_npz[relative]
        with zipfile.ZipFile(io.BytesIO(raw), "r") as archive:
            info = archive.getinfo(f"{member}.npy")
            if type(info.file_size) is not int or info.file_size <= 0:
                raise ValueError("invalid registered NPZ member size")
            with archive.open(info, "r") as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError(f"unsupported registered NPY version {version}")
                if tuple(shape) != spec["shape"]:
                    raise ValueError(f"NPZ shape mismatch for {member}")
                if (dtype.str != spec["dtype"] or dtype.hasobject or dtype.fields is not None
                        or dtype.subdtype is not None or dtype.kind != "f" or dtype.itemsize != 8):
                    raise ValueError(f"NPZ dtype mismatch for {member}")
                header_bytes = stream.tell()
                element_count = int(np.prod(shape, dtype=np.int64))
                payload_bytes = element_count * dtype.itemsize
                if header_bytes + payload_bytes != info.file_size:
                    raise ValueError(f"NPZ member size mismatch for {member}")
                payload = stream.read(payload_bytes)
                if len(payload) != payload_bytes:
                    raise EOFError(f"short registered NPZ payload for {member}")
        array = np.frombuffer(payload, dtype=dtype, count=element_count).reshape(
            tuple(shape), order="F" if fortran_order else "C"
        )
        if array.flags.writeable:
            raise AssertionError("decoded saved arrays must remain read-only")
        if not np.isfinite(array).all():
            raise ValueError(f"non-finite values in registered NPZ member {member}")
        return array

    @staticmethod
    def _replace_descriptors(value: Any, arrays: Mapping[str, np.ndarray]) -> Any:
        if isinstance(value, dict):
            key = value.get("array")
            if (isinstance(key, str) and key in arrays
                    and "shape" in value and "dtype" in value):
                return arrays[key]
            return {name: SavedEnergyInputs._replace_descriptors(child, arrays)
                    for name, child in value.items()}
        if isinstance(value, list):
            return [SavedEnergyInputs._replace_descriptors(child, arrays) for child in value]
        return value

    def load(self, name: str) -> dict[str, Any]:
        """Load one registered NPZ stem and replace only its allowed descriptors."""
        self._require_scope()
        if self._failed:
            raise SavedEnergyInputsError("reader is closed after an earlier failure")
        if not self._verified:
            raise SavedEnergyInputsError("verify saved-source metadata before loading arrays")
        if name not in self._members:
            raise SavedEnergyInputsError("unknown saved NPZ stem")
        if name in self._loaded:
            return self._loaded[name]

        arrays: dict[str, np.ndarray] = {}
        try:
            for member in self._members[name]:
                self._member_attempted += 1
                self._total_probe_calls += 1
                try:
                    value = self._ops.call(
                        "probe", lambda member=member: self._decode_member(name, member),
                        tag="saved_source_member_decode",
                    )
                except BaseException:
                    self._member_failed += 1
                    raise
                self._member_returned += 1
                key = f"{name}__{member}"
                if key in self._decoded_member_keys:
                    raise AssertionError("duplicate saved member payload decode")
                self._decoded_member_keys.add(key)
                arrays[member] = value
            json_rel = f"results/{RUN_ID}/{name}.json"
            result = self._replace_descriptors(self._json[json_rel], arrays)
            self._loaded[name] = result
            return result
        except BaseException:
            self._failed = True
            raise

    def _verification_report(self) -> dict[str, Any]:
        return {
            "status": "verified",
            "source_file_count": EXPECTED_FILE_COUNT,
            "source_files_verified": self._file_returned,
            "json_file_count": EXPECTED_JSON_COUNT,
            "npz_file_count": EXPECTED_NPZ_COUNT,
            "registered_member_count": EXPECTED_MEMBER_COUNT,
            "registered_member_descriptors_verified": (
                EXPECTED_MEMBER_COUNT if self._verified else 0
            ),
            "json_array_descriptors_seen": sum(
                row.get("array_descriptor_count", 0) for row in self._file_report.values()
            ),
            "file_sha256": {path: row["sha256"] for path, row in self._file_report.items()},
        }

    @property
    def stems(self) -> tuple[str, ...]:
        """The fixed NPZ stem order to preload inside the reserved scope."""
        return tuple(self._members)

    def report(self) -> dict[str, Any]:
        """Return read counters without opening another source or spending a probe."""
        return {
            "verified": self._verified,
            "failed": self._failed,
            "source_files_registered": EXPECTED_FILE_COUNT,
            "source_files_attempted": self._file_attempted,
            "source_files_verified": self._file_returned,
            "source_files_failed": self._file_failed,
            "npz_files_registered": EXPECTED_NPZ_COUNT,
            "npz_files_loaded": len(self._loaded),
            "loaded_stems": sorted(self._loaded),
            "array_members_registered": EXPECTED_MEMBER_COUNT,
            "array_members_attempted": self._member_attempted,
            "array_members_decoded": self._member_returned,
            "array_members_failed": self._member_failed,
            "duplicate_payload_reads": 0,
            "probe_calls": self._total_probe_calls,
        }
