"""One-pass, manifest-bound access to the saved train energy-cap comparison inputs."""
from __future__ import annotations

import ast
import hashlib
import io
import json
import math
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np


class EnergyCapSavedInputs:
    """Decode only explicit members; failed reads are spent and cannot be repeated."""

    def __init__(self, root, manifest, operations):
        self.root = Path(root).resolve(strict=True)
        self.files = manifest["files"]
        self.operations = operations
        self.attempted = set()
        self.decoded = set()
        self.records = []
        self.bytes_read = 0
        self.npy_header_bytes = 0
        self.logical_bytes = 0
        runs = {source["run_id"] for source in manifest["sources"].values()}
        if not self.files or not runs:
            raise ValueError("empty saved-source manifest")
        for relative, spec in self.files.items():
            parts = PurePosixPath(relative).parts
            if ("\\" in relative or len(parts) < 3 or parts[0] != "results"
                    or parts[1] not in runs or ".." in parts
                    or spec["kind"] not in {"json", "npz"}
                    or not relative.endswith("." + spec["kind"])
                    or type(spec["size_bytes"]) is not int or spec["size_bytes"] <= 0
                    or len(spec["sha256"]) != 64):
                raise ValueError("invalid saved-source file contract")
            int(spec["sha256"], 16)
            if spec["kind"] == "npz":
                if not spec["members"]:
                    raise ValueError("NPZ needs a nonempty member whitelist")
                for name, member in spec["members"].items():
                    if (not name or "/" in name or "\\" in name or "." in name
                            or member["dtype"] not in {"<f8", "<i8"}
                            or not isinstance(member["shape"], list)
                            or any(type(n) is not int or n <= 0 for n in member["shape"])):
                        raise ValueError("invalid saved-array member contract")

    def _bytes(self, relative):
        if relative not in self.files:
            raise PermissionError("source file is outside the frozen whitelist")
        if relative in self.attempted:
            raise PermissionError("saved-source read has already been spent")
        self.attempted.add(relative)
        spec = self.files[relative]
        returned = {}

        def read():
            expected = self.root.joinpath(*PurePosixPath(relative).parts)
            path = expected.resolve(strict=True)
            if path != expected or not path.is_relative_to(self.root / "results"):
                raise PermissionError("saved-source path escapes its bound location")
            if path.stat().st_size != spec["size_bytes"]:
                raise ValueError("saved-source byte size differs")
            payload = path.read_bytes()
            self.bytes_read += len(payload)
            digest = hashlib.sha256(payload).hexdigest()
            if len(payload) != spec["size_bytes"] or digest != spec["sha256"]:
                raise ValueError("saved-source bytes differ from their frozen hash")
            record = {"path": relative, "size_bytes": len(payload), "sha256": digest}
            self.records.append(record)
            returned["payload"] = payload
            return record

        self.operations.call("probe", read, tag="source_bytes:" + relative)
        return returned["payload"]

    def read_json(self, relative):
        if self.files.get(relative, {}).get("kind") != "json":
            raise PermissionError("not a registered JSON source")
        return json.loads(self._bytes(relative).decode("utf-8"))

    def read_arrays(self, relative):
        if self.files.get(relative, {}).get("kind") != "npz":
            raise PermissionError("not a registered NPZ source")
        payload = self._bytes(relative)
        arrays = {}
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError("duplicate ZIP members")
            for name, spec in self.files[relative]["members"].items():
                member = name + ".npy"
                if member not in names:
                    raise ValueError("registered array is missing")

                def decode(name=name, spec=spec, member=member):
                    key = relative + ":" + name
                    if key in self.decoded:
                        raise PermissionError("saved-array decode has already been spent")
                    self.decoded.add(key)
                    with archive.open(member) as stream:
                        magic = stream.read(8)
                        self.npy_header_bytes += len(magic)
                        if magic[:6] != b"\x93NUMPY" or magic[6:] not in {b"\x01\x00", b"\x02\x00", b"\x03\x00"}:
                            raise ValueError("unsupported NPY format")
                        size_length = 2 if magic[6] == 1 else 4
                        size_bytes = stream.read(size_length)
                        self.npy_header_bytes += len(size_bytes)
                        if len(size_bytes) != size_length:
                            raise ValueError("truncated saved-array header length")
                        header_size = int.from_bytes(size_bytes, "little")
                        if not 0 < header_size <= 4096:
                            raise ValueError("saved-array header exceeds its fixed byte limit")
                        header_bytes = stream.read(header_size)
                        self.npy_header_bytes += len(header_bytes)
                        if len(header_bytes) != header_size:
                            raise ValueError("truncated saved-array header")
                        header = ast.literal_eval(header_bytes.decode("utf-8" if magic[6] == 3 else "latin1"))
                        nbytes = math.prod(spec["shape"]) * np.dtype(spec["dtype"]).itemsize
                        if (not isinstance(header, dict) or set(header) != {"descr", "fortran_order", "shape"}
                                or header["descr"] != spec["dtype"] or type(header["fortran_order"]) is not bool
                                or header["shape"] != tuple(spec["shape"])
                                or archive.getinfo(member).file_size != 8 + size_length + header_size + nbytes):
                            raise ValueError("saved-array shape, dtype or finiteness contract differs before payload decode")
                        raw = stream.read(nbytes)
                        self.logical_bytes += len(raw)
                        if len(raw) != nbytes:
                            raise ValueError("truncated saved-array payload")
                    value = np.frombuffer(raw, dtype=spec["dtype"]).reshape(
                        spec["shape"], order="F" if header["fortran_order"] else "C")
                    if not np.isfinite(value).all():
                        raise ValueError("saved-array shape, dtype or finiteness differs")
                    if spec.get("array_sha256") is not None:
                        digest = hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
                        if digest != spec["array_sha256"]:
                            raise ValueError("saved-array decoded bytes differ")
                    value.flags.writeable = False
                    return value

                arrays[name] = self.operations.call("probe", decode, tag="source_member:" + relative + ":" + name)
        return arrays

    def snapshot(self):
        return {"files_attempted": len(self.attempted), "members_attempted": len(self.decoded),
                "bytes_read": self.bytes_read, "npy_header_bytes": self.npy_header_bytes,
                "logical_bytes": self.logical_bytes, "verified_files": self.records,
                "decoded_members": sorted(self.decoded)}
