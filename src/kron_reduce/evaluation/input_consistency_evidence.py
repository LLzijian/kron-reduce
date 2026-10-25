"""Exclusive, durable A17 result records; no source provider or model calls."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path

import numpy as np


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1048576):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def pack(value, arrays, prefix="root"):
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("object arrays are forbidden in result evidence")
        if prefix in arrays:
            raise ValueError("duplicate output array identity")
        arrays[prefix] = value
        return {"array_member": prefix, "shape": list(value.shape), "dtype": value.dtype.str,
                "sha256_c_order_bytes": hashlib.sha256(value.tobytes(order="C")).hexdigest()}
    if isinstance(value, np.dtype):
        return {"numpy_dtype": value.str}
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: pack(getattr(value, field.name), arrays, f"{prefix}__{field.name}")
                for field in fields(value)}
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("result mapping keys must be explicit strings")
        return {key: pack(item, arrays, f"{prefix}__{key}") for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [pack(item, arrays, f"{prefix}__{index}") for index, item in enumerate(value)]
    if isinstance(value, np.generic):
        return pack(value.item(), arrays, prefix)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, float) and not math.isfinite(value):
        return {"value": None, "reason": "nonfinite_saved_return", "representation": repr(value)}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported result evidence type: {type(value).__name__}")


def save_record(directory, name, value):
    if not isinstance(name, str) or re.fullmatch(r"[A-Za-z0-9_-]+", name) is None:
        raise ValueError("result record name must be a flat frozen identifier")
    arrays = {}
    metadata = pack(value, arrays)
    archive, document = Path(directory) / f"{name}.npz", Path(directory) / f"{name}.json"
    with archive.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    write_json(document, metadata)
    return {archive.name: sha256_file(archive), document.name: sha256_file(document)}


def error_record(error):
    record = {"error_type": type(error).__name__, "message": str(error), "retry_allowed": False}
    for attribute in ("partial_record", "input_consistency_partial_record",
                      "input_consistency_operation_partial", "input_consistency_paths_partial_record",
                      "input_consistency_log_error", "input_consistency_campaign_partial",
                      "input_consistency_construction_partial", "input_consistency_seal_partial_record",
                      "input_consistency_seal_phase"):
        if hasattr(error, attribute):
            record[attribute] = getattr(error, attribute)
    chain, current, seen = [], error.__cause__ or error.__context__, {id(error)}
    while current is not None and id(current) not in seen and len(chain) < 8:
        seen.add(id(current))
        chain.append({"type": type(current).__name__, "message": str(current)})
        current = current.__cause__ or current.__context__
    record["exception_chain"] = chain
    return record


def save_manifest(directory):
    directory = Path(directory)
    manifest = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
            raise PermissionError("result manifest cannot follow linked evidence")
        if path.is_file() and path.name != "artifact_hashes.json":
            manifest[path.relative_to(directory).as_posix()] = sha256_file(path)
    write_json(directory / "artifact_hashes.json", manifest)
    return manifest
