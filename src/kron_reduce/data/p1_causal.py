"""F1 allowlisted, fixed-window reads without loading labels or future Pm.

The harness must verify the dataset and candidate artifacts before scientific
execution. This reader enforces access, not authenticity of a caller's dataset.
It intentionally does not use CanonicalStore.load_event / adapt_event.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from kron_reduce.physics import coi_project, coi_project_power, complete_incidence
from kron_reduce.physics.units import effective_mass

P1_ROLES = ("fit", "selection", "internal_audit")
LOCKED_ROLES = {"external_validation", "postfreeze_safety_train", "postfreeze_safety_validation", "test"}
P1_TIME = np.arange(1001, dtype=float) * 0.01
P1_TIME.flags.writeable = False


def id_hash(ids: Iterable[str]) -> str:
    return hashlib.sha256(json.dumps(sorted(ids), separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class P1EventRole:
    event_id: str
    source_dataset: str
    fault_line: str
    role: str
    valid_length: int

    def __post_init__(self) -> None:
        if not self.event_id or "/" in self.event_id or "\\" in self.event_id or self.event_id in {".", ".."}:
            raise ValueError("event ID must be a single nonempty HDF5 key")
        if not self.source_dataset or not self.fault_line or self.role not in P1_ROLES:
            raise ValueError("invalid P1 source, fault line, or role")
        if type(self.valid_length) is not int or self.valid_length < len(P1_TIME):
            raise ValueError("P1 needs the complete registered 10-second window")


@dataclass(frozen=True)
class P1RoleManifest:
    records: tuple[P1EventRole, ...]
    fold: str
    source_sha256: str

    def __post_init__(self) -> None:
        if self.fold != "F1":
            raise ValueError("only F1 may be opened during P1")
        if len(self.source_sha256) != 64 or any(c not in "0123456789abcdef" for c in self.source_sha256):
            raise ValueError("a frozen role-manifest SHA256 is required")
        if not self.records or len({row.event_id for row in self.records}) != len(self.records):
            raise ValueError("empty or duplicate role IDs")
        if {row.role for row in self.records} != set(P1_ROLES):
            raise ValueError("all three P1 roles must be explicitly registered")
        groups: dict[tuple[str, str], str] = {}
        for row in self.records:
            key = (row.source_dataset, row.fault_line)
            if groups.setdefault(key, row.role) != row.role:
                raise ValueError("fault-line role leakage")

    @classmethod
    def from_csv(cls, path: Path, *, fold: str, expected_sha256: str) -> P1RoleManifest:
        if fold != "F1":
            raise ValueError("only F1 may be opened during P1")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError("role-manifest hash mismatch")
        reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
        fields = reader.fieldnames or []
        required = {"event_id", "source_dataset", "fault_line", "role", "canonical_role", "valid_length"}
        if len(fields) != len(set(fields)) or not required.issubset(fields):
            raise ValueError("invalid role-manifest header")
        records = []
        seen = set()
        for row in reader:
            if row["event_id"] in seen:
                raise ValueError("duplicate role IDs")
            seen.add(row["event_id"])
            if row["role"] in LOCKED_ROLES:
                continue  # Saved role metadata may enumerate locked IDs; arrays may not be read.
            if row["canonical_role"] != "train":
                raise ValueError("P1 development roles must come from canonical train")
            records.append(P1EventRole(row["event_id"], row["source_dataset"], row["fault_line"], row["role"], int(row["valid_length"])))
        return cls(tuple(records), fold, expected_sha256)

    def ids(self, role: str) -> tuple[str, ...]:
        if role not in P1_ROLES:
            raise ValueError(f"locked or unknown role: {role}")
        return tuple(sorted(row.event_id for row in self.records if row.role == role))

    @property
    def role_id_hashes(self) -> dict[str, str]:
        return {role: id_hash(self.ids(role)) for role in P1_ROLES}

    def authorize(self, role: str, event_ids: Iterable[str]) -> tuple[P1EventRole, ...]:
        expected = self.ids(role)
        requested = tuple(event_ids)
        if len(requested) != len(set(requested)) or set(requested) != set(expected):
            raise ValueError("request must contain the complete registered role, with no foreign IDs")
        lookup = {row.event_id: row for row in self.records}
        return tuple(lookup[event_id] for event_id in expected)


class P1CausalReader:
    def __init__(self, h5_path: Path, inertia_prior: np.ndarray, manifest: P1RoleManifest):
        self.h5_path = Path(h5_path)
        self.mass = effective_mass(inertia_prior)
        if self.mass.shape != (10,) or not np.isfinite(self.mass).all():
            raise ValueError("the IEEE39 reader requires ten finite positive mass priors")
        self.manifest = manifest
        self.accesses: list[dict[str, Any]] = []

    def read_role(self, role: str, *, event_ids: Iterable[str] | None = None) -> list[dict[str, Any]]:
        # Complete authorization happens before opening even the HDF5 file.
        records = self.manifest.authorize(role, self.manifest.ids(role) if event_ids is None else event_ids)
        samples = []
        with h5py.File(self.h5_path, "r") as handle:
            for row in records:
                group = handle["events"][row.event_id]

                def read(name: str, selection: Any, *, row: P1EventRole = row, group: h5py.Group = group) -> np.ndarray:
                    self.accesses.append({"event_id": row.event_id, "role": role, "dataset": name, "selection": str(selection)})
                    return np.asarray(group[name][selection], dtype=float)

                time = read("postclear/t", slice(0, 1001))
                delta = read("postclear/delta", slice(0, 1001))
                omega = read("postclear/omega", slice(0, 1001))
                pm_clear = read("postclear/Pm", 0)
                equilibrium_delta = read("equilibrium/delta0", (...,))
                equilibrium_key = "Pm_equilibrium" if "Pm_equilibrium" in group["equilibrium"] else "Pm0"
                equilibrium_pm = read(f"equilibrium/{equilibrium_key}", (...,))
                if time.shape != P1_TIME.shape or not np.allclose(time, P1_TIME, rtol=0, atol=1e-12):
                    raise ValueError(f"{row.event_id}: incomplete or changed sampling grid")
                for name, array, shape in (
                    ("delta", delta, (1001, 10)), ("omega", omega, (1001, 10)),
                    ("Pm_clear", pm_clear, (10,)), ("equilibrium_delta", equilibrium_delta, (10,)),
                    ("Pm_equilibrium", equilibrium_pm, (10,)),
                ):
                    if array.shape != shape or not np.isfinite(array).all():
                        raise ValueError(f"{row.event_id}: invalid {name}")
                delta, omega = coi_project(delta, self.mass), coi_project(omega, self.mass)
                pm_clear = coi_project_power(pm_clear, self.mass)
                incidence, _edges = complete_incidence(10)
                static = {
                    "delta0": delta[[0]].copy(), "omega0": omega[[0]].copy(),
                    "Pm_clear": pm_clear[None, :].copy(), "Pm0": pm_clear[None, :].copy(),
                    "equilibrium_delta": coi_project(equilibrium_delta, self.mass),
                    "Pm_equilibrium": coi_project_power(equilibrium_pm, self.mass),
                    "mass": self.mass.copy(), "incidence": incidence,
                }
                samples.append({
                    "event_id": row.event_id,
                    "model_inputs": {"static": static, "exogenous": {"t": time.copy(), "Pm": np.repeat(pm_clear[None, :], len(time), axis=0)}},
                    "training_context": {
                        "targets": {"t": time, "delta": delta, "omega": omega},
                        "metadata": {"event_id": row.event_id, "source_dataset": row.source_dataset, "fault_line": row.fault_line},
                        "role": role, "role_manifest_sha256": self.manifest.source_sha256,
                        "claim_type": "causal_zoh_rollout", "representation": "generator_10",
                    },
                })
        validate_p1_samples(samples, self.manifest, role)
        return samples


def validate_p1_samples(samples: Iterable[Mapping[str, Any]], manifest: P1RoleManifest, role: str) -> None:
    """Reject role mixing and feature expansion before an optimizer or rollout."""
    samples = tuple(samples)
    records = manifest.authorize(role, [sample["event_id"] for sample in samples])
    lookup = {row.event_id: row for row in records}
    static_fields = {"delta0", "omega0", "Pm_clear", "Pm0", "equilibrium_delta", "Pm_equilibrium", "mass", "incidence"}
    context_fields = {"targets", "metadata", "role", "role_manifest_sha256", "claim_type", "representation"}
    for sample in samples:
        row = lookup[sample["event_id"]]
        if set(sample) != {"event_id", "model_inputs", "training_context"}:
            raise ValueError("unregistered sample fields")
        inputs, context = sample["model_inputs"], sample["training_context"]
        if set(inputs) != {"static", "exogenous"} or set(inputs["static"]) != static_fields or set(inputs["exogenous"]) != {"t", "Pm"}:
            raise ValueError("unregistered model input fields")
        if set(context) != context_fields or context["role"] != role or context["role_manifest_sha256"] != manifest.source_sha256:
            raise ValueError("unregistered training context or role mismatch")
        if context["claim_type"] != "causal_zoh_rollout" or context["metadata"] != {"event_id": row.event_id, "source_dataset": row.source_dataset, "fault_line": row.fault_line}:
            raise ValueError("conditional input or metadata mismatch")
        targets = context["targets"]
        if set(targets) != {"t", "delta", "omega"}:
            raise ValueError("unregistered targets")
        for time in (targets["t"], inputs["exogenous"]["t"]):
            if np.shape(time) != P1_TIME.shape or not np.allclose(time, P1_TIME, rtol=0, atol=1e-12):
                raise ValueError("all three complete registered horizons are mandatory")
        static = inputs["static"]
        mass = np.asarray(static["mass"])
        n_nodes = mass.size
        if mass.shape != (n_nodes,) or n_nodes < 2 or not np.isfinite(mass).all() or np.any(mass <= 0):
            raise ValueError("invalid mass")
        incidence = np.asarray(static["incidence"])
        if incidence.ndim != 2 or incidence.shape[0] != n_nodes or incidence.shape[1] < 1 or not np.isfinite(incidence).all():
            raise ValueError("invalid incidence")
        for name in ("delta", "omega"):
            array = np.asarray(targets[name])
            initial = np.asarray(static[f"{name}0"])
            if array.shape != (1001, n_nodes) or initial.shape not in {(n_nodes,), (1, n_nodes)} or not np.isfinite(array).all() or not np.array_equal(initial.reshape(-1), array[0]):
                raise ValueError("invalid trajectory or clearing state")
            if np.max(np.abs(array @ mass)) > 1e-10:
                raise ValueError("non-COI target")
        pm = np.asarray(inputs["exogenous"]["Pm"])
        clear = np.asarray(static["Pm_clear"])
        if clear.shape not in {(n_nodes,), (1, n_nodes)} or pm.shape != (1001, n_nodes) or not np.isfinite(pm).all() or not np.array_equal(pm, np.broadcast_to(clear.reshape(-1), pm.shape)):
            raise ValueError("future Pm schedule is forbidden on the causal path")
        if not np.array_equal(np.asarray(static["Pm0"]).reshape(-1), clear.reshape(-1)) or abs(float(clear.sum())) > 1e-10:
            raise ValueError("invalid clearing power alias or balance")
        for name in ("equilibrium_delta", "Pm_equilibrium"):
            value = np.asarray(static[name])
            if value.shape != (n_nodes,) or not np.isfinite(value).all():
                raise ValueError("invalid equilibrium prior")
