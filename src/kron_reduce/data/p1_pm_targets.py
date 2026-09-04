"""Guarded P1 future-Pm observations for closure fitting and diagnostics.

This is deliberately separate from :mod:`p1_causal`: future mechanical power
is a target/diagnostic observation, never a causal main-model feature.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from kron_reduce.physics import coi_project_power

from .p1_causal import (
    P1_ROLES,
    P1_SOURCE_POINTS,
    P1RoleManifest,
    apply_p1_time_grid,
    prepare_p1_time_grid,
    validate_p1_samples,
)


class PmReadPurpose(str, Enum):
    FIT_CLOSURE_TARGET = "fit_closure_target"
    CONDITIONAL_DIAGNOSTIC = "conditional_diagnostic"


@dataclass(frozen=True)
class PmObservationBatch:
    role: str
    purpose: PmReadPurpose
    role_manifest_sha256: str
    event_ids: tuple[str, ...]
    pm_by_event: Mapping[str, np.ndarray]
    accesses: tuple[dict[str, Any], ...]
    core_checkpoint_sha256: str | None
    main_model_input_allowed: bool = False
    time_grid_audits: tuple[dict[str, Any], ...] = ()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_conditional_checkpoint(core_checkpoint: Path | None, expected_core_sha256: str | None) -> str:
    if core_checkpoint is None or expected_core_sha256 is None:
        raise ValueError("conditional diagnostics require a core checkpoint and its SHA256")
    checkpoint = Path(core_checkpoint)
    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise ValueError("conditional diagnostic checkpoint must be a non-symlink regular file")
    if len(expected_core_sha256) != 64 or any(char not in "0123456789abcdef" for char in expected_core_sha256):
        raise ValueError("conditional diagnostic checkpoint SHA256 must be lowercase hexadecimal")
    actual = _sha256_file(checkpoint)
    if actual != expected_core_sha256:
        raise ValueError("conditional diagnostic checkpoint SHA256 mismatch")
    return actual


def read_pm_batch(
    h5_path: Path,
    manifest: P1RoleManifest,
    samples: Sequence[dict],
    *,
    role: str,
    purpose: PmReadPurpose,
    core_checkpoint: Path | None = None,
    expected_core_sha256: str | None = None,
) -> PmObservationBatch:
    """Read only authorized Pm trajectories after all non-HDF5 gates succeed."""
    # This must remain the first operation: it authorizes the complete role and
    # rejects causal-path expansion before a target file can be opened.
    validate_p1_samples(samples, manifest, role)

    if not isinstance(purpose, PmReadPurpose):
        raise TypeError("Pm read purpose must be a PmReadPurpose")
    if purpose is PmReadPurpose.FIT_CLOSURE_TARGET:
        if role != "fit":
            raise ValueError("fit closure targets are restricted to the fit role")
        checkpoint_sha256 = None
    elif purpose is PmReadPurpose.CONDITIONAL_DIAGNOSTIC:
        if role not in P1_ROLES:
            raise ValueError("conditional diagnostics require an F1 development role")
        checkpoint_sha256 = _verify_conditional_checkpoint(core_checkpoint, expected_core_sha256)
    else:  # Defensive for future Enum extensions.
        raise ValueError("unsupported Pm read purpose")

    event_ids = manifest.ids(role)
    sample_by_id = {sample["event_id"]: sample for sample in samples}
    values: dict[str, np.ndarray] = {}
    accesses: list[dict[str, Any]] = []
    time_grid_audits: list[dict[str, Any]] = []
    with h5py.File(Path(h5_path), "r") as handle:
        for event_id in event_ids:
            group = handle["events"][event_id]

            def read(name: str, *, event_id: str = event_id, group: h5py.Group = group) -> np.ndarray:
                selection = slice(0, P1_SOURCE_POINTS)
                accesses.append({"event_id": event_id, "role": role, "dataset": name, "selection": str(selection)})
                return np.asarray(group[name][selection], dtype=float)

            time = read("postclear/t")
            plan = prepare_p1_time_grid(time, event_id=event_id, role=role)
            pm = apply_p1_time_grid(plan, read("postclear/Pm"), channel="Pm_observed")
            time_grid_audits.append(plan.audit)
            mass = np.asarray(sample_by_id[event_id]["model_inputs"]["static"]["mass"], dtype=float)
            n_nodes = mass.size
            if pm.shape != (1001, n_nodes) or not np.isfinite(pm).all():
                raise ValueError(f"{event_id}: invalid Pm target")
            with np.errstate(over="ignore", invalid="ignore"):
                projected = coi_project_power(pm, mass)
                net_power = projected.sum(axis=1)
            if not np.isfinite(projected).all() or np.max(np.abs(net_power)) > 1e-10:
                raise ValueError(f"{event_id}: invalid projected Pm target")
            clear = np.asarray(sample_by_id[event_id]["model_inputs"]["static"]["Pm_clear"], dtype=float).reshape(-1)
            if not np.allclose(projected[0], clear, rtol=0, atol=1e-10):
                raise ValueError(f"{event_id}: projected Pm target disagrees with Pm_clear")
            observed = np.array(projected, dtype=float, copy=True)
            observed.flags.writeable = False
            values[event_id] = observed

    return PmObservationBatch(
        role=role,
        purpose=purpose,
        role_manifest_sha256=manifest.source_sha256,
        event_ids=event_ids,
        pm_by_event=values,
        accesses=tuple(accesses),
        core_checkpoint_sha256=checkpoint_sha256,
        time_grid_audits=tuple(time_grid_audits),
    )
