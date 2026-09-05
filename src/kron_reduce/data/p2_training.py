"""Strict causal training and prediction views for the frozen P2 contract.

The ordinary repository sample dictionaries intentionally contain richer
training context.  P2 models never receive those dictionaries: this module
copies the allowlisted clearing-state inputs into frozen dataclasses and keeps
source/fault-line identifiers beside the targets only for hierarchical loss
aggregation.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from kron_reduce.data.p1_causal import (
    P1_SOURCE_POINTS,
    P1_TIME,
    apply_p1_time_grid,
    prepare_p1_time_grid,
)
from kron_reduce.data.p2_roles import P2_ROLES, P2RoleRow
from kron_reduce.physics import coi_project, coi_project_power, complete_incidence
from kron_reduce.physics.units import effective_mass


def _frozen_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"invalid P2 {name}")
    array.flags.writeable = False
    return array


@dataclass(frozen=True)
class P2PredictionInput:
    """The complete and exclusive input surface visible during prediction."""

    delta0: np.ndarray
    omega0: np.ndarray
    pm_equilibrium: np.ndarray
    pm_clear: np.ndarray
    equilibrium_delta: np.ndarray
    mass: np.ndarray
    incidence: np.ndarray
    time: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "delta0", _frozen_array(self.delta0, (10,), "delta0"))
        object.__setattr__(self, "omega0", _frozen_array(self.omega0, (10,), "omega0"))
        object.__setattr__(
            self,
            "pm_equilibrium",
            _frozen_array(self.pm_equilibrium, (10,), "Pm_equilibrium"),
        )
        object.__setattr__(self, "pm_clear", _frozen_array(self.pm_clear, (10,), "Pm_clear"))
        object.__setattr__(
            self,
            "equilibrium_delta",
            _frozen_array(self.equilibrium_delta, (10,), "equilibrium_delta"),
        )
        mass = _frozen_array(self.mass, (10,), "mass")
        if np.any(mass <= 0.0):
            raise ValueError("P2 mass must be positive")
        object.__setattr__(self, "mass", mass)
        incidence = np.array(self.incidence, dtype=np.float64, copy=True)
        if incidence.ndim != 2 or incidence.shape[0] != 10 or not np.isfinite(incidence).all():
            raise ValueError("invalid P2 incidence")
        incidence.flags.writeable = False
        object.__setattr__(self, "incidence", incidence)
        time = np.array(self.time, dtype=np.float64, copy=True)
        if (
            time.ndim != 1
            or not len(time)
            or time[0] != 0.0
            or not np.isfinite(time).all()
            or np.any(np.diff(time) <= 0.0)
            or (len(time) > 1 and float(np.max(np.diff(time))) > 0.0100000001)
        ):
            raise ValueError("P2 prediction time must start at clearing on the registered clock")
        time.flags.writeable = False
        object.__setattr__(self, "time", time)
        if max(abs(float(self.pm_clear.sum())), abs(float(self.pm_equilibrium.sum()))) > 1e-10:
            raise ValueError("P2 power inputs must be balanced")
        if (
            max(
                float(np.max(np.abs(self.delta0 @ mass))),
                float(np.max(np.abs(self.omega0 @ mass))),
            )
            > 1e-10
        ):
            raise ValueError("P2 clearing state must be in the mass-weighted COI quotient")

    @property
    def allowed_field_names(self) -> tuple[str, ...]:
        return (
            "delta0",
            "omega0",
            "Pm_equilibrium",
            "Pm_clear",
            "equilibrium_delta",
            "mass",
            "incidence",
            "time",
        )

    def static_dict(self) -> dict[str, np.ndarray]:
        return {
            "delta0": self.delta0,
            "omega0": self.omega0,
            "Pm_equilibrium": self.pm_equilibrium,
            "Pm_clear": self.pm_clear,
            "equilibrium_delta": self.equilibrium_delta,
            "mass": self.mass,
            "incidence": self.incidence,
        }


@dataclass(frozen=True)
class P2TrainingSample:
    """Causal model inputs plus fit-only targets and non-routing strata."""

    event_id: str
    source_dataset: str
    fault_line: str
    role: str
    prediction_input: P2PredictionInput
    target_delta: np.ndarray
    target_omega: np.ndarray

    def __post_init__(self) -> None:
        if not self.event_id or not self.source_dataset or not self.fault_line:
            raise ValueError("P2 training sample needs event/source/fault-line identity")
        if self.role not in P2_ROLES:
            raise ValueError("P2 training sample has a locked or unknown role")
        shape = (len(self.prediction_input.time), 10)
        delta = _frozen_array(self.target_delta, shape, "target delta")
        omega = _frozen_array(self.target_omega, shape, "target omega")
        if not np.allclose(delta[0], self.prediction_input.delta0, rtol=0.0, atol=1e-14):
            raise ValueError("P2 target delta does not begin at clearing")
        if not np.allclose(omega[0], self.prediction_input.omega0, rtol=0.0, atol=1e-14):
            raise ValueError("P2 target omega does not begin at clearing")
        mass = self.prediction_input.mass
        if (
            max(
                float(np.max(np.abs(delta @ mass))),
                float(np.max(np.abs(omega @ mass))),
            )
            > 1e-10
        ):
            raise ValueError("P2 targets must remain in the mass-weighted COI quotient")
        object.__setattr__(self, "target_delta", delta)
        object.__setattr__(self, "target_omega", omega)


def hierarchical_event_weights(samples: Sequence[P2TrainingSample]) -> dict[str, float]:
    """Return source -> line -> event equal weights that sum to one."""

    if not samples or len({sample.event_id for sample in samples}) != len(samples):
        raise ValueError("P2 hierarchical weights require unique nonempty events")
    groups: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for sample in samples:
        groups[sample.source_dataset][sample.fault_line].append(sample.event_id)
    source_count = len(groups)
    weights = {
        event_id: 1.0 / source_count / len(lines) / len(event_ids)
        for lines in groups.values()
        for event_ids in lines.values()
        for event_id in event_ids
    }
    if not np.isclose(sum(weights.values()), 1.0, rtol=0.0, atol=1e-14):
        raise RuntimeError("P2 hierarchical event weights do not sum to one")
    return weights


def hierarchical_row_weights(
    samples: Sequence[P2TrainingSample], row_counts: Sequence[int]
) -> np.ndarray:
    """Expand event weights so every event's time windows are equal in total."""

    if len(samples) != len(row_counts) or any(
        type(count) is not int or count <= 0 for count in row_counts
    ):
        raise ValueError("P2 row counts must be positive and aligned to samples")
    event_weights = hierarchical_event_weights(samples)
    output = np.concatenate(
        [
            np.full(count, event_weights[sample.event_id] / count, dtype=np.float64)
            for sample, count in zip(samples, row_counts, strict=True)
        ]
    )
    if not np.isclose(float(output.sum()), 1.0, rtol=0.0, atol=1e-14):
        raise RuntimeError("P2 hierarchical row weights do not sum to one")
    return output


class P2CausalReader:
    """Read only P2-authorized F1 arrays and only Pm at clearing index zero."""

    def __init__(
        self,
        dataset_dir: str | Path,
        inertia_prior: np.ndarray,
        rows: Sequence[P2RoleRow],
        *,
        role_manifest_sha256: str,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.h5_path = self.dataset_dir / "ieee39_postclear_canonical_v1.h5"
        self.mass = effective_mass(inertia_prior)
        if self.mass.shape != (10,) or not np.isfinite(self.mass).all():
            raise ValueError("P2 requires ten finite mass priors")
        self.rows = tuple(rows)
        if not self.rows or len({row.event_id for row in self.rows}) != len(self.rows):
            raise ValueError("P2 reader requires unique registered roles")
        if len(role_manifest_sha256) != 64:
            raise ValueError("P2 reader requires the frozen role-manifest hash")
        self.role_manifest_sha256 = role_manifest_sha256
        self.accesses: list[dict[str, Any]] = []
        self.time_grid_audits: list[dict[str, Any]] = []

    def _authorize(self, role: str, event_ids: Iterable[str] | None) -> tuple[P2RoleRow, ...]:
        if role not in P2_ROLES:
            raise ValueError("P2 reader refuses locked or unknown roles")
        registered = tuple(sorted(row.event_id for row in self.rows if row.role == role))
        requested = registered if event_ids is None else tuple(event_ids)
        if len(requested) != len(set(requested)) or set(requested) != set(registered):
            raise ValueError("P2 reads must contain the complete registered role")
        lookup = {row.event_id: row for row in self.rows}
        return tuple(lookup[event_id] for event_id in registered)

    def read_role(
        self, role: str, *, event_ids: Iterable[str] | None = None
    ) -> tuple[P2TrainingSample, ...]:
        records = self._authorize(role, event_ids)
        samples: list[P2TrainingSample] = []
        incidence, _ = complete_incidence(10)
        with h5py.File(self.h5_path, "r") as handle:
            for row in records:
                group = handle["events"][row.event_id]

                def read(
                    name: str,
                    selection: Any,
                    *,
                    current_row: P2RoleRow = row,
                    current_group: h5py.Group = group,
                ) -> np.ndarray:
                    self.accesses.append(
                        {
                            "event_id": current_row.event_id,
                            "role": role,
                            "dataset": name,
                            "selection": str(selection),
                        }
                    )
                    return np.asarray(current_group[name][selection], dtype=np.float64)

                source_time = read("postclear/t", slice(0, P1_SOURCE_POINTS))
                plan = prepare_p1_time_grid(source_time, event_id=row.event_id, role=role)
                delta = apply_p1_time_grid(
                    plan,
                    read("postclear/delta", slice(0, P1_SOURCE_POINTS)),
                    channel="delta",
                )
                omega = apply_p1_time_grid(
                    plan,
                    read("postclear/omega", slice(0, P1_SOURCE_POINTS)),
                    channel="omega",
                )
                pm_clear = read("postclear/Pm", 0)
                equilibrium_delta = read("equilibrium/delta0", (...,))
                equilibrium_name = (
                    "Pm_equilibrium" if "Pm_equilibrium" in group["equilibrium"] else "Pm0"
                )
                pm_equilibrium = read(f"equilibrium/{equilibrium_name}", (...,))
                delta = coi_project(delta, self.mass)
                omega = coi_project(omega, self.mass)
                request = P2PredictionInput(
                    delta0=delta[0],
                    omega0=omega[0],
                    pm_equilibrium=coi_project_power(pm_equilibrium, self.mass),
                    pm_clear=coi_project_power(pm_clear, self.mass),
                    equilibrium_delta=coi_project(equilibrium_delta, self.mass),
                    mass=self.mass,
                    incidence=incidence,
                    time=P1_TIME,
                )
                samples.append(
                    P2TrainingSample(
                        event_id=row.event_id,
                        source_dataset=row.source_dataset,
                        fault_line=row.fault_line,
                        role=role,
                        prediction_input=request,
                        target_delta=delta,
                        target_omega=omega,
                    )
                )
                self.time_grid_audits.append(plan.audit)
        return tuple(samples)

    def access_audit(self) -> dict[str, Any]:
        datasets = Counter(str(row["dataset"]) for row in self.accesses)
        future_pm_reads = [
            row
            for row in self.accesses
            if row["dataset"] == "postclear/Pm" and row["selection"] not in {"0", "0.0"}
        ]
        return {
            "schema": "kron_reduce.p2_causal_reader_access.v1",
            "dataset_read_counts": dict(sorted(datasets.items())),
            "access_count": len(self.accesses),
            "future_pm_reads": future_pm_reads,
            "observed_pm_primary_reads": future_pm_reads,
            "labels_read": False,
            "canonical_validation_ids_read": [],
            "test_ids_read": [],
            "other_folds_read": [],
            "roles_read": sorted({str(row["role"]) for row in self.accesses}),
            "role_manifest_sha256": self.role_manifest_sha256,
            "time_grid_audit_count": len(self.time_grid_audits),
        }
