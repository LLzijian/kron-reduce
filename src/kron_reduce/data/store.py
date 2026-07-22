from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterable, Iterator

import h5py
import numpy as np

from .schema import DatasetMetadata, EventMetadata, EventSequence


def _bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _optional_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


class CanonicalStore:
    """Path-only lazy HDF5 store safe to pickle into spawn workers."""

    def __init__(self, dataset_dir: Path):
        self.dataset_dir = Path(dataset_dir).resolve()
        self.h5_path = self.dataset_dir / "ieee39_postclear_canonical_v1.h5"
        self.index_path = self.dataset_dir / "dataset_index.csv"
        if not self.h5_path.is_file() or not self.index_path.is_file():
            raise FileNotFoundError(f"canonical dataset incomplete: {self.dataset_dir}")
        with self.index_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self._rows = {row["event_id"]: row for row in rows}
        if len(self._rows) != 315:
            raise ValueError(f"expected 315 canonical events, got {len(self._rows)}")

    @classmethod
    def open(cls, dataset_path: str | Path) -> "CanonicalStore":
        return cls(Path(dataset_path))

    def __getstate__(self) -> dict[str, object]:
        return {"dataset_dir": self.dataset_dir}

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__init__(Path(state["dataset_dir"]))

    @property
    def dataset_metadata(self) -> DatasetMetadata:
        schema = json.loads((self.dataset_dir / "schema.json").read_text(encoding="utf-8"))
        with h5py.File(self.h5_path, "r") as handle:
            attrs = handle["metadata"].attrs
            return DatasetMetadata(
                dataset_revision=str(attrs["dataset_revision"]),
                build_id=str(attrs["build_id"]),
                schema_version=str(attrs["schema_version"]),
                M_prior=np.asarray(schema["M_prior"], dtype=np.float64),
                omega_s=float(schema["omega_s_rad_per_s"]),
            )

    def event_ids(self, view: str = "combined_dynamics") -> list[str]:
        if view == "combined_dynamics":
            return sorted(self._rows)
        if view == "general_dynamics":
            return sorted(
                eid for eid, row in self._rows.items()
                if row["source_dataset"] == "ieee39_highorder_dataset"
            )
        if view == "binary_stability":
            return sorted(
                eid for eid, row in self._rows.items()
                if row["source_dataset"] == "ieee39_near_critical_unstable_v1_dense"
            )
        raise ValueError(f"unknown canonical view: {view}")

    def metadata(self, event_id: str) -> EventMetadata:
        row = self._rows[event_id]
        return EventMetadata(
            event_id=event_id,
            source_dataset=row["source_dataset"],
            fault_line=row["fault_line"],
            outcome_label=row["outcome_label"],
            observation_schema=row["observation_schema"],
            valid_length=int(row["valid_length"]),
            available_horizon_s=float(row["available_horizon_s"]),
            event_observed=_bool(row.get("event_observed", "")),
            event_time_s=_optional_float(row.get("event_time_s")),
            source_npz_sha256=row["source_npz_sha256"],
            event_signature_sha256=row["event_signature_sha256"],
            quality_status=row["quality_status"],
            attributes=dict(row),
        )

    def load_event(
        self,
        event_id: str,
        *,
        include_offline: bool = False,
        include_extended: bool = False,
    ) -> EventSequence:
        metadata = self.metadata(event_id)
        with h5py.File(self.h5_path, "r") as handle:
            group = handle["events"][event_id]
            post = group["postclear"]
            required = {
                key: np.asarray(post[key], dtype=np.float64)
                for key in ("t", "delta", "omega", "Pm")
            }
            equilibrium = {
                key: np.asarray(value, dtype=np.float64)
                for key, value in group["equilibrium"].items()
            }
            offline = {}
            if include_offline:
                offline = {
                    key: np.asarray(post[key], dtype=np.float64)
                    for key in ("Pe", "bus_v", "bus_a")
                    if key in post
                }
            extended = {}
            if include_extended and "extended" in group:
                extended = {
                    key: np.asarray(value, dtype=np.float64)
                    for key, value in group["extended"].items()
                }
        t = required["t"]
        if len(t) != metadata.valid_length or t[0] != 0.0 or np.any(np.diff(t) <= 0):
            raise ValueError(f"invalid canonical time contract for {event_id}")
        for key in ("delta", "omega", "Pm"):
            if required[key].shape != (len(t), 10) or not np.isfinite(required[key]).all():
                raise ValueError(f"invalid {key} for {event_id}")
        return EventSequence(
            event_id=event_id,
            t=t,
            delta=required["delta"],
            omega=required["omega"],
            Pm=required["Pm"],
            equilibrium=equilibrium,
            metadata=metadata,
            offline=offline,
            extended=extended,
        )

    def iter_events(self, event_ids: Iterable[str], **kwargs: object) -> Iterator[EventSequence]:
        for event_id in event_ids:
            yield self.load_event(event_id, **kwargs)
