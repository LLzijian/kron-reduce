"""Leakage-resistant, source-stratified P2 development-role allocation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py

P2_ROLES = ("fit", "selection", "internal_audit")
HIGHORDER_SOURCE = "ieee39_highorder_dataset"
NEAR_CRITICAL_SOURCE = "ieee39_near_critical_unstable_v1_dense"
EXPECTED_LINE_ALLOCATIONS = {
    HIGHORDER_SOURCE: (10, 3, 3),
    NEAR_CRITICAL_SOURCE: (3, 1, 3),
}


@dataclass(frozen=True)
class P2RoleRow:
    event_id: str
    source_dataset: str
    fault_line: str
    role: str
    canonical_role: str
    valid_length: int


def _line_digest(dataset_revision: str, source: str, line: str, seed: int) -> str:
    payload = f"{dataset_revision}|{source}|{line}|{seed}".encode()
    return hashlib.sha256(payload).hexdigest()


def _training_ids(dataset_dir: Path, fold: str) -> tuple[str, ...]:
    if fold != "F1":
        raise ValueError("P2-A01 may only use F1")
    path = dataset_dir / "splits" / "disturbance_calibration" / f"combined_{fold}.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("split_name") != "combined_line_5fold_v1" or value.get("fold") != fold:
        raise ValueError("P2-A01 disturbance calibration identity mismatch")
    ids = value.get("training_event_ids")
    if not isinstance(ids, list) or len(ids) != len(set(ids)) or not ids:
        raise ValueError("P2-A01 requires unique F1 training IDs")
    recorded_hash = value.get("training_event_ids_sha256")
    actual_hash = hashlib.sha256(
        json.dumps(ids).encode("utf-8")
    ).hexdigest()
    if actual_hash != recorded_hash:
        raise ValueError("P2-A01 F1 training-ID hash mismatch")
    return tuple(str(event_id) for event_id in ids)


def build_p2_role_rows(
    dataset_dir: str | Path,
    *,
    fold: str = "F1",
    seed: int = 324,
    minimum_steps: int = 1001,
) -> tuple[tuple[P2RoleRow, ...], dict[str, object]]:
    """Allocate complete fault lines using only train IDs and three HDF5 attrs."""
    dataset_dir = Path(dataset_dir)
    training_ids = _training_ids(dataset_dir, fold)
    h5_path = dataset_dir / "ieee39_postclear_canonical_v1.h5"
    metadata: list[tuple[str, str, str, int]] = []
    with h5py.File(h5_path, "r") as handle:
        events = handle["events"]
        for event_id in training_ids:
            if event_id not in events:
                raise ValueError(f"P2-A01 missing registered training event: {event_id}")
            attrs = events[event_id].attrs
            source = str(attrs["source_dataset"])
            line = str(attrs["fault_line"])
            valid_length = int(attrs["valid_length"])
            if source not in EXPECTED_LINE_ALLOCATIONS or not line:
                raise ValueError("P2-A01 found an unexpected source or empty fault line")
            if valid_length >= minimum_steps:
                metadata.append((event_id, source, line, valid_length))

    lines_by_source: dict[str, set[str]] = defaultdict(set)
    for _event_id, source, line, _valid_length in metadata:
        lines_by_source[source].add(line)
    line_roles: dict[tuple[str, str], str] = {}
    ordered_lines: dict[str, list[str]] = {}
    for source, allocations in EXPECTED_LINE_ALLOCATIONS.items():
        lines = sorted(
            lines_by_source[source],
            key=lambda line: (_line_digest("ieee39_postclear_canonical_v1", source, line, seed), line),
        )
        if len(lines) != sum(allocations):
            raise ValueError(
                f"P2-A01 {source} line count changed: {len(lines)} != {sum(allocations)}"
            )
        ordered_lines[source] = lines
        cursor = 0
        for role, count in zip(P2_ROLES, allocations, strict=True):
            for line in lines[cursor : cursor + count]:
                line_roles[(source, line)] = role
            cursor += count

    rows = tuple(
        sorted(
            (
                P2RoleRow(
                    event_id=event_id,
                    source_dataset=source,
                    fault_line=line,
                    role=line_roles[(source, line)],
                    canonical_role="train",
                    valid_length=valid_length,
                )
                for event_id, source, line, valid_length in metadata
            ),
            key=lambda row: (P2_ROLES.index(row.role), row.source_dataset, row.fault_line, row.event_id),
        )
    )
    if not rows or len({row.event_id for row in rows}) != len(rows):
        raise ValueError("P2-A01 role allocation is empty or duplicated")
    observed: dict[tuple[str, str], str] = {}
    for row in rows:
        key = (row.source_dataset, row.fault_line)
        if observed.setdefault(key, row.role) != row.role:
            raise ValueError("P2-A01 fault-line role leakage")
    line_counts = {
        source: {
            role: sum(
                line_roles[(source, line)] == role for line in lines_by_source[source]
            )
            for role in P2_ROLES
        }
        for source in EXPECTED_LINE_ALLOCATIONS
    }
    for counts in line_counts.values():
        if counts["internal_audit"] < 3:
            raise ValueError("P2-A01 source-level uncertainty needs three internal lines")
    row_payload = [asdict(row) for row in rows]
    audit: dict[str, object] = {
        "schema": "kron_reduce.p2_role_allocation.v1",
        "fold": fold,
        "seed": seed,
        "dataset_revision": "ieee39_postclear_canonical_v1",
        "minimum_steps": minimum_steps,
        "training_id_count_read": len(training_ids),
        "eligible_event_count": len(rows),
        "eligible_event_counts_by_role": dict(Counter(row.role for row in rows)),
        "line_counts_by_source_and_role": line_counts,
        "ordered_lines_by_source": ordered_lines,
        "role_manifest_sha256": hashlib.sha256(
            json.dumps(row_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "metadata_fields_read": ["source_dataset", "fault_line", "valid_length"],
        "trajectory_channels_read": [],
        "labels_read": False,
        "canonical_validation_ids_read": [],
        "test_ids_read": [],
        "other_folds_read": [],
    }
    return rows, audit
