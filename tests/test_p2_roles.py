from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from kron_reduce.data.p2_roles import (
    EXPECTED_LINE_ALLOCATIONS,
    P2_ROLES,
    build_p2_role_rows,
)


def _dataset() -> Path:
    return Path(__file__).resolve().parents[1] / "data/ieee39_postclear_canonical_v1"


def test_p2_role_allocation_is_frozen_metadata_only_and_source_stratified() -> None:
    rows, audit = build_p2_role_rows(_dataset())
    assert len(rows) == 178
    assert Counter(row.role for row in rows) == {
        "fit": 83,
        "selection": 27,
        "internal_audit": 68,
    }
    assert audit["role_manifest_sha256"] == (
        "27281dacbef6b71b7d41514c9d0de26401d5df5b5917f1caa87691ed4437eaa7"
    )
    assert audit["metadata_fields_read"] == [
        "source_dataset",
        "fault_line",
        "valid_length",
    ]
    assert audit["trajectory_channels_read"] == []
    assert audit["labels_read"] is False
    assert audit["canonical_validation_ids_read"] == []
    assert audit["test_ids_read"] == []
    assert audit["other_folds_read"] == []


def test_p2_role_allocation_keeps_complete_lines_and_three_internal_per_source() -> None:
    rows, audit = build_p2_role_rows(_dataset())
    line_roles: dict[tuple[str, str], set[str]] = {}
    for row in rows:
        line_roles.setdefault((row.source_dataset, row.fault_line), set()).add(row.role)
    assert all(len(roles) == 1 for roles in line_roles.values())
    for source, allocation in EXPECTED_LINE_ALLOCATIONS.items():
        expected = dict(zip(P2_ROLES, allocation, strict=True))
        assert audit["line_counts_by_source_and_role"][source] == expected
        assert expected["internal_audit"] >= 3


def test_p2_role_allocation_refuses_other_folds_or_shorter_contract() -> None:
    with pytest.raises(ValueError, match="only use F1"):
        build_p2_role_rows(_dataset(), fold="F2")
    with pytest.raises(ValueError, match="line count changed"):
        build_p2_role_rows(_dataset(), minimum_steps=2000)


def test_p2_role_seed_changes_assignment_but_not_registered_counts() -> None:
    rows_324, audit_324 = build_p2_role_rows(_dataset(), seed=324)
    rows_325, audit_325 = build_p2_role_rows(_dataset(), seed=325)
    assert audit_324["line_counts_by_source_and_role"] == audit_325[
        "line_counts_by_source_and_role"
    ]
    roles_324 = {row.event_id: row.role for row in rows_324}
    roles_325 = {row.event_id: row.role for row in rows_325}
    assert roles_324 != roles_325
