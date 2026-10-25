"""Metadata-only regression tests for the frozen A17 production plan."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from kron_reduce.evaluation.input_consistency_plan import (
    EXPECTED_LIMITS,
    SOURCE_RUNS,
    validate_production_plan,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "configs/proposals/kiki_a17_input_consistency_reads.json"


@pytest.fixture(scope="module")
def production_plan() -> dict:
    """Read only the designated JSON control plan, never referenced payloads."""
    return json.loads(PLAN_PATH.read_text(encoding="utf-8"))


def _context(plan: dict, identity: str) -> dict:
    return next(row for row in plan["contexts"] if row["context_id"] == identity)


def test_current_frozen_plan_validates_as_metadata_only(production_plan: dict) -> None:
    result = validate_production_plan(production_plan)

    assert result["execution_authorized"] is False
    assert result["source_runs"] == SOURCE_RUNS
    assert result["counts"] == EXPECTED_LIMITS
    assert len(result["npz_identities"]) == 3358
    assert len(result["metadata_identities"]) == 3376
    assert result["counts"]["npz_contexts"] == 3616
    assert result["counts"]["descriptors"] == 20406


@pytest.mark.parametrize(
    ("identity", "mutate"),
    [
        ("inputs", lambda row: row.update(source="A15")),
        ("static-324", lambda row: row.update(phase=2)),
        ("integral-324-000", lambda row: row.update(path=row["path"].replace("integral.npz", "wrong.npz"))),
        ("weighted-324-000", lambda row: row["members"][0].update(name="unexpected_i8")),
        ("impulse-000", lambda row: row["members"][0].update(shape=[10, 2])),
        ("clock-014", lambda row: row["members"][0].update(dtype="|i1")),
        ("reference-324-014-h005", lambda row: row["members"].pop()),
        ("reference-324-014-h005", lambda row: row["members"][0].update(sha256_c_order_bytes="x" * 64)),
    ],
)
def test_context_contract_mutations_rejected(production_plan: dict, identity: str, mutate) -> None:
    changed = copy.deepcopy(production_plan)
    mutate(_context(changed, identity))
    with pytest.raises((ValueError, PermissionError)):
        validate_production_plan(changed)


def test_duplicate_context_rejected_without_changing_total(production_plan: dict) -> None:
    changed = copy.deepcopy(production_plan)
    changed["contexts"][-1] = copy.deepcopy(changed["contexts"][0])
    with pytest.raises((ValueError, PermissionError)):
        validate_production_plan(changed)


def test_reordered_population_freeze_or_source_identity_rejected(production_plan: dict) -> None:
    changed = copy.deepcopy(production_plan)
    changed["freeze_contract"]["slot_ids"].reverse()
    with pytest.raises(ValueError):
        validate_production_plan(changed)

    changed = copy.deepcopy(production_plan)
    changed["source_runs"]["A09"] = "results/other-run"
    with pytest.raises(ValueError):
        validate_production_plan(changed)

    changed = copy.deepcopy(production_plan)
    changed["population"]["weights"][0] = 1.0
    with pytest.raises(ValueError):
        validate_production_plan(changed)


@pytest.mark.parametrize(
    "change",
    [
        lambda plan: plan["limits"].update(npz_unique_files=3359),
        lambda plan: plan["metadata_files"].pop(),
        lambda plan: plan["metadata_files"][0].update(path="C:outside.json"),
        lambda plan: plan["metadata_files"][0].update(path="results//invalid.json"),
        lambda plan: plan["metadata_files"][0].update(path="results/./invalid.json"),
        lambda plan: plan["metadata_files"][0].update(sha256="0" * 64),
    ],
)
def test_fixed_counts_and_metadata_whitelist_are_strict(production_plan: dict, change) -> None:
    changed = copy.deepcopy(production_plan)
    change(changed)
    with pytest.raises((ValueError, PermissionError)):
        validate_production_plan(changed)


def test_fixed_controls_are_bound_to_exact_known_identities(production_plan: dict) -> None:
    changed = copy.deepcopy(production_plan)
    row = next(
        item for item in changed["metadata_files"]
        if item["path"].endswith("/frozen_plan.json")
    )
    row["sha256"] = hashlib.sha256(b"not the bound A16 plan").hexdigest()
    with pytest.raises(ValueError):
        validate_production_plan(changed)


def test_same_npz_path_cannot_have_conflicting_opaque_identity(production_plan: dict) -> None:
    changed = copy.deepcopy(production_plan)
    rows = [row for row in changed["contexts"] if row["path"].endswith("future-0.005.npz")]
    assert len(rows) > 1
    rows[0]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        validate_production_plan(changed)
