from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from kron_reduce.data.p1_causal import (
    P1_TIME,
    apply_p1_time_grid,
    prepare_p1_time_grid,
)
from kron_reduce.workflows import p1_a04_time_grid_retry as retry


def _shifted_time(offset: float = 0.005) -> np.ndarray:
    return np.concatenate(([0.0], offset + 0.01 * np.arange(1001)))


def test_exact_grid_is_copied_bit_for_bit() -> None:
    values = np.column_stack((np.sin(P1_TIME), np.cos(P1_TIME)))
    plan = prepare_p1_time_grid(P1_TIME, event_id="exact", role="fit")
    output = apply_p1_time_grid(plan, values, channel="delta")
    assert plan.exact_grid is True
    assert np.array_equal(output, values)
    assert plan.audit["interpolation_applied"] is False
    assert plan.audit["extrapolation_applied"] is False


def test_shifted_linear_channels_are_recovered_without_changing_t0() -> None:
    source_time = _shifted_time()
    values = np.column_stack((2.0 * source_time + 3.0, -0.5 * source_time + 1.0))
    plan = prepare_p1_time_grid(source_time, event_id="shifted", role="internal_audit")
    output = apply_p1_time_grid(plan, values, channel="Pm_observed")
    expected = np.column_stack((2.0 * P1_TIME + 3.0, -0.5 * P1_TIME + 1.0))
    assert plan.exact_grid is False and plan.bracketing_index == 1001
    assert np.array_equal(output[0], values[0])
    assert np.allclose(output, expected, rtol=0.0, atol=4e-15)
    assert plan.audit["interpolation_applied"] is True
    assert plan.audit["extrapolation_applied"] is False


@pytest.mark.parametrize(
    "attack,pattern",
    [
        ("nonzero_start", "start exactly"),
        ("duplicate", "strictly increasing"),
        ("missing_bracket", "bracket"),
        ("large_gap", "gap exceeds"),
        ("irregular_postfirst", "post-first"),
        ("nonfinite", "invalid source"),
    ],
)
def test_time_grid_attacks_refuse_before_interpolation(attack: str, pattern: str) -> None:
    time = _shifted_time()
    if attack == "nonzero_start":
        time[0] = 1e-6
    elif attack == "duplicate":
        time[50] = time[49]
    elif attack == "missing_bracket":
        time = time[:1001]
    elif attack == "large_gap":
        time[500:] += 0.001
    elif attack == "irregular_postfirst":
        time[500:] += 5e-12
    else:
        time[20] = np.nan
    with pytest.raises(ValueError, match=pattern):
        prepare_p1_time_grid(time, event_id="attack", role="selection")


def test_time_grid_rejects_channel_length_or_nonfinite_attack() -> None:
    plan = prepare_p1_time_grid(_shifted_time(), event_id="shifted", role="fit")
    with pytest.raises(ValueError, match="invalid omega"):
        apply_p1_time_grid(plan, np.zeros((1001, 2)), channel="omega")
    values = np.zeros((1002, 2))
    values[10, 0] = np.inf
    with pytest.raises(ValueError, match="invalid omega"):
        apply_p1_time_grid(plan, values, channel="omega")


def test_a04_frozen_config_and_registry_are_single_model_no_extrapolation() -> None:
    root = Path(__file__).resolve().parents[1]
    config_path, config = retry._load_config(root, {"config": retry.CONFIG_NAME})
    assert retry.file_sha256(config_path) == retry.CONFIG_SHA256
    assert config["model_count"] == 1 and config["source_routing"] is False
    assert config["time_grid"]["extrapolation"] is False
    assert config["time_grid"]["maximum_source_points_read"] == 1002
    project = json.loads((root / "configs/project_manifest.json").read_text(encoding="utf-8"))
    for stage in ("retry_registered", "model_unit_gate", "controlled_experiment"):
        entry = project["experiments"][f"paper_ieee39_modular_energy_p1_a04_{stage}_f1_seed324"]
        assert entry["status"] == "gated"
        assert entry["module"].endswith("p1_a04_time_grid_retry")


def test_a04_time_grid_summary_rejects_omission_or_extrapolation() -> None:
    rows = []
    for index in range(retry.EXPECTED_EVENT_COUNT):
        rows.append(
            {
                "event_id": f"event-{index}",
                "role": ("fit", "selection", "internal_audit")[index % 3],
                "source_dataset": "source",
                "first_positive_time_seconds": 0.005 if index < retry.EXPECTED_SHIFTED_COUNT else 0.01,
                "bracketing_index": 1001 if index < retry.EXPECTED_SHIFTED_COUNT else 1000,
                "maximum_source_gap_seconds": 0.01,
                "interpolation_applied": index < retry.EXPECTED_SHIFTED_COUNT,
                "extrapolation_applied": False,
            }
        )
    assert retry._summarize_time_grid(rows)["shifted_event_count"] == retry.EXPECTED_SHIFTED_COUNT
    with pytest.raises(ValueError, match="178"):
        retry._summarize_time_grid(rows[:-1])
    rows[0]["extrapolation_applied"] = True
    with pytest.raises(ValueError, match="extrapolation"):
        retry._summarize_time_grid(rows)
