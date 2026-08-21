from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

from kron_reduce.evaluation import DmdMode
from kron_reduce.workflows.m0_afsd_v8_modal_identifiability import (
    _decision,
    _regime_envelope_map,
    _validate_scope,
)

ROOT = Path(__file__).resolve().parents[1]


def test_v8_scope_locks_validation_test_folds_and_observed_pm_name() -> None:
    valid = {
        "fold": "F1", "seed": 324, "pm_mode": "observed",
        "test_access": "forbidden", "other_folds_access": "forbidden",
        "canonical_mutation": "forbidden",
        "rollout_label": "conditional rollout with observed Pm",
    }
    _validate_scope(valid)
    for key, value in (
        ("fold", "F2"), ("test_access", "allowed"),
        ("pm_mode", "zoh"), ("canonical_mutation", "allowed"),
    ):
        with pytest.raises(RuntimeError):
            _validate_scope({**valid, key: value})


def _complete_map() -> dict:
    return {
        "complete_nonoverlapping_assignment": True,
        "eligible_events": 10,
        "assigned_events": 10,
        "envelopes": [{"envelope_id": "A-low__R-low"}],
    }


def test_v8_routes_nonuniform_modes_to_envelope_map() -> None:
    modal = {
        "fit_consensus": [{"passes_all_roles": False}],
        "threshold_sensitivity": [{"modes_passing_fit_80pct": 0} for _ in range(9)],
        "regime_envelope_map": _complete_map(),
    }
    residual = {
        "selection_improvement_1s": 0.17,
        "selection_improvement_10s": -0.004,
        "audit_improvement_10s": -0.14,
    }
    state, gates = _decision(
        modal, residual, {"case": {"convergent": True}},
        {"minimum_model_improvement": 0.02},
    )
    assert state == "MODAL_ENVELOPE_MAP_CREATED"
    assert gates["low_rank_short_horizon_gain_is_not_sufficient"]
    assert not gates["cross_role_consensus_mode_found"]


def test_v8_identifies_only_consensus_and_heldout_rollout_gain() -> None:
    modal = {
        "fit_consensus": [{"passes_all_roles": True}],
        "threshold_sensitivity": [{"modes_passing_fit_80pct": 1}],
        "regime_envelope_map": _complete_map(),
    }
    residual = {
        "selection_improvement_1s": 0.04,
        "selection_improvement_10s": 0.03,
        "audit_improvement_10s": 0.025,
        "low_rank_parameter_identifiable": True,
    }
    state, _ = _decision(
        modal, residual, {"case": {"convergent": True}},
        {"minimum_model_improvement": 0.02},
    )
    assert state == "MODAL_STRUCTURE_IDENTIFIED"


def test_v8_refuses_modal_promotion_when_loading_is_not_identifiable() -> None:
    modal = {
        "fit_consensus": [{"passes_all_roles": True}],
        "threshold_sensitivity": [{"modes_passing_fit_80pct": 1}],
        "regime_envelope_map": _complete_map(),
    }
    residual = {
        "selection_improvement_1s": 0.04,
        "selection_improvement_10s": 0.03,
        "audit_improvement_10s": 0.025,
        "low_rank_parameter_identifiable": False,
    }
    state, gates = _decision(
        modal, residual, {"case": {"convergent": True}},
        {"minimum_model_improvement": 0.02},
    )
    assert state == "MODAL_ENVELOPE_MAP_CREATED"
    assert not gates["low_rank_parameters_identifiable"]


def test_v8_incomplete_envelope_map_remains_inconclusive() -> None:
    modal = {
        "fit_consensus": [],
        "threshold_sensitivity": [{"modes_passing_fit_80pct": 0} for _ in range(9)],
        "regime_envelope_map": {
            "complete_nonoverlapping_assignment": False,
            "eligible_events": 10, "assigned_events": 9,
            "envelopes": [{"envelope_id": "partial"}],
        },
    }
    residual = {
        "selection_improvement_1s": 0.1,
        "selection_improvement_10s": -0.01,
        "audit_improvement_10s": -0.1,
    }
    state, _ = _decision(
        modal, residual, {"case": {"convergent": True}},
        {"minimum_model_improvement": 0.02},
    )
    assert state == "MODAL_IDENTIFIABILITY_AUDIT_INCONCLUSIVE"


@pytest.mark.parametrize(
    ("consensus", "sensitivity", "short_gain"),
    [
        (False, False, -0.01),
        (False, True, 0.00),
        (True, True, 0.01),
    ],
)
def test_v8_routes_every_unpromoted_global_candidate_to_complete_envelopes(
    consensus: bool,
    sensitivity: bool,
    short_gain: float,
) -> None:
    modal = {
        "fit_consensus": [{"passes_all_roles": consensus}],
        "threshold_sensitivity": [
            {"modes_passing_fit_80pct": int(sensitivity)}
        ],
        "regime_envelope_map": _complete_map(),
    }
    residual = {
        "selection_improvement_1s": short_gain,
        "selection_improvement_10s": 0.01,
        "audit_improvement_10s": -0.02,
    }
    state, _ = _decision(
        modal, residual, {"case": {"convergent": True}},
        {"minimum_model_improvement": 0.02},
    )
    assert state == "MODAL_ENVELOPE_MAP_CREATED"


def _mode() -> DmdMode:
    return DmdMode(
        frequency_hz=0.8, damping_ratio=0.05, decay_rate=-0.2, strength=1.0,
        shape_real=(1.0, -1.0), shape_imag=(0.0, 0.0),
    )


def test_v8_regime_map_authorizes_only_predeclared_supported_envelope() -> None:
    rows = []
    roles = {}
    empirical = {role: {} for role in ("fit", "selection", "audit")}
    for index, role in enumerate(("fit", "selection", "audit")):
        event, line = f"event-{index}", f"Line_{index}"
        roles[line] = role
        rows.append({
            "event_id": event, "fault_line": line,
            "amplitude_level": "A-low", "richness_level": "R-low",
            "excitation_amplitude": 0.1+0.01*index,
            "richness_score": 1.0, "nonlinearity_ratio": 0.02,
            "spectral_entropy": 0.3,
        })
        empirical[role] = {line: {event: [_mode()]}}
    config = {
        "minimum_regime_fit_fault_lines": 1,
        "minimum_regime_selection_fault_lines": 1,
        "minimum_regime_audit_fault_lines": 1,
        "modal_frequency_tolerance": 0.15,
        "modal_minimum_mac": 0.6,
        "modal_minimum_fault_line_support": 0.8,
        "modal_minimum_event_fraction_per_line": 0.5,
    }
    result = _regime_envelope_map(rows, roles, empirical, config)
    assert result["complete_nonoverlapping_assignment"]
    assert result["envelopes"][0]["status"] == "MODAL_CONSTRAINT_AUTHORIZED"
    assert result["envelopes"][0]["next_route"] == "ESC_MODAL_PORT_CANDIDATE"


def test_v8_regime_map_rejects_duplicate_event_assignment() -> None:
    row = {
        "event_id": "event-0", "fault_line": "Line_0",
        "amplitude_level": "A-low", "richness_level": "R-low",
        "excitation_amplitude": 0.1, "richness_score": 1.0,
        "nonlinearity_ratio": 0.02, "spectral_entropy": 0.3,
    }
    empirical = {role: {} for role in ("fit", "selection", "audit")}
    empirical["fit"] = {"Line_0": {"event-0": [_mode()]}}
    config = {
        "minimum_regime_fit_fault_lines": 1,
        "minimum_regime_selection_fault_lines": 0,
        "minimum_regime_audit_fault_lines": 0,
        "modal_frequency_tolerance": 0.15,
        "modal_minimum_mac": 0.6,
        "modal_minimum_fault_line_support": 0.8,
        "modal_minimum_event_fraction_per_line": 0.5,
    }
    result = _regime_envelope_map(
        [row, dict(row)], {"Line_0": "fit"}, empirical, config
    )
    assert result["duplicate_assignments"] == ["event-0"]
    assert not result["complete_nonoverlapping_assignment"]


def test_v8_manifest_and_workflow_require_explicit_merge_gate() -> None:
    manifest = json.loads((ROOT/"configs/project_manifest.json").read_text(encoding="utf-8"))
    workflow = json.loads((ROOT/"configs/workflows/m0_afsd_v8_modal_identifiability.json").read_text(encoding="utf-8"))
    config = tomllib.loads((ROOT/"configs/experiments/paper_m0_afsd_v8_modal_identifiability_f1_seed324.toml").read_text(encoding="utf-8"))
    names = {
        "paper_m0_afsd_v8_hypothesis_f1_seed324",
        "paper_m0_afsd_v8_controlled_f1_seed324",
        "paper_m0_afsd_v8_residual_f1_seed324",
        "paper_m0_afsd_v8_adversarial_f1_seed324",
        "paper_m0_afsd_v8_acceptance_f1_seed324",
    }
    assert names.issubset(manifest["experiments"])
    assert workflow["approval_required"] == ["merge_gate"]
    assert "frozen_F1_validation_performance" in workflow["locked_claims"]
    assert config["claim_boundaries"]["frozen_F1_validation_read"] is False
