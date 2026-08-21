from __future__ import annotations

import tomllib
from pathlib import Path

from kron_reduce.evaluation import physical_model_modes
from kron_reduce.workflows.m0_afsd_v7_modal_structure_discovery import _fit_ssdm
from kron_reduce.workflows.m0_esc_v1_excitation_conditioned_selection import (
    _confirmed_branch,
    _hypothesis_registry,
    _select_ladder,
    _synthetic_gate,
    _synthetic_samples,
    _validate_scope,
)

ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict:
    path = (
        ROOT
        / "configs"
        / "experiments"
        / "paper_m0_esc_v1_excitation_conditioned_selection_f1_seed324.toml"
    )
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _summary(rmse: float, failure: float = 0.0) -> list[dict]:
    return [{"horizon_steps": 1000, "mean_rmse": rmse, "failure_rate": failure}]


def test_esc_scope_and_hypothesis_are_frozen_before_metrics() -> None:
    config = _config()
    _validate_scope(config)
    registry = _hypothesis_registry(config, "unit-run")
    assert registry["registered_before_controlled_metrics"] is True
    assert registry["registered_ladder"] == ["B2", "B3", "SS-DM"]
    assert registry["modal_evidence_roles"]["trajectory_DMD"].endswith("only")
    assert registry["test_ids_read"] == []
    assert registry["other_folds_read"] == []


def test_ladder_requires_incremental_two_percent_gain() -> None:
    config = _config()
    physics = {branch: {"status": "pass"} for branch in ("B2", "B3", "SS-DM")}
    decision = _select_ladder(
        {
            "B2": _summary(1.0),
            "B3": _summary(0.97),
            "SS-DM": _summary(0.96),
        },
        physics,
        config,
    )
    assert decision["selected_branch"] == "B3"
    assert decision["decisions"][0]["promoted"] is True
    assert decision["decisions"][1]["promoted"] is False


def test_ladder_rejects_failed_or_nonphysical_candidate() -> None:
    config = _config()
    physics = {
        "B2": {"status": "pass"},
        "B3": {"status": "fail"},
        "SS-DM": {"status": "pass"},
    }
    decision = _select_ladder(
        {
            "B2": _summary(1.0),
            "B3": _summary(0.5),
            "SS-DM": _summary(0.4, failure=0.1),
        },
        physics,
        config,
    )
    assert decision["selected_branch"] == "B2"


def test_disjoint_audit_rejects_unconfirmed_selection_promotion() -> None:
    config = _config()
    entry = {
        "selection": {"selected_branch": "SS-DM"},
        "audit_aggregate": {
            "B2": _summary(1.0),
            "SS-DM": _summary(0.99),
        },
    }
    confirmation = _confirmed_branch(entry, config)
    assert confirmation["selection_branch"] == "SS-DM"
    assert confirmation["confirmed_branch"] == "B2"
    assert confirmation["promotion_confirmed"] is False


def test_esc_synthetic_gate_separates_poles_from_participation() -> None:
    config = _config()
    config["synthetic_bootstrap_draws"] = 20
    evidence = _synthetic_gate(config)
    assert evidence["status"] == "PASS"
    assert evidence["B2_modal_recovery"]["uniform_fault_line_DMD_consensus_count"] == 0
    assert len(evidence["B2_modal_recovery"]["matches"]) == 3
    assert evidence["outside_envelope_probe"]["status"] == "REFUSE_AND_REQUEST_DATA"


def test_augmented_ssdm_equation_has_auditable_quotient_modes() -> None:
    config = _config()
    _, samples = _synthetic_samples(config)
    train = [sample for index, sample in enumerate(samples) if index % 4 != 3]
    validation = [sample for index, sample in enumerate(samples) if index % 4 == 3]
    model = _fit_ssdm(train, validation, config)
    modes, diagnostics = physical_model_modes(
        model,
        train[0]["model_inputs"]["static"],
        min_frequency_hz=0.02,
        max_frequency_hz=3.0,
    )
    assert model.physics_audit()["status"] == "pass"
    assert len(modes) >= 3
    assert diagnostics["equilibrium_residual_inf_norm"] < 1e-10
    assert diagnostics["linearization"] == "central_difference_COI_quotient"
