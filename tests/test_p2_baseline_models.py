from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from kron_reduce.data.p2_training import P2PredictionInput, P2TrainingSample
from kron_reduce.evaluation.p2_metrics import P2FitScales
from kron_reduce.models.p2_baselines import (
    BASELINE_CLASSES,
    P2_BASELINE_NAMES,
    B2Baseline,
    LinearOdeBaseline,
    NodeMlpBaseline,
    SindycOdeBaseline,
    load_p2_baseline,
    new_p2_baseline,
)
from kron_reduce.physics import complete_incidence


def _truth_samples() -> tuple[tuple[P2TrainingSample, ...], tuple[P2TrainingSample, ...]]:
    incidence, _ = complete_incidence(10)
    truth = B2Baseline(np.full(10, 0.4), np.full(incidence.shape[1], 0.2))
    time = np.arange(21, dtype=float) * 0.01
    samples = []
    definitions = [
        ("fit-a1", "A", "L1", "fit", 0.010),
        ("fit-a2", "A", "L2", "fit", 0.015),
        ("fit-b1", "B", "L3", "fit", 0.020),
        ("fit-b2", "B", "L4", "fit", 0.025),
        ("sel-a1", "A", "L5", "selection", 0.012),
        ("sel-a2", "A", "L6", "selection", 0.018),
        ("sel-b1", "B", "L7", "selection", 0.022),
        ("sel-b2", "B", "L8", "selection", 0.028),
    ]
    pattern = np.linspace(-1.0, 1.0, 10)
    pattern -= np.mean(pattern)
    for event_id, source, line, role, amplitude in definitions:
        delta0 = amplitude * pattern
        pm_clear = 0.05 * amplitude * pattern
        request = P2PredictionInput(
            delta0=delta0,
            omega0=np.zeros(10),
            pm_equilibrium=np.zeros(10),
            pm_clear=pm_clear,
            equilibrium_delta=np.zeros(10),
            mass=np.ones(10),
            incidence=incidence,
            time=time,
        )
        delta, omega = truth.predict(request)
        samples.append(
            P2TrainingSample(
                event_id=event_id,
                source_dataset=source,
                fault_line=line,
                role=role,
                prediction_input=request,
                target_delta=delta,
                target_omega=omega,
            )
        )
    return tuple(samples[:4]), tuple(samples[4:])


@pytest.mark.parametrize("method", P2_BASELINE_NAMES)
def test_seven_p2_baselines_share_interface_and_roundtrip(method: str, tmp_path: Path) -> None:
    fit, selection = _truth_samples()
    scales = P2FitScales((0.02,) * 10, (0.02,) * 10, "a" * 64)
    model = new_p2_baseline(method)
    record = model.fit(fit, selection, scales, seed=324)
    assert record["method"] == method
    assert record["observed_pm_reads"] == []
    assert record["source_routing"] is False
    audit = model.access_audit()
    assert audit["future_pm_reads"] == []
    assert audit["forbidden_field_reads"] == []
    prediction = model.predict(selection[0].prediction_input)
    assert prediction[0].shape == prediction[1].shape == (21, 10)
    assert np.max(np.abs(prediction[0] @ np.ones(10))) < 1e-10
    assert np.max(np.abs(prediction[1] @ np.ones(10))) < 1e-10

    path = tmp_path / f"{method}.npz"
    model.save(path)
    restored = load_p2_baseline(method, path)
    replay = restored.predict(selection[0].prediction_input)
    assert np.array_equal(prediction[0], replay[0])
    assert np.array_equal(prediction[1], replay[1])


def test_p2_prediction_rejects_legacy_dictionary_with_future_pm() -> None:
    fit, selection = _truth_samples()
    model = B2Baseline()
    model.fit(fit, selection, P2FitScales((1.0,) * 10, (1.0,) * 10, "b" * 64))
    request = selection[0].prediction_input
    with pytest.raises(TypeError, match="only P2PredictionInput"):
        model.predict(
            {
                **request.static_dict(),
                "Pm": np.repeat(request.pm_clear[None, :], len(request.time), axis=0),
            }
        )


def test_p2_source_label_cannot_route_predictions() -> None:
    fit, selection = _truth_samples()
    model = LinearOdeBaseline()
    model.fit(fit, selection, P2FitScales((1.0,) * 10, (1.0,) * 10, "c" * 64))
    changed = replace(selection[0], source_dataset="attacker-controlled-source")
    left = model.predict(selection[0].prediction_input)
    right = model.predict(changed.prediction_input)
    assert np.array_equal(left[0], right[0])
    assert np.array_equal(left[1], right[1])


def test_p2_stochastic_baseline_is_deterministic_at_registered_seed() -> None:
    fit, selection = _truth_samples()
    scales = P2FitScales((1.0,) * 10, (1.0,) * 10, "d" * 64)
    first, second = NodeMlpBaseline(), NodeMlpBaseline()
    first.fit(fit, selection, scales, seed=324)
    second.fit(fit, selection, scales, seed=324)
    assert first.fit_record["selected_spec"] == second.fit_record["selected_spec"]
    for name in first._arrays():
        assert np.array_equal(first._arrays()[name], second._arrays()[name])


def test_p2_method_specific_contract_audits() -> None:
    fit, selection = _truth_samples()
    scales = P2FitScales((1.0,) * 10, (1.0,) * 10, "e" * 64)
    linear = LinearOdeBaseline()
    linear.fit(fit, selection, scales)
    assert linear.spectral_audit()["status"] == "PASS"
    sindy = SindycOdeBaseline()
    sindy.fit(fit, selection, scales)
    assert sindy.library_audit()["status"] == "PASS"
    node = NodeMlpBaseline()
    node.fit(fit, selection, scales)
    assert node.deterministic_audit()["status"] == "PASS"
    assert set(BASELINE_CLASSES) == set(P2_BASELINE_NAMES)
