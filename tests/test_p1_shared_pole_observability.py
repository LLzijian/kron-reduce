from __future__ import annotations

import copy
import hashlib
import json
import shutil

import numpy as np
import pytest
from test_p1_counterfactual_execution import (
    saved_nonzero_counterfactual as _saved_nonzero_counterfactual_fixture,
)

import kron_reduce.evaluation.p1_shared_pole_observability as audit_module
from kron_reduce.evaluation.modal import DmdMode
from kron_reduce.evaluation.p1_shared_pole_observability import (
    SharedEquationPole,
    StaticFamilyError,
    audit_local_observability_events,
    extract_shared_equation_poles,
    finite_window_energy_share,
    match_poles_one_to_one,
    shared_static_signature,
    summarize_local_observability,
)
from kron_reduce.models.prunable_energy_model import PrunableEnergyModel
from kron_reduce.workflows.p1_counterfactual_execution import _seal
from kron_reduce.workflows.p1_modal_observability_execution import (
    evaluate_saved_modal_observability,
    verify_modal_observability_artifacts,
)


@pytest.fixture(scope="module")
def saved_modal_bundle(tmp_path_factory):
    return _saved_nonzero_counterfactual_fixture.__wrapped__(tmp_path_factory)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _pole(index: int, frequency: float, damping: float, shape: np.ndarray) -> SharedEquationPole:
    imaginary = 2.0 * np.pi * frequency
    real = -damping * imaginary / np.sqrt(1.0 - damping**2)
    return SharedEquationPole(
        pole_id=_hash(f"pole:{index}"),
        checkpoint_sha256=_hash("checkpoint"),
        static_signature_sha256=_hash("placeholder-static"),
        eigenvalue_real=real,
        eigenvalue_imag=imaginary,
        frequency_hz=frequency,
        damping_ratio=damping,
        speed_visibility=0.2,
        input_output_residue_rank=1,
        equilibrium_residual_inf_norm=1e-10,
        shape_real=tuple(np.asarray(shape).real),
        shape_imag=tuple(np.asarray(shape).imag),
    )


def _static(n_nodes: int = 6) -> dict[str, np.ndarray]:
    incidence = np.zeros((n_nodes, n_nodes - 1))
    for edge in range(n_nodes - 1):
        incidence[edge, edge] = 1.0
        incidence[edge + 1, edge] = -1.0
    return {
        "mass": np.ones(n_nodes),
        "incidence": incidence,
        "equilibrium_delta": np.zeros(n_nodes),
        "Pm_equilibrium": np.zeros(n_nodes),
    }


def _signal(
    shapes_and_poles: list[tuple[np.ndarray, SharedEquationPole]],
    *,
    amplitude: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    time = np.arange(1001, dtype=float) * 0.01
    values = np.zeros((len(time), len(shapes_and_poles[0][0])))
    for shape, pole in shapes_and_poles:
        exponent = (pole.eigenvalue_real + 1j * pole.eigenvalue_imag) * time
        values += amplitude * np.real(np.exp(exponent)[:, None] * shape[None, :])
    return time, values


def _sample(
    event_id: str,
    view: str,
    line: str,
    signal: np.ndarray,
    time: np.ndarray,
    *,
    static: dict[str, np.ndarray] | None = None,
) -> dict:
    static = _static(signal.shape[1]) if static is None else static
    return {
        "event_id": event_id,
        "model_inputs": {"static": static},
        "training_context": {
            "metadata": {
                "event_id": event_id,
                "source_dataset": view,
                "fault_line": line,
            },
            "targets": {"t": time, "omega": signal},
        },
    }


def _bind_static(poles: tuple[SharedEquationPole, ...], signature: str):
    return tuple(
        SharedEquationPole(
            **{
                **pole.as_dict(),
                "static_signature_sha256": signature,
            }
        )
        for pole in poles
    )


def _mode(frequency: float, damping: float, shape: np.ndarray) -> DmdMode:
    imaginary = 2.0 * np.pi * frequency
    real = -damping * imaginary / np.sqrt(1.0 - damping**2)
    return DmdMode(
        frequency,
        damping,
        real,
        1.0,
        tuple(np.asarray(shape).real),
        tuple(np.asarray(shape).imag),
    )


def test_known_modes_are_supported_by_different_views_of_one_shared_spectrum() -> None:
    shape_a = np.asarray([1.0, -1.0, 1j, -1j, 0.0, 0.0, 0.0, 0.0])
    shape_b = np.asarray([0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 1j, -1j])
    raw_poles = (_pole(0, 0.8, 0.04, shape_a), _pole(1, 1.4, 0.08, shape_b))
    samples = []
    for view, pole_index in (("view_a", 0), ("view_b", 1)):
        shape = (shape_a, shape_b)[pole_index]
        for line_index in range(3):
            time, values = _signal([(shape, raw_poles[pole_index])])
            samples.append(_sample(f"{view}-{line_index}", view, f"line-{line_index}", values, time))
    signature, _ = shared_static_signature(samples)
    poles = _bind_static(raw_poles, signature)

    audited = audit_local_observability_events(
        samples, poles, expected_static_signature_sha256=signature
    )
    expected = [
        {
            "event_id": sample["event_id"],
            "view_name": sample["training_context"]["metadata"]["source_dataset"],
            "fault_line": sample["training_context"]["metadata"]["fault_line"],
        }
        for sample in samples
    ]
    summary = summarize_local_observability(
        audited["event_rows"], expected, [pole.pole_id for pole in poles]
    )

    assert summary["pole_statuses"] == {
        poles[0].pole_id: "LOCALLY_OBSERVED",
        poles[1].pole_id: "LOCALLY_OBSERVED",
    }
    statuses = {(row["pole_id"], row["view_name"]): row for row in summary["view_rows"]}
    assert statuses[(poles[0].pole_id, "view_a")]["status"] == "SUPPORTED"
    assert statuses[(poles[0].pole_id, "view_b")]["status"] == "UNSUPPORTED"
    assert statuses[(poles[1].pole_id, "view_a")]["status"] == "UNSUPPORTED"
    assert statuses[(poles[1].pole_id, "view_b")]["status"] == "SUPPORTED"
    assert all(
        row["dmd_rank_passes"] == 3
        for row in audited["event_rows"]
        if row["event_supported"]
    )


def test_same_frequency_wrong_shape_and_zero_energy_do_not_create_support() -> None:
    reference_shape = np.asarray([1.0, -1.0, 1j, -1j, 0.0, 0.0, 0.0, 0.0])
    wrong_shape = np.asarray([0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 1j, -1j])
    raw = _pole(0, 1.0, 0.05, reference_shape)
    time, values = _signal([(wrong_shape, raw)])
    samples = [_sample(f"event-{i}", "wrong", f"line-{i}", values, time) for i in range(3)]
    signature, _ = shared_static_signature(samples)
    pole = _bind_static((raw,), signature)[0]
    audited = audit_local_observability_events(
        samples, (pole,), expected_static_signature_sha256=signature
    )

    assert all(not row["event_supported"] for row in audited["event_rows"])
    assert all(row["finite_window_energy_share"] < 1e-12 for row in audited["event_rows"])
    assert all(not row["match_passed"] for row in audited["rank_rows"])
    assert finite_window_energy_share(np.zeros_like(values), reference_shape) == 0.0


def test_global_assignment_prevents_one_candidate_from_supporting_two_poles() -> None:
    shape_a = np.asarray([1.0, 1j, 0.0, 0.0])
    shape_b = np.asarray([1.0, 1j, 0.01, 0.0])
    poles = (_pole(0, 1.0, 0.05, shape_a), _pole(1, 1.001, 0.05, shape_b))
    candidate = _mode(1.0, 0.05, shape_a / np.linalg.norm(shape_a))

    rows = match_poles_one_to_one(poles, (candidate,), np.ones(4))

    assert sum(row["candidate_index"] is not None for row in rows) == 1
    assert sum(row["match_passed"] for row in rows) == 1
    assert {row["status"] for row in rows} == {"MATCHED_PASS", "UNMATCHED"}


def test_single_line_and_copied_events_remain_undetermined() -> None:
    pole_id = _hash("single-line-pole")
    expected = [
        {"event_id": f"e-{index}", "view_name": "one", "fault_line": "same-line"}
        for index in range(12)
    ]
    rows = [{**row, "pole_id": pole_id, "event_supported": True} for row in expected]

    summary = summarize_local_observability(rows, expected, (pole_id,))

    view = summary["view_rows"][0]
    assert view["independent_lines"] == 1
    assert view["line_support_fraction"] == 1.0
    assert view["bootstrap_support_probability"] == 1.0
    assert view["status"] == "UNDETERMINED"
    assert summary["pole_statuses"][pole_id] == "UNDETERMINED"


def test_bootstrap_is_order_invariant_and_requires_complete_event_cells() -> None:
    pole_id = _hash("order-pole")
    expected = [
        {"event_id": f"e-{index}", "view_name": "v", "fault_line": f"l-{index}"}
        for index in range(4)
    ]
    rows = [
        {**row, "pole_id": pole_id, "event_supported": index != 3}
        for index, row in enumerate(expected)
    ]
    forward = summarize_local_observability(rows, expected, (pole_id,))
    reverse = summarize_local_observability(rows[::-1], expected[::-1], (pole_id,))

    assert forward == reverse
    assert forward["view_rows"][0]["status"] == "UNSUPPORTED"
    with pytest.raises(ValueError, match="complete registered role"):
        summarize_local_observability(rows[:-1], expected, (pole_id,))


def test_static_family_is_checked_before_equation_spectrum_extraction() -> None:
    time = np.arange(1001) * 0.01
    zero = np.zeros((1001, 6))
    first = _sample("a", "v", "l1", zero, time)
    changed = _static()
    changed["equilibrium_delta"] = np.asarray([1e-12, 0.0, 0.0, 0.0, 0.0, 0.0])
    second = _sample("b", "v", "l2", zero, time, static=changed)

    with pytest.raises(StaticFamilyError, match="INVALID_STATIC_FAMILY"):
        shared_static_signature((first, second))


def test_equation_extraction_does_not_need_trajectory_or_view_metadata() -> None:
    sample = {"event_id": "only", "model_inputs": {"static": _static()}}

    class FakeModel:
        def modal_spectrum(self, _static_value, **_kwargs):
            return {
                "internal_poles": [{"frequency_hz": 1.0}],
                "equation_reportable_modes": [
                    {
                        "eigenvalue_real": -0.1,
                        "eigenvalue_imag": 2 * np.pi,
                        "frequency_hz": 1.0,
                        "damping_ratio": 0.1 / np.sqrt(0.1**2 + (2 * np.pi) ** 2),
                        "speed_visibility": 0.2,
                        "input_output_residue_rank": 1,
                        "shape_real": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "shape_imag": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                        "linearization": {"equilibrium_residual_inf_norm": 1e-10},
                    }
                ],
                "diagnostics": {
                    "modal_diagnostics_complete": True,
                    "energy_hessian_positive_definite": True,
                },
            }

    poles, document = extract_shared_equation_poles(
        FakeModel(), (sample,), _hash("fake-checkpoint")  # type: ignore[arg-type]
    )

    assert len(poles) == 1
    assert document["trajectory_modes_used_to_define_poles"] is False
    assert document["static_signature"]["reference_event_id"] == "only"


def test_dmd_failure_is_retained_as_three_failed_rank_cells(monkeypatch) -> None:
    raw = _pole(0, 1.0, 0.05, np.asarray([1.0, -1.0, 1j, -1j, 0.0, 0.0]))
    time, values = _signal([(raw.shape, raw)])
    samples = [_sample("event", "view", "line", values, time)]
    signature, _ = shared_static_signature(samples)
    pole = _bind_static((raw,), signature)[0]

    def fail(*_args, **_kwargs):
        raise np.linalg.LinAlgError("deliberate DMD failure")

    monkeypatch.setattr(audit_module, "exact_dmd_modes", fail)
    result = audit_local_observability_events(
        samples, (pole,), expected_static_signature_sha256=signature
    )

    assert len(result["rank_rows"]) == 3
    assert all(row["status"] == "UNMATCHED" for row in result["rank_rows"])
    assert all("deliberate DMD failure" in row["dmd_error"] for row in result["rank_rows"])
    assert result["event_rows"][0]["event_supported"] is False


def test_saved_workflow_is_read_only_and_seals_all_registered_artifacts(
    saved_modal_bundle, tmp_path, monkeypatch
) -> None:
    directory, manifest, samples, base, _counter = saved_modal_bundle

    def forbidden(*_args, **_kwargs):
        raise AssertionError("modal audit must not fit or estimate scales")

    monkeypatch.setattr(PrunableEnergyModel, "fit", forbidden)
    monkeypatch.setattr(PrunableEnergyModel, "_training_scales", forbidden)
    output = tmp_path / "modal"
    result = evaluate_saved_modal_observability(
        directory / "base",
        base["artifact_manifest_sha256"],
        manifest,
        samples["internal_audit"],
        output,
    )

    assert result["scientific_verdict"] is None
    assert result["model_unit_gate_passed"] is False
    assert result["promotion_decision"] == "KEEP_BASELINE"
    assert result["event_count"] == len(manifest.ids("internal_audit"))
    assert result["event_rank_match_cells"] == 3 * result["event_pole_cells"]
    verify_modal_observability_artifacts(output, result["artifact_manifest_sha256"])
    lineage = json.loads((output / "lineage.json").read_text())
    assert lineage["source_routing"] is False
    assert lineage["test_ids_read"] == lineage["other_folds_read"] == []
    with pytest.raises(FileExistsError):
        evaluate_saved_modal_observability(
            directory / "base",
            base["artifact_manifest_sha256"],
            manifest,
            samples["internal_audit"],
            output,
        )


def test_saved_workflow_preserves_failed_static_family_without_success_manifest(
    saved_modal_bundle, tmp_path
) -> None:
    directory, manifest, samples, base, _counter = saved_modal_bundle
    changed = copy.deepcopy(samples["internal_audit"])
    changed[-1]["model_inputs"]["static"]["equilibrium_delta"][0] += 1e-12
    output = tmp_path / "failed-modal"

    with pytest.raises(StaticFamilyError, match="INVALID_STATIC_FAMILY"):
        evaluate_saved_modal_observability(
            directory / "base",
            base["artifact_manifest_sha256"],
            manifest,
            changed,
            output,
        )

    failure = json.loads((output / "FAILED.json").read_text())
    assert failure["promotion_decision"] == "KEEP_BASELINE"
    assert failure["model_unit_gate_passed"] is False
    assert not (output / "artifact_hashes.json").exists()


@pytest.mark.parametrize("attack", ["changed", "added", "missing"])
def test_frozen_modal_artifacts_reject_tree_mutation(
    saved_modal_bundle, tmp_path, attack
) -> None:
    directory, manifest, samples, base, _counter = saved_modal_bundle
    source = tmp_path / "source-modal"
    result = evaluate_saved_modal_observability(
        directory / "base",
        base["artifact_manifest_sha256"],
        manifest,
        samples["internal_audit"],
        source,
    )
    changed = tmp_path / f"changed-{attack}"
    shutil.copytree(source, changed)
    if attack == "changed":
        (changed / "modal_audit_summary.json").write_bytes(b"changed")
    elif attack == "added":
        (changed / "unregistered.txt").write_text("not registered")
    else:
        (changed / "modal_bootstrap.json").unlink()

    with pytest.raises(ValueError):
        verify_modal_observability_artifacts(changed, result["artifact_manifest_sha256"])


def test_rehashed_but_semantically_incomplete_modal_tree_is_rejected(
    saved_modal_bundle, tmp_path
) -> None:
    directory, manifest, samples, base, _counter = saved_modal_bundle
    source = tmp_path / "semantic-source"
    evaluate_saved_modal_observability(
        directory / "base",
        base["artifact_manifest_sha256"],
        manifest,
        samples["internal_audit"],
        source,
    )
    changed = tmp_path / "semantic-changed"
    shutil.copytree(source, changed)
    summary_path = changed / "modal_audit_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["event_rank_match_cells"] += 1
    summary_path.write_text(json.dumps(summary, sort_keys=True))
    (changed / "artifact_hashes.json").unlink()
    digest = _seal(changed)

    with pytest.raises(ValueError, match="event match cells"):
        verify_modal_observability_artifacts(changed, digest)
