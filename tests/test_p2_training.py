from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

from kron_reduce.data.p2_roles import P2RoleRow
from kron_reduce.data.p2_training import (
    P2CausalReader,
    P2PredictionInput,
    P2TrainingSample,
    hierarchical_event_weights,
    hierarchical_row_weights,
)
from kron_reduce.physics import complete_incidence


def _request() -> P2PredictionInput:
    incidence, _ = complete_incidence(10)
    return P2PredictionInput(
        delta0=np.zeros(10),
        omega0=np.zeros(10),
        pm_equilibrium=np.zeros(10),
        pm_clear=np.zeros(10),
        equilibrium_delta=np.zeros(10),
        mass=np.ones(10),
        incidence=incidence,
        time=np.arange(11) * 0.01,
    )


def _sample(event_id: str, source: str, line: str) -> P2TrainingSample:
    request = _request()
    target = np.zeros((len(request.time), 10))
    return P2TrainingSample(
        event_id=event_id,
        source_dataset=source,
        fault_line=line,
        role="fit",
        prediction_input=request,
        target_delta=target,
        target_omega=target,
    )


def test_p2_hierarchical_weights_equalize_source_line_event_and_time_rows() -> None:
    samples = (
        _sample("a1", "A", "L1"),
        _sample("a2", "A", "L1"),
        _sample("a3", "A", "L2"),
        _sample("b1", "B", "L3"),
    )
    weights = hierarchical_event_weights(samples)
    assert weights == {"a1": 0.125, "a2": 0.125, "a3": 0.25, "b1": 0.5}
    expanded = hierarchical_row_weights(samples, (2, 4, 1, 5))
    totals: dict[str, float] = defaultdict(float)
    cursor = 0
    for sample, count in zip(samples, (2, 4, 1, 5), strict=True):
        totals[sample.event_id] = float(expanded[cursor : cursor + count].sum())
        cursor += count
    assert totals == weights
    assert np.isclose(expanded.sum(), 1.0)


def test_p2_prediction_input_has_no_future_schedule_or_routing_fields() -> None:
    request = _request()
    assert request.allowed_field_names == (
        "delta0",
        "omega0",
        "Pm_equilibrium",
        "Pm_clear",
        "equilibrium_delta",
        "mass",
        "incidence",
        "time",
    )
    assert not hasattr(request, "Pm_observed")
    assert not hasattr(request, "source_dataset")
    assert not hasattr(request, "fault_line")


def test_p2_reader_reads_only_clearing_pm_on_synthetic_hdf(tmp_path: Path) -> None:
    h5_path = tmp_path / "ieee39_postclear_canonical_v1.h5"
    rows = []
    with h5py.File(h5_path, "w") as handle:
        events = handle.create_group("events")
        for role, event_id in (
            ("fit", "fit-event"),
            ("selection", "selection-event"),
            ("internal_audit", "audit-event"),
        ):
            row = P2RoleRow(event_id, "synthetic", f"line-{role}", role, "train", 1001)
            rows.append(row)
            event = events.create_group(event_id)
            postclear = event.create_group("postclear")
            postclear.create_dataset("t", data=np.arange(1001) * 0.01)
            postclear.create_dataset("delta", data=np.zeros((1001, 10)))
            postclear.create_dataset("omega", data=np.zeros((1001, 10)))
            postclear.create_dataset("Pm", data=np.zeros((1001, 10)))
            equilibrium = event.create_group("equilibrium")
            equilibrium.create_dataset("delta0", data=np.zeros(10))
            equilibrium.create_dataset("Pm_equilibrium", data=np.zeros(10))
    reader = P2CausalReader(
        tmp_path,
        np.ones(10),
        rows,
        role_manifest_sha256="a" * 64,
    )
    samples = reader.read_role("fit")
    assert len(samples) == 1
    audit = reader.access_audit()
    assert audit["future_pm_reads"] == []
    assert audit["observed_pm_primary_reads"] == []
    assert audit["dataset_read_counts"]["postclear/Pm"] == 1
    pm_access = next(row for row in reader.accesses if row["dataset"] == "postclear/Pm")
    assert pm_access["selection"] == "0"
