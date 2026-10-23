import h5py
import numpy as np
import pytest
from test_kiki_history import build_sources

from kron_reduce.data.kiki_history import KikiHistorySamples, allocate_roles


def test_short_history_origin_future_cap_and_power_allowlist(tmp_path):
    build_sources(tmp_path, {"e0": 11, "e1": 12, "e2": 400})
    with h5py.File(tmp_path / "A" / "dataset.h5", "r+") as handle:
        for event in handle["events"].values():
            pm = event["Pm"]
            pm[...] = np.nan
            pm[10, :] = np.linspace(0, 0.09, 10)
    rows, bindings = allocate_roles(tmp_path, ["A"])
    rows = [dict(row, role="train") for row in rows]
    reader = KikiHistorySamples(
        tmp_path, rows, bindings, history_count=11, power_index=10
    )
    mass, samples = reader.load("train")
    assert not any(sample.event_id.endswith("/e0") for sample in samples)
    exact = next(sample for sample in samples if sample.event_id.endswith("/e1"))
    assert exact.history.time[0] == pytest.approx(-0.1)
    assert exact.history.time[-1] == pytest.approx(0.0)
    assert exact.forecast_time[0] == pytest.approx(0.0)
    assert len(exact.forecast_time) == 2
    assert np.all(np.diff(exact.forecast_time) == pytest.approx(0.01))
    assert np.isfinite(exact.history.constant_power).all()
    pattern = np.linspace(0, 0.09, 10)
    np.testing.assert_allclose(
        exact.history.constant_power, pattern - pattern.sum() * mass / mass.sum(),
        rtol=0, atol=1e-12,
    )
    capped = next(sample for sample in samples if sample.event_id.endswith("/e2"))
    assert len(capped.forecast_time) == 301
    assert capped.forecast_time[-1] == pytest.approx(3.0)
    assert np.all(np.diff(capped.forecast_time) == pytest.approx(0.01))
    audit = reader.access_audit()
    assert {item["selection"] for item in audit["accesses"] if item["dataset"] == "Pm"} == {"10"}
    assert not {item["dataset"] for item in audit["accesses"]} & {"Pe", "te"}
    coverage = {item["raw_id"]: item for item in audit["coverage"]}
    assert coverage["e0"]["history_eligible"] is False
    assert coverage["e1"]["history_eligible"] is True
    assert coverage["e2"]["future_count"] == 300


@pytest.mark.parametrize("raw_id,role", [("val_only", "train"), ("test_only", "train")])
def test_short_reader_rejects_source_holdout_ids(tmp_path, raw_id, role):
    build_sources(tmp_path)
    rows, bindings = allocate_roles(tmp_path, ["A"])
    forged = dict(rows[0], raw_id=raw_id, role=role, source_role="train")
    with pytest.raises(ValueError, match="allowlist"):
        KikiHistorySamples(tmp_path, [forged], bindings, history_count=11, power_index=10)


@pytest.mark.parametrize("history_count,power_index", [(11, 0), (21, 10), (10, 10), (True, 10), (11, True)])
def test_short_reader_rejects_unregistered_contract(tmp_path, history_count, power_index):
    build_sources(tmp_path)
    rows, bindings = allocate_roles(tmp_path, ["A"])
    with pytest.raises(ValueError, match="contract"):
        KikiHistorySamples(
            tmp_path, rows, bindings, history_count=history_count, power_index=power_index
        )
