import json
import math

import pytest

from kron_reduce.evaluation.recorded_error_audit import (
    audit_rows,
    join_horizons,
    paired_contributions,
    run,
    weights,
)
from project_harness.evidence import sha256_file, verify_artifact_directory


def rows():
    return [{"event_id": e, "source_dataset": s, "fault_line": l, "role": "train", "status": "ok",
                 "normalized_mse": m, "delta_mse_rad2": 4*m, "omega_mse_rad2_per_s2": 9*m,
                 "actual_duration_s": t} for e, s, l, m, t in (
        ("a", "s1", "l1", 1., .5), ("b", "s1", "l1", 3., 2.),
        ("c", "s1", "l2", 5., 3.), ("d", "s2", "l3", 7., 3.))]


LABELS = {"a": "stable", "b": "unstable", "c": "stable", "d": "unstable"}


def test_same_denominator_partition_and_physical_units_hand_calculation():
    data = rows()
    assert weights(data) == pytest.approx([.125, .125, .25, .5])
    result = audit_rows(data, LABELS, math.sqrt(5.25))
    stable = result["groups"]["label"]["stable"]
    assert stable["weight_mass"] == .375
    assert stable["same_denominator_mse_contribution"] == 1.375
    assert stable["conditional_nrmse"] == pytest.approx(math.sqrt(1.375/.375))
    assert stable["squared_error_share"] == pytest.approx(1.375/5.25)
    assert result["physical"]["delta_mse_rad2"]["weighted_mse"] == 21
    for groups in result["groups"].values():
        assert sum(x["same_denominator_mse_contribution"] for x in groups.values()) == 5.25


def score():
    h3 = rows()
    h1 = [{**r, "actual_duration_s": min(1, r["actual_duration_s"]),
           "normalized_mse": 2*r["normalized_mse"]} for r in h3]
    return {"available": {"horizons": {"1.0": {"rows": h1, "normalized_rmse": math.sqrt(10.5)},
                                       "3.0": {"rows": h3, "normalized_rmse": math.sqrt(5.25)}},
                           "multi_horizon_normalized_rmse": math.sqrt(7.875)}}


def test_join_averages_mse_once_and_pairs_without_reweighting():
    joined, result = join_horizons(score(), LABELS)
    assert result["normalized_mse"] == 7.875
    candidate = [{**r, "normalized_mse": r["normalized_mse"]+2} for r in joined]
    diff = paired_contributions(candidate, joined, LABELS)
    assert diff["difference"] == pytest.approx(2)
    assert diff["same_denominator_group_differences"]["label"]["stable"] == .75
    candidate[0]["actual_duration_s"] = .4
    with pytest.raises(ValueError, match="clocks differ"):
        paired_contributions(candidate, joined, LABELS)


@pytest.mark.parametrize("change", [{"status": "failed"}, {"normalized_mse": float("nan")},
                                    {"normalized_mse": -1}, {"role": "test"}])
def test_bad_records_are_not_dropped_or_imputed(change):
    data = rows()
    data[0].update(change)
    with pytest.raises(ValueError):
        audit_rows(data, LABELS, 1.)


def test_unknown_duplicate_and_inconsistent_reports_refuse():
    with pytest.raises(ValueError):
        audit_rows(rows(), {**LABELS, "a": "unknown"}, 1.)
    with pytest.raises(ValueError):
        audit_rows(rows()+[rows()[0]], LABELS, 1.)
    with pytest.raises(ValueError, match="stored report"):
        audit_rows(rows(), LABELS, 1.)
    bad = score()
    bad["available"]["horizons"]["1.0"]["rows"][1]["fault_line"] = "forged"
    with pytest.raises(ValueError):
        join_horizons(bad, LABELS)


def test_actual_cli_run_binds_source_and_never_overwrites(tmp_path):
    source = tmp_path/"source"
    source.mkdir()
    coverage = [{**r, "history_eligible": True, "label": LABELS[r["event_id"]]} for r in rows()]
    val = [{**r, "event_id": "v"+r["event_id"], "role": "val"} for r in coverage]
    files = {"coverage.json": {"rows": coverage+val}, "scientific_plan.json": {"candidate_settings": {}}}
    for c in "ABCD":
        for s in (324, 325, 326):
            model = f"{c}-{s}"
            files["scientific_plan.json"]["candidate_settings"][model] = {"history_weight": 0}
            files[f"candidate-{model}/fit_record.json"] = {"fit_record": {"loss_trace": []}}
            files[f"candidate-{model}/train_scores.json"] = score()
            vs = score()
            for section in vs["available"]["horizons"].values():
                section["rows"] = [{**r, "event_id": "v"+r["event_id"], "role": "val"} for r in section["rows"]]
            files[f"candidate-{model}/val_scores.json"] = vs
    for relative, obj in files.items():
        path = source/relative
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(obj), encoding="utf-8")
    (source/"artifact_hashes.json").write_text(json.dumps({p: sha256_file(source/p) for p in files}), encoding="utf-8")
    before = {p: (source/p).read_bytes() for p in files}
    out = run(source, "audit-1", tmp_path/"results", sha256_file(source/"artifact_hashes.json"))
    verify_artifact_directory(out, strict=True)
    audit = json.loads((out/"audit_summary.json").read_text(encoding="utf-8"))
    assert len(audit["results"]) == 24
    assert audit["optimizer_updates"] == audit["predictions_computed"] == 0
    assert before == {p: (source/p).read_bytes() for p in files}
    with pytest.raises(FileExistsError):
        run(source, "audit-1", tmp_path/"results")
    with pytest.raises(ValueError, match="unsafe"):
        run(source, "../escape", tmp_path/"results")
    with pytest.raises(ValueError, match="separate"):
        run(source, "child", source)
