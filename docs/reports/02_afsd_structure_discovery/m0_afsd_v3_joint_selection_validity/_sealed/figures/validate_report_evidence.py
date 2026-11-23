"""Fail-fast checks for the report's numerical evidence snapshots."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
V2 = ROOT / "results/paper_m0_afsd_v2_controlled_f1_seed324-controlled_experiment-20260812T152745Z-6a24d166"
V2_ACCEPTANCE = ROOT / "results/paper_m0_afsd_v2_acceptance_f1_seed324-acceptance-20260812T160526Z-6a24d166"
V3 = ROOT / "results/paper_m0_afsd_v3_eligibility_f1_seed324-eligibility_audit-20260812T170816Z-4c81daed"
V3_ACCEPTANCE = ROOT / "results/paper_m0_afsd_v3_acceptance_f1_seed324-acceptance-20260812T170911Z-4c81daed"


def check_v2_snapshot() -> None:
    original = pd.read_csv(V2 / "aggregate_metrics.csv")
    original = original[
        (original["cohort"] == "C3->C3")
        & original["branch"].isin(["B2", "SS-DM"])
        & original["horizon_steps"].isin([100, 500, 1000])
    ].copy()
    original["horizon_s"] = original["horizon_steps"] / 100
    snapshot = pd.read_csv(HERE / "v2_conditional_rollout_metrics.csv")
    mapping = {
        "branch": "model",
        "mean_rmse": "mean_rmse",
        "median_rmse": "median_rmse",
        "pooled_rmse": "pooled_rmse",
        "failure_rate": "failure_rate",
        "mean_spectral_overlap": "spectral_overlap",
        "median_modal_frequency_relative_error": "frequency_relative_error",
        "median_modal_damping_absolute_error": "damping_absolute_error",
        "median_modal_mac": "mac",
    }
    left = original[[*mapping, "horizon_s"]].rename(columns=mapping)
    columns = ["model", "horizon_s", *[value for value in mapping.values() if value != "model"]]
    left = left[columns].sort_values(["model", "horizon_s"]).reset_index(drop=True)
    right = snapshot[columns].sort_values(["model", "horizon_s"]).reset_index(drop=True)
    assert left["model"].tolist() == right["model"].tolist()
    assert np.allclose(left["horizon_s"].to_numpy(float), right["horizon_s"].to_numpy(float))
    assert np.allclose(
        left.drop(columns=["model"]).to_numpy(float),
        right.drop(columns=["model"]).to_numpy(float),
        equal_nan=True,
    )
    acceptance = json.loads((V2_ACCEPTANCE / "acceptance.json").read_text(encoding="utf-8"))
    assert acceptance["selected_route"] == "SS-DM"
    assert np.isclose(acceptance["relative_improvement_vs_B2"], 0.5135325099395907)


def check_v3_snapshot() -> None:
    original = pd.read_csv(V3 / "source_excitation_table.csv")
    original["source"] = original["source_dataset"].replace(
        {
            "ieee39_highorder_dataset": "high-order",
            "ieee39_near_critical_unstable_v1_dense": "dense",
        }
    )
    original = original.rename(
        columns={
            "amplitude_level": "amplitude",
            "richness_level": "richness",
        }
    )
    columns = [
        "source",
        "amplitude",
        "richness",
        "n_events",
        "n_fault_line_clusters",
        "comparable_cell",
        "minimum_clusters_met",
    ]
    snapshot = pd.read_csv(HERE / "source_excitation_fault_lines.csv")
    left = original[columns].sort_values(["source", "amplitude", "richness"]).reset_index(drop=True)
    right = snapshot[columns].sort_values(["source", "amplitude", "richness"]).reset_index(drop=True)
    assert left.equals(right)
    acceptance = json.loads((V3_ACCEPTANCE / "acceptance.json").read_text(encoding="utf-8"))
    assert acceptance["status"] == "PASS"
    assert acceptance["decision"] == "ACCEPT_NEGATIVE_CONCLUSION"
    assert acceptance["scientific_conclusion"] == "SOURCE_EXCITATION_CONFOUNDED"
    assert acceptance["joint_effective_branch_selected"] is False


def check_outputs() -> None:
    for suffix in (".svg", ".pdf", ".png", ".tiff"):
        path = HERE / f"fig1_afsd_v3_evidence{suffix}"
        assert path.is_file() and path.stat().st_size > 0
    report = ROOT / "docs/reports/02_afsd_structure_discovery/m0_afsd_v3_joint_selection_validity/_sealed/m0_afsd_v3_joint_selection_validity_research_report_zh.md"
    text = report.read_text(encoding="utf-8")
    for required in (
        "SOURCE_EXCITATION_CONFOUNDED",
        "0.1251",
        "26.5%",
        "observed \\(P_m(t)\\)",
        "test/F2–F5：未读取",
    ):
        assert required in text


if __name__ == "__main__":
    check_v2_snapshot()
    check_v3_snapshot()
    check_outputs()
    print("report evidence validation: PASS")
