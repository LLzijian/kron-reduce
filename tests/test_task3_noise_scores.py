"""Synthetic checks for paired fault-group uncertainty and coverage reporting."""

import numpy as np

from kron_reduce.evaluation.task3_noise_scores import (
    _bootstrap_metric_draws,
    _group_bootstrap_data,
    paired_group_bootstrap,
    score_pair,
    summarize_reps,
)


def _row(event_id, group, label, total):
    return {
        "id": event_id,
        "group": group,
        "label": label,
        "status": "stable" if label == "stable" else "unstable",
        "stable_8s_nrmse": 0.1 if label == "stable" else None,
        "unstable_available_nrmse": 0.2 if label == "unstable" else None,
        "positive_peak_deficit_rad_s": 0.03 if label == "unstable" else None,
        "exact_cut": True if label == "unstable" else None,
        "cut_jaccard": 0.5 if label == "unstable" else None,
        "energy": {"mechanical": total + 1.0, "memory": total + 2.0, "total": total},
    }


def test_auroc_bootstrap_matches_brute_multiplicity_with_unequal_groups_and_ties():
    groups = ["g0", "g1", "g2"]
    reps = [
        [
            _row("r0a", "g0", "unstable", 0.8), _row("r0b", "g0", "stable", 0.8),
            _row("r0c", "g1", "unstable", 0.9), _row("r0d", "g1", "stable", 0.1),
            _row("r0e", "g1", "stable", 0.5), _row("r0f", "g2", "stable", 0.4),
        ],
        [
            _row("r1a", "g0", "unstable", 0.5), _row("r1b", "g0", "stable", 0.2),
            _row("r1c", "g1", "unstable", 0.7), _row("r1d", "g1", "unstable", 0.6),
            _row("r1e", "g1", "stable", 0.7), _row("r1f", "g2", "stable", 0.3),
            _row("r1g", "g2", "stable", 0.3),
        ],
    ]
    grouped = [_group_bootstrap_data(rows, groups) for rows in reps]
    multiplicities = np.asarray([[[2, 0, 1], [2, 0, 1]]], dtype=np.int16)
    actual = _bootstrap_metric_draws(multiplicities, grouped)["total_energy_auroc"][0]

    sampled = []
    for rows in reps:
        for group, copies in zip(groups, (2, 0, 1), strict=True):
            sampled.extend(row for _ in range(copies) for row in rows if row["group"] == group)
    positive = [row["energy"]["total"] for row in sampled if row["label"] == "unstable"]
    negative = [row["energy"]["total"] for row in sampled if row["label"] == "stable"]
    expected = sum((a > b) + 0.5 * (a == b) for a in positive for b in negative) / (len(positive) * len(negative))
    assert actual == expected


def test_replicate_summary_includes_counts_energy_and_endpoint_strata():
    mass = np.ones(10)
    x = np.zeros((811, 20))
    events = [
        {"id": "a", "label": "stable", "group": "g1", "x": x, "raw_x": x,
         "p0": np.zeros(10), "mass": mass, "equilibrium": np.zeros(10), "omega_s": 2 * np.pi * 60},
        {"id": "b", "label": "unstable", "group": "g2", "x": x, "raw_x": x,
         "p0": np.zeros(10), "mass": mass, "equilibrium": np.zeros(10), "omega_s": 2 * np.pi * 60},
    ]
    paths = {0.0025: np.zeros((2, 801, 20)), 0.00125: np.zeros((2, 801, 20))}
    energies = {step: np.asarray([[1, 2, 3], [2, 1, 4]], dtype=np.float64) for step in paths}
    scored = score_pair(events, paths, energies, endpoint=[0, 10])
    summary = summarize_reps([scored, scored])["by_step"]["0.0025"]
    assert set(summary) == {"all", "0", "10"}
    assert summary["all"]["tp"]["mean"] == 0
    assert summary["all"]["tn"]["mean"] == 1
    assert summary["all"]["uncovered"]["mean"] == 0
    assert summary["all"]["mechanical_energy_auroc"]["mean"] == 1
    assert summary["all"]["memory_energy_auroc"]["mean"] == 0
    assert summary["all"]["tp"]["worst_rep_index"] == 0
    for step in scored["steps"].values():
        step.pop("rows")
    assert summarize_reps([scored, scored])["by_step"]["0.0025"] == summary


def test_bootstrap_reuses_one_group_draw_vector_for_every_noise_rep():
    # Equal candidate/original data gives a zero paired delta for every shared draw.
    def make_rep(offset):
        rows = []
        for group in range(21):
            for index in range(4):
                label = "unstable" if index < 2 else "stable"
                rows.append(_row(f"g{group}-{index}", f"g{group:02d}", label,
                                 float(index + group / 10 + offset)))
        return {"steps": {step: {"rows": rows} for step in ("0.0025", "0.00125")}}

    candidate = [make_rep(0.0) for _ in range(3)]
    original = [make_rep(0.0) for _ in range(3)]
    result = paired_group_bootstrap(candidate, original)
    for step in ("0.0025", "0.00125"):
        assert result["by_step"][step]["total_energy_auroc"]["candidate_minus_original"] == 0.0
        assert result["by_step"][step]["total_energy_auroc"]["valid_replicates"] == 10_000
    assert result["shared_group_draws_sha256"]

