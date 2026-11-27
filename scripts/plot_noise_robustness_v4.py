#!/usr/bin/env python3
"""Render ES-ROM V4 figures from normalized aggregate metrics and timing JSON."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import plot_noise_robustness as base
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import ScaledTranslation

V4_MODEL = "m3b_g2_robust_mhe_v4"
MODEL_ORDER = (*base.MODEL_ORDER[:5], V4_MODEL)
PEER_MODELS = {
    "task3_baselines/campaign2/graph_gru_extension1/native/VALID_COMPLETE/los_rule": "Graph-GRU",
    "task3_baselines/campaign2/gru_seq2seq_extension1/native/UNDEROPTIMIZED_WITH_RESULTS/los_rule": "GRU Seq2Seq",
    "task3_baselines/campaign2/latent_neural_ode_extension1/native/VALID_COMPLETE/los_rule": "Latent Neural ODE",
    "task3_baselines/latest_algorithm_others/lass_small_p0/native/VALID_COMPLETE/los_rule": "LASS P0",
    "task3_baselines/campaign2/gru_classifier/native/VALID_COMPLETE/fixed": "GRU classifier (fixed)",
    "task3_baselines/campaign2/extra_trees/native/VALID_COMPLETE/fixed": "Extra Trees (fixed)",
}
MARKERS = ("o", "s", "^", "v", "D", "P", "<", ">", "h", "p", "d", "X", "8", "*")
SCENARIO_LABELS = {
    **base.DISPLAY_SCENARIOS,
    "gauss_pmu_steady": "Steady\nPMU", "gauss_dse_dynamic": "Dynamic\nDSE",
    "gauss_dse_dynamic_pm": "DSE + Pm", "nongauss_student_t3": "Student t(3)",
    "outlier_random5pct_10sigma": "Random\npulse", "outlier_endpoint_10sigma": "Endpoint\npulse",
}
ACCURACY_PANELS = (
    ("stable_8s_nrmse", "Stable trajectories, 8 s", "NRMSE"),
    ("unstable_available_nrmse", "Unstable trajectories, available window", "NRMSE"),
    ("confirmed_recall", "Confirmed unstable recall", "Rate"),
    ("false_alarm_rate", "Stable false alarm rate", "Rate"),
)
CUT_ENERGY_PANELS = (
    ("exact_cut_rate", "Exact cut identification", "Rate"),
    ("cut_jaccard_mean", "Cut Jaccard similarity", "Mean Jaccard"),
    ("mechanical_energy_auroc", "Mechanical energy at forecast origin", "AUROC"),
    ("memory_energy_auroc", "Memory energy at forecast origin", "AUROC"),
    ("total_energy_auroc", "Total energy at forecast origin", "AUROC"),
)
MISSING_COLOR = "#777f86"
INIT_COLOR = "#7b9aaa"
ROLLOUT_COLOR = "#334e68"
METRICS_NOTE = "Points show repeat means. Whiskers show ±1 sample SD when available. × indicates a missing value."
PEER_METRICS_NOTE = "Peer models use native steps. Fixed classifiers provide rates. GRU Seq2Seq training was underoptimized."


def _label(model, labels):
    if model == V4_MODEL:
        return "ES-ROM Robust MHE V4"
    return PEER_MODELS.get(model, labels.get(model, base._model_label(model)))


def _model_style(model, index, *, dense=False):
    size = 3.2 if model == V4_MODEL else 2.35
    return {
        "color": base.MODEL_COLORS["m3b_g2_robust_mhe"] if model == V4_MODEL else base._model_color(model, index),
        "marker": MARKERS[index % len(MARKERS)],
        "markersize": size if dense else size + 0.75,
        "markeredgewidth": 0.55,
    }


def _error(ax, center, mean, sd, style):
    if mean is None:
        offset = ScaledTranslation(0, -5.5 / 72, ax.figure.dpi_scale_trans)
        ax.plot(center, 0, marker="x", markersize=2.3, markeredgewidth=0.55, linestyle="none",
                color=MISSING_COLOR, transform=ax.get_xaxis_transform() + offset, clip_on=False)
        return
    ax.errorbar(center, mean, yerr=sd if sd is not None and sd > 0 else None, linestyle="none",
                elinewidth=0.6, capsize=1.3, zorder=3, **style)


def _axes_style(ax, title, ylabel):
    ax.set_title(title, loc="left", pad=5)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#e1e5e9", linewidth=0.45)
    ax.set_axisbelow(True)


def _metrics_panel(ax, data, scenarios, models, step, spec, *, show_labels=True):
    metric, title, ylabel = spec
    group_width = 0.87
    x = np.arange(len(scenarios))
    for index, model in enumerate(models):
        offset = group_width * ((index + 0.5) / len(models) - 0.5)
        style = _model_style(model, index, dense=len(models) > 8)
        model_step = "native" if model in PEER_MODELS else step
        for center, scenario in zip(x + offset, scenarios):
            mean, sd = base._summary_metric(data, scenario, model, model_step, metric)
            _error(ax, center, mean, sd, style)
    ax.set_xlim(-0.5, len(scenarios) - 0.5)
    if show_labels:
        ax.set_xticks(x, [SCENARIO_LABELS[name] for name in scenarios])
        ax.tick_params(axis="x", pad=7)
    else:
        ax.set_xticks(x, [""] * len(scenarios))
    _axes_style(ax, title, ylabel)
    if ylabel != "NRMSE":
        low, high = ax.get_ylim()
        floor = {"exact_cut_rate": 0.75, "cut_jaccard_mean": 0.75, "confirmed_recall": 0.5}.get(metric, 0)
        ax.set_ylim(min(floor, low), max(0.05 if metric == "false_alarm_rate" else 1.04, high))


def _endpoints_panel(ax, data, models, step, endpoint, title):
    for index, model in enumerate(models):
        model_step = "native" if model in PEER_MODELS else step
        mean, sd = base._summary_metric(data, "outlier_endpoint_10sigma", model, model_step,
                                        "positive_peak_deficit_rad_s", endpoint=str(endpoint))
        _error(ax, index, mean, sd, _model_style(model, index))
    ax.set_xticks(range(len(models)), [""] * len(models))
    ax.set_xlim(-0.5, len(models) - 0.5)
    _axes_style(ax, title, "Median peak deficit (rad/s)")


def _finalize(fig, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    base._add_panel_labels(fig.axes)
    fig.canvas.draw()
    require_matplotlib_panel_alignment = base._alignment_helper()
    require_matplotlib_panel_alignment(
        fig, json_out=output.with_suffix(".alignment.json"),
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, require_panel_labels=True, strict=True,
    )
    paths = [output.with_suffix(".pdf"), output.with_suffix(".png")]
    fig.savefig(paths[0])
    fig.savefig(paths[1], dpi=300)
    plt.close(fig)
    return paths


def _model_legend(fig, models, labels, y):
    handles = [Line2D([], [], linestyle="none", label=_label(model, labels), **_model_style(model, index))
               for index, model in enumerate(models)]
    fig.legend(handles=handles, ncol=min(len(models), 4), loc="lower center", bbox_to_anchor=(0.5, y),
               handletextpad=0.35, handlelength=1.2, columnspacing=1.35)


def _plot_metrics(roster, data, local_models, peer_models, labels, step, out):
    heading = f"{roster} · ES-ROM step {float(step) * 1000:g} ms"
    figures = (
        ("gaussian_degradation", "Gaussian measurement noise", base.GAUSSIAN_SCENARIOS,
         ACCURACY_PANELS, 2, 2, 4.15),
        ("heavy_tail_pulse", "Heavy-tailed noise and pulse outliers", base.STRESS_SCENARIOS,
         ACCURACY_PANELS, 3, 2, 5.35),
        ("cut_energy_auroc", "Loss-of-synchronism cuts and energy discrimination", base.SCENARIO_ORDER,
         CUT_ENERGY_PANELS, 5, 1, 7.25),
    )
    outputs = []
    for name, title, scenarios, specs, rows, columns, height in figures:
        cut_energy = name == "cut_energy_auroc"
        models = local_models if cut_energy else local_models + peer_models
        notes = [METRICS_NOTE]
        if cut_energy:
            notes.append("Cut and physical energy metrics are available for ES-ROM models.")
        elif peer_models:
            notes.append(PEER_METRICS_NOTE)
        fig, axes = plt.subplots(rows, columns, figsize=(7.2, height), squeeze=False)
        for index, spec in enumerate(specs):
            _metrics_panel(axes.flat[index], data, scenarios, models, step, spec,
                           show_labels=columns > 1 or index == len(specs) - 1)
        if name == "heavy_tail_pulse":
            for ax, endpoint, endpoint_title in zip(axes[-1], (0, 10), ("First input sample", "Last input sample")):
                _endpoints_panel(ax, data, models, step, endpoint, endpoint_title)
            low = min(0, *(ax.get_ylim()[0] for ax in axes[-1]))
            high = max(ax.get_ylim()[1] for ax in axes[-1])
            for ax in axes[-1]:
                ax.set_ylim(low, high)
            notes.append("Endpoint values summarize the five repeat medians.")
        legend_y = 0.08 + 0.11 * len(notes)
        legend_height = 0.15 * ((len(models) + 3) // 4)
        _model_legend(fig, models, labels, legend_y / height)
        fig.text(0.5, 0.09 / height, "\n".join(notes), ha="center", va="bottom", fontsize=6, linespacing=1.2)
        fig.suptitle(f"{title} · {heading}", fontsize=8, x=0.5, y=1 - 0.09 / height)
        fig.subplots_adjust(left=0.108, right=0.98, bottom=(0.38 + legend_y + legend_height) / height,
                            top=1 - 0.49 / height, hspace=0.65 if columns > 1 else 0.5, wspace=0.33)
        outputs.extend(_finalize(fig, out / f"{roster}_{name}"))
    return outputs


def _total_latency(cell):
    totals = cell.get("total_s")
    if isinstance(totals, list):
        mean, sd = base._mean_and_sd(totals)
        if mean is not None and all(value >= 0 for value in totals):
            return mean, sd
    if isinstance(totals, dict):
        mean, sd = base._finite(totals.get("mean")), base._finite(totals.get("sample_std"))
        if mean is not None and mean >= 0:
            return mean, sd if sd is not None and sd >= 0 else None
    return None, None


def _latency_row(axes, by_model, models, labels, step, batches):
    stages = set()
    for ax, batch in zip(axes, batches):
        for row, model in enumerate(models):
            model_data = by_model.get(model, {})
            step_key = base._step_key(model_data, step)
            cell = model_data.get(step_key, {}).get(batch, {})
            split = base._latency_samples(cell) if step != "native" else None
            if split is not None:
                stages.add("split")
                init, rollout = split
                init_mean, rollout_mean = float(np.mean(init)), float(np.mean(rollout))
                mean, sd = base._mean_and_sd((init + rollout).tolist())
                ax.barh(row, init_mean, height=0.64, color=INIT_COLOR, zorder=3)
                ax.barh(row, rollout_mean, height=0.64, left=init_mean, color=ROLLOUT_COLOR, zorder=3)
            else:
                mean, sd = _total_latency(cell)
                if mean is not None:
                    stages.add("total")
                    ax.barh(row, mean, height=0.64, facecolor="none", edgecolor="#515b64",
                            linewidth=0.9, linestyle="--", zorder=3)
            if mean is None:
                ax.plot(0.014, row, marker="x", markersize=3, markeredgewidth=0.65, linestyle="none",
                        color=MISSING_COLOR, transform=ax.get_yaxis_transform(), clip_on=False)
            elif sd is not None and sd > 0:
                ax.errorbar(mean, row, xerr=sd, fmt="none", ecolor="#272b30", capsize=1.4,
                            elinewidth=0.65, zorder=4)
            if step == "native" and mean is not None and mean < 0.01:
                ax.annotate(f"{mean * 1000:.2f} ms", (mean + (sd or 0), row), xytext=(6, 0),
                            textcoords="offset points", fontsize=5.8, ha="left", va="center")
        ax.set_title(f"Batch {batch}", loc="left", pad=5)
        ax.set_yticks(range(len(models)), [_label(model, labels) for model in models])
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color="#e1e5e9", linewidth=0.45)
        ax.set_axisbelow(True)
    axes[0].set_ylim(len(models) - 0.45, -0.55)
    if stages:
        axes[0].set_xlim(left=min(0, axes[0].get_xlim()[0]))
    else:
        axes[0].set_xticks([])
    return stages


def _plot_latency(roster, latency, local_models, peer_models, labels, step, out):
    by_model = latency.get("by_model", {})
    local_scope = (f"ES-ROM · val dynamic DSE Gaussian noise · CUDA initialization and rollout"
                   f" · step {float(step) * 1000:g} ms")
    timing_rows = [(local_models, step, ("1", "64"), local_scope)]
    if peer_models:
        timing_rows.append((peer_models, "native", ("1", "32"),
                            "TASK3 baselines · val clean · native inference with host assembly and transfers"))
    height = 4.9 if peer_models else 2.9
    fig, axes = plt.subplots(len(timing_rows), 2, figsize=(7.2, height), squeeze=False,
                             sharex="row", sharey="row")
    stages = set()
    for row_axes, (models, model_step, batches, _) in zip(axes, timing_rows):
        stages.update(_latency_row(row_axes, by_model, models, labels, model_step, batches))

    notes = ["Whiskers show ±1 timing-trial SD when available. × indicates a missing timing value."]
    if peer_models:
        notes.append("Peer forecast steps are 10 ms for Graph-GRU and GRU Seq2Seq and 5 ms for Latent Neural ODE.")
    handles = []
    if "split" in stages:
        handles.extend([Patch(facecolor=INIT_COLOR, label="Initialization"),
                        Patch(facecolor=ROLLOUT_COLOR, label="Free rollout")])
    if "total" in stages:
        handles.append(Patch(facecolor="none", edgecolor="#515b64", linestyle="--", label="Total only"))
    if handles:
        fig.legend(handles=handles, ncol=3, loc="lower center",
                   bbox_to_anchor=(0.5, (0.11 + 0.11 * len(notes)) / height))
        fig.supxlabel("Mean latency per batch (s)", y=(0.40 + 0.11 * len(notes)) / height, fontsize=6.5)
    else:
        notes[0] = "× indicates a missing timing value."
    fig.text(0.5, 0.08 / height, "\n".join(notes), ha="center", va="bottom", fontsize=6, linespacing=1.2)
    fig.suptitle("Inference latency by timing protocol", fontsize=8, y=1 - 0.09 / height)
    bottom = (0.70 if handles else 0.23) + 0.11 * len(notes)
    fig.subplots_adjust(left=0.24, right=0.98, bottom=bottom / height,
                        top=1 - 0.71 / height, hspace=0.46, wspace=0.28)
    for row_axes, (_, _, _, scope) in zip(axes, timing_rows):
        top = row_axes[0].get_position().y1
        fig.text(0.24, top + 0.29 / height, scope, fontsize=6.5, ha="left", va="bottom")
    return _finalize(fig, out / f"{roster}_latency_initialization_rollout")


def render(metrics: dict[str, Any], latency: dict[str, Any], roster: str, step: str, out: Path):
    rosters = metrics.get("rosters", {})
    data = rosters.get(roster)
    if not isinstance(data, dict) or not data:
        raise ValueError(f"metrics JSON has no populated {roster!r} roster")
    available = set(base._roster_model_names(data))
    local_models = [model for model in MODEL_ORDER if model in available]
    peer_models = [model for model in PEER_MODELS if model in available]
    if not local_models:
        raise ValueError(f"metrics JSON has no ES-ROM models for roster {roster!r}")
    labels = {**latency.get("model_labels", {}), **metrics.get("model_labels", {})}
    with plt.rc_context():
        base._publication_style()
        return [*_plot_metrics(roster, data, local_models, peer_models, labels, step, out),
                *_plot_latency(roster, latency, local_models, peer_models, labels, step, out)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-json", type=Path, required=True)
    parser.add_argument("--latency-json", type=Path)
    parser.add_argument("--roster", choices=base.ROSTER_ORDER, required=True)
    parser.add_argument("--step", choices=base.STEP_ORDER, default="0.0025")
    parser.add_argument("--out", "--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    metrics = base._load_json(args.metrics_json)
    latency = base._load_json(args.latency_json) if args.latency_json else {}
    for path in render(metrics, latency, args.roster, args.step, args.out):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
