#!/usr/bin/env python3
"""Render ES-ROM V4 figures from normalized aggregate metrics and timing JSON."""

from __future__ import annotations

import argparse
import textwrap
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


def _ordered_models(models):
    available = set(models)
    return [model for model in MODEL_ORDER if model in available] + sorted(available - set(MODEL_ORDER))


def _label(model, labels):
    return labels.get(model, "ES-ROM Robust MHE V4" if model == V4_MODEL else base._model_label(model))


def _model_style(model, index):
    return {
        "color": base.MODEL_COLORS["m3b_g2_robust_mhe"] if model == V4_MODEL else base._model_color(model, index),
        "marker": MARKERS[index % len(MARKERS)],
        "markersize": 4 if model == V4_MODEL else 3,
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
    group_width = 0.78
    x = np.arange(len(scenarios))
    for index, model in enumerate(models):
        offset = group_width * ((index + 0.5) / len(models) - 0.5)
        style = _model_style(model, index)
        for center, scenario in zip(x + offset, scenarios):
            mean, sd = base._metric_value(data, scenario, model, step, metric)
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
        ax.set_ylim(min(0, low), max(1.04, high))


def _endpoints_panel(ax, data, models, step, endpoint, title):
    for index, model in enumerate(models):
        mean, sd = base._summary_metric(data, "outlier_endpoint_10sigma", model, step,
                                        "positive_peak_deficit_rad_s", endpoint=str(endpoint))
        if mean is None:
            mean, sd = base._endpoint_value(data, model, step, endpoint)
        _error(ax, index, mean, sd, _model_style(model, index))
    ax.set_xticks(range(len(models)), [""] * len(models))
    ax.set_xlim(-0.5, len(models) - 0.5)
    _axes_style(ax, title, "Mean peak deficit (rad/s)")


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


def _plot_metrics(roster, data, models, labels, step, out):
    heading = f"{roster} · integration step {float(step) * 1000:g} ms"
    legend_rows = (len(models) + 3) // 4
    legend_bottom = 0.37 + 0.15 * (legend_rows - 1)
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
        fig, axes = plt.subplots(rows, columns, figsize=(7.2, height), squeeze=False)
        for index, spec in enumerate(specs):
            _metrics_panel(axes.flat[index], data, scenarios, models, step, spec,
                           show_labels=columns > 1 or index == len(specs) - 1)
        note = METRICS_NOTE
        if name == "heavy_tail_pulse":
            for ax, endpoint, endpoint_title in zip(axes[-1], (0, 10), ("First input sample", "Last input sample")):
                _endpoints_panel(ax, data, models, step, endpoint, endpoint_title)
            low = min(0, *(ax.get_ylim()[0] for ax in axes[-1]))
            high = max(ax.get_ylim()[1] for ax in axes[-1])
            for ax in axes[-1]:
                ax.set_ylim(low, high)
            note += "\nEndpoint panels use the same model colors and markers as the other panels."
        _model_legend(fig, models, labels, 0.28 / height)
        fig.text(0.5, 0.09 / height, note, ha="center", va="bottom", fontsize=6, linespacing=1.2)
        fig.suptitle(f"{title} · {heading}", fontsize=8, x=0.5, y=1 - 0.09 / height)
        fig.subplots_adjust(left=0.108, right=0.98, bottom=(0.41 + legend_bottom) / height,
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


def _plot_latency(roster, latency, models, labels, step, out):
    by_model = latency.get("by_model", {})
    models = _ordered_models(set(models) | set(by_model))
    notes = ["Whiskers show ±1 sample SD of total time when available. × indicates a missing timing value."]
    common_scope = latency.get("scope")
    if common_scope:
        notes.extend(textwrap.wrap(f"Timing scope — {common_scope}.", width=110))
    labels_by_scope = {}
    for model in models:
        scope = by_model.get(model, {}).get("scope")
        if scope and scope != common_scope:
            labels_by_scope.setdefault(scope, []).append(_label(model, labels))
    for scope, model_labels in labels_by_scope.items():
        notes.extend(textwrap.wrap(f"{', '.join(model_labels)} — {scope}.", width=110))

    height = max(2.9, 1.23 + len(models) * 0.23) + 0.11 * (len(notes) - 1)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, height), sharey=True, sharex=True)
    has_split = has_total_only = False
    for ax, batch in zip(axes, ("1", "64")):
        for row, model in enumerate(models):
            model_data = by_model.get(model, {})
            actual_step = base._step_key(model_data, step)
            cell = model_data.get(actual_step, {}).get(batch, {})
            samples = base._latency_samples(cell)
            if samples is not None:
                has_split = True
                init, rollout = samples
                init_mean, rollout_mean = float(np.mean(init)), float(np.mean(rollout))
                mean, sd = base._mean_and_sd((init + rollout).tolist())
                ax.barh(row, init_mean, height=0.64, color=INIT_COLOR, zorder=3)
                ax.barh(row, rollout_mean, height=0.64, left=init_mean, color=ROLLOUT_COLOR, zorder=3)
            else:
                mean, sd = _total_latency(cell)
                if mean is not None:
                    has_total_only = True
                    ax.barh(row, mean, height=0.64, facecolor="none", edgecolor="#515b64",
                            linewidth=0.9, linestyle="--", zorder=3)
            if mean is None:
                ax.plot(0.014, row, marker="x", markersize=3, markeredgewidth=0.65, linestyle="none",
                        color=MISSING_COLOR, transform=ax.get_yaxis_transform(), clip_on=False)
            elif sd is not None and sd > 0:
                ax.errorbar(mean, row, xerr=sd, fmt="none", ecolor="#272b30", capsize=1.4,
                            elinewidth=0.65, zorder=4)
        ax.set_title(f"Batch {batch}", loc="left", pad=5)
        ax.set_yticks(range(len(models)), [_label(model, labels) for model in models])
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color="#e1e5e9", linewidth=0.45)
        ax.set_axisbelow(True)
    axes[0].set_ylim(len(models) - 0.45, -0.55)
    has_timings = has_split or has_total_only
    if has_timings:
        axes[0].set_xlim(left=min(0, axes[0].get_xlim()[0]))
        handles = []
        if has_split:
            handles.extend([Patch(facecolor=INIT_COLOR, label="Initialization"),
                            Patch(facecolor=ROLLOUT_COLOR, label="Free rollout")])
        if has_total_only:
            handles.append(Patch(facecolor="none", edgecolor="#515b64", linestyle="--", label="Total only"))
        fig.legend(handles=handles, ncol=3, loc="lower center",
                   bbox_to_anchor=(0.5, (0.11 + 0.11 * len(notes)) / height))
        fig.supxlabel("Mean latency per batch (s)", y=(0.40 + 0.11 * len(notes)) / height, fontsize=6.5)
    else:
        axes[0].set_xticks([])
        notes[0] = "× indicates a missing timing value."
    fig.text(0.5, 0.08 / height, "\n".join(notes), ha="center", va="bottom", fontsize=6, linespacing=1.2)
    timing_roster = latency.get("source_roster", "unspecified roster" if by_model else f"{roster} timings unavailable")
    fig.suptitle(f"Initialization and rollout latency · {timing_roster} · step {float(step) * 1000:g} ms",
                 fontsize=8, y=1 - 0.09 / height)
    bottom = (0.70 if has_timings else 0.23) + 0.11 * len(notes)
    fig.subplots_adjust(left=0.24, right=0.98, bottom=bottom / height,
                        top=1 - 0.45 / height, wspace=0.28)
    return _finalize(fig, out / f"{roster}_latency_initialization_rollout")


def render(metrics: dict[str, Any], latency: dict[str, Any], roster: str, step: str, out: Path):
    rosters = metrics.get("rosters", {})
    data = rosters.get(roster)
    if not isinstance(data, dict) or not data:
        raise ValueError(f"metrics JSON has no populated {roster!r} roster")
    models = _ordered_models(base._roster_model_names(data))
    if not models:
        raise ValueError(f"metrics JSON has no models for roster {roster!r}")
    labels = {**latency.get("model_labels", {}), **metrics.get("model_labels", {})}
    with plt.rc_context():
        base._publication_style()
        return [*_plot_metrics(roster, data, models, labels, step, out),
                *_plot_latency(roster, latency, models, labels, step, out)]


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
