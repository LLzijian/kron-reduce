#!/usr/bin/env python3
"""Render TASK3 noise/outlier figures from aggregate JSON only.

This script does not open trajectories or source records. It writes metrics
figures per available roster and an optional timing figure from latency JSON.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.transforms import ScaledTranslation

SCENARIO_ORDER = (
    "clean", "gauss_pmu_steady", "gauss_dse_dynamic", "gauss_dse_dynamic_pm",
    "gauss_snr30", "gauss_snr20", "nongauss_laplace", "nongauss_student_t3",
    "outlier_random5pct_10sigma", "outlier_endpoint_10sigma",
)
GAUSSIAN_SCENARIOS = SCENARIO_ORDER[:6]
STRESS_SCENARIOS = SCENARIO_ORDER[6:]
DISPLAY_SCENARIOS = {
    "clean": "Clean", "gauss_pmu_steady": "Steady PMU",
    "gauss_dse_dynamic": "Dynamic DSE", "gauss_dse_dynamic_pm": "DSE + Pm",
    "gauss_snr30": "30 dB", "gauss_snr20": "20 dB",
    "nongauss_laplace": "Laplace", "nongauss_student_t3": "Student t(3)",
    "outlier_random5pct_10sigma": "Random pulse",
    "outlier_endpoint_10sigma": "Endpoint pulse",
}
ROSTER_ORDER = ("val", "test", "blind_v2")
STEP_ORDER = ("0.0025", "0.00125")
MODEL_ORDER = (
    "v4_hard_r", "v4_soft_2r", "m3b_g2_orig", "m3b_g2_sg_gn",
    "m3b_g2_joint_l2_mhe", "m3b_g2_robust_mhe", "m3b_g2_robust_mhe__B",
    "m3b_g2_robust_mhe__C",
)
MODEL_LABELS = {
    "v4_hard_r": "V4 hard-R", "v4_soft_2r": "V4 soft-2R",
    "m3b_g2_orig": "ES-ROM original", "m3b_g2_sg_gn": "ES-ROM SG + GN",
    "m3b_g2_joint_l2_mhe": "ES-ROM joint L2",
    "m3b_g2_robust_mhe": "Robust MHE selected",
    "m3b_g2_robust_mhe__B": "Robust MHE candidate B",
    "m3b_g2_robust_mhe__C": "Robust MHE candidate C",
}
MODEL_COLORS = {
    "v4_hard_r": "#626b73", "v4_soft_2r": "#a5adb4",
    "m3b_g2_orig": "#334e68", "m3b_g2_sg_gn": "#3d8b88",
    "m3b_g2_joint_l2_mhe": "#c17b32", "m3b_g2_robust_mhe": "#a64253",
    "m3b_g2_robust_mhe__B": "#d07a82", "m3b_g2_robust_mhe__C": "#7f526d",
}
MODEL_STYLES = {"m3b_g2_robust_mhe__B": "--", "m3b_g2_robust_mhe__C": ":"}


def _publication_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.size": 6.5, "axes.labelsize": 6.5, "axes.titlesize": 7,
        "xtick.labelsize": 5.8, "ytick.labelsize": 5.8, "legend.fontsize": 5.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.65, "lines.linewidth": 1.15, "lines.markersize": 3,
        "legend.frameon": False, "savefig.facecolor": "white", "figure.facecolor": "white",
    })


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _step_key(mapping: dict[str, Any], requested: str) -> str | None:
    if requested in mapping:
        return requested
    for key in mapping:
        try:
            if math.isclose(float(key), float(requested), rel_tol=0, abs_tol=1e-12):
                return key
        except (TypeError, ValueError):
            continue
    return None


def _mean_and_sd(values: list[Any]) -> tuple[float | None, float | None]:
    """Return a complete-case mean and sample SD; incomplete sets stay missing."""
    clean = [_finite(value) for value in values]
    if not clean or any(value is None for value in clean):
        return None, None
    values_array = np.asarray(clean, dtype=float)
    sd = float(np.std(values_array, ddof=1)) if len(values_array) > 1 else 0.0
    return float(np.mean(values_array)), sd


def _summary_metric(roster_data: dict[str, Any], scenario: str, model: str,
                    step: str, metric: str, endpoint: str = "all"
                    ) -> tuple[float | None, float | None]:
    record = roster_data.get(scenario, {}).get(model)
    if not isinstance(record, dict):
        return None, None
    summary = record.get("summary", {})
    by_step = summary.get("by_step", {}) if isinstance(summary, dict) else {}
    actual = _step_key(by_step, step) if isinstance(by_step, dict) else None
    step_data = by_step.get(actual, {}) if actual else {}
    item = None
    if isinstance(step_data, dict):
        endpoint_metrics = step_data.get(endpoint)
        if isinstance(endpoint_metrics, dict):
            item = endpoint_metrics.get(metric)
        if item is None and endpoint == "all":
            item = step_data.get(metric)
    if not isinstance(item, dict):
        return None, None
    mean, sd = _finite(item.get("mean")), _finite(item.get("sample_std"))
    return (mean, sd) if mean is not None else (None, None)


def _repeat_metric_values(roster_data: dict[str, Any], scenario: str, model: str,
                          step: str, extract) -> list[Any]:
    record = roster_data.get(scenario, {}).get(model)
    repeats = record.get("repeats", []) if isinstance(record, dict) else []
    if not isinstance(repeats, list):
        return []
    output = []
    for repeat in repeats:
        steps = repeat.get("steps", {}) if isinstance(repeat, dict) else {}
        actual = _step_key(steps, step) if isinstance(steps, dict) else None
        output.append(extract(steps.get(actual, {}) if actual else {}))
    return output


def _metric_from_repeat(step_output: dict[str, Any], metric: str) -> float | None:
    metrics = step_output.get("metrics", {})
    if not isinstance(metrics, dict):
        return None
    if metric in {"confirmed_recall", "false_alarm_rate", "coverage"}:
        return _finite(metrics.get(metric))
    if metric in {"stable_8s_nrmse", "unstable_available_nrmse"}:
        item = metrics.get(metric)
        return _finite(item.get("mean")) if isinstance(item, dict) else None
    if metric == "positive_peak_deficit_rad_s":
        item = metrics.get(metric)
        return _finite(item.get("median")) if isinstance(item, dict) else None
    if metric in {"exact_cut_rate", "cut_jaccard_mean"}:
        field = "exact_cut" if metric == "exact_cut_rate" else "cut_jaccard"
        item = metrics.get(field)
        subfield = "exact_rate" if metric == "exact_cut_rate" else "mean"
        return _finite(item.get(subfield)) if isinstance(item, dict) else None
    if metric.endswith("_energy_auroc"):
        component = metric.removesuffix("_energy_auroc")
        energy = metrics.get("energy", {})
        item = energy.get(component) if isinstance(energy, dict) else None
        return _finite(item.get("auroc")) if isinstance(item, dict) else None
    return None


def _metric_value(roster_data: dict[str, Any], scenario: str, model: str,
                  step: str, metric: str) -> tuple[float | None, float | None]:
    value = _summary_metric(roster_data, scenario, model, step, metric)
    if value[0] is not None:
        return value
    vals = _repeat_metric_values(
        roster_data, scenario, model, step,
        lambda step_data: _metric_from_repeat(step_data, metric),
    )
    return _mean_and_sd(vals)


def _endpoint_recall(step_output: dict[str, Any], endpoint: int) -> float | None:
    strata = step_output.get("endpoint_strata", {})
    stratum = strata.get(str(endpoint), {}) if isinstance(strata, dict) else {}
    all_events = stratum.get("all", {}) if isinstance(stratum, dict) else {}
    return _finite(all_events.get("confirmed_recall")) if isinstance(all_events, dict) else None


def _endpoint_value(roster_data: dict[str, Any], scenario: str, model: str,
                    step: str, endpoint: int) -> tuple[float | None, float | None]:
    summary = _summary_metric(
        roster_data, scenario, model, step, "confirmed_recall", endpoint=str(endpoint)
    )
    if summary[0] is not None:
        return summary
    vals = _repeat_metric_values(
        roster_data, scenario, model, step,
        lambda step_data: _endpoint_recall(step_data, endpoint),
    )
    return _mean_and_sd(vals)


def _roster_model_names(roster_data: dict[str, Any]) -> list[str]:
    available = {model for by_model in roster_data.values() if isinstance(by_model, dict)
                 for model in by_model}
    return [model for model in MODEL_ORDER if model in available] + sorted(available - set(MODEL_ORDER))


def _model_label(model: str) -> str:
    return MODEL_LABELS.get(model, model.replace("_", " "))


def _model_color(model: str, index: int) -> str:
    return MODEL_COLORS.get(model, matplotlib.colors.to_hex(plt.get_cmap("tab10")(index % 10)))


def _scenario_list(data: dict[str, Any], allowed: tuple[str, ...]) -> list[str]:
    return [scenario for scenario in allowed if scenario in data]


def _add_model_lines(ax, data: dict[str, Any], scenarios: list[str], models: list[str],
                     step: str, metric: str) -> None:
    x = np.arange(len(scenarios), dtype=float)
    for index, model in enumerate(models):
        values, spread = [], []
        for scenario in scenarios:
            mean, sd = _metric_value(data, scenario, model, step, metric)
            values.append(np.nan if mean is None else mean)
            spread.append(np.nan if mean is None or sd is None else sd)
        y, sd_values = np.asarray(values, float), np.asarray(spread, float)
        available = np.isfinite(y)
        if not available.any():
            continue
        color = _model_color(model, index)
        ax.plot(x, y, color=color, linestyle=MODEL_STYLES.get(model, "-"), marker="o",
                label=_model_label(model), zorder=3)
        with_error = available & np.isfinite(sd_values) & (sd_values > 0)
        if with_error.any():
            ax.errorbar(x[with_error], y[with_error], yerr=sd_values[with_error], fmt="none",
                        ecolor=color, elinewidth=0.65, capsize=1.8, alpha=0.8, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_SCENARIOS.get(name, name) for name in scenarios], rotation=28,
                       ha="right", rotation_mode="anchor")
    ax.grid(axis="y", color="#d9dee3", linewidth=0.45)
    ax.set_axisbelow(True)


def _add_shared_legend(fig, axes) -> None:
    handles, labels, seen = [], [], set()
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in seen:
                seen.add(label)
                handles.append(handle)
                labels.append(label)
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)),
                   bbox_to_anchor=(0.5, 0.012), handlelength=2.0, columnspacing=1.0,
                   labelspacing=0.3)


def _add_panel_labels(axes) -> None:
    for index, ax in enumerate(axes):
        offset = ScaledTranslation(-7 / 72, 2 / 72, ax.figure.dpi_scale_trans)
        ax.text(0, 1, chr(ord("a") + index), transform=ax.transAxes + offset,
                fontsize=8, fontweight="bold", ha="right", va="bottom", clip_on=False)


def _alignment_helper():
    scripts_dir = os.environ.get("NATURE_FIGURE_SKILL_SCRIPTS")
    if not scripts_dir:
        scripts_dir = str(Path.home() / ".agents" / "skills" / "nature-figure" / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    try:
        module = importlib.import_module("audit_panel_alignment")
    except ImportError as exc:
        raise RuntimeError(
            "panel alignment QA needs the nature-figure scripts directory; set "
            "NATURE_FIGURE_SKILL_SCRIPTS to that directory"
        ) from exc
    return module.require_matplotlib_panel_alignment


def _finalize(fig, base: Path, dpi: int = 300) -> list[str]:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.10, 1, 0.97), h_pad=1.35, w_pad=1.0)
    fig.canvas.draw()
    _add_panel_labels(fig.axes)
    fig.canvas.draw()
    align = _alignment_helper()
    align(
        fig,
        json_out=base.with_name(base.name + ".alignment.json"),
        overlay_svg=base.with_name(base.name + ".alignment.svg"),
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
    )
    paths = []
    for extension, options in (("pdf", {"bbox_inches": "tight"}),
                               ("png", {"dpi": dpi, "bbox_inches": "tight"}),
                               ("svg", {"bbox_inches": "tight"})):
        output = base.with_suffix(f".{extension}")
        fig.savefig(output, **options)
        paths.append(str(output))
    plt.close(fig)
    return paths


def _plot_gaussian(roster: str, data: dict[str, Any], step: str, out_dir: Path) -> list[str]:
    scenarios = _scenario_list(data, GAUSSIAN_SCENARIOS)
    if not scenarios:
        return []
    models = _roster_model_names(data)
    specs = (
        ("stable_8s_nrmse", "Stable 8 s NRMSE", None),
        ("unstable_available_nrmse", "Unstable available NRMSE", None),
        ("confirmed_recall", "Confirmed unstable recall", (0, 1)),
        ("false_alarm_rate", "Stable false alarm rate", (0, 1)),
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.3))
    for ax, (metric, title, limits) in zip(axes.flat, specs):
        _add_model_lines(ax, data, scenarios, models, step, metric)
        ax.set_title(title, loc="left", pad=4)
        ax.set_ylabel("NRMSE" if "nrmse" in metric else "Rate")
        if limits:
            ax.set_ylim(*limits)
    for ax in axes[-1, :]:
        ax.set_xlabel("Gaussian condition")
    _add_shared_legend(fig, axes.flat)
    fig.suptitle(f"Gaussian measurement-noise degradation · {roster} · step {step} s", y=0.995)
    return _finalize(fig, out_dir / f"{roster}_gaussian_degradation")


def _plot_heavy_tail_pulse(roster: str, data: dict[str, Any], step: str,
                           out_dir: Path) -> list[str]:
    scenarios = _scenario_list(data, STRESS_SCENARIOS)
    if not scenarios:
        return []
    models = _roster_model_names(data)
    specs = (
        ("stable_8s_nrmse", "Stable 8 s NRMSE", None),
        ("unstable_available_nrmse", "Unstable available NRMSE", None),
        ("confirmed_recall", "Confirmed unstable recall", (0, 1)),
        ("false_alarm_rate", "Stable false alarm rate", (0, 1)),
    )
    fig, axes = plt.subplots(3, 2, figsize=(7.4, 7.0))
    for ax, (metric, title, limits) in zip(axes[:2].flat, specs):
        _add_model_lines(ax, data, scenarios, models, step, metric)
        ax.set_title(title, loc="left", pad=4)
        ax.set_ylabel("NRMSE" if "nrmse" in metric else "Rate")
        if limits:
            ax.set_ylim(*limits)
    x = np.arange(len(scenarios), dtype=float)
    for ax, endpoint, endpoint_name in (
        (axes[2, 0], 0, "First input sample (index 0)"),
        (axes[2, 1], 10, "Last input sample (index 10)"),
    ):
        for index, model in enumerate(models):
            values, spread = [], []
            for scenario in scenarios:
                mean, sd = _endpoint_value(data, scenario, model, step, endpoint)
                values.append(np.nan if mean is None else mean)
                spread.append(np.nan if mean is None or sd is None else sd)
            y, s = np.asarray(values, float), np.asarray(spread, float)
            available = np.isfinite(y)
            if not available.any():
                continue
            color = _model_color(model, index)
            ax.plot(x, y, color=color, linestyle=MODEL_STYLES.get(model, "-"), marker="o",
                    label=_model_label(model), zorder=3)
            with_error = available & np.isfinite(s) & (s > 0)
            if with_error.any():
                ax.errorbar(x[with_error], y[with_error], yerr=s[with_error], fmt="none",
                            ecolor=color, elinewidth=0.65, capsize=1.8, alpha=0.8, zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels([DISPLAY_SCENARIOS.get(name, name) for name in scenarios], rotation=28,
                           ha="right", rotation_mode="anchor")
        ax.set_title(f"Confirmed recall · {endpoint_name}", loc="left", pad=4)
        ax.set_ylabel("Recall")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", color="#d9dee3", linewidth=0.45)
        ax.set_axisbelow(True)
    for ax in axes[-1, :]:
        ax.set_xlabel("Heavy-tail or pulse condition")
    _add_shared_legend(fig, axes.flat)
    fig.suptitle(f"Heavy-tail and pulse robustness · {roster} · step {step} s", y=0.995)
    return _finalize(fig, out_dir / f"{roster}_heavy_tail_pulse")


def _plot_cut_energy(roster: str, data: dict[str, Any], step: str, out_dir: Path) -> list[str]:
    scenarios = [name for name in SCENARIO_ORDER if name != "clean" and name in data]
    if not scenarios:
        return []
    models = _roster_model_names(data)
    specs = (
        ("exact_cut_rate", "Exact cut rate", "Rate"),
        ("cut_jaccard_mean", "Cut Jaccard", "Mean Jaccard"),
        ("mechanical_energy_auroc", "Mechanical energy AUROC", "AUROC"),
        ("memory_energy_auroc", "Memory energy AUROC", "AUROC"),
        ("total_energy_auroc", "Total energy AUROC", "AUROC"),
    )
    fig, axes = plt.subplots(len(specs), 1, figsize=(8.6, 8.6), sharex=True)
    for ax, (metric, title, ylabel) in zip(axes, specs):
        _add_model_lines(ax, data, scenarios, models, step, metric)
        ax.set_title(title, loc="left", pad=3)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1)
    axes[-1].set_xlabel("Noise or outlier condition")
    _add_shared_legend(fig, axes)
    fig.suptitle(f"Failure timing and energy discrimination · {roster} · step {step} s", y=0.995)
    return _finalize(fig, out_dir / f"{roster}_cut_energy_auroc")


def _latency_samples(item: Any) -> tuple[np.ndarray, np.ndarray] | None:
    if not isinstance(item, dict):
        return None
    init, rollout = item.get("init_s"), item.get("rollout_s")
    if not isinstance(init, list) or not isinstance(rollout, list) or len(init) != len(rollout):
        return None
    init_clean, rollout_clean = [_finite(value) for value in init], [_finite(value) for value in rollout]
    if not init_clean or any(value is None for value in init_clean + rollout_clean):
        return None
    init_array, rollout_array = np.asarray(init_clean), np.asarray(rollout_clean)
    if np.any(init_array < 0) or np.any(rollout_array < 0):
        return None
    return init_array, rollout_array


def _plot_latency(latency: dict[str, Any], step: str, out_dir: Path) -> list[str]:
    by_model = latency.get("by_model")
    if not isinstance(by_model, dict) or not by_model:
        return []
    models = [model for model in MODEL_ORDER if model in by_model]
    models.extend(sorted(set(by_model) - set(MODEL_ORDER)))
    source_roster = str(latency.get("source_roster", "unspecified"))
    fig, axes = plt.subplots(1, 2, figsize=(8.7, 4.1), sharey=True)
    colors = {"init": "#7b9aaa", "rollout": "#334e68"}
    hatches = {"0.0025": "", "0.00125": "//"}
    bar_group_width = 0.82
    bar_width = bar_group_width / len(STEP_ORDER)
    x = np.arange(len(models), dtype=float)
    for ax, batch in zip(axes, ("1", "64")):
        for model_index, model in enumerate(models):
            model_data = by_model.get(model, {})
            for step_index, requested_step in enumerate(STEP_ORDER):
                actual_step = _step_key(model_data, requested_step) if isinstance(model_data, dict) else None
                step_data = model_data.get(actual_step, {}) if actual_step else {}
                batch_data = step_data.get(batch, {}) if isinstance(step_data, dict) else {}
                samples = _latency_samples(batch_data)
                if samples is None:
                    continue
                init, rollout = samples
                center = x[model_index] + (step_index - 0.5) * bar_width
                init_mean, rollout_mean = float(np.mean(init)), float(np.mean(rollout))
                total = init + rollout
                total_mean = float(np.mean(total))
                total_sd = float(np.std(total, ddof=1)) if len(total) > 1 else 0.0
                ax.bar(center, init_mean, width=bar_width * 0.86, color=colors["init"],
                       edgecolor="white", linewidth=0.35, hatch=hatches[requested_step], zorder=3)
                ax.bar(center, rollout_mean, bottom=init_mean, width=bar_width * 0.86,
                       color=colors["rollout"], edgecolor="white", linewidth=0.35,
                       hatch=hatches[requested_step], zorder=3)
                ax.errorbar(center, total_mean, yerr=total_sd, fmt="none", ecolor="#20252a",
                            elinewidth=0.65, capsize=1.6, zorder=4)
        ax.set_title(f"Batch {batch}", loc="left", pad=4)
        ax.set_xticks(x)
        ax.set_xticklabels([_model_label(model) for model in models], rotation=40,
                           ha="right", rotation_mode="anchor")
        ax.set_xlabel("Model")
        ax.grid(axis="y", color="#d9dee3", linewidth=0.45)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Mean time per trajectory (s)")
    legend = [
        Patch(facecolor=colors["init"], label="Initialization"),
        Patch(facecolor=colors["rollout"], label="Free rollout"),
        Patch(facecolor="white", edgecolor="#555", hatch=hatches["0.0025"], label="Step 0.0025 s"),
        Patch(facecolor="white", edgecolor="#555", hatch=hatches["0.00125"], label="Step 0.00125 s"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.01),
               handlelength=1.6, columnspacing=1.0)
    fig.suptitle(f"End-to-end latency split by stage · timing roster: {source_roster}", y=0.995)
    return _finalize(fig, out_dir / "latency_initialization_rollout")


def render(metrics: dict[str, Any], latency: dict[str, Any] | None, step: str,
           out_dir: Path) -> list[str]:
    rosters = metrics.get("rosters")
    if not isinstance(rosters, dict):
        raise TypeError("metrics JSON must contain an object at top-level key 'rosters'")
    rosters_available = [name for name in ROSTER_ORDER if isinstance(rosters.get(name), dict)]
    if not rosters_available:
        raise ValueError("metrics JSON has no val, test, or blind_v2 roster")
    outputs = []
    for roster in rosters_available:
        data = rosters[roster]
        outputs.extend(_plot_gaussian(roster, data, step, out_dir))
        outputs.extend(_plot_heavy_tail_pulse(roster, data, step, out_dir))
        outputs.extend(_plot_cut_energy(roster, data, step, out_dir))
    if latency is not None:
        outputs.extend(_plot_latency(latency, step, out_dir))
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-json", type=Path, required=True,
                        help="Aggregate metrics JSON; no trajectory rows are read.")
    parser.add_argument("--latency-json", type=Path, help="Optional aggregate timing JSON.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Destination for PDF, PNG, and SVG outputs.")
    parser.add_argument("--step", choices=STEP_ORDER, default="0.0025",
                        help="Forecast integration step represented in the figures.")
    parser.add_argument("--dpi", type=int, default=300, help="PNG resolution (default: 300 dpi).")
    args = parser.parse_args(argv)
    if args.dpi < 300:
        parser.error("--dpi must be at least 300 for paper-quality raster output")
    _publication_style()
    metrics = _load_json(args.metrics_json)
    latency = _load_json(args.latency_json) if args.latency_json else None
    outputs = render(metrics, latency, args.step, args.output_dir)
    if not outputs:
        raise ValueError("no figures rendered; check the supplied aggregate metric fields")
    print(f"Rendered {len(outputs) // 3} figure(s) in {args.output_dir}")
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
