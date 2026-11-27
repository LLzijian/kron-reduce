"""Render the ES-ROM V2 synthetic endpoint gate from its score receipt."""

from __future__ import annotations

import argparse
import json
import runpy
from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

NAMES = ("orig", "v1", *"ABCDEFGH")
LABELS = ("GN", "V1", *"ABCDEFGH")
STEPS = ("0.0025", "0.00125")


def render(diagnostics):
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": .6,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig = Figure(figsize=(7.2, 2.9), facecolor="white")
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 3)
    fig.subplots_adjust(left=.09, right=.98, bottom=.25, top=.81, wspace=.43)
    x = np.arange(len(NAMES))
    colors = ["#747f8b", "#25384a", *["#187a78"] * 6, *["#ab632d"] * 2]
    rows = [diagnostics["steps"][key] for key in STEPS]
    specs = [
        ("First input frame", "Peak error (rad/s)", .9,
         [[row["endpoint_target_peak_means"]["0"][name] for name in NAMES] for row in rows]),
        ("Last input frame", "Peak error (rad/s)", .9,
         [[row["endpoint_target_peak_means"]["10"][name] for name in NAMES] for row in rows]),
        ("8 s free rollout", "NRMSE", 1.,
         [[row["models"][name]["aggregate_nrmse"] for name in NAMES] for row in rows]),
    ]
    markers = (("o", -.14, 20), ("x", .14, 23))
    for ax, panel, (title, ylabel, boundary_ratio, values) in zip(axes, "abc", specs, strict=True):
        values = np.asarray(values)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("the log-scale figure requires finite positive errors for every model")
        for (marker, offset, size), y in zip(markers, values, strict=True):
            ax.scatter(x + offset, y, c=colors, marker=marker, s=size, linewidths=.8, zorder=3)
        for limit in sorted(set(values[:, 1] * boundary_ratio)):
            ax.axhline(limit, color="#6d7480", linestyle=(0, (3, 2)), linewidth=.7, zorder=2)
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5), numticks=12))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.4g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xticks(x, LABELS)
        ax.set_xlim(-.6, len(NAMES) - .4)
        ax.set_title(title, loc="left")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", which="major", color="#dce2e8", linewidth=.45)
        ax.set_axisbelow(True)
        ax.text(-.13, 1.025, panel, transform=ax.transAxes, fontsize=8, fontweight="bold")
    low = min(axes[0].get_ylim()[0], axes[1].get_ylim()[0])
    high = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
    for ax in axes[:2]:
        ax.set_ylim(low, high)
    handles = [
        Line2D([], [], color="#25384a", linestyle="none", marker="o", markersize=4, label="2.5 ms"),
        Line2D([], [], color="#25384a", linestyle="none", marker="x", markersize=4, label="1.25 ms"),
        Line2D([], [], color="#6d7480", linestyle="--", linewidth=.7, label="Limit versus V1"),
    ]
    fig.suptitle("Synthetic endpoint pulse gate", x=.09, y=.973, ha="left", fontsize=10)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.54, .07),
               ncol=3, frameon=False, handlelength=1.7)
    return fig


def save(fig, output, *, editable=False):
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(output.with_suffix(".png"), dpi=300)
    if editable:
        fig.savefig(output.with_suffix(".svg"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("diagnostics", type=Path)
    parser.add_argument("output", type=Path, help="Destination without an extension")
    parser.add_argument("--editable", action="store_true")
    parser.add_argument("--alignment-auditor", type=Path)
    parser.add_argument("--alignment-json", type=Path)
    args = parser.parse_args()
    fig = render(json.loads(args.diagnostics.read_bytes()))
    if args.alignment_auditor:
        auditor = runpy.run_path(str(args.alignment_auditor))
        require_matplotlib_panel_alignment = auditor["require_matplotlib_panel_alignment"]
        require_matplotlib_panel_alignment(fig, require_panel_labels=True, json_out=args.alignment_json)
    save(fig, args.output, editable=args.editable)


if __name__ == "__main__":
    main()
