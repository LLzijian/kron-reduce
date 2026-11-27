"""Render the two registered V3 synthetic tracks and both RK4 step sizes."""

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
STEPS = (.0025, .00125)


def render(summary):
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .6, "pdf.fonttype": 42, "svg.fonttype": "none",
    })
    fig = Figure(figsize=(7.2, 5.5), facecolor="white")
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 3)
    fig.subplots_adjust(left=.095, right=.98, bottom=.14, top=.84, wspace=.40, hspace=.72)
    rows = {(row["track"], row["step_s"]): row for row in summary["tracks"]}
    for index, track in enumerate(("A", "B")):
        first, last = (lambda row, ep=ep: row["endpoint_target_peak_means"][ep] for ep in ("0", "10"))
        specs = (
            ("First input frame", "Peak error (rad/s)", first),
            ("Last input frame", "Peak error (rad/s)", last),
            ("8 s free rollout", "NRMSE", lambda row: {
                name: model["aggregate_nrmse"] for name, model in row["models"].items()
            }),
        )
        for col, (ax, (title, ylabel, values)) in enumerate(zip(axes[index], specs, strict=True)):
            for step, marker, offset in zip(STEPS, ("o", "x"), (-.14, .14), strict=True):
                measurements = values(rows[track, step])
                for position, name in enumerate(NAMES):
                    value = measurements[name]
                    if value is None:
                        continue
                    if not np.isfinite(value) or value <= 0:
                        raise ValueError(f"{track}/{step}/{name}: log scale requires finite positive scores")
                    color = {"orig": "#79818c", "v1": "#24384f", "F": "#be5c36"}.get(name, "#227d83")
                    ax.scatter(position + offset, value, color=color, marker=marker, s=18,
                               linewidths=.8, zorder=3)
            data = values(rows[track, STEPS[0]])
            if data["orig"] is not None and data["v1"] is not None:
                original_limit = (.9 if track == "A" else 1.) * data["orig"]
                ax.axhline(min(original_limit, data["v1"]), color="#6d7480",
                           linestyle=(0, (3, 2)), linewidth=.7, zorder=2)
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5), numticks=12))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.4g}"))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.set_xticks(range(len(NAMES)), LABELS)
            ax.set_xlim(-.6, len(NAMES) - .4)
            ax.set_title(title, loc="left")
            ax.set_ylabel(ylabel)
            ax.grid(axis="y", which="major", color="#dce2e8", linewidth=.45)
            ax.set_axisbelow(True)
            ax.text(-.14, 1.065, "abcdef"[3 * index + col], transform=ax.transAxes, fontweight="bold")
    for column in axes.T:
        low = min(ax.get_ylim()[0] for ax in column)
        high = max(ax.get_ylim()[1] for ax in column)
        for ax in column:
            ax.set_ylim(low, high)
    for index, heading in enumerate(("Track A   Matched model", "Track B   +8% stiffness and damping")):
        fig.text(.095, axes[index, 0].get_position().y1 + .075, heading, fontsize=9, fontweight="bold")
    handles = [
        Line2D([], [], color="#24384f", linestyle="none", marker="o", markersize=4, label="2.5 ms"),
        Line2D([], [], color="#24384f", linestyle="none", marker="x", markersize=4, label="1.25 ms"),
        Line2D([], [], color="#6d7480", linestyle="--", linewidth=.7, label="Comparison limit"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.53, .025),
               ncol=3, frameon=False, handlelength=1.7)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--editable", action="store_true")
    parser.add_argument("--alignment-auditor", type=Path)
    parser.add_argument("--alignment-json", type=Path)
    args = parser.parse_args()
    fig = render(json.loads(args.summary.read_bytes()))
    if args.alignment_auditor:
        auditor = runpy.run_path(str(args.alignment_auditor))
        require_matplotlib_panel_alignment = auditor["require_matplotlib_panel_alignment"]
        require_matplotlib_panel_alignment(fig, require_panel_labels=True, json_out=args.alignment_json)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(args.output.with_suffix(".png"), dpi=300)
    if args.editable:
        fig.savefig(args.output.with_suffix(".svg"))


if __name__ == "__main__":
    main()
