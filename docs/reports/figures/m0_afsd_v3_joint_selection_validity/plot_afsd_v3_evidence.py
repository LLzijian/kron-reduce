"""Generate the main evidence figure for the M0-AFSD v3 Chinese report.

The plotted values are immutable snapshots copied from the registered v2 and
v3 result artifacts.  The script checks the headline values before rendering
and exports editable/vector and high-resolution raster formats.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch


HERE = Path(__file__).resolve().parent
OUT_STEM = HERE / "fig1_afsd_v3_evidence"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.08, 1.05, label, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")


def rounded_box(ax: plt.Axes, xy: tuple[float, float], width: float, height: float,
                facecolor: str, title: str, body: str) -> None:
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        linewidth=0.8,
        edgecolor="#2F3B45",
        facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(xy[0] + width / 2, xy[1] + height * 0.67, title,
            ha="center", va="center", fontsize=8, fontweight="bold")
    ax.text(xy[0] + width / 2, xy[1] + height * 0.32, body,
            ha="center", va="center", fontsize=6.7, linespacing=1.25)


def panel_workflow(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    add_panel_label(ax, "a")
    ax.set_title("证据递进与失效保护", loc="left", pad=6, fontweight="bold")
    boxes = [
        (0.04, 0.73, "v1 探索性联合筛选", "空间 × 动态分枝\n发现候选规律", "#DDEBF7"),
        (0.04, 0.45, "v2 固定支撑消融", "七节点同支撑下\n耗散 + 记忆必要", "#DFF2E1"),
        (0.04, 0.17, "v3 选择有效性审计", "Q-A、故障线路外层 CV\n源 × 激励重叠门", "#FDE6D8"),
    ]
    for x, y, title, body, color in boxes:
        rounded_box(ax, (x, y), 0.72, 0.20, color, title, body)
    for y1, y2 in [(0.73, 0.63), (0.45, 0.35)]:
        ax.annotate("", xy=(0.40, y2 + 0.10), xytext=(0.40, y1),
                    arrowprops=dict(arrowstyle="-|>", lw=0.9, color="#52616B"))
    status = FancyBboxPatch((0.04, 0.01), 0.72, 0.10,
                            boxstyle="round,pad=0.01,rounding_size=0.02",
                            linewidth=1.0, edgecolor="#B23A2B", facecolor="#FFF3F0")
    ax.add_patch(status)
    ax.text(0.40, 0.06, "FAIL-CLOSED\n源—激励混杂",
            color="#B23A2B", ha="center", va="center", fontsize=6.0,
            linespacing=1.1, fontweight="bold")


def panel_overlap(ax: plt.Axes, overlap: pd.DataFrame) -> None:
    add_panel_label(ax, "b")
    ax.set_title("Q-A 独立故障线路簇的源—激励覆盖", loc="left", pad=6, fontweight="bold")
    column_order = [
        ("A-low", "R-low"),
        ("A-low", "R-high"),
        ("A-mid", "R-low"),
        ("A-mid", "R-high"),
        ("A-high", "R-low"),
    ]
    source_order = ["high-order", "dense"]
    matrix = np.zeros((2, len(column_order)), dtype=int)
    for i, source in enumerate(source_order):
        for j, (amp, rich) in enumerate(column_order):
            hit = overlap[(overlap.source == source) & (overlap.amplitude == amp) & (overlap.richness == rich)]
            matrix[i, j] = int(hit.n_fault_line_clusters.iloc[0]) if len(hit) else 0
    cmap = LinearSegmentedColormap.from_list("qa_counts", ["#F7F7F7", "#B9D8EA", "#2171B5"])
    ax.imshow(matrix, cmap=cmap, vmin=0, vmax=max(11, matrix.max()), aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            ax.text(j, i, str(value), ha="center", va="center",
                    color="white" if value >= 7 else "#1B2631", fontsize=8, fontweight="bold")
            if value < 5:
                ax.add_patch(plt.Rectangle((j - 0.48, i - 0.48), 0.96, 0.96,
                                           fill=False, edgecolor="#C43C39", lw=0.9))
    ax.add_patch(plt.Rectangle((-0.49, -0.49), 0.98, 1.98, fill=False,
                               edgecolor="#E69F00", lw=1.8))
    ax.set_xticks(range(len(column_order)), [f"{a}\n{r}" for a, r in column_order])
    ax.set_yticks(range(2), ["high-order", "dense"])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def panel_rollout(ax: plt.Axes, metrics: pd.DataFrame) -> None:
    add_panel_label(ax, "c")
    ax.set_title("v2 条件重积分：轨迹改善不等于模态通过", loc="left", pad=6, fontweight="bold")
    colors = {"B2": "#6B7280", "SS-DM": "#0072B2"}
    markers = {"B2": "o", "SS-DM": "s"}
    for model in ["B2", "SS-DM"]:
        part = metrics[metrics.model == model].sort_values("horizon_s")
        ax.plot(part.horizon_s, part.mean_rmse, marker=markers[model], color=colors[model],
                lw=1.6, ms=4.5, label=model)
    ax.set_xticks([1, 5, 10])
    ax.set_xlabel("重积分视界（s）")
    ax.set_ylabel("mean RMSE")
    ax.set_ylim(0, 0.34)
    ax.grid(axis="y", color="#D9D9D9", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    b2_10 = float(metrics[(metrics.model == "B2") & (metrics.horizon_s == 10)].mean_rmse.iloc[0])
    dm_10 = float(metrics[(metrics.model == "SS-DM") & (metrics.horizon_s == 10)].mean_rmse.iloc[0])
    improvement = (b2_10 - dm_10) / b2_10
    ax.annotate(f"10 s：−{improvement:.1%}", xy=(10, dm_10), xytext=(6.4, 0.09),
                arrowprops=dict(arrowstyle="->", lw=0.8, color="#0072B2"),
                color="#0072B2", fontsize=7, fontweight="bold")
    gate_text = (
        "10 s 绝对模态门（SS-DM）\n"
        "频率误差 26.5%  > 10%\n"
        "阻尼误差 0.068 > 0.05\n"
        "MAC 0.388          < 0.80"
    )
    ax.text(0.03, 0.05, gate_text, transform=ax.transAxes, fontsize=6.4, va="bottom",
            bbox=dict(boxstyle="round,pad=0.30", facecolor="#FFF4E6", edgecolor="#D98E04", lw=0.7))


def main() -> None:
    configure_style()
    overlap = pd.read_csv(HERE / "source_excitation_fault_lines.csv")
    metrics = pd.read_csv(HERE / "v2_conditional_rollout_metrics.csv")
    assert int(overlap.n_events.sum()) == 95
    # A fault line can populate more than one amplitude--richness cell, so cell
    # counts are not additive.  The registered v3 population contains 23 unique
    # fault-line clusters; here we instead verify the decisive overlap cell.
    decisive = overlap[(overlap.amplitude == "A-low") & (overlap.richness == "R-low")]
    assert decisive.set_index("source").n_fault_line_clusters.to_dict() == {
        "high-order": 11,
        "dense": 1,
    }
    b2 = float(metrics[(metrics.model == "B2") & (metrics.horizon_s == 10)].mean_rmse.iloc[0])
    dm = float(metrics[(metrics.model == "SS-DM") & (metrics.horizon_s == 10)].mean_rmse.iloc[0])
    assert np.isclose(b2, 0.2571116093909681)
    assert np.isclose(dm, 0.12507643928581663)

    width_in = 183 / 25.4
    fig = plt.figure(figsize=(width_in, width_in * 0.51), constrained_layout=False)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.02, 1.32, 1.28], wspace=0.38,
                            left=0.035, right=0.985, top=0.90, bottom=0.15)
    panel_workflow(fig.add_subplot(grid[0, 0]))
    panel_overlap(fig.add_subplot(grid[0, 1]), overlap)
    panel_rollout(fig.add_subplot(grid[0, 2]), metrics)

    fig.savefig(OUT_STEM.with_suffix(".svg"))
    fig.savefig(OUT_STEM.with_suffix(".pdf"))
    fig.savefig(OUT_STEM.with_suffix(".png"), dpi=600)
    fig.savefig(OUT_STEM.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


if __name__ == "__main__":
    main()
