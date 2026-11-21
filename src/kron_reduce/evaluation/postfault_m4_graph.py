"""Compare frozen ten-machine effective graphs to stable postfault speed coherency."""

from io import BytesIO
from pathlib import Path

import numpy as np

from kron_reduce.physics.coordinates import coi_project
from kron_reduce.physics.graph import complete_incidence

N_MACHINES = 10
G10_INDEX = 9
STABLE_TAIL = slice(610, 811)  # Forecast seconds 6–8 on the stored 10 ms clock, including both endpoints.


def _array(tensor):
    return tensor.detach().cpu().numpy()


def _partition(incidence, weights, left, right):
    laplacian = (incidence * weights) @ incidence.T
    values, vectors = np.linalg.eigh(laplacian)
    scale = float(values[-1])
    near_fiedler = np.abs(values[1:] - values[1]) <= 1e-10 * scale
    eigenspace = vectors[:, 1:][:, near_fiedler]
    if eigenspace.shape[1] == 1:
        fiedler = eigenspace[:, 0].copy()
    else:
        # Use the first nonzero projected coordinate to resolve a repeated eigenvalue.
        projector = eigenspace @ eigenspace.T
        pivot = int(np.flatnonzero(np.diag(projector) > 1e-12)[0])
        fiedler = projector[:, pivot] / np.sqrt(projector[pivot, pivot])
    fiedler *= np.sign(fiedler[np.argmax(np.abs(fiedler))])
    positive = fiedler > 1e-12
    groups = sorted([np.flatnonzero(group).tolist() for group in (positive, ~positive)])
    crossing = positive[left] != positive[right]
    return {
        "laplacian": laplacian.tolist(),
        "eigenvalues": values.tolist(),
        "fiedler_vector": fiedler.tolist(),
        "fiedler_gap": float(values[2] - values[1]),
        "fiedler_relative_gap": float((values[2] - values[1]) / scale),
        "fiedler_eigenspace_dimension": int(eigenspace.shape[1]),
        "groups": groups,
        "crossing_edges": np.flatnonzero(crossing).tolist(),
        "crossing_weight_fraction": float(weights[crossing].sum() / weights.sum()),
    }


def _compare(groups, reference, left, right):
    first, second = (np.isin(np.arange(N_MACHINES), cut[0]) for cut in (groups, reference))
    a, b = first[left] == first[right], second[left] == second[right]
    total = len(a)
    matched = int(np.count_nonzero(a == b))
    count_a, count_b = int(a.sum()), int(b.sum())
    expected = count_a * count_b / total
    maximum = (count_a + count_b) / 2.
    ari = (int((a & b).sum()) - expected) / (maximum - expected) if maximum > expected else 1.
    return {"matched_pairs": matched, "total_pairs": total, "pair_agreement": matched / total,
            "adjusted_rand_index": ari, "same_unordered_cut": matched == total}


def _stable_coherency(events, mass, incidence, left, right):
    profiles, tail_rms, omitted = [], {}, []
    for event in sorted(events, key=lambda row: row["id"]):
        event_id, x = event["id"], np.asarray(event["x"])
        if (event.get("role") != "val" or event.get("label") != "stable" or event_id in tail_rms
                or x.shape != (STABLE_TAIL.stop, 2 * N_MACHINES)):
            raise ValueError("coherency requires distinct stable val events with complete 8 s observations")
        tail = coi_project(x[STABLE_TAIL, N_MACHINES:], mass)
        if not np.isfinite(tail).all():
            raise ValueError(f"nonfinite stable val tail in {event_id}")
        centered = tail - tail.mean(axis=0)
        rms = float(np.sqrt(np.mean(centered**2)))
        tail_rms[event_id] = rms
        if rms <= 1e-12:
            omitted.append(event_id)
            continue
        difference = (centered[:, left] - centered[:, right]) / rms
        profiles.append(np.mean(difference**2, axis=0))
    distance = np.mean(profiles, axis=0) if profiles else None
    affinity = np.exp(-.5 * distance) if distance is not None else None
    return {
        "role": "val",
        "truth_label": "stable",
        "forecast_tail_s": [6., 8.],
        "tail_samples": STABLE_TAIL.stop - STABLE_TAIL.start,
        "method": "Remove each COI speed temporal mean. Average pairwise mean squared speed differences "
                  "divided by event-wide mean square with equal event weights. Use exp(-distance/2) affinity.",
        "events_allocated": len(tail_rms),
        "events_used": len(profiles),
        "tail_centered_rms_rad_s": tail_rms,
        "rms_floor_rad_s": 1e-12,
        "uninformative_events": omitted,
        "distance_by_edge": distance.tolist() if distance is not None else None,
        "affinity_by_edge": affinity.tolist() if affinity is not None else None,
        "spectrum": _partition(incidence, affinity, left, right) if affinity is not None else None,
    }


def evaluate_graph(models, stable_events):
    """Compute two graph cuts and their comparisons on the stable val cohort."""
    hard_r, m3 = models["hard_r"], models["m3"]
    incidence, edges = complete_incidence(N_MACHINES)
    left, right = np.asarray(edges).T
    a06 = _array(m3.stiffness)
    mass = _array(m3.mass)
    if (not all(np.array_equal(_array(model.incidence), incidence) for model in (hard_r, m3))
            or not np.array_equal(a06, _array(hard_r.stiffness))
            or not np.array_equal(mass, _array(hard_r.mass))):
        raise ValueError("the frozen models must share the A06 edge and machine ordering")
    weights = {"a06": a06, "hard_r": _array(hard_r.effective_stiffness), "m3": _array(m3.effective_stiffness)}
    if any(k.shape != (len(edges),) or not np.isfinite(k).all() or np.any(k <= 0.) for k in weights.values()):
        raise ValueError("each graph requires 45 finite positive edge stiffnesses")
    relative = {"m3_vs_a06": weights["m3"] / a06 - 1.,
                "m3_vs_hard_r": weights["m3"] / weights["hard_r"] - 1.,
                "hard_r_vs_a06": weights["hard_r"] / a06 - 1.}
    rows = [
        {"edge_index": i, "machine_indices": list(pair), "touches_g10": G10_INDEX in pair,
         "stiffness": {key: float(k[i]) for key, k in weights.items()},
         "relative_change": {key: float(delta[i]) for key, delta in relative.items()}}
        for i, pair in enumerate(edges)
    ]
    by_a06 = sorted(rows, key=lambda row: (-abs(row["relative_change"]["m3_vs_a06"]), row["edge_index"]))
    top_10 = by_a06[:10]
    graphs = {key: _partition(incidence, k, left, right) for key, k in weights.items()}
    observed = _stable_coherency(stable_events, mass, incidence, left, right)
    classic = [list(range(G10_INDEX)), [G10_INDEX]]
    obs_groups = observed["spectrum"]["groups"] if observed["spectrum"] is not None else None
    comparisons = {}
    for key, graph in graphs.items():
        cut = graph["groups"]
        comparisons[key] = {
            "stable_val": _compare(cut, obs_groups, left, right) if obs_groups is not None else None,
            "g10_vs_remaining": _compare(cut, classic, left, right),
        }
    return {
        "graph_type": "ten-machine effective static coupling",
        "laplacian_formula": "B diag(k) B.T",
        "g10_machine_index": G10_INDEX,
        "edge_rows": rows,
        "top_10_by_absolute_change_from_a06": top_10,
        "g10_incident_edges_in_top_10": sum(row["touches_g10"] for row in top_10),
        "hard_r_vs_a06_max_abs_relative_change": float(np.max(np.abs(relative["hard_r_vs_a06"]))),
        "graphs": graphs,
        "stable_val": observed,
        "g10_reference": {
            "groups": classic,
            "stable_val": _compare(classic, obs_groups, left, right) if obs_groups is not None else None,
        },
        "comparisons": comparisons,
    }


def plot_topology(graph_result, destination):
    """Save the effective-coupling schematic and signed changes as PNG and editable SVG."""
    import matplotlib as mpl
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.font_manager import FontProperties, fontManager
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    windows_fonts = Path("C:/Windows/Fonts")
    family = "DejaVu Sans"
    for filenames in (("ARIALN.TTF", "ARIALNB.TTF"), ("arial.ttf", "arialbd.ttf")):
        regular, bold = (windows_fonts / name for name in filenames)
        if regular.is_file():
            for file in (regular, bold):
                if file.is_file():
                    fontManager.addfont(file)
            family = FontProperties(fname=regular).get_name()
            break

    def font(size):
        return FontProperties(family=[family, "Microsoft YaHei", "DejaVu Sans"], size=size, weight="bold")

    def edge_color(change):
        return "#006A9C" if change > 0. else "#AF4817" if change < 0. else "#666666"

    rows = graph_result["top_10_by_absolute_change_from_a06"]
    primary = np.asarray([100. * row["relative_change"]["m3_vs_a06"] for row in rows])
    other = np.asarray([100. * row["relative_change"]["m3_vs_hard_r"] for row in rows])
    colors = [edge_color(change) for change in primary]
    height = np.arange(len(rows))
    g10 = graph_result["g10_machine_index"]
    angle = np.pi / 2. - 2. * np.pi * np.arange(N_MACHINES) / N_MACHINES
    points = np.column_stack((np.cos(angle), np.sin(angle)))
    destination = Path(destination)
    png, svg = destination.with_suffix(".png"), destination.with_suffix(".svg")

    opts = {"font.family": [family], "font.sans-serif": [family, "Microsoft YaHei", "DejaVu Sans"],
            "axes.unicode_minus": False, "axes.grid": False, "svg.fonttype": "none",
            "svg.hashsalt": "postfault_m4_topology"}
    with mpl.rc_context(opts):
        fig = Figure(figsize=(7.2, 3.55), layout="constrained")
        FigureCanvasAgg(fig)
        network, changes = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1., 1.15]})
        for ax in (network, changes):
            ax.spines[:].set_visible(True)
            ax.spines[:].set_edgecolor("black")
            ax.tick_params(direction="in", which="both", top=False, right=False)
            ax.grid(False)

        span = max(float(np.max(np.abs(primary))), 1e-12)
        for i in reversed(range(len(rows))):
            segment = points[rows[i]["machine_indices"]]
            network.plot(*segment.T, lw=.55 + 2.45 * abs(primary[i]) / span, color=colors[i],
                         solid_capstyle="round", linestyle="--" if primary[i] < 0. else "-")
        network.scatter(*points.T, s=160, facecolors="white", edgecolors="#303030", zorder=3)
        network.scatter(*points[g10], s=230, facecolors="white", edgecolors="#202020", linewidths=1.6, zorder=4)
        for i, (x, y) in enumerate(points):
            network.text(x, y, f"G{i + 1}", ha="center", va="center", fontproperties=font(8), zorder=5)
        network.set(xlim=(-1.2, 1.2), ylim=(-1.20, 1.2), aspect="equal", xticks=[], yticks=[])
        network.set_title("(a) Ten-machine effective coupling", fontproperties=font(10))

        changes.barh(height, primary, height=.55, color=colors)
        overlaps = graph_result["hard_r_vs_a06_max_abs_relative_change"] <= 1e-10
        handles = [Patch(facecolor="#666666", label="vs A06 (Hard-R overlaps)" if overlaps else "vs A06")]
        if not overlaps:
            changes.scatter(other, height, marker="o", facecolors="white", edgecolors="#202020", s=19, zorder=3)
            handles.append(Line2D([], [], marker="o", ls="", markersize=4,
                                  markerfacecolor="white", markeredgecolor="#202020", label="vs Hard-R"))
        changes.axvline(0., color="#303030", lw=.65, zorder=1)
        changes.set_yticks(height, ["–".join(f"G{j + 1}" for j in row["machine_indices"]) for row in rows])
        changes.set_ylim(len(rows) - .45, -1.8)
        changes.margins(x=.11)
        changes.set_xlabel("M3 relative stiffness change (%)", fontproperties=font(9))
        changes.set_title("(b) Top 10 by absolute change from A06", fontproperties=font(10))
        for label in changes.get_xticklabels() + changes.get_yticklabels():
            label.set_fontproperties(font(8))
        changes.legend(handles=handles, loc="upper right", frameon=False, prop=font(8))

        with png.open("xb") as stream:
            fig.savefig(stream, format="png", dpi=600)
        buffer = BytesIO()
        fig.savefig(buffer, format="svg", metadata={"Date": None})
        lines = buffer.getvalue().decode("utf-8").splitlines()
        with svg.open("xb") as stream:
            stream.write(("\n".join(line.rstrip() for line in lines) + "\n").encode("utf-8"))
    return png, svg
