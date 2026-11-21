"""Inertia-weighted slow coherency and forecast-origin mechanical and memory storage."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np

from kron_reduce.evaluation.postfault_m4 import STEPS, _auc
from kron_reduce.evaluation.postfault_m4_graph import _compare
from kron_reduce.physics.graph import complete_incidence

_GRAPHS = ("a06", "hard_r", "m3")
_MODELS = ("hard_r", "soft_2r", "m3")
_COMPONENTS = ("kinetic", "potential", "mechanical", "memory", "total")


def _spectral_cut(laplacian, mass, reference, left, right):
    inv_root = 1./np.sqrt(mass)
    symmetric = inv_root[:, None]*laplacian*inv_root[None, :]
    values, modes = np.linalg.eigh(symmetric)
    scale = float(values[-1])
    if not (scale > 0 and abs(values[0]) <= 1e-10*scale and values[1] > 1e-10*scale):
        raise ValueError("effective graph must have one rigid mode and positive quotient stiffness")
    physical = inv_root[:, None]*modes
    # Give the largest physical component a positive sign to make all exported vectors reproducible.
    pivot = np.argmax(np.abs(physical), axis=0)
    physical *= np.sign(physical[pivot, np.arange(len(mass))])
    gram_error = float(np.linalg.norm(physical.T @ (mass[:, None]*physical)-np.eye(len(mass)), ord=np.inf))
    residual = laplacian @ physical-mass[:, None]*physical*values
    relative_residual = float(np.linalg.norm(residual, ord="fro")/(
        np.linalg.norm(laplacian, 2)*np.linalg.norm(physical, ord="fro")
        +np.linalg.norm(mass[:, None]*physical*values, ord="fro")))
    if gram_error > 1e-10 or relative_residual > 1e-10:
        raise FloatingPointError("generalized eigensystem failed its orthogonality or residual check")
    u2 = physical[:, 1]
    eigenspace_dimension = int(np.count_nonzero(abs(values-values[1]) <= 1e-10*scale))
    ambiguous = np.flatnonzero(abs(u2) <= 1e-10*max(abs(u2)))
    identifiable = eigenspace_dimension == 1 and not len(ambiguous)
    groups = sorted([np.flatnonzero(u2 < 0).tolist(), np.flatnonzero(u2 > 0).tolist()]) if identifiable else None
    return {"eigenvalues": values.tolist(), "physical_eigenvectors_by_column": physical.tolist(), "u2": u2.tolist(),
            "lambda3_over_lambda2": float(values[2]/values[1]), "lambda3_minus_lambda2": float(values[2]-values[1]),
            "eigenspace_dimension": eigenspace_dimension, "near_zero_u2_machine_indices": ambiguous.tolist(),
            "identifiable_sign_cut": identifiable, "groups": groups,
            "reference_comparison": _compare(groups, reference, left, right) if identifiable else None,
            "mass_orthogonality_inf_error": gram_error, "relative_residual_frobenius": relative_residual}


def slow_coherency(graph_result, mass):
    """Compare the cuts of L and L u = λ M u to the single frozen late-speed reference partition."""
    mass = np.asarray(mass, dtype=np.float64)
    if mass.shape != (10,) or not np.isfinite(mass).all() or np.any(mass <= 0):
        raise ValueError("slow coherency requires ten positive finite inertias")
    incidence, edges = complete_incidence(10)
    left, right = np.asarray(edges).T
    rows = graph_result["edge_rows"]
    if (len(rows) != len(edges) or any(row["edge_index"] != i or list(pair) != row["machine_indices"]
                                      for i, (row, pair) in enumerate(zip(rows, edges, strict=True)))):
        raise ValueError("effective graph must use the canonical 45 undirected ten-machine edges")
    empirical = graph_result["stable_val"]
    if (empirical["role"] != "val" or empirical["truth_label"] != "stable" or empirical["spectrum"] is None
            or not 0 < empirical["events_used"] <= empirical["events_allocated"]):
        raise ValueError("a measured stable val reference with explicit tail coverage is required")
    reference = empirical["spectrum"]["groups"]
    if len(reference) != 2 or not all(reference) or sorted(reference[0]+reference[1]) != list(range(10)):
        raise ValueError("measured coherency reference must partition all ten machines")
    result = {"reference": {"groups": reference, "events_used": empirical["events_used"],
                            "events_allocated": empirical["events_allocated"],
                            "forecast_tail_s": empirical["forecast_tail_s"]},
              "mass": mass.tolist(), "machine_index_origin": 0, "graphs": {},
              "graph_operator": "B diag(k) B^T on the complete effective ten-machine graph",
              "generalized_problem": "L u = lambda M u; y = M^(1/2) u; u^T M u = 1"}
    for name in _GRAPHS:
        weights = np.asarray([row["stiffness"][name] for row in rows])
        if np.any(weights <= 0.) or not np.isfinite(weights).all():
            raise ValueError("edge stiffness must be positive and finite")
        laplacian = (incidence*weights) @ incidence.T
        standard = _spectral_cut(laplacian, np.ones_like(mass), reference, left, right)
        inertial = _spectral_cut(laplacian, mass, reference, left, right)
        archived = graph_result["graphs"][name]
        if (not np.allclose(laplacian, archived["laplacian"], rtol=1e-12, atol=1e-12)
                or not np.allclose(standard["eigenvalues"], archived["eigenvalues"], rtol=1e-12, atol=1e-12)
                or (standard["identifiable_sign_cut"]
                    and not _compare(standard["groups"], archived["groups"], left, right)["same_unordered_cut"])):
            raise ValueError("reconstructed ordinary effective graph differs from frozen M4 evidence")
        result["graphs"][name] = {"laplacian": laplacian.tolist(), "ordinary": standard, "inertia_weighted": inertial}
    return result


def _group_stats(values, unstable):
    stable_q = np.quantile(values[~unstable], [.25, .5, .75])
    unstable_q = np.quantile(values[unstable], [.25, .5, .75])
    return {"auroc": _auc(unstable, values), "stable_q25_median_q75": stable_q.tolist(),
            "unstable_q25_median_q75": unstable_q.tolist(),
            "unstable_to_stable_median_ratio": float(unstable_q[1]/stable_q[1]) if stable_q[1] != 0. else None}


def separate_storage(profiles):
    """Compare K, U, K+U, memory and total storage on the same labeled events for all models."""
    if set(profiles) != {"train", "val"}:
        raise ValueError("stored energies must cover train and val")
    output = {"splits": {}, "time_origin": "forecast origin after the observed 100 ms",
              "positive_class": "unstable", "median_ratio": "unstable median divided by stable median",
              "auc": "event-equal positive versus negative ranking; ties count one half"}
    seen = set()
    for role in ("train", "val"):
        if set(profiles[role]) != set(STEPS):
            raise ValueError("both frozen scoring step sizes are required")
        output["splits"][role] = {}
        ids_labels = None
        for step in STEPS:
            profiles_at_step = profiles[role][step]
            if set(profiles_at_step) != set(_MODELS):
                raise ValueError("expected hard_r, soft_2r and m3 matched energies")
            models, mechanical, hidden, total = {}, {}, {}, {}
            for name in _MODELS:
                rows = profiles_at_step[name]["rows"]
                keys = [(row["id"], row["label"]) for row in rows]
                if ids_labels is None:
                    ids_labels = keys
                    ids = {key for key, _ in keys}
                    if len(ids) != len(keys) or ids & seen or {label for _, label in keys} != {"stable", "unstable"}:
                        raise ValueError("expected unique, disjoint event IDs and both truth classes")
                    seen.update(ids)
                if keys != ids_labels:
                    raise ValueError("step/model energy rows differ in event order or labels")
                unstable = np.array([label == "unstable" for _, label in keys])
                components = {k: np.asarray([row[k] for row in rows], dtype=np.float64)
                              for k in ("kinetic", "potential", "memory", "total")}
                components["mechanical"] = components["kinetic"]+components["potential"]
                if (not all(np.isfinite(values).all() for values in components.values())
                        or np.any(components["kinetic"] < 0) or np.any(components["memory"] < 0)
                        or not np.allclose(components["total"], components["mechanical"]+components["memory"],
                                           rtol=2e-12, atol=1e-12)):
                    raise ValueError("invalid storage or inconsistent total = kinetic + potential + memory")
                stats = {key: _group_stats(components[key], unstable) for key in _COMPONENTS}
                mechanical[name], hidden[name], total[name] = (components[key] for key in ("mechanical", "memory", "total"))
                models[name] = {"n_stable": int((~unstable).sum()), "n_unstable": int(unstable.sum()),
                    "components": stats, "total_minus_mechanical_auroc": stats["total"]["auroc"]-stats["mechanical"]["auroc"],
                    "rows": [row | {"mechanical": float(value)} for row, value in zip(rows, components["mechanical"], strict=True)]}
            # Compare like-for-like event energies. Subtracting AUROCs is descriptive, not an additive attribution.
            delta_mechanical = mechanical["soft_2r"]-mechanical["hard_r"]
            delta_memory = hidden["soft_2r"]-hidden["hard_r"]
            delta_total = total["soft_2r"]-total["hard_r"]
            control = {"same_mechanical_each_event": bool(np.allclose(mechanical["soft_2r"], mechanical["hard_r"],
                                                                      rtol=2e-12, atol=1e-12)),
                       "max_abs_mechanical_difference": float(np.max(np.abs(delta_mechanical))),
                       "max_abs_total_minus_memory_difference": float(np.max(np.abs(delta_total-delta_memory))),
                       "soft_minus_hard_total_auroc": models["soft_2r"]["components"]["total"]["auroc"]
                                                     -models["hard_r"]["components"]["total"]["auroc"]}
            output["splits"][role][step] = {"models": models, "paired_v4": control}
    return output


def _figure_font(size):
    from matplotlib.font_manager import FontProperties, fontManager

    font_dir = Path("C:/Windows/Fonts")
    for regular, bold in (("ARIALN.TTF", "ARIALNB.TTF"), ("arial.ttf", "arialbd.ttf")):
        if (font_dir/regular).is_file() and (font_dir/bold).is_file():
            for file in (regular, bold):
                fontManager.addfont(font_dir/file)
            name = FontProperties(fname=font_dir/regular).get_name()
            return FontProperties(family=[name, "Microsoft YaHei", "DejaVu Sans"], size=size, weight="bold")
    return FontProperties(family=["DejaVu Sans"], size=size, weight="bold")


def _format_axis(ax, tick_font, *, heatmap=False):
    ax.spines[:].set_visible(True)
    ax.spines[:].set_color("black")
    ax.tick_params(direction="out" if heatmap else "in", which="both", top=False, right=False)
    for label in ax.get_xticklabels()+ax.get_yticklabels():
        label.set_fontproperties(tick_font)
    ax.grid(False)


def _save_figure(fig, prefix):
    import matplotlib as mpl

    prefix = Path(prefix)
    png, svg = prefix.with_suffix(".png"), prefix.with_suffix(".svg")
    with mpl.rc_context({"svg.fonttype": "none", "svg.hashsalt": "postfault_m4_v2", "axes.unicode_minus": False}):
        with png.open("xb") as stream:
            fig.savefig(stream, format="png", dpi=600)
        buffer = BytesIO()
        fig.savefig(buffer, format="svg", metadata={"Date": None})
        lines = buffer.getvalue().decode("utf-8").splitlines()
        with svg.open("xb") as stream:
            stream.write(("\n".join(line.rstrip() for line in lines)+"\n").encode("utf-8"))
    return png, svg


def plot_slow_coherency(result, prefix):
    """Show which machines share G10's cut side and the scale-free spectral separation."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import ListedColormap
    from matplotlib.figure import Figure

    font, small = _figure_font(10), _figure_font(8)
    fig = Figure(figsize=(7.2, 3.2), layout="constrained")
    FigureCanvasAgg(fig)
    partition, gap = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.7, 1.]})
    labels = {"a06": "A06", "hard_r": "Hard R", "m3": "M3"}
    groups = [result["reference"]["groups"]]
    row_names = ["Measured 6–8 s"]
    for method, title in (("ordinary", "ordinary"), ("inertia_weighted", "inertia")):
        for name in _GRAPHS:
            groups.append(result["graphs"][name][method]["groups"])
            row_names.append(f"{labels[name]} · {title}")
    membership = np.full((len(groups), 10), .5)
    for i, cut in enumerate(groups):
        if cut is not None:
            membership[i] = np.isin(np.arange(10), next(group for group in cut if 9 in group))
    partition.pcolormesh(np.arange(11)-.5, np.arange(len(groups)+1)-.5, membership,
                         cmap=ListedColormap(["#f4f5f5", "#b6b7b8", "#216a91"]), vmin=0., vmax=1.,
                         shading="flat", rasterized=False)
    partition.invert_yaxis()
    for i, cut in enumerate(groups):
        if cut is None:
            partition.text(4.5, i, "Ambiguous", ha="center", va="center", fontproperties=small)
    partition.set_xticks(np.arange(10), [f"G{i}" for i in range(1, 11)])
    partition.set_yticks(np.arange(len(row_names)), row_names)
    partition.set_title("(a) Machines grouped with G10", fontproperties=font)
    _format_axis(partition, small, heatmap=True)

    for method, label, offset, color, marker in (
            ("ordinary", "Ordinary", -.09, "#686c70", "o"),
            ("inertia_weighted", "Inertia-weighted", .09, "#216a91", "D")):
        ratios = [result["graphs"][name][method]["lambda3_over_lambda2"] for name in _GRAPHS]
        gap.scatter(np.arange(3)+offset, ratios, c=color, marker=marker, s=32, label=label, zorder=3)
    gap.set_xticks(np.arange(3), list(labels.values()))
    gap.set_xlim(-.45, 2.45)
    low, high = gap.get_ylim()
    gap.set_ylim(min(.96, low), high+.28*(high-min(.96, low)))
    gap.axhline(1., color="#888888", lw=.7, ls=":")
    gap.set_ylabel("Eigenvalue ratio λ3 / λ2", fontproperties=font)
    gap.set_title("(b) Slow-mode separation", fontproperties=font)
    gap.legend(loc="upper right", frameon=False, prop=small)
    _format_axis(gap, small)
    return _save_figure(fig, prefix)


def plot_energy_decomposition(result, prefix):
    """Show the event ranking and class-median contrast for both registered clock steps."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    font, small = _figure_font(10), _figure_font(8)
    fig = Figure(figsize=(7.2, 5.2), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 2, sharex="col")
    model_style = (("hard_r", "Hard R", "#4d5459", "s"), ("soft_2r", "Soft 2R", "#b36128", "^"),
                   ("m3", "M3", "#216a91", "o"))
    for col, role in enumerate(("train", "val")):
        auc_axis, ratio_axis = axes[:, col]
        steps = result["splits"][role]
        counts = steps[STEPS[0]]["models"][_MODELS[0]]
        auc_axis.set_title(f"({chr(97+col)}) {role} · {counts['n_stable']} stable, {counts['n_unstable']} unstable",
                           fontproperties=font)
        ratio_axis.set_title(f"({chr(99+col)}) {role} · median contrast", fontproperties=font)
        ratio_values = []
        for i, (name, label, color, marker) in enumerate(model_style):
            for j, step in enumerate(STEPS):
                stats = steps[step]["models"][name]["components"]
                x = np.arange(len(_COMPONENTS))+.22*(i-1)+.075*(j-.5)
                auc = [stats[k]["auroc"] for k in _COMPONENTS]
                ratios = [stats[k]["unstable_to_stable_median_ratio"] for k in _COMPONENTS]
                ratio_values.extend(v for v in ratios if v is not None)
                style = {"marker": marker, "s": 22, "edgecolors": color, "linewidths": .9,
                         "facecolors": color if j == 0 else "white", "zorder": 3}
                auc_axis.scatter(x, auc, label=label if j == 0 else None, **style)
                ratio_axis.scatter(x, [np.nan if v is None else v for v in ratios], **style)
        ratio_axis.axhline(1., color="#888888", lw=.7, ls=":")
        auc_axis.axhline(.5, color="#888888", lw=.7, ls=":")
        auc_axis.set_ylim(-.02, 1.04)
        auc_axis.legend(loc="lower right", prop=small, frameon=False)
        auc_axis.set_ylabel("AUROC", fontproperties=font)
        ratio_axis.set_ylabel("Unstable / stable median", fontproperties=font)
        ratio_axis.set_xticks(np.arange(5), ["Kinetic", "Potential", "Mechanical", "Memory", "Total"],
                              rotation=22, ha="right")
        if ratio_values and min(ratio_values) > 0. and max(ratio_values)/min(ratio_values) > 6.:
            ratio_axis.set_yscale("log")
        for ax in (auc_axis, ratio_axis):
            ax.set_xlim(-.5, 4.5)
            _format_axis(ax, small)
    fig.supxlabel("Filled marks   2.5 ms      Open marks   1.25 ms", fontproperties=small)
    return _save_figure(fig, prefix)
