from __future__ import annotations

import numpy as np


def fault_line_cluster_bootstrap(
    rows: list[dict], metric: str, *, n_bootstrap: int = 1000, seed: int = 324
) -> dict[str, float | int | str]:
    pairs = [row for row in rows if np.isfinite(float(row.get(metric, np.nan)))]
    lines = sorted({row["fault_line"] for row in pairs})
    if len(lines) < 2:
        return {"status": "insufficient_support", "n_pairs": len(pairs), "n_fault_line_clusters": len(lines)}
    by_line = {line: [float(row[metric]) for row in pairs if row["fault_line"] == line] for line in lines}
    rng = np.random.default_rng(seed)
    observed = float(np.mean([value for values in by_line.values() for value in values]))
    draws = []
    for _ in range(n_bootstrap):
        selected = rng.choice(lines, size=len(lines), replace=True)
        draws.append(np.mean([value for line in selected for value in by_line[line]]))
    low, high = np.quantile(draws, [0.025, 0.975])
    return {"status": "completed", "n_pairs": len(pairs), "n_fault_line_clusters": len(lines), "paired_effect": observed, "ci95_low": float(low), "ci95_high": float(high), "bootstrap_seed": seed}
