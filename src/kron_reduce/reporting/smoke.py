from __future__ import annotations

from pathlib import Path


def write_smoke_report(path: Path, run_id: str, summaries: list[dict], energy_gate: dict) -> None:
    rows = "\n".join(
        f"| {row['method']} | {row['status']} | {row.get('rmse', float('nan')):.6g} | {row.get('reported_name', row['method'])} |"
        for row in summaries
    )
    path.write_text(
        f"""---
title: "kron-reduce model smoke"
evidence_status: smoke
run_id: "{run_id}"
---

# Model smoke report

This artifact verifies real train, checkpoint, load, and closed-loop inference
paths. It is not paper evidence and must not be cited as a full experiment.

| Method | Status | RMSE | Reported name |
|---|---|---:|---|
{rows}

Energy gate: `{energy_gate['reported_name']}`. A Lyapunov name is permitted only
when `certificate_passed=true`; this smoke does not promote any manuscript claim.
""",
        encoding="utf-8",
    )
