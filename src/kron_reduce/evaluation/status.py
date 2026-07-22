from __future__ import annotations

import math


def evaluation_status(required_population: int, valid_predictions: int, required_metrics: dict[str, float]) -> str:
    if required_population <= 0:
        return "ineligible_by_observation_window"
    if valid_predictions <= 0:
        return "attempted_failed_with_evidence"
    if any(not math.isfinite(float(value)) for value in required_metrics.values()):
        return "attempted_failed_with_evidence"
    return "completed"
