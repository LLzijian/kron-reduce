from __future__ import annotations


def prospective_eligibility(metadata: dict, decision_time_s: float, has_real_prefix: bool) -> tuple[bool, str]:
    if not has_real_prefix:
        return False, "ineligible_short_prefix"
    event_time = metadata.get("event_time_s")
    if metadata.get("event_observed") and event_time is not None and float(event_time) <= decision_time_s:
        return False, "event_observed_during_input"
    return True, "prospective_risk_set"
