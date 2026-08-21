"""Certified excitation-envelope routing with mandatory outside-union refusal."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ExcitationEnvelope:
    envelope_id: str
    model_id: str
    bounds: dict[str, tuple[float, float]]
    earned_name: str
    evidence_run_id: str

    def __post_init__(self) -> None:
        if not self.envelope_id or not self.model_id or not self.bounds:
            raise ValueError("an envelope needs identities and at least one feature bound")
        for feature, interval in self.bounds.items():
            if (
                not feature
                or len(interval) != 2
                or not np.isfinite(interval).all()
                or float(interval[0]) > float(interval[1])
            ):
                raise ValueError("envelope bounds must be finite ordered pairs")

    def contains(self, features: dict[str, float]) -> bool:
        if set(features) != set(self.bounds):
            return False
        return all(
            float(lower) <= float(features[name]) <= float(upper)
            for name, (lower, upper) in self.bounds.items()
        )

    @property
    def log_volume(self) -> float:
        widths = np.asarray(
            [
                max(float(upper) - float(lower), np.finfo(float).eps)
                for lower, upper in self.bounds.values()
            ]
        )
        return float(np.sum(np.log(widths)))

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EnvelopeRoute:
    status: str
    envelope_id: str | None
    model_id: str | None
    earned_name: str | None
    reason: str

    def as_dict(self) -> dict[str, str | None]:
        return asdict(self)


class ExcitationEnvelopeRouter:
    """Route causal-prefix features to the smallest covering certified envelope."""

    def __init__(self, envelopes: list[ExcitationEnvelope]) -> None:
        identities = [item.envelope_id for item in envelopes]
        if len(identities) != len(set(identities)):
            raise ValueError("envelope identities must be unique")
        feature_sets = {tuple(sorted(item.bounds)) for item in envelopes}
        if len(feature_sets) > 1:
            raise ValueError("all envelopes in one router must use the same causal features")
        self.envelopes = tuple(envelopes)

    def route(self, causal_prefix_features: dict[str, float]) -> EnvelopeRoute:
        if not self.envelopes:
            return EnvelopeRoute(
                status="REFUSE_AND_REQUEST_DATA",
                envelope_id=None,
                model_id=None,
                earned_name=None,
                reason="no certified excitation envelope is registered",
            )
        expected = set(self.envelopes[0].bounds)
        if set(causal_prefix_features) != expected or not all(
            np.isfinite(float(value)) for value in causal_prefix_features.values()
        ):
            return EnvelopeRoute(
                status="REFUSE_AND_REQUEST_DATA",
                envelope_id=None,
                model_id=None,
                earned_name=None,
                reason="causal prefix features are missing, extra, or non-finite",
            )
        covering = [item for item in self.envelopes if item.contains(causal_prefix_features)]
        if not covering:
            return EnvelopeRoute(
                status="REFUSE_AND_REQUEST_DATA",
                envelope_id=None,
                model_id=None,
                earned_name=None,
                reason="probe lies outside the union of certified excitation envelopes",
            )
        selected = min(covering, key=lambda item: (item.log_volume, item.envelope_id))
        return EnvelopeRoute(
            status="ROUTED_INSIDE_CERTIFIED_ENVELOPE",
            envelope_id=selected.envelope_id,
            model_id=selected.model_id,
            earned_name=selected.earned_name,
            reason="smallest covering certified envelope selected",
        )
