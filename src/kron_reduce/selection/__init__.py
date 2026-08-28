"""Leakage-free model-selection and deployment routing primitives."""

from .clusterability import (
    ClusterAuditResult,
    ClusterDecision,
    ClusterThresholds,
    audit_cluster_decision,
    data_level_group_split,
    decide_clustering,
)
from .envelopes import EnvelopeRoute, ExcitationEnvelope, ExcitationEnvelopeRouter

__all__ = [
    "ClusterAuditResult",
    "ClusterDecision",
    "ClusterThresholds",
    "EnvelopeRoute",
    "ExcitationEnvelope",
    "ExcitationEnvelopeRouter",
    "audit_cluster_decision",
    "data_level_group_split",
    "decide_clustering",
]
