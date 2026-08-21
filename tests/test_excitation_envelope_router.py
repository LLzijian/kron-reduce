from __future__ import annotations

from kron_reduce.selection import ExcitationEnvelope, ExcitationEnvelopeRouter


def _envelope(identifier: str, upper: float) -> ExcitationEnvelope:
    return ExcitationEnvelope(
        envelope_id=identifier,
        model_id=f"model-{identifier}",
        bounds={"prefix_amplitude": (0.0, upper), "prefix_entropy": (0.1, 0.5)},
        earned_name="rollout-accurate equivalent",
        evidence_run_id="synthetic-gate",
    )


def test_router_selects_smallest_covering_envelope() -> None:
    router = ExcitationEnvelopeRouter([_envelope("wide", 2.0), _envelope("narrow", 1.0)])
    route = router.route({"prefix_amplitude": 0.7, "prefix_entropy": 0.3})
    assert route.status == "ROUTED_INSIDE_CERTIFIED_ENVELOPE"
    assert route.envelope_id == "narrow"


def test_router_refuses_outside_union_and_malformed_prefix() -> None:
    router = ExcitationEnvelopeRouter([_envelope("known", 1.0)])
    outside = router.route({"prefix_amplitude": 1.5, "prefix_entropy": 0.3})
    assert outside.status == "REFUSE_AND_REQUEST_DATA"
    missing = router.route({"prefix_amplitude": 0.5})
    assert missing.status == "REFUSE_AND_REQUEST_DATA"


def test_router_rejects_incompatible_feature_schemas() -> None:
    first = _envelope("first", 1.0)
    second = ExcitationEnvelope(
        envelope_id="second",
        model_id="model-second",
        bounds={"different_feature": (0.0, 1.0)},
        earned_name="rollout-accurate equivalent",
        evidence_run_id="synthetic-gate",
    )
    try:
        ExcitationEnvelopeRouter([first, second])
    except ValueError as error:
        assert "same causal features" in str(error)
    else:
        raise AssertionError("incompatible envelope schemas must be rejected")
