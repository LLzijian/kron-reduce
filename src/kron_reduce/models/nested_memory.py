"""Deterministic, train-input-only r2/r4 starts for the bounded A08 comparison.

This helper performs matrix algebra only. The caller owns frozen train IDs,
source checkpoint hashes and causal feature construction. It neither fits nor
rolls out a model and it never reads trajectory or label files.
"""

from __future__ import annotations

import numpy as np

from kron_reduce.models.nonlinear_latent import (
    NonlinearPassiveClosure,
    _array_sha256,
    _validated_full_start,
)

FULL_ARRAY_NAMES = (
    "mass", "edge_stiffness", "damping", "coupling", "latent_skew",
    "latent_dissipation", "encoder", "feature_scale", "target_scale",
)
COEFFICIENT_RATIO = 0.05
RANK_RELATIVE_TOLERANCE = 1e-10
NEW_LATENT_DISSIPATION = float(np.logaddexp(0.0, -0.5) + 1e-5)
BASE_ENCODER_PENALTY = 1e-4
ENCODER_PENALTY_DENOMINATOR = 160


def _validate_source(source_arrays):
    if not isinstance(source_arrays, dict) or set(source_arrays) != set(FULL_ARRAY_NAMES):
        raise ValueError("nested memory requires exactly the nine full-start arrays")
    source = {key: np.array(source_arrays[key], dtype=np.float64, copy=True)
              for key in FULL_ARRAY_NAMES}
    shapes = {"mass": (10,), "edge_stiffness": (45,), "damping": (10, 10),
              "coupling": (10, 2), "latent_skew": (2, 2),
              "latent_dissipation": (2, 2), "encoder": (2, 80),
              "feature_scale": (80,), "target_scale": (20,)}
    if any(source[key].shape != shape or not np.isfinite(source[key]).all()
           for key, shape in shapes.items()):
        raise ValueError("nested memory source must be finite ten-node r2/80-feature arrays")
    NonlinearPassiveClosure(**{key: source[key] for key in FULL_ARRAY_NAMES[:6]})
    _validated_full_start(source, source["mass"], source["feature_scale"],
                          source["target_scale"], 2)
    return source


def nested_memory_start(source_arrays, normalized_train_features, train_event_weights, *, group):
    """Return independent ``(full_start, audit)`` arrays for group ``r2`` or ``r4``.

    Features are the existing 80 causal inputs divided by the frozen source
    feature scale. Weights are the original source-line-event weights, in the
    same frozen train order. Both groups must pass the same residual-rank check.
    The SVD initializes new encoder rows once; optimization does not recompute
    it or enforce their subsequent orthogonality. Adam must be fresh in both
    groups, with the unchanged mean future objective.
    """
    if group not in ("r2", "r4"):
        raise ValueError("nested memory group must be r2 or r4")
    source = _validate_source(source_arrays)
    features = np.array(normalized_train_features, dtype=np.float64, copy=True)
    weights = np.array(train_event_weights, dtype=np.float64, copy=True)
    if (features.ndim != 2 or features.shape[1] != 80 or features.shape[0] < 2
            or weights.shape != (len(features),)
            or not np.isfinite(features).all() or not np.isfinite(weights).all()
            or np.any(weights <= 0)
            or not np.isclose(weights.sum(), 1.0, rtol=0, atol=1e-12)):
        raise ValueError("train causal features and positive normalized weights must align")

    encoder = source["encoder"]
    _, old_singular, old_right = np.linalg.svd(encoder, full_matrices=False)
    if old_singular[0] <= 0 or old_singular[1] <= RANK_RELATIVE_TOLERANCE * old_singular[0]:
        raise ValueError("source encoder lacks two independent rows")
    # SVD computes the same orthogonal projector as E.T @ inv(E @ E.T) @ E,
    # without explicitly inverting the Gram matrix.
    projector = np.eye(80) - old_right.T @ old_right
    weighted_features = np.sqrt(weights)[:, None] * features
    input_largest_singular = float(np.linalg.svd(weighted_features, compute_uv=False)[0])
    residual = weighted_features @ projector
    _, residual_singular, right = np.linalg.svd(residual, full_matrices=False)
    # Reference the original input too: cancellation roundoff in an exactly
    # exhausted feature space must not become two apparently supported modes.
    residual_rank_threshold = RANK_RELATIVE_TOLERANCE * max(
        input_largest_singular, float(residual_singular[0])
    )
    if (residual_singular[0] <= 0
            or residual_singular[1] <= residual_rank_threshold):
        raise ValueError("train causal features lack two independent residual directions")
    directions = right[:2].copy()
    for row in directions:
        if row[int(np.argmax(np.abs(row)))] < 0:
            row *= -1
    overlap = float(np.linalg.norm(encoder @ directions.T, ord="fro"))
    orthogonality = float(np.linalg.norm(directions @ directions.T - np.eye(2), ord="fro"))
    if (overlap > 1e-10 * max(1.0, float(np.linalg.norm(encoder, ord="fro")))
            or orthogonality > 1e-12):
        raise ValueError("computed residual directions do not satisfy orthogonality")
    row_norm = COEFFICIENT_RATIO * float(np.linalg.norm(encoder, ord="fro")) / np.sqrt(2)
    new_encoder = row_norm * directions

    start = {key: value.copy() for key, value in source.items()}
    dimension = 2 if group == "r2" else 4
    if group == "r4":
        start["coupling"] = np.column_stack((source["coupling"], np.zeros((10, 2))))
        start["latent_skew"] = np.zeros((4, 4))
        start["latent_skew"][:2, :2] = source["latent_skew"]
        start["latent_dissipation"] = np.zeros((4, 4))
        start["latent_dissipation"][:2, :2] = source["latent_dissipation"]
        start["latent_dissipation"][2:, 2:] = NEW_LATENT_DISSIPATION * np.eye(2)
        start["encoder"] = np.vstack((encoder, new_encoder))
    NonlinearPassiveClosure(**{key: start[key] for key in FULL_ARRAY_NAMES[:6]})
    _validated_full_start(start, start["mass"], start["feature_scale"],
                          start["target_scale"], dimension)

    old_norm_squared = float(np.sum(encoder**2))
    new_norm_squared = float(np.sum(new_encoder**2))
    penalty_coefficient = BASE_ENCODER_PENALTY / ENCODER_PENALTY_DENOMINATOR
    audit = {
        "schema": "kron_reduce.nested_memory_start.v1",
        "group": group,
        "source_latent_dimension": 2,
        "target_latent_dimension": dimension,
        "train_event_count": len(features),
        "feature_count": 80,
        "coefficient_ratio": COEFFICIENT_RATIO,
        "rank_relative_tolerance": RANK_RELATIVE_TOLERANCE,
        "rank_reference_input_largest_singular_value": input_largest_singular,
        "residual_rank_threshold": residual_rank_threshold,
        "numpy_version": np.__version__,
        "new_latent_dissipation": NEW_LATENT_DISSIPATION,
        "source_array_sha256": {key: _array_sha256(value) for key, value in source.items()},
        "normalized_train_features_sha256": _array_sha256(features),
        "train_event_weights_sha256": _array_sha256(weights),
        "full_start_array_sha256": {key: _array_sha256(value) for key, value in start.items()},
        "projector_sha256": _array_sha256(projector),
        "residual_directions_sha256": _array_sha256(directions),
        "new_encoder_sha256": _array_sha256(new_encoder),
        "old_encoder_singular_values": old_singular.tolist(),
        "residual_singular_values": residual_singular.tolist(),
        "residual_directions": directions.tolist(),
        "new_encoder": new_encoder.tolist(),
        "new_encoder_rows_applied": group == "r4",
        "old_encoder_direction_overlap_frobenius": overlap,
        "direction_orthogonality_frobenius": orthogonality,
        "new_to_old_encoder_squared_norm_ratio": new_norm_squared / old_norm_squared,
        "encoder_penalty_denominator": ENCODER_PENALTY_DENOMINATOR,
        "encoder_penalty_setting": BASE_ENCODER_PENALTY * dimension / 2,
        "initial_encoder_penalty": penalty_coefficient * float(np.sum(start["encoder"]**2)),
        "initial_encoder_penalty_increase": (
            penalty_coefficient * new_norm_squared if group == "r4" else 0.0
        ),
        "initial_visible_embedding_exact_by_construction": True,
        "svd_recomputed_during_optimization": False,
        "optimizer_initialization": "fresh_adam_zero_moments_step_zero",
        "exact_optimizer_resume": False,
        "optimizer_updated_scalars": 54 + 90 * dimension + dimension**2,
        "future_targets_used": False,
        "labels_used": False,
    }
    return start, audit
