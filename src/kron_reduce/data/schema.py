from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EventMetadata:
    event_id: str
    source_dataset: str
    fault_line: str
    outcome_label: str
    observation_schema: str
    valid_length: int
    available_horizon_s: float
    event_observed: bool
    event_time_s: float | None
    source_npz_sha256: str
    event_signature_sha256: str
    quality_status: str
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class EventSequence:
    event_id: str
    t: np.ndarray
    delta: np.ndarray
    omega: np.ndarray
    Pm: np.ndarray
    equilibrium: dict[str, np.ndarray]
    metadata: EventMetadata
    offline: dict[str, np.ndarray] = field(default_factory=dict)
    extended: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass(frozen=True)
class CanonicalSplit:
    view: str
    split_name: str
    fold: str
    train_ids: tuple[str, ...]
    val_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    disturbance_calibration: dict[str, Any]


@dataclass(frozen=True)
class DatasetMetadata:
    dataset_revision: str
    build_id: str
    schema_version: str
    M_prior: np.ndarray
    omega_s: float
