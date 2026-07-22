from __future__ import annotations

from pathlib import Path

import pytest

from kron_reduce.data import CanonicalStore


@pytest.fixture(scope="session")
def root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def dataset_dir(root: Path) -> Path:
    return root / "data" / "ieee39_postclear_canonical_v1"


@pytest.fixture(scope="session")
def store(dataset_dir: Path) -> CanonicalStore:
    return CanonicalStore.open(dataset_dir)
