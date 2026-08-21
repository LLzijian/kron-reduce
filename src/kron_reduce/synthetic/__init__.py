"""Deterministic synthetic evidence with known dynamical ground truth."""

from .benchmarks import (
    PassiveNetworkBenchmark,
    SyntheticTrajectory,
    benchmark_registry_v1,
)

__all__ = [
    "PassiveNetworkBenchmark",
    "SyntheticTrajectory",
    "benchmark_registry_v1",
]
