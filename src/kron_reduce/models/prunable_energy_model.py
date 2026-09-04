"""One B2-rooted, block-prunable passive dynamic model.

Every optional dynamic block is reciprocal: the same loading is used to
drive its state and to return force to the swing balance.  This makes the
storage/supply identity structural instead of a fitted penalty.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares, lsq_linear, minimize, root

from kron_reduce.physics.coordinates import coi_basis, coi_project, coi_project_power

from .base import DynamicsModel, FitResult, RunContext, sample_rollout_inputs
from .m0 import _pm_equilibrium, _trapz

B2_BACKBONE = "B2_BACKBONE"
NONLINEAR_POTENTIAL = "NONLINEAR_POTENTIAL"
EDGE_DAMPING = "EDGE_DAMPING"
NODE_HIGH_ORDER = "NODE_HIGH_ORDER"
EDGE_MEMORY = "EDGE_MEMORY"
RESONANT_MEMORY = "RESONANT_MEMORY"
OPTIONAL_BLOCKS = (
    NONLINEAR_POTENTIAL,
    EDGE_DAMPING,
    NODE_HIGH_ORDER,
    EDGE_MEMORY,
    RESONANT_MEMORY,
)


@dataclass(frozen=True)
class BlockSpec:
    name: str
    state_dimension: str
    input_port: str
    output_port: str
    parameter_constraint: str
    storage: str
    dissipation: str
    linearization: str
    deletion_method: str
    removable: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateBlockInstance:
    """One independently removable, energy-compatible candidate block."""

    instance_id: str
    family: str
    time_constant: float | None = None
    frequency_hz: float | None = None
    damping_ratio: float | None = None
    factor: tuple[tuple[float, ...], ...] = ()
    port_rank: int = 0
    source_support: tuple[tuple[str, float], ...] = ()
    energy_constraint: str = "passive_by_construction"
    removable: bool = True

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["source_support"] = dict(self.source_support)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CandidateBlockInstance:
        support = value.get("source_support", {})
        if isinstance(support, dict):
            support = tuple(sorted((str(key), float(item)) for key, item in support.items()))
        return cls(
            instance_id=str(value["instance_id"]),
            family=str(value["family"]),
            time_constant=(
                None if value.get("time_constant") is None else float(value["time_constant"])
            ),
            frequency_hz=(
                None if value.get("frequency_hz") is None else float(value["frequency_hz"])
            ),
            damping_ratio=(
                None if value.get("damping_ratio") is None else float(value["damping_ratio"])
            ),
            factor=tuple(tuple(float(entry) for entry in row) for row in value.get("factor", ())),
            port_rank=int(value.get("port_rank", 0)),
            source_support=tuple(support),
            energy_constraint=str(value.get("energy_constraint", "passive_by_construction")),
            removable=bool(value.get("removable", True)),
        )


@dataclass(frozen=True)
class PortRealization:
    """Numerically minimal reciprocal realization of one edge-port factor.

    ``proposed_rank`` is the number of columns supplied by discovery or a
    checkpoint.  It is intentionally distinct from the candidate metadata:
    old checkpoints retain their proposed ``port_rank`` even when a numerical
    re-realization removes dependent or null columns.
    """

    factor: np.ndarray
    proposed_rank: int
    realized_rank: int
    tolerance: float
    discarded_rank: int
    zero_port: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposed_rank": self.proposed_rank,
            "realized_rank": self.realized_rank,
            "tolerance": self.tolerance,
            "discarded_rank": self.discarded_rank,
            "zero_port": self.zero_port,
        }


def _minimal_port_factor(factor: np.ndarray) -> PortRealization:
    """Return a factor with no numerically null port coordinates.

    The thin SVD form ``U S`` preserves the PSD port operator ``F F.T``
    within the numerical SVD rank tolerance while avoiding unforced hidden
    states.  The threshold follows the conventional matrix-rank tolerance and
    is kept deliberately independent of any data-dependent selection
    threshold.
    """

    original = np.asarray(factor, dtype=float)
    if original.ndim != 2:
        raise ValueError("reciprocal port factor must be a two-dimensional array")
    if not np.isfinite(original).all():
        raise ValueError("reciprocal port factor must contain only finite values")
    proposed_rank = int(original.shape[1])
    if original.shape[0] == 0 or proposed_rank == 0:
        empty = np.zeros((original.shape[0], 0), dtype=float)
        empty.setflags(write=False)
        return PortRealization(
            empty,
            proposed_rank,
            0,
            0.0,
            proposed_rank,
            True,
        )
    left, singular_values, _ = np.linalg.svd(original, full_matrices=False)
    smax = float(singular_values[0]) if singular_values.size else 0.0
    tolerance = float(max(original.shape) * np.finfo(float).eps * smax)
    if smax == 0.0:
        realized = np.zeros((original.shape[0], 0), dtype=float)
    else:
        keep = singular_values > tolerance
        left = left[:, keep]
        # SVD signs are arbitrary.  Canonicalize them once so an otherwise
        # identical frozen factor cannot silently flip hidden-state axes.
        for column in range(left.shape[1]):
            pivot = int(np.argmax(np.abs(left[:, column])))
            if left[pivot, column] < 0.0:
                left[:, column] *= -1.0
        realized = left * singular_values[keep][None, :]
    realized.setflags(write=False)
    realized_rank = int(realized.shape[1])
    return PortRealization(
        realized,
        proposed_rank,
        realized_rank,
        tolerance,
        max(0, proposed_rank - realized_rank),
        realized_rank == 0,
    )


@dataclass(frozen=True)
class StructureSelectionResult:
    cluster_decision: dict[str, Any]
    active_blocks: tuple[str, ...]
    pruning_trace: tuple[dict[str, Any], ...]
    rollout_metrics: dict[str, Any]
    modes: tuple[dict[str, Any], ...]
    stability: dict[str, Any]
    model_path: str | None = None
    upstream_cluster_hash: str | None = None
    source_balancing: dict[str, Any] | None = None
    active_block_instances: tuple[dict[str, Any], ...] = ()
    candidate_port_ranks: dict[str, int] | None = None
    module_excitation_support: dict[str, dict[str, Any]] | None = None
    module_identifiability: dict[str, dict[str, Any]] | None = None
    module_deletion_effect: dict[str, dict[str, Any]] | None = None
    module_cross_fold_stability: dict[str, dict[str, Any]] | None = None
    model_count: int = 1
    source_routing: bool = False
    internal_audit_status: str = "not_evaluated"
    external_audit_status: str = "not_evaluated"
    terminal_state: str | None = None

    def __post_init__(self) -> None:
        if self.model_count != 1:
            raise ValueError("one supplied corpus must produce exactly one final model")
        if self.source_routing:
            raise ValueError("source or excitation routing is forbidden for the final model")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def block_registry() -> dict[str, BlockSpec]:
    return {
        B2_BACKBONE: BlockSpec(
            B2_BACKBONE,
            "2*n",
            "node power",
            "node speed",
            "M>0, D>=0, K>=0",
            "kinetic + anchored sine potential",
            "omega.T D omega",
            "analytic swing Jacobian on the COI quotient",
            "never deleted",
            False,
        ),
        NONLINEAR_POTENTIAL: BlockSpec(
            NONLINEAR_POTENTIAL,
            "0",
            "edge angle",
            "edge conservative force",
            "all coefficients >=0",
            "quadratic + quartic + log-cosh edge potential",
            "none",
            "edge-potential Hessian",
            "remove all three potential features and refit every remaining block",
        ),
        EDGE_DAMPING: BlockSpec(
            EDGE_DAMPING,
            "0",
            "edge speed",
            "edge dissipative force",
            "all coefficients >=0",
            "none",
            "nu*f_D(eta,nu) >=0",
            "local edge damping Jacobian",
            "remove all three damping features and refit every remaining block",
        ),
        NODE_HIGH_ORDER: BlockSpec(
            NODE_HIGH_ORDER,
            "n",
            "node speed",
            "node force",
            "tau>0, reciprocal gain >=0",
            "0.5*tau*h.T h",
            "h.T h",
            "first-order reciprocal port Jacobian",
            "remove node state and both reciprocal ports, then refit",
        ),
        EDGE_MEMORY: BlockSpec(
            EDGE_MEMORY,
            "n_tau*n_edge",
            "edge speed",
            "edge force",
            "tau>0, reciprocal gain >=0",
            "sum 0.5*tau*z.T z",
            "sum z.T z",
            "stable real-pole reciprocal port Jacobian",
            "remove all real-pole states and both reciprocal ports, then refit",
        ),
        RESONANT_MEMORY: BlockSpec(
            RESONANT_MEMORY,
            "2*n_mode*n_edge",
            "edge speed",
            "edge force",
            "frequency>0, 0<zeta<1, reciprocal gain >=0",
            "sum 0.5*(p.T p + q.T Omega^2 q)",
            "sum 2*zeta*Omega*p.T p",
            "damped second-order reciprocal port Jacobian",
            "remove all oscillator states and both reciprocal ports, then refit",
        ),
    }


def _first_order_filter(values: np.ndarray, time: np.ndarray, tau: float) -> np.ndarray:
    signal = np.asarray(values, dtype=float)
    t = np.asarray(time, dtype=float)
    state = signal[0].copy()
    output = [state.copy()]
    for index, step in enumerate(np.diff(t)):
        if step <= 0:
            raise ValueError("filter time must be strictly increasing")
        u0, u1 = signal[index], signal[index + 1]
        uh = 0.5 * (u0 + u1)

        def field(value: np.ndarray, drive: np.ndarray) -> np.ndarray:
            return (drive - value) / float(tau)

        k1 = field(state, u0)
        k2 = field(state + 0.5 * step * k1, uh)
        k3 = field(state + 0.5 * step * k2, uh)
        k4 = field(state + step * k3, u1)
        state = state + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        output.append(state.copy())
    return np.stack(output)


def _resonant_filter(
    values: np.ndarray, time: np.ndarray, frequency_hz: float, damping_ratio: float
) -> tuple[np.ndarray, np.ndarray]:
    signal = np.asarray(values, dtype=float)
    t = np.asarray(time, dtype=float)
    angular = 2.0 * np.pi * float(frequency_hz)
    position = signal[0] / angular**2
    velocity = np.zeros_like(position)
    q_out, p_out = [position.copy()], [velocity.copy()]

    def field(q: np.ndarray, p: np.ndarray, drive: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return p, drive - angular**2 * q - 2.0 * damping_ratio * angular * p

    for index, step in enumerate(np.diff(t)):
        if step <= 0:
            raise ValueError("filter time must be strictly increasing")
        u0, u1 = signal[index], signal[index + 1]
        uh = 0.5 * (u0 + u1)
        k1 = field(position, velocity, u0)
        k2 = field(position + 0.5 * step * k1[0], velocity + 0.5 * step * k1[1], uh)
        k3 = field(position + 0.5 * step * k2[0], velocity + 0.5 * step * k2[1], uh)
        k4 = field(position + step * k3[0], velocity + step * k3[1], u1)
        position = position + step * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6.0
        velocity = velocity + step * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6.0
        q_out.append(position.copy())
        p_out.append(velocity.copy())
    return np.stack(q_out), np.stack(p_out)


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=float).reshape(-1)
    b = np.asarray(right, dtype=float).reshape(-1)
    if len(a) != len(b) or len(a) < 4:
        return 0.0
    a = a - np.mean(a)
    b = b - np.mean(b)
    scale = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / scale) if scale > 1e-14 else 0.0


def _merge_relative(values: Sequence[float], *, tolerance: float = 0.10) -> list[float]:
    """Deterministically merge positive pole candidates within a relative band."""

    ordered = sorted(float(value) for value in values if np.isfinite(value) and value > 0.0)
    groups: list[list[float]] = []
    for value in ordered:
        if not groups or abs(value - float(np.median(groups[-1]))) / value > tolerance:
            groups.append([value])
        else:
            groups[-1].append(value)
    return [float(np.median(group)) for group in groups]


def _sample_source(sample: dict[str, Any]) -> str:
    return str(
        sample.get("training_context", {})
        .get("metadata", {})
        .get("source_dataset", "unknown")
    )


def _sample_group(sample: dict[str, Any]) -> str:
    metadata = sample.get("training_context", {}).get("metadata", {})
    return str(metadata.get("fault_line") or metadata.get("independent_group") or sample["event_id"])


def _balanced_event_weights(samples: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Equal source, then equal line, then equal event weighting."""

    by_source: dict[str, dict[str, list[str]]] = {}
    for sample in samples:
        by_source.setdefault(_sample_source(sample), {}).setdefault(
            _sample_group(sample), []
        ).append(str(sample["event_id"]))
    weights: dict[str, float] = {}
    source_count = max(1, len(by_source))
    for groups in by_source.values():
        group_count = max(1, len(groups))
        for event_ids in groups.values():
            event_count = max(1, len(event_ids))
            weight = 1.0 / (source_count * group_count * event_count)
            for event_id in event_ids:
                weights[event_id] = weight
    return weights


def _psd_low_rank_factor(
    inputs: np.ndarray,
    outputs: np.ndarray,
    weights: np.ndarray,
    *,
    energy_fraction: float = 0.95,
    relative_floor: float = 0.05,
    rank_cap: int = 3,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit a reciprocal residual operator and project it to a low-rank PSD factor."""

    x = np.asarray(inputs, dtype=float)
    y = np.asarray(outputs, dtype=float)
    w = np.sqrt(np.maximum(np.asarray(weights, dtype=float).reshape(-1), 0.0))
    if x.ndim != 2 or y.shape != x.shape or len(w) != len(x):
        raise ValueError("PSD port estimation requires matching two-dimensional arrays")
    if len(x) == 0:
        return np.zeros((x.shape[1], 0)), {"rank": 0, "positive_energy": 0.0}
    operator, *_ = np.linalg.lstsq(x * w[:, None], y * w[:, None], rcond=None)
    symmetric = 0.5 * (operator + operator.T)
    values, vectors = np.linalg.eigh(symmetric)
    order = np.argsort(values)[::-1]
    values = np.maximum(values[order], 0.0)
    vectors = vectors[:, order]
    if values.size == 0 or values[0] <= np.finfo(float).eps:
        return np.zeros((x.shape[1], 0)), {
            "rank": 0,
            "positive_energy": 0.0,
            "raw_min_eigenvalue": float(np.min(np.linalg.eigvalsh(symmetric))),
        }
    eligible = np.flatnonzero(values >= float(relative_floor) * values[0])
    maximum_rank = min(int(rank_cap), len(eligible))
    total = float(np.sum(values))
    rank = maximum_rank
    for candidate_rank in range(1, maximum_rank + 1):
        if float(np.sum(values[:candidate_rank])) / max(total, np.finfo(float).eps) >= float(
            energy_fraction
        ):
            rank = candidate_rank
            break
    factor = vectors[:, :rank] * np.sqrt(values[:rank])[None, :]
    return factor, {
        "rank": int(rank),
        "positive_energy": total,
        "explained_positive_energy": float(np.sum(values[:rank])) / max(total, np.finfo(float).eps),
        "leading_eigenvalue": float(values[0]),
        "retained_eigenvalues": values[:rank].tolist(),
        "raw_min_eigenvalue": float(np.min(np.linalg.eigvalsh(symmetric))),
    }


class PrunableEnergyModel(DynamicsModel):
    method_name = "B2-prunable-energy"

    def __init__(
        self,
        *,
        active_blocks: Iterable[str] | None = None,
        block_instances: Sequence[CandidateBlockInstance | dict[str, Any]] | None = None,
        memory_taus: Sequence[float] = (0.05, 0.2, 1.0),
        resonances: Sequence[tuple[float, float]] = ((1.0, 0.10),),
        node_tau: float = 0.2,
        potential_scale: float = 0.25,
        max_internal_step: float = 0.01,
        enforce_identifiability: bool = False,
        minimum_equation_parameter_ratio: float = 3.0,
        maximum_condition_number: float = 1e8,
    ) -> None:
        requested = tuple(dict.fromkeys(active_blocks if active_blocks is not None else OPTIONAL_BLOCKS))
        unknown = set(requested) - set(OPTIONAL_BLOCKS)
        if unknown:
            raise ValueError(f"unknown optional blocks: {sorted(unknown)}")
        self.memory_taus = tuple(sorted({float(value) for value in memory_taus}))
        self.resonances = tuple(
            sorted({(float(frequency), float(damping)) for frequency, damping in resonances})
        )
        self.node_tau = float(node_tau)
        self.potential_scale = float(potential_scale)
        self.max_internal_step = float(max_internal_step)
        self.enforce_identifiability = bool(enforce_identifiability)
        self.minimum_equation_parameter_ratio = float(minimum_equation_parameter_ratio)
        self.maximum_condition_number = float(maximum_condition_number)
        if any(value <= 0 for value in self.memory_taus) or self.node_tau <= 0:
            raise ValueError("memory time constants must be positive")
        if any(frequency <= 0 or not 0 < damping < 1 for frequency, damping in self.resonances):
            raise ValueError("resonance candidates require f>0 and 0<zeta<1")
        if self.potential_scale <= 0 or self.max_internal_step <= 0:
            raise ValueError("potential scale and integration step must be positive")
        if block_instances is None:
            instances: list[CandidateBlockInstance] = []
            if NONLINEAR_POTENTIAL in requested:
                instances.append(
                    CandidateBlockInstance(
                        "potential:edge_nonlinear",
                        NONLINEAR_POTENTIAL,
                        energy_constraint="explicit_nonnegative_edge_potential",
                    )
                )
            if EDGE_DAMPING in requested:
                instances.append(
                    CandidateBlockInstance(
                        "damping:edge_nonlinear",
                        EDGE_DAMPING,
                        energy_constraint="nonnegative_edge_dissipation",
                    )
                )
            if NODE_HIGH_ORDER in requested:
                instances.append(
                    CandidateBlockInstance(
                        "node_high_order:tau",
                        NODE_HIGH_ORDER,
                        time_constant=self.node_tau,
                        energy_constraint="reciprocal_node_storage",
                    )
                )
            for index, tau in enumerate(self.memory_taus):
                if EDGE_MEMORY in requested:
                    instances.append(
                        CandidateBlockInstance(
                            f"edge_memory:{index}:tau={tau:.12g}",
                            EDGE_MEMORY,
                            time_constant=tau,
                            energy_constraint="reciprocal_real_pole_storage",
                        )
                    )
            for index, (frequency, damping) in enumerate(self.resonances):
                if RESONANT_MEMORY in requested:
                    instances.append(
                        CandidateBlockInstance(
                            f"resonant_memory:{index}:f={frequency:.12g}",
                            RESONANT_MEMORY,
                            frequency_hz=frequency,
                            damping_ratio=damping,
                            energy_constraint="reciprocal_oscillator_storage",
                        )
                    )
        else:
            instances = [
                value
                if isinstance(value, CandidateBlockInstance)
                else CandidateBlockInstance.from_dict(value)
                for value in block_instances
            ]
            unknown_instances = {value.family for value in instances} - set(OPTIONAL_BLOCKS)
            if unknown_instances:
                raise ValueError(f"unknown candidate families: {sorted(unknown_instances)}")
            if len({value.instance_id for value in instances}) != len(instances):
                raise ValueError("candidate block instance IDs must be unique")
            self.memory_taus = tuple(
                float(value.time_constant)
                for value in instances
                if value.family == EDGE_MEMORY and value.time_constant is not None
            )
            self.resonances = tuple(
                (float(value.frequency_hz), float(value.damping_ratio))
                for value in instances
                if value.family == RESONANT_MEMORY
                and value.frequency_hz is not None
                and value.damping_ratio is not None
            )
        self.active_instances = tuple(instances)
        self.active_blocks = tuple(
            family for family in OPTIONAL_BLOCKS if any(row.family == family for row in instances)
        )
        self.node_damping: np.ndarray | None = None
        self.coupling: np.ndarray | None = None
        self.potential_coefficients: np.ndarray | None = None
        self.edge_damping_coefficients: np.ndarray | None = None
        self.node_high_order_gain: np.ndarray | None = None
        self.memory_gains: np.ndarray | None = None
        self.resonant_gains: np.ndarray | None = None
        self.memory_factors: list[np.ndarray] = []
        self.resonant_factors: list[np.ndarray] = []
        self._port_realization_cache: dict[
            tuple[str, int], tuple[np.ndarray, PortRealization]
        ] = {}
        self.selected_ridge: float | None = None
        self.validation_scales = {"omega": 1.0, "edge_angle": 1.0}
        self.diagnostics: dict[str, Any] = {}
        self.pruning_trace: list[dict[str, Any]] = []

    @property
    def selected_blocks(self) -> tuple[str, ...]:
        return (B2_BACKBONE, *self.active_blocks)

    @property
    def selected_block_instances(self) -> tuple[CandidateBlockInstance, ...]:
        return self.active_instances

    def _instances(self, family: str) -> tuple[CandidateBlockInstance, ...]:
        return tuple(value for value in self.active_instances if value.family == family)

    def reciprocal_connections(self) -> dict[str, Any]:
        """Return the fitted PSD port factors used at both input and output."""

        _, n_edges = self._require_fitted()
        memory = [self._memory_factor(index, n_edges) for index in range(len(self._instances(EDGE_MEMORY)))]
        resonance = [
            self._resonant_factor(index, n_edges)
            for index in range(len(self._instances(RESONANT_MEMORY)))
        ]
        return {
            NODE_HIGH_ORDER: np.diag(
                np.sqrt(np.maximum(self.node_high_order_gain, 0.0))
            ),
            EDGE_MEMORY: memory,
            RESONANT_MEMORY: resonance,
        }

    def _port_realization(
        self, index: int, family: str, n_edges: int
    ) -> PortRealization:
        if family == EDGE_MEMORY:
            factors = self.memory_factors
            gains = self.memory_gains
        elif family == RESONANT_MEMORY:
            factors = self.resonant_factors
            gains = self.resonant_gains
        else:
            raise ValueError(f"unsupported reciprocal port family: {family}")
        if index < len(factors) and factors[index].shape[0] == n_edges:
            raw = np.asarray(factors[index], dtype=float)
        elif gains is not None and index < len(gains):
            raw = np.diag(np.sqrt(np.maximum(gains[index], 0.0)))
        else:
            instance = self._instances(family)[index]
            raw = np.asarray(instance.factor, dtype=float)
            if raw.ndim != 2 or raw.shape[0] != n_edges:
                raw = np.eye(n_edges)
        cache_key = (family, int(index))
        cached = self._port_realization_cache.get(cache_key)
        if cached is not None and np.array_equal(cached[0], raw):
            return cached[1]
        realization = _minimal_port_factor(raw)
        # Keep a value snapshot rather than an object identity: calibration
        # may replace a factor list, while a truly unchanged frozen factor must
        # retain exactly the same hidden coordinate system.
        self._port_realization_cache[cache_key] = (raw.copy(), realization)
        return realization

    def _memory_factor(self, index: int, n_edges: int) -> np.ndarray:
        return self._port_realization(index, EDGE_MEMORY, n_edges).factor

    def _resonant_factor(self, index: int, n_edges: int) -> np.ndarray:
        return self._port_realization(index, RESONANT_MEMORY, n_edges).factor

    def _port_realization_diagnostics(self, n_edges: int) -> dict[str, dict[str, Any]]:
        output: dict[str, dict[str, Any]] = {}
        for family in (EDGE_MEMORY, RESONANT_MEMORY):
            for index, instance in enumerate(self._instances(family)):
                realization = self._port_realization(index, family, n_edges)
                output[instance.instance_id] = {
                    **realization.to_dict(),
                    "candidate_port_rank": int(instance.port_rank),
                }
        return output

    def _require_fitted(self) -> tuple[int, int]:
        if self.node_damping is None or self.coupling is None:
            raise RuntimeError("prunable model is not fitted")
        return len(self.node_damping), len(self.coupling)

    @staticmethod
    def _training_scales(samples: Sequence[dict[str, Any]]) -> dict[str, float]:
        omega = []
        edge = []
        for sample in samples:
            target = sample["training_context"]["targets"]
            incidence = np.asarray(sample["model_inputs"]["static"]["incidence"], dtype=float)
            omega.append(np.asarray(target["omega"], dtype=float))
            edge.append(np.asarray(target["delta"], dtype=float) @ incidence)
        return {
            "omega": max(float(np.std(np.concatenate(omega))), 1e-8),
            "edge_angle": max(float(np.std(np.concatenate(edge))), 1e-8),
        }

    @staticmethod
    def propose_candidates(
        samples: Sequence[dict[str, Any]],
        base_damping: np.ndarray,
        base_coupling: np.ndarray,
        *,
        max_steps: int = 1001,
        correlation_threshold: float = 0.20,
        bootstrap_draws: int = 200,
        seed: int = 324,
    ) -> dict[str, Any]:
        """Generate supported candidates from derivative-free integral residuals."""

        paths: list[dict[str, Any]] = []
        residual_ratios: list[float] = []
        frequency_candidates: list[tuple[float, float]] = []
        tau_candidates: list[float] = []
        for sample in samples:
            target = sample["training_context"]["targets"]
            static = sample["model_inputs"]["static"]
            use = min(int(max_steps), len(target["t"]))
            if use < 8:
                continue
            time = np.asarray(target["t"][:use], dtype=float)
            delta = np.asarray(target["delta"][:use], dtype=float)
            omega = np.asarray(target["omega"][:use], dtype=float)
            pm = np.asarray(sample["model_inputs"]["exogenous"]["Pm"][:use], dtype=float)
            mass = np.asarray(static["mass"], dtype=float)
            incidence = np.asarray(static["incidence"], dtype=float)
            equilibrium = np.asarray(static["equilibrium_delta"], dtype=float)
            eta = delta @ incidence
            eta0 = equilibrium @ incidence
            nu = omega @ incidence
            known = (
                pm
                - _pm_equilibrium(static)[None, :]
                - omega * base_damping[None, :]
                - (base_coupling[None, :] * (np.sin(eta) - np.sin(eta0)[None, :]))
                @ incidence.T
            )
            interval_impulse = 0.5 * (known[:-1] + known[1:]) * np.diff(time)[:, None]
            interval_impulse -= mass[None, :] * np.diff(omega, axis=0)
            interval_impulse = np.stack(
                [coi_project_power(row, mass) for row in interval_impulse]
            )
            total = pm - _pm_equilibrium(static)[None, :]
            total_impulse = 0.5 * (total[:-1] + total[1:]) * np.diff(time)[:, None]
            ratio = float(
                np.linalg.norm(interval_impulse)
                / max(np.linalg.norm(total_impulse), 1e-12)
            )
            residual_ratios.append(ratio)
            if ratio <= 1e-3:
                continue
            duration = np.diff(time)[:, None]
            edge_rate = (interval_impulse / duration) @ np.linalg.pinv(incidence.T)
            centered = edge_rate - np.mean(edge_rate, axis=0, keepdims=True)
            scale = float(np.sum(centered**2))
            if scale > 1e-14:
                maximum_lag = min(
                    len(centered) // 3,
                    max(2, round(2.0 / np.median(np.diff(time)))),
                )
                correlation = np.asarray(
                    [
                        float(np.sum(centered[:-lag] * centered[lag:]) / scale)
                        for lag in range(1, maximum_lag + 1)
                    ]
                )
                crossings = np.flatnonzero(correlation <= np.exp(-1.0))
                if crossings.size:
                    tau_candidates.append(
                        float((crossings[0] + 1) * np.median(np.diff(time)))
                    )
            spectrum = np.abs(np.fft.rfft(centered, axis=0))
            frequencies = np.fft.rfftfreq(use, d=float(np.median(np.diff(time))))
            if len(frequencies) != len(spectrum):
                frequencies = np.fft.rfftfreq(len(centered), d=float(np.median(np.diff(time))))
            band = (frequencies >= 0.1) & (frequencies <= 3.0)
            if np.any(band):
                band_power = np.sum(spectrum[band] ** 2, axis=1)
                concentration = float(np.max(band_power) / max(np.sum(band_power), 1e-12))
                if concentration >= 0.10:
                    local_frequency = frequencies[band]
                    selected_peaks: list[float] = []
                    for peak_index in np.argsort(band_power)[::-1]:
                        peak_frequency = float(local_frequency[int(peak_index)])
                        if any(
                            abs(peak_frequency - previous)
                            / max(peak_frequency, previous)
                            <= 0.10
                            for previous in selected_peaks
                        ):
                            continue
                        selected_peaks.append(peak_frequency)
                        half_power = band_power >= 0.5 * band_power[int(peak_index)]
                        support_frequency = local_frequency[half_power]
                        bandwidth = (
                            float(np.max(support_frequency) - np.min(support_frequency))
                            if support_frequency.size >= 2
                            else float(np.median(np.diff(frequencies)))
                        )
                        damping = float(
                            np.clip(
                                bandwidth / max(2.0 * peak_frequency, 1e-12),
                                0.01,
                                0.49,
                            )
                        )
                        frequency_candidates.append((peak_frequency, damping))
                        if len(selected_peaks) >= 3:
                            break
            speed_centered = nu - np.mean(nu, axis=0, keepdims=True)
            speed_spectrum = np.abs(np.fft.rfft(speed_centered, axis=0))
            speed_frequencies = np.fft.rfftfreq(
                len(speed_centered), d=float(np.median(np.diff(time)))
            )
            speed_band = (speed_frequencies >= 0.1) & (speed_frequencies <= 3.0)
            if np.any(speed_band):
                speed_power = np.sum(speed_spectrum[speed_band] ** 2, axis=1)
                local_speed_frequency = speed_frequencies[speed_band]
                selected_speed_peaks: list[float] = []
                for peak_index in np.argsort(speed_power)[::-1]:
                    peak_frequency = float(local_speed_frequency[int(peak_index)])
                    if any(
                        abs(peak_frequency - previous)
                        / max(peak_frequency, previous)
                        <= 0.10
                        for previous in selected_speed_peaks
                    ):
                        continue
                    selected_speed_peaks.append(peak_frequency)
                    frequency_candidates.append((peak_frequency, 0.10))
                    if len(selected_speed_peaks) >= 3:
                        break
            paths.append(
                {
                    "event_id": str(sample["event_id"]),
                    "source": _sample_source(sample),
                    "group": _sample_group(sample),
                    "time": time,
                    "known": known,
                    "mass": mass,
                    "incidence": incidence,
                    "omega": omega,
                    "theta": eta - eta0[None, :],
                    "nu": nu,
                }
            )

        def cap_three(values: list[float]) -> list[float]:
            if len(values) <= 3:
                return values
            indices = np.linspace(0, len(values) - 1, 3).round().astype(int)
            return [values[int(index)] for index in np.unique(indices)]

        memory_taus = cap_three(_merge_relative(tau_candidates) or [0.05, 0.2, 1.0])
        merged_frequencies = cap_three(
            _merge_relative([row[0] for row in frequency_candidates]) or [1.0]
        )
        resonances: list[tuple[float, float]] = []
        for frequency in merged_frequencies:
            nearby = [
                damping
                for candidate, damping in frequency_candidates
                if abs(candidate - frequency) / max(candidate, frequency) <= 0.10
            ]
            resonances.append((frequency, float(np.median(nearby)) if nearby else 0.10))

        records: list[dict[str, Any]] = []
        event_weights = _balanced_event_weights(samples)
        for path in paths:
            time = path["time"]
            limit = len(time)
            widths = [value for value in (1, 5, 20, 80) if value < limit]
            node_filtered = _first_order_filter(path["omega"], time, 0.2)
            memory_filtered = [_first_order_filter(path["nu"], time, tau) for tau in memory_taus]
            resonant_filtered = [
                _resonant_filter(path["nu"], time, frequency, damping)[1]
                for frequency, damping in resonances
            ]
            rows: dict[str, list[np.ndarray]] = {
                "edge_residual": [],
                "node_residual": [],
                "potential_linear": [],
                "potential_cubic": [],
                "potential_saturation": [],
                "damping_linear": [],
                "damping_quadratic": [],
                "damping_angle": [],
                "node_high_order": [],
            }
            rows.update({f"memory:{index}": [] for index in range(len(memory_taus))})
            rows.update({f"resonance:{index}": [] for index in range(len(resonances))})
            for width in widths:
                starts = np.arange(limit - width)
                if len(starts) > 16:
                    starts = np.unique(np.linspace(0, len(starts) - 1, 16).round().astype(int))
                for start in starts:
                    stop = int(start + width)
                    segment = slice(int(start), stop + 1)
                    node_residual = _trapz(path["known"][segment], time[segment])
                    node_residual -= path["mass"] * (
                        path["omega"][stop] - path["omega"][int(start)]
                    )
                    node_residual = coi_project_power(node_residual, path["mass"])
                    rows["node_residual"].append(node_residual)
                    rows["edge_residual"].append(
                        node_residual @ np.linalg.pinv(path["incidence"].T)
                    )
                    theta = path["theta"][segment]
                    nu = path["nu"][segment]
                    rows["potential_linear"].append(_trapz(theta, time[segment]))
                    rows["potential_cubic"].append(_trapz(theta**3, time[segment]))
                    rows["potential_saturation"].append(
                        _trapz(np.tanh(theta / 0.25), time[segment])
                    )
                    rows["damping_linear"].append(_trapz(nu, time[segment]))
                    rows["damping_quadratic"].append(
                        _trapz(nu * np.abs(nu), time[segment])
                    )
                    rows["damping_angle"].append(_trapz(nu * theta**2, time[segment]))
                    rows["node_high_order"].append(
                        _trapz(node_filtered[segment], time[segment])
                    )
                    for index, filtered in enumerate(memory_filtered):
                        rows[f"memory:{index}"].append(_trapz(filtered[segment], time[segment]))
                    for index, filtered in enumerate(resonant_filtered):
                        rows[f"resonance:{index}"].append(_trapz(filtered[segment], time[segment]))
            if rows["edge_residual"]:
                records.append(
                    {
                        "event_id": path["event_id"],
                        "source": path["source"],
                        "group": path["group"],
                        "event_weight": event_weights.get(path["event_id"], 1.0),
                        **{key: np.stack(value) for key, value in rows.items()},
                    }
                )

        candidate_keys = [NONLINEAR_POTENTIAL, EDGE_DAMPING, NODE_HIGH_ORDER]
        candidate_keys += [f"memory:{index}" for index in range(len(memory_taus))]
        candidate_keys += [f"resonance:{index}" for index in range(len(resonances))]

        def scores(chosen: Sequence[dict[str, Any]]) -> dict[str, float]:
            if not chosen:
                return {key: 0.0 for key in candidate_keys}

            def joined(name: str) -> np.ndarray:
                blocks = []
                for row in chosen:
                    weight = np.sqrt(float(row["event_weight"]) / max(1, len(row[name])))
                    blocks.append(np.asarray(row[name]) * weight)
                return np.concatenate(blocks)

            edge = joined("edge_residual")
            def median_event_correlation(residual_name: str, feature_name: str) -> float:
                return float(
                    np.median(
                        [
                            abs(_safe_corr(row[residual_name], row[feature_name]))
                            for row in chosen
                        ]
                    )
                )

            output = {
                NONLINEAR_POTENTIAL: max(
                    abs(_safe_corr(edge, joined("potential_linear"))),
                    abs(_safe_corr(edge, joined("potential_cubic"))),
                    abs(_safe_corr(edge, joined("potential_saturation"))),
                ),
                EDGE_DAMPING: max(
                    abs(_safe_corr(edge, joined("damping_linear"))),
                    abs(_safe_corr(edge, joined("damping_quadratic"))),
                    abs(_safe_corr(edge, joined("damping_angle"))),
                ),
                NODE_HIGH_ORDER: min(
                    1.0,
                    5.0 * median_event_correlation("node_residual", "node_high_order"),
                ),
            }
            output.update(
                {
                    f"memory:{index}": min(
                        1.0,
                        4.0
                        * median_event_correlation("edge_residual", f"memory:{index}"),
                    )
                    for index in range(len(memory_taus))
                }
            )
            output.update(
                {
                    f"resonance:{index}": min(
                        1.0,
                        4.0
                        * median_event_correlation("edge_residual", f"resonance:{index}"),
                    )
                    for index in range(len(resonances))
                }
            )
            return output

        sources = sorted({row["source"] for row in records})
        support_by_source: dict[str, dict[str, float]] = {}
        for source_index, source in enumerate(sources):
            local = [row for row in records if row["source"] == source]
            groups = sorted({row["group"] for row in local})
            by_group = {group: [row for row in local if row["group"] == group] for group in groups}
            counts = {key: 0 for key in candidate_keys}
            draws = max(1, int(bootstrap_draws))
            rng = np.random.default_rng(int(seed) + 1009 * source_index)
            for _ in range(draws):
                chosen_groups = groups if bootstrap_draws <= 0 else [
                    groups[int(index)] for index in rng.integers(0, len(groups), len(groups))
                ]
                chosen = [row for group in chosen_groups for row in by_group[group]]
                for key, score in scores(chosen).items():
                    counts[key] += int(score >= correlation_threshold)
            support_by_source[source] = {key: counts[key] / draws for key in candidate_keys}

        def support(key: str) -> dict[str, float]:
            return {source: support_by_source[source].get(key, 0.0) for source in sources}

        instances: list[CandidateBlockInstance] = []
        static_specs = (
            (NONLINEAR_POTENTIAL, "potential:edge_nonlinear", "explicit_nonnegative_edge_potential"),
            (EDGE_DAMPING, "damping:edge_nonlinear", "nonnegative_edge_dissipation"),
            (NODE_HIGH_ORDER, "node_high_order:tau=0.2", "reciprocal_node_storage"),
        )
        for key, instance_id, energy in static_specs:
            local_support = support(key)
            if max(local_support.values(), default=0.0) >= 0.80:
                instances.append(
                    CandidateBlockInstance(
                        instance_id,
                        key,
                        time_constant=0.2 if key == NODE_HIGH_ORDER else None,
                        source_support=tuple(sorted(local_support.items())),
                        energy_constraint=energy,
                    )
                )

        factor_diagnostics: dict[str, Any] = {}
        for family, values, prefix in (
            (EDGE_MEMORY, memory_taus, "memory"),
            (RESONANT_MEMORY, resonances, "resonance"),
        ):
            for index, value in enumerate(values):
                key = f"{prefix}:{index}"
                local_support = support(key)
                if max(local_support.values(), default=0.0) < 0.80 or not records:
                    continue
                x = np.concatenate([row[key] for row in records])
                y = np.concatenate([row["edge_residual"] for row in records])
                weights = np.concatenate(
                    [
                        np.full(len(row[key]), float(row["event_weight"]) / max(1, len(row[key])))
                        for row in records
                    ]
                )
                factor, diagnostics = _psd_low_rank_factor(x, y, weights)
                if factor.shape[1] == 0:
                    covariance = (y * weights[:, None]).T @ y / max(
                        float(np.sum(weights)), np.finfo(float).eps
                    )
                    covariance = 0.5 * (covariance + covariance.T)
                    values_eig, vectors_eig = np.linalg.eigh(covariance)
                    order_eig = np.argsort(values_eig)[::-1]
                    values_eig = np.maximum(values_eig[order_eig], 0.0)
                    vectors_eig = vectors_eig[:, order_eig]
                    if values_eig.size == 0 or values_eig[0] <= np.finfo(float).eps:
                        continue
                    eligible_eig = np.flatnonzero(values_eig >= 0.05 * values_eig[0])
                    rank = min(3, len(eligible_eig))
                    total_eig = float(np.sum(values_eig))
                    for candidate_rank in range(1, rank + 1):
                        if float(np.sum(values_eig[:candidate_rank])) / max(
                            total_eig, np.finfo(float).eps
                        ) >= 0.95:
                            rank = candidate_rank
                            break
                    factor = vectors_eig[:, :rank] * np.sqrt(values_eig[:rank])[None, :]
                    diagnostics = {
                        "rank": int(rank),
                        "positive_energy": total_eig,
                        "explained_positive_energy": float(np.sum(values_eig[:rank]))
                        / max(total_eig, np.finfo(float).eps),
                        "leading_eigenvalue": float(values_eig[0]),
                        "retained_eigenvalues": values_eig[:rank].tolist(),
                        "fallback": "PSD_residual_spatial_covariance",
                    }
                instance_id = (
                    f"edge_memory:{index}:tau={float(value):.12g}"
                    if family == EDGE_MEMORY
                    else f"resonant_memory:{index}:f={float(value[0]):.12g}"
                )
                factor_diagnostics[instance_id] = diagnostics
                instances.append(
                    CandidateBlockInstance(
                        instance_id,
                        family,
                        time_constant=float(value) if family == EDGE_MEMORY else None,
                        frequency_hz=float(value[0]) if family == RESONANT_MEMORY else None,
                        damping_ratio=float(value[1]) if family == RESONANT_MEMORY else None,
                        factor=tuple(tuple(float(entry) for entry in row) for row in factor),
                        port_rank=int(factor.shape[1]),
                        source_support=tuple(sorted(local_support.items())),
                        energy_constraint=(
                            "reciprocal_real_pole_storage"
                            if family == EDGE_MEMORY
                            else "reciprocal_oscillator_storage"
                        ),
                    )
                )

        family_support = {
            family: {
                source: max(
                    [
                        value
                        for key, value in support_by_source[source].items()
                        if key == family
                        or (family == EDGE_MEMORY and key.startswith("memory:"))
                        or (family == RESONANT_MEMORY and key.startswith("resonance:"))
                    ],
                    default=0.0,
                )
                for source in sources
            }
            for family in OPTIONAL_BLOCKS
        }
        selected = tuple(
            family for family in OPTIONAL_BLOCKS if any(row.family == family for row in instances)
        )
        return {
            "active_blocks": selected,
            "candidate_instances": [row.to_dict() for row in instances],
            "support": {family: max(values.values(), default=0.0) for family, values in family_support.items()},
            "support_by_source": family_support,
            "support_unit": "fault_line_within_source",
            "support_method": "source_separate_fault_line_bootstrap_integral_residual",
            "bootstrap_draws": int(bootstrap_draws),
            "minimum_support": 0.80,
            "median_residual_ratio": float(np.median(residual_ratios)) if residual_ratios else 0.0,
            "memory_taus": memory_taus,
            "resonances": [list(row) for row in resonances],
            "port_factor_diagnostics": factor_diagnostics,
            "residual_definition": "integral_impulse_without_numerical_differentiation",
            "source_balancing": "equal_source_then_equal_fault_line_then_equal_event_then_equal_window",
            "maximum_port_rank": 3,
            "positive_energy_target": 0.95,
            "relative_eigenvalue_floor": 0.05,
        }

    def _parameter_layout(self, n_nodes: int, n_edges: int) -> dict[str, slice]:
        cursor = 0
        layout: dict[str, slice] = {}

        def add(name: str, count: int) -> None:
            nonlocal cursor
            layout[name] = slice(cursor, cursor + int(count))
            cursor += int(count)

        add("node_damping", n_nodes)
        add("coupling", n_edges)
        for instance in self.active_instances:
            if instance.family in {NONLINEAR_POTENTIAL, EDGE_DAMPING}:
                add(instance.instance_id, 3 * n_edges)
            elif instance.family == NODE_HIGH_ORDER:
                add(instance.instance_id, n_nodes)
            elif instance.family in {EDGE_MEMORY, RESONANT_MEMORY}:
                factor = np.asarray(instance.factor, dtype=float)
                add(instance.instance_id, 1 if factor.shape[0] == n_edges and factor.size else n_edges)
        layout["all"] = slice(0, cursor)
        return layout

    def _design_path(self, sample: dict[str, Any], limit: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        target = sample["training_context"]["targets"]
        static = sample["model_inputs"]["static"]
        time = np.asarray(target["t"][:limit], dtype=float)
        delta = np.asarray(target["delta"][:limit], dtype=float)
        omega = np.asarray(target["omega"][:limit], dtype=float)
        pm = np.asarray(sample["model_inputs"]["exogenous"]["Pm"][:limit], dtype=float)
        mass = np.asarray(static["mass"], dtype=float)
        incidence = np.asarray(static["incidence"], dtype=float)
        equilibrium = np.asarray(static["equilibrium_delta"], dtype=float)
        eta = delta @ incidence
        eta0 = equilibrium @ incidence
        theta = eta - eta0[None, :]
        nu = omega @ incidence
        n_nodes, n_edges = len(mass), incidence.shape[1]
        layout = self._parameter_layout(n_nodes, n_edges)
        design = np.zeros((limit, n_nodes, layout["all"].stop), dtype=float)
        projector = np.eye(n_nodes) - np.outer(mass, np.ones(n_nodes)) / np.sum(mass)
        for index in range(limit):
            design[index, :, layout["node_damping"]] = projector @ np.diag(omega[index])
            design[index, :, layout["coupling"]] = incidence * (
                np.sin(eta[index]) - np.sin(eta0)
            )
        if NONLINEAR_POTENTIAL in self.active_blocks:
            features = np.stack(
                (theta, theta**3, np.tanh(theta / self.potential_scale)), axis=2
            )
            block = layout[self._instances(NONLINEAR_POTENTIAL)[0].instance_id]
            for index in range(limit):
                design[index, :, block] = np.concatenate(
                    [incidence * features[index, :, feature] for feature in range(3)], axis=1
                )
        if EDGE_DAMPING in self.active_blocks:
            features = np.stack((nu, nu * np.abs(nu), nu * theta**2), axis=2)
            block = layout[self._instances(EDGE_DAMPING)[0].instance_id]
            for index in range(limit):
                design[index, :, block] = np.concatenate(
                    [incidence * features[index, :, feature] for feature in range(3)], axis=1
                )
        if NODE_HIGH_ORDER in self.active_blocks:
            filtered = _first_order_filter(omega, time, self.node_tau)
            block = layout[self._instances(NODE_HIGH_ORDER)[0].instance_id]
            for index in range(limit):
                design[index, :, block] = projector @ np.diag(filtered[index])
        if EDGE_MEMORY in self.active_blocks:
            for instance in self._instances(EDGE_MEMORY):
                tau = float(instance.time_constant)
                filtered = _first_order_filter(nu, time, tau)
                block = layout[instance.instance_id]
                factor = np.asarray(instance.factor, dtype=float)
                if factor.shape[0] == n_edges and factor.size:
                    operator_signal = filtered @ (factor @ factor.T)
                    design[:, :, block] = (operator_signal @ incidence.T)[:, :, None]
                else:
                    for index in range(limit):
                        design[index, :, block] = incidence * filtered[index]
        if RESONANT_MEMORY in self.active_blocks:
            for instance in self._instances(RESONANT_MEMORY):
                frequency = float(instance.frequency_hz)
                damping = float(instance.damping_ratio)
                _, velocity = _resonant_filter(nu, time, frequency, damping)
                block = layout[instance.instance_id]
                factor = np.asarray(instance.factor, dtype=float)
                if factor.shape[0] == n_edges and factor.size:
                    operator_signal = velocity @ (factor @ factor.T)
                    design[:, :, block] = (operator_signal @ incidence.T)[:, :, None]
                else:
                    for index in range(limit):
                        design[index, :, block] = incidence * velocity[index]
        target_vector = pm - _pm_equilibrium(static)[None, :]
        return time, design, target_vector

    def build_weak_system(
        self, samples: Sequence[dict[str, Any]], context: RunContext
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        rows = []
        targets = []
        blocks = 0
        event_weights = _balanced_event_weights(samples[: context.max_cases])
        for sample in samples[: context.max_cases]:
            target = sample["training_context"]["targets"]
            static = sample["model_inputs"]["static"]
            mass = np.asarray(static["mass"], dtype=float)
            limit = min(
                len(target["t"]),
                int(context.fit_max_steps) if context.fit_max_steps > 0 else len(target["t"]),
            )
            time, designs, delta_pm = self._design_path(sample, limit)
            widths = [value for value in sorted(set(context.integral_windows)) if 0 < value < limit]
            for width in widths:
                starts = np.arange(limit - width)
                if context.weak_max_windows_per_case_horizon > 0 and len(starts) > context.weak_max_windows_per_case_horizon:
                    starts = np.unique(
                        np.linspace(
                            0, len(starts) - 1, context.weak_max_windows_per_case_horizon
                        ).round().astype(int)
                    )
                weight = np.sqrt(
                    event_weights[str(sample["event_id"])]
                    / max(1, len(widths) * len(starts) * len(mass))
                )
                blocks += 1
                omega = np.asarray(target["omega"][:limit], dtype=float)
                for start in starts:
                    stop = int(start + width)
                    segment = slice(int(start), stop + 1)
                    rows.append(weight * _trapz(designs[segment], time[segment]))
                    targets.append(
                        weight
                        * (
                            _trapz(delta_pm[segment], time[segment])
                            - mass * (omega[stop] - omega[int(start)])
                        )
                    )
        if not rows:
            raise ValueError("no weak-form windows for prunable model")
        matrix = np.concatenate(rows)
        vector = np.concatenate(targets)
        return matrix, vector, {
            "n_scalar_equations": int(matrix.shape[0]),
            "n_parameters": int(matrix.shape[1]),
            "n_equal_case_window_blocks": int(blocks),
            "weak_window_weighting": "equal_source_equal_line_equal_event_equal_window_width",
            "source_balance": {
                source: float(
                    sum(
                        event_weights[str(sample["event_id"])]
                        for sample in samples[: context.max_cases]
                        if _sample_source(sample) == source
                    )
                )
                for source in sorted({_sample_source(sample) for sample in samples[: context.max_cases]})
            },
        }

    def _assign_parameters(self, theta: np.ndarray, n_nodes: int, n_edges: int) -> None:
        layout = self._parameter_layout(n_nodes, n_edges)
        self.node_damping = theta[layout["node_damping"]].copy()
        self.coupling = theta[layout["coupling"]].copy()
        self.potential_coefficients = (
            theta[layout[self._instances(NONLINEAR_POTENTIAL)[0].instance_id]].reshape(3, n_edges)
            if NONLINEAR_POTENTIAL in self.active_blocks
            else np.zeros((3, n_edges))
        )
        self.edge_damping_coefficients = (
            theta[layout[self._instances(EDGE_DAMPING)[0].instance_id]].reshape(3, n_edges)
            if EDGE_DAMPING in self.active_blocks
            else np.zeros((3, n_edges))
        )
        self.node_high_order_gain = (
            theta[layout[self._instances(NODE_HIGH_ORDER)[0].instance_id]].copy()
            if NODE_HIGH_ORDER in self.active_blocks
            else np.zeros(n_nodes)
        )
        self.memory_factors = []
        memory_gains = []
        for instance in self._instances(EDGE_MEMORY):
            values = theta[layout[instance.instance_id]]
            base_factor = np.asarray(instance.factor, dtype=float)
            if base_factor.shape[0] == n_edges and base_factor.size:
                factor = np.sqrt(max(float(values[0]), 0.0)) * base_factor
                gains = np.diag(factor @ factor.T)
            else:
                gains = np.maximum(values, 0.0)
                factor = np.diag(np.sqrt(gains))
            self.memory_factors.append(_minimal_port_factor(factor).factor)
            memory_gains.append(gains)
        self.memory_gains = np.stack(memory_gains) if memory_gains else np.zeros((0, n_edges))
        self.resonant_factors = []
        resonant_gains = []
        for instance in self._instances(RESONANT_MEMORY):
            values = theta[layout[instance.instance_id]]
            base_factor = np.asarray(instance.factor, dtype=float)
            if base_factor.shape[0] == n_edges and base_factor.size:
                factor = np.sqrt(max(float(values[0]), 0.0)) * base_factor
                gains = np.diag(factor @ factor.T)
            else:
                gains = np.maximum(values, 0.0)
                factor = np.diag(np.sqrt(gains))
            self.resonant_factors.append(_minimal_port_factor(factor).factor)
            resonant_gains.append(gains)
        self.resonant_gains = np.stack(resonant_gains) if resonant_gains else np.zeros((0, n_edges))

    def _solve(
        self,
        matrix: np.ndarray,
        vector: np.ndarray,
        ridge: float,
        normal_system: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if normal_system is None:
            scale = np.sqrt(np.mean(matrix**2, axis=0))
            scale = np.where(scale > 1e-14, scale, 1.0)
            normalized = matrix / scale
            gram = normalized.T @ normalized
            normal_rhs = normalized.T @ vector
        else:
            scale, gram, normal_rhs = normal_system
        regularized = gram + float(ridge) * np.eye(gram.shape[0])
        jitter = np.finfo(float).eps * max(1.0, float(np.trace(regularized)))
        upper = np.linalg.cholesky(regularized + jitter * np.eye(gram.shape[0])).T
        compressed_rhs = np.linalg.solve(upper.T, normal_rhs)
        result = lsq_linear(
            upper,
            compressed_rhs,
            bounds=(0.0, np.inf),
            max_iter=3000,
            lsq_solver="exact",
        )
        theta = result.x / scale
        eigenvalues = np.maximum(np.linalg.eigvalsh(0.5 * (gram + gram.T)), 0.0)
        positive = eigenvalues[eigenvalues > np.max(eigenvalues, initial=0.0) * 1e-12]
        return theta, {
            "solver_success": bool(result.success),
            "solver_iterations": int(result.nit),
            "effective_rank": len(positive),
            "condition_number": (
                float(np.sqrt(np.max(positive) / np.min(positive)))
                if len(positive)
                else float("inf")
            ),
            "weak_residual_rmse": float(np.sqrt(np.mean((matrix @ theta - vector) ** 2))),
            "solver_form": "equivalent_normal_equation_cholesky_nonnegative_least_squares",
        }

    def validation_summary(
        self, samples: Sequence[dict[str, Any]], context: RunContext
    ) -> dict[str, Any]:
        losses = []
        raw = []
        failures = 0
        horizon_losses: dict[float, list[float]] = {1.0: [], 5.0: [], 10.0: []}
        for sample in samples[: context.max_cases]:
            target = sample["training_context"]["targets"]
            steps = min(len(target["t"]), context.horizon_steps)
            static, time, exogenous = sample_rollout_inputs(sample)
            try:
                delta, omega = self.rollout(
                    static,
                    time[:steps],
                    {"Pm": exogenous["Pm"][:steps], "interpolation": exogenous["interpolation"]},
                )
            except (FloatingPointError, ValueError):
                failures += 1
                continue
            incidence = np.asarray(static["incidence"], dtype=float)
            edge_error = (delta - np.asarray(target["delta"][:steps])) @ incidence
            omega_error = omega - np.asarray(target["omega"][:steps])
            standardized_pointwise = (
                np.mean(
                    (edge_error / self.validation_scales["edge_angle"]) ** 2,
                    axis=1,
                )
                + np.mean(
                    (omega_error / self.validation_scales["omega"]) ** 2,
                    axis=1,
                )
            )
            losses.append(float(np.mean(standardized_pointwise)))
            relative_time = np.asarray(time[:steps], dtype=float) - float(time[0])
            for horizon_seconds, values in horizon_losses.items():
                stop = int(np.searchsorted(relative_time, horizon_seconds, side="right"))
                if stop >= 2:
                    values.append(float(np.mean(standardized_pointwise[:stop])))
            raw.append(
                float(
                    np.sqrt(
                        np.mean(
                            np.concatenate((edge_error.reshape(-1), omega_error.reshape(-1)))
                            ** 2
                        )
                    )
                )
            )
        return {
            "standardized_loss": float(np.mean(losses)) if losses else float("inf"),
            "raw_mean_rmse": float(np.mean(raw)) if raw else float("inf"),
            "failure_rate": failures / max(1, min(len(samples), context.max_cases)),
            "horizon_standardized_loss": {
                f"{horizon:g}s": float(np.mean(values)) if values else float("inf")
                for horizon, values in horizon_losses.items()
            },
        }

    @staticmethod
    def _rollout_noninferior(
        reference: dict[str, Any], candidate: dict[str, Any], tolerance: float
    ) -> bool:
        if candidate["failure_rate"] != 0.0:
            return False
        for horizon, reference_loss in reference["horizon_standardized_loss"].items():
            if not np.isfinite(reference_loss):
                continue
            candidate_loss = candidate["horizon_standardized_loss"].get(horizon, float("inf"))
            if not np.isfinite(candidate_loss):
                return False
            if candidate_loss > reference_loss * (1.0 + float(tolerance)) + 1e-15:
                return False
        return True

    @staticmethod
    def source_stratified_degradation(
        reference: PrunableEnergyModel,
        candidate: PrunableEnergyModel,
        samples: Sequence[dict[str, Any]],
        context: RunContext,
        *,
        tolerance: float = 0.02,
        bootstrap_draws: int = 200,
        confidence: float = 0.95,
        seed: int = 324,
    ) -> dict[str, Any]:
        """Audit deletion loss by source and horizon with fault-line bootstrap."""

        rows: list[dict[str, Any]] = []
        failures: list[dict[str, str]] = []
        for sample in samples[: context.max_cases]:
            target = sample["training_context"]["targets"]
            static, time, exogenous = sample_rollout_inputs(sample)
            steps = min(len(time), int(context.horizon_steps))
            try:
                reference_delta, reference_omega = reference.rollout(
                    static,
                    time[:steps],
                    {"Pm": exogenous["Pm"][:steps], "interpolation": exogenous["interpolation"]},
                )
                candidate_delta, candidate_omega = candidate.rollout(
                    static,
                    time[:steps],
                    {"Pm": exogenous["Pm"][:steps], "interpolation": exogenous["interpolation"]},
                )
            except (FloatingPointError, ValueError) as error:
                failures.append({"event_id": str(sample["event_id"]), "error": type(error).__name__})
                continue
            incidence = np.asarray(static["incidence"], dtype=float)
            truth_delta = np.asarray(target["delta"][:steps], dtype=float) @ incidence
            truth_omega = np.asarray(target["omega"][:steps], dtype=float)
            reference_state = np.concatenate((reference_delta @ incidence, reference_omega), axis=1)
            candidate_state = np.concatenate((candidate_delta @ incidence, candidate_omega), axis=1)
            truth_state = np.concatenate((truth_delta, truth_omega), axis=1)
            relative_time = np.asarray(time[:steps], dtype=float) - float(time[0])
            for horizon in (1.0, 5.0, 10.0):
                stop = int(np.searchsorted(relative_time, horizon, side="right"))
                if stop < 2:
                    continue
                truth_slice = truth_state[:stop]
                centered = truth_slice - np.mean(truth_slice, axis=0, keepdims=True)
                rows.append(
                    {
                        "event_id": str(sample["event_id"]),
                        "source": _sample_source(sample),
                        "group": _sample_group(sample),
                        "horizon_seconds": horizon,
                        "reference_sse": float(np.sum((reference_state[:stop] - truth_slice) ** 2)),
                        "candidate_sse": float(np.sum((candidate_state[:stop] - truth_slice) ** 2)),
                        "truth_sst": float(np.sum(centered**2)),
                    }
                )
        audits: list[dict[str, Any]] = []
        rng = np.random.default_rng(int(seed))
        for source in sorted({row["source"] for row in rows}):
            for horizon in (1.0, 5.0, 10.0):
                local = [
                    row
                    for row in rows
                    if row["source"] == source and row["horizon_seconds"] == horizon
                ]
                if not local:
                    continue
                numerator = sum(row["candidate_sse"] - row["reference_sse"] for row in local)
                denominator = sum(row["truth_sst"] for row in local)
                point = max(0.0, float(numerator)) / max(float(denominator), np.finfo(float).eps)
                groups = sorted({row["group"] for row in local})
                by_group = {group: [row for row in local if row["group"] == group] for group in groups}
                bootstrap = []
                for _ in range(max(1, int(bootstrap_draws))):
                    chosen = [groups[int(index)] for index in rng.integers(0, len(groups), len(groups))]
                    selected = [row for group in chosen for row in by_group[group]]
                    draw_num = sum(
                        row["candidate_sse"] - row["reference_sse"] for row in selected
                    )
                    draw_den = sum(row["truth_sst"] for row in selected)
                    bootstrap.append(
                        max(0.0, float(draw_num))
                        / max(float(draw_den), np.finfo(float).eps)
                    )
                upper = float(np.quantile(bootstrap, float(confidence)))
                audits.append(
                    {
                        "source": source,
                        "horizon_seconds": horizon,
                        "point_degradation": point,
                        "bootstrap_upper": upper,
                        "fault_lines": len(groups),
                        "events": len(local),
                        "passes": bool(point <= tolerance and upper <= tolerance),
                    }
                )
        return {
            "rows": audits,
            "passes": bool(not failures and audits and all(row["passes"] for row in audits)),
            "failures": failures,
            "maximum_point_degradation": max(
                [row["point_degradation"] for row in audits], default=float("inf")
            ),
            "maximum_bootstrap_upper": max(
                [row["bootstrap_upper"] for row in audits], default=float("inf")
            ),
            "bootstrap_draws": int(bootstrap_draws),
            "confidence": float(confidence),
            "normalization": "positive_SSE_increase_over_truth_SST",
        }

    def fit(
        self, train: list[dict[str, Any]], val: list[dict[str, Any]], run_context: RunContext
    ) -> FitResult:
        if not train or not val:
            raise ValueError("prunable model requires nonempty fit and selection data")
        self.validation_scales = self._training_scales(train)
        self.max_internal_step = float(run_context.max_internal_step)
        matrix, vector, system = self.build_weak_system(train, run_context)
        n_nodes = len(train[0]["model_inputs"]["static"]["mass"])
        n_edges = train[0]["model_inputs"]["static"]["incidence"].shape[1]
        scale = np.sqrt(np.mean(matrix**2, axis=0))
        scale = np.where(scale > 1e-14, scale, 1.0)
        normalized = matrix / scale
        gram = normalized.T @ normalized
        normal_rhs = normalized.T @ vector
        normal_system = (scale, gram, normal_rhs)
        candidates = []
        for ridge in run_context.ridge_values:
            theta, diagnostics = self._solve(
                matrix, vector, float(ridge), normal_system=normal_system
            )
            backbone_parameters = n_nodes + n_edges
            active_mask = np.ones(matrix.shape[1], dtype=bool)
            if matrix.shape[1] > backbone_parameters:
                optional = np.asarray(theta[backbone_parameters:], dtype=float)
                floor = max(float(np.max(optional)), 1.0) * 1e-8
                active_mask[backbone_parameters:] = optional > floor
            active_gram = gram[np.ix_(active_mask, active_mask)]
            active_eigenvalues = np.maximum(
                np.linalg.eigvalsh(0.5 * (active_gram + active_gram.T)), 0.0
            )
            active_positive = active_eigenvalues[
                active_eigenvalues
                > np.max(active_eigenvalues, initial=0.0) * 1e-12
            ]
            diagnostics["effective_rank"] = len(active_positive)
            diagnostics["condition_number"] = (
                float(np.sqrt(np.max(active_positive) / np.min(active_positive)))
                if len(active_positive)
                else float("inf")
            )
            diagnostics["active_parameter_count"] = int(np.sum(active_mask))
            diagnostics["screened_parameter_count"] = int(np.sum(~active_mask))
            diagnostics["equation_parameter_ratio"] = float(
                matrix.shape[0] / max(1, np.sum(active_mask))
            )
            diagnostics["identifiability_pass"] = bool(
                diagnostics["effective_rank"] >= int(np.sum(active_mask))
                and diagnostics["equation_parameter_ratio"]
                >= self.minimum_equation_parameter_ratio
                and diagnostics["condition_number"] <= self.maximum_condition_number
            )
            self._assign_parameters(theta, n_nodes, n_edges)
            summary = self.validation_summary(val, run_context)
            candidates.append(
                (
                    summary["standardized_loss"],
                    float(ridge),
                    theta,
                    diagnostics,
                    summary,
                    active_mask,
                )
            )
        eligible = [row for row in candidates if row[3]["identifiability_pass"]]
        if self.enforce_identifiability and not eligible:
            best = min(candidates, key=lambda row: row[0])
            raise ValueError(
                "identifiability gate failed: "
                f"rank={best[3]['effective_rank']}/{best[3]['active_parameter_count']}, "
                f"equation_ratio={best[3]['equation_parameter_ratio']:.3g}, "
                f"condition={best[3]['condition_number']:.3g}"
            )
        loss, ridge, theta, diagnostics, summary, active_mask = min(
            eligible or candidates, key=lambda row: row[0]
        )
        if np.any(~active_mask):
            refit, refit_diagnostics = self._solve(
                matrix[:, active_mask], vector, float(ridge)
            )
            screened_theta = np.zeros_like(theta)
            screened_theta[active_mask] = refit
            theta = screened_theta
            diagnostics["unbiased_active_refit"] = refit_diagnostics
        self._assign_parameters(theta, n_nodes, n_edges)
        self.selected_ridge = ridge
        instance_norms = {
            name: float(np.linalg.norm(theta[section]))
            for name, section in self._parameter_layout(n_nodes, n_edges).items()
            if name not in {"all", "node_damping", "coupling"}
        }
        block_norms = {
            family: float(
                np.sqrt(
                    sum(
                        instance_norms[row.instance_id] ** 2
                        for row in self._instances(family)
                    )
                )
            )
            for family in self.active_blocks
        }
        self.diagnostics = {
            **system,
            **diagnostics,
            "selected_ridge": ridge,
            "active_blocks": list(self.active_blocks),
            "active_block_instances": [row.to_dict() for row in self.active_instances],
            "block_parameter_norms": block_norms,
            "instance_parameter_norms": instance_norms,
            "candidate_validation_losses": {str(row[1]): float(row[0]) for row in candidates},
            "validation": summary,
            "passive_reciprocity_by_construction": True,
            "reciprocal_port_ranks": {
                name: [int(np.linalg.matrix_rank(factor)) for factor in factors]
                if isinstance(factors, list)
                else int(np.linalg.matrix_rank(factors))
                for name, factors in self.reciprocal_connections().items()
            },
            "port_realizations": self._port_realization_diagnostics(n_edges),
        }
        return FitResult(
            self.method_name,
            "completed",
            tuple(sample["event_id"] for sample in train[: run_context.max_cases]),
            tuple(sample["event_id"] for sample in val[: run_context.max_cases]),
            float(loss),
            dict(self.diagnostics),
        )

    def _refresh_parameter_diagnostics(self) -> None:
        """Keep diagnostic norms synchronized after group-scale refinement."""

        _, n_edges = self._require_fitted()
        instance_norms: dict[str, float] = {}
        family_values = {
            NONLINEAR_POTENTIAL: self.potential_coefficients,
            EDGE_DAMPING: self.edge_damping_coefficients,
            NODE_HIGH_ORDER: self.node_high_order_gain,
        }
        for family, value in family_values.items():
            instances = self._instances(family)
            if instances:
                instance_norms[instances[0].instance_id] = float(np.linalg.norm(value))
        for family in (EDGE_MEMORY, RESONANT_MEMORY):
            for index, instance in enumerate(self._instances(family)):
                factor = self._port_realization(index, family, n_edges).factor
                instance_norms[instance.instance_id] = float(np.linalg.norm(factor @ factor.T))
        self.diagnostics["instance_parameter_norms"] = instance_norms
        self.diagnostics["block_parameter_norms"] = {
            family: float(
                np.sqrt(
                    sum(
                        instance_norms.get(instance.instance_id, 0.0) ** 2
                        for instance in self._instances(family)
                    )
                )
            )
            for family in self.active_blocks
        }
        self.diagnostics["reciprocal_port_ranks"] = {
            name: [int(factor.shape[1]) for factor in factors]
            if isinstance(factors, list)
            else int(np.linalg.matrix_rank(factors))
            for name, factors in self.reciprocal_connections().items()
        }
        self.diagnostics["port_realizations"] = self._port_realization_diagnostics(n_edges)
        self.diagnostics["parameter_norms_updated_after_refinement"] = True

    def refine_closed_loop(
        self,
        samples: Sequence[dict[str, Any]],
        context: RunContext,
        *,
        maximum_cases: int = 4,
        maximum_evaluations: int = 16,
    ) -> dict[str, Any]:
        """Calibrate all fitted block groups against closed-loop reintegration.

        The weak-form solve has already re-estimated every scalar coefficient.
        This second pass jointly adjusts one positive scale per complete block,
        which preserves every passivity constraint and keeps refinement small
        enough to repeat after each deletion trial.
        """

        self._require_fitted()
        if not samples or maximum_cases <= 0 or maximum_evaluations <= 0:
            return {"status": "skipped", "reason": "no_refinement_budget"}
        count = min(int(maximum_cases), len(samples))
        indices = np.unique(
            np.linspace(0, len(samples) - 1, count).round().astype(int)
        )
        selected = [samples[int(index)] for index in indices]
        names = ["node_damping", "coupling"]
        block_arrays = {
            NONLINEAR_POTENTIAL: "potential_coefficients",
            EDGE_DAMPING: "edge_damping_coefficients",
            NODE_HIGH_ORDER: "node_high_order_gain",
            EDGE_MEMORY: "memory_port_operator",
            RESONANT_MEMORY: "resonant_port_operator",
        }
        names.extend(block_arrays[block] for block in self.active_blocks)
        ordinary_names = [
            name
            for name in names
            if name not in {"memory_port_operator", "resonant_port_operator"}
        ]
        reference = {
            name: np.asarray(getattr(self, name), dtype=float).copy()
            for name in ordinary_names
        }
        reference_memory_factors = [factor.copy() for factor in self.memory_factors]
        reference_resonant_factors = [factor.copy() for factor in self.resonant_factors]
        reference_memory_gains = np.asarray(self.memory_gains).copy()
        reference_resonant_gains = np.asarray(self.resonant_gains).copy()

        residual_sizes = []
        for sample in selected:
            target = sample["training_context"]["targets"]
            steps = min(len(target["t"]), context.horizon_steps)
            n_edges = sample["model_inputs"]["static"]["incidence"].shape[1]
            n_nodes = len(sample["model_inputs"]["static"]["mass"])
            residual_sizes.append((steps, n_edges, n_nodes))

        def assign(log_scales: np.ndarray) -> None:
            for name, log_scale in zip(names, log_scales, strict=True):
                scale = np.exp(float(log_scale))
                if name == "memory_port_operator":
                    self.memory_factors = [
                        factor * np.sqrt(scale) for factor in reference_memory_factors
                    ]
                    self.memory_gains = reference_memory_gains * scale
                elif name == "resonant_port_operator":
                    self.resonant_factors = [
                        factor * np.sqrt(scale) for factor in reference_resonant_factors
                    ]
                    self.resonant_gains = reference_resonant_gains * scale
                else:
                    setattr(self, name, reference[name] * scale)

        def residual(log_scales: np.ndarray) -> np.ndarray:
            assign(log_scales)
            rows = []
            for sample, (steps, n_edges, n_nodes) in zip(
                selected, residual_sizes, strict=True
            ):
                static, time, exogenous = sample_rollout_inputs(sample)
                target = sample["training_context"]["targets"]
                try:
                    delta, omega = self.rollout(
                        static,
                        time[:steps],
                        {
                            "Pm": exogenous["Pm"][:steps],
                            "interpolation": exogenous["interpolation"],
                        },
                    )
                    edge_error = (
                        delta - np.asarray(target["delta"][:steps], dtype=float)
                    ) @ np.asarray(static["incidence"], dtype=float)
                    omega_error = omega - np.asarray(target["omega"][:steps], dtype=float)
                    normalizer = np.sqrt(max(1, steps * (n_edges + n_nodes) * len(selected)))
                    rows.extend(
                        (
                            (edge_error / self.validation_scales["edge_angle"] / normalizer).reshape(-1),
                            (omega_error / self.validation_scales["omega"] / normalizer).reshape(-1),
                        )
                    )
                except (FloatingPointError, ValueError):
                    rows.append(np.full(steps * (n_edges + n_nodes), 1e3))
            return np.concatenate(rows)

        initial = np.zeros(len(names), dtype=float)
        baseline = residual(initial)
        result = least_squares(
            residual,
            initial,
            bounds=(np.log(0.25), np.log(4.0)),
            max_nfev=int(maximum_evaluations),
        )
        refined = residual(result.x)
        accepted = bool(
            result.success
            and np.isfinite(refined).all()
            and float(refined @ refined) <= float(baseline @ baseline)
        )
        if not accepted:
            assign(initial)
        audit = {
            "status": "accepted" if accepted else "rejected",
            "case_ids": [sample["event_id"] for sample in selected],
            "group_scales": {
                name: float(np.exp(value))
                for name, value in zip(names, result.x if accepted else initial, strict=True)
            },
            "baseline_loss": float(baseline @ baseline),
            "refined_loss": float(refined @ refined),
            "solver_evaluations": int(result.nfev),
            "solver_success": bool(result.success),
            "all_coefficients_refit_by_weak_form_before_group_calibration": True,
            "passivity_preserved": True,
        }
        self.diagnostics["closed_loop_refinement"] = audit
        self._refresh_parameter_diagnostics()
        return audit

    def _hidden_shapes(self, n_nodes: int, n_edges: int) -> dict[str, tuple[int, ...]]:
        memory_size = sum(
            self._memory_factor(index, n_edges).shape[1]
            for index in range(len(self._instances(EDGE_MEMORY)))
        )
        resonance_size = sum(
            self._resonant_factor(index, n_edges).shape[1]
            for index in range(len(self._instances(RESONANT_MEMORY)))
        )
        return {
            "node": (n_nodes,) if NODE_HIGH_ORDER in self.active_blocks else (0,),
            "memory": (memory_size,),
            "resonance_q": (resonance_size,),
            "resonance_p": (resonance_size,),
        }

    def _split_port_state(
        self, vector: np.ndarray, family: str, n_edges: int
    ) -> list[np.ndarray]:
        factors = (
            [self._memory_factor(index, n_edges) for index in range(len(self._instances(family)))]
            if family == EDGE_MEMORY
            else [
                self._resonant_factor(index, n_edges)
                for index in range(len(self._instances(family)))
            ]
        )
        output: list[np.ndarray] = []
        cursor = 0
        flat = np.asarray(vector, dtype=float).reshape(-1)
        for factor in factors:
            rank = factor.shape[1]
            output.append(flat[cursor : cursor + rank])
            cursor += rank
        if cursor != len(flat):
            raise ValueError("port state does not match reciprocal factor ranks")
        return output

    def initial_hidden(
        self, omega: np.ndarray, static: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        _, n_edges = self._require_fitted()
        speed = np.asarray(static["incidence"], dtype=float).T @ np.asarray(omega, dtype=float)
        node_root = np.sqrt(np.maximum(self.node_high_order_gain, 0.0))
        node = node_root * np.asarray(omega) if NODE_HIGH_ORDER in self.active_blocks else np.zeros(0)
        memory = (
            np.concatenate(
                [
                    self._memory_factor(index, n_edges).T @ speed
                    for index in range(len(self._instances(EDGE_MEMORY)))
                ]
            )
            if EDGE_MEMORY in self.active_blocks
            else np.zeros(0)
        )
        if RESONANT_MEMORY in self.active_blocks:
            positions = []
            for index, instance in enumerate(self._instances(RESONANT_MEMORY)):
                angular = 2 * np.pi * float(instance.frequency_hz)
                positions.append(self._resonant_factor(index, n_edges).T @ speed / angular**2)
            q = np.concatenate(positions)
            p = np.zeros_like(q)
        else:
            q = p = np.zeros(0)
        return {"node": node, "memory": memory, "resonance_q": q, "resonance_p": p}

    @staticmethod
    def _pack_hidden(hidden: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate([np.asarray(hidden[name]).reshape(-1) for name in (
            "node", "memory", "resonance_q", "resonance_p"
        )])

    def _unpack_hidden(self, vector: np.ndarray, n_nodes: int, n_edges: int) -> dict[str, np.ndarray]:
        shapes = self._hidden_shapes(n_nodes, n_edges)
        cursor = 0
        output = {}
        for name in ("node", "memory", "resonance_q", "resonance_p"):
            size = int(np.prod(shapes[name]))
            output[name] = np.asarray(vector[cursor : cursor + size]).reshape(shapes[name])
            cursor += size
        if cursor != len(vector):
            raise ValueError("hidden vector size does not match active blocks")
        return output

    def _edge_forces(self, delta: np.ndarray, omega: np.ndarray, static: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        incidence = np.asarray(static["incidence"], dtype=float)
        equilibrium = np.asarray(static["equilibrium_delta"], dtype=float)
        eta = incidence.T @ delta
        eta0 = incidence.T @ equilibrium
        theta = eta - eta0
        nu = incidence.T @ omega
        conservative = self.coupling * (np.sin(eta) - np.sin(eta0))
        if NONLINEAR_POTENTIAL in self.active_blocks:
            conservative = conservative + (
                self.potential_coefficients[0] * theta
                + self.potential_coefficients[1] * theta**3
                + self.potential_coefficients[2] * np.tanh(theta / self.potential_scale)
            )
        dissipative = np.zeros_like(nu)
        if EDGE_DAMPING in self.active_blocks:
            dissipative = (
                self.edge_damping_coefficients[0] * nu
                + self.edge_damping_coefficients[1] * nu * np.abs(nu)
                + self.edge_damping_coefficients[2] * nu * theta**2
            )
        return conservative, dissipative, nu

    def vector_field_augmented(
        self,
        delta: np.ndarray,
        omega: np.ndarray,
        hidden: dict[str, np.ndarray],
        pm: np.ndarray,
        static: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        self._require_fitted()
        mass = np.asarray(static["mass"], dtype=float)
        incidence = np.asarray(static["incidence"], dtype=float)
        conservative, dissipative, nu = self._edge_forces(delta, omega, static)
        force = (
            np.asarray(pm, dtype=float).reshape(-1)
            - _pm_equilibrium(static)
            - self.node_damping * omega
            - incidence @ (conservative + dissipative)
        )
        derivatives = {name: np.zeros_like(value) for name, value in hidden.items()}
        if NODE_HIGH_ORDER in self.active_blocks:
            root_gain = np.sqrt(np.maximum(self.node_high_order_gain, 0.0))
            force = force - root_gain * hidden["node"]
            derivatives["node"] = (-hidden["node"] + root_gain * omega) / self.node_tau
        if EDGE_MEMORY in self.active_blocks:
            states = self._split_port_state(hidden["memory"], EDGE_MEMORY, incidence.shape[1])
            state_derivatives = []
            edge_force = np.zeros(incidence.shape[1])
            for index, (instance, state) in enumerate(
                zip(self._instances(EDGE_MEMORY), states, strict=True)
            ):
                factor = self._memory_factor(index, incidence.shape[1])
                edge_force += factor @ state
                state_derivatives.append(
                    (-state + factor.T @ nu) / float(instance.time_constant)
                )
            force = force - incidence @ edge_force
            derivatives["memory"] = np.concatenate(state_derivatives)
        if RESONANT_MEMORY in self.active_blocks:
            q_states = self._split_port_state(hidden["resonance_q"], RESONANT_MEMORY, incidence.shape[1])
            p_states = self._split_port_state(hidden["resonance_p"], RESONANT_MEMORY, incidence.shape[1])
            q_derivatives = []
            p_derivatives = []
            edge_force = np.zeros(incidence.shape[1])
            for index, (instance, q_state, p_state) in enumerate(
                zip(self._instances(RESONANT_MEMORY), q_states, p_states, strict=True)
            ):
                factor = self._resonant_factor(index, incidence.shape[1])
                angular = 2 * np.pi * float(instance.frequency_hz)
                damping = float(instance.damping_ratio)
                edge_force += factor @ p_state
                q_derivatives.append(p_state)
                p_derivatives.append(
                    factor.T @ nu
                    - angular**2 * q_state
                    - 2 * damping * angular * p_state
                )
            force = force - incidence @ edge_force
            derivatives["resonance_q"] = np.concatenate(q_derivatives)
            derivatives["resonance_p"] = np.concatenate(p_derivatives)
        return omega, coi_project_power(force, mass) / mass, derivatives

    def _flat_field(
        self,
        delta: np.ndarray,
        omega: np.ndarray,
        hidden_vector: np.ndarray,
        pm: np.ndarray,
        static: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_nodes, n_edges = self._require_fitted()
        hidden = self._unpack_hidden(hidden_vector, n_nodes, n_edges)
        dd, dw, dh = self.vector_field_augmented(delta, omega, hidden, pm, static)
        return dd, dw, self._pack_hidden(dh)

    def rollout_augmented(
        self,
        initial_state: dict[str, np.ndarray],
        time_grid: np.ndarray,
        exogenous: dict[str, np.ndarray],
        *,
        max_step: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._require_fitted()
        time = np.asarray(time_grid, dtype=float)
        pm = np.asarray(exogenous["Pm"], dtype=float)
        if len(time) != len(pm) or len(time) < 1:
            raise ValueError("time and Pm schedules must have equal positive length")
        mass = np.asarray(initial_state["mass"], dtype=float)
        delta = coi_project(np.asarray(initial_state["delta0"]).reshape(-1), mass)
        omega = coi_project(np.asarray(initial_state["omega0"]).reshape(-1), mass)
        hidden = self._pack_hidden(self.initial_hidden(omega, initial_state))
        output_delta, output_omega, output_hidden = [delta.copy()], [omega.copy()], [hidden.copy()]
        step_limit = self.max_internal_step if max_step is None else float(max_step)
        requested = exogenous.get("interpolation", "auto")
        mode = (
            "zoh"
            if requested == "zoh" or (requested == "auto" and np.allclose(pm, pm[[0]]))
            else "linear"
        )
        for index, interval in enumerate(np.diff(time)):
            if interval <= 0:
                raise ValueError("rollout time must be strictly increasing")
            substeps = max(1, int(np.ceil(float(interval) / step_limit)))
            step = float(interval) / substeps
            pm_left = pm[index]
            pm_right = pm[index + 1]
            for substep in range(substeps):
                q0 = substep / substeps
                qh = (substep + 0.5) / substeps
                q1 = (substep + 1.0) / substeps
                input0 = pm_left if mode == "zoh" else pm_left + q0 * (pm_right - pm_left)
                input_half = (
                    pm_left if mode == "zoh" else pm_left + qh * (pm_right - pm_left)
                )
                input1 = pm_left if mode == "zoh" else pm_left + q1 * (pm_right - pm_left)

                state = (delta, omega, hidden)
                k1 = self._flat_field(*state, input0, initial_state)
                k2 = self._flat_field(
                    *(value + 0.5 * step * slope for value, slope in zip(state, k1, strict=True)),
                    input_half,
                    initial_state,
                )
                k3 = self._flat_field(
                    *(value + 0.5 * step * slope for value, slope in zip(state, k2, strict=True)),
                    input_half,
                    initial_state,
                )
                k4 = self._flat_field(
                    *(value + step * slope for value, slope in zip(state, k3, strict=True)),
                    input1,
                    initial_state,
                )
                delta, omega, hidden = tuple(
                    value + step * (a + 2 * b + 2 * c + d) / 6.0
                    for value, a, b, c, d in zip(state, k1, k2, k3, k4, strict=True)
                )
                delta = coi_project(delta, mass)
                omega = coi_project(omega, mass)
                if not all(np.isfinite(value).all() for value in (delta, omega, hidden)):
                    raise FloatingPointError("nonfinite prunable-model rollout")
            output_delta.append(delta.copy())
            output_omega.append(omega.copy())
            output_hidden.append(hidden.copy())
        return np.stack(output_delta), np.stack(output_omega), np.stack(output_hidden)

    def rollout(
        self,
        initial_state: dict[str, np.ndarray],
        time_grid: np.ndarray,
        exogenous: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        delta, omega, _ = self.rollout_augmented(initial_state, time_grid, exogenous)
        return delta, omega

    def _potential_value(
        self, delta: np.ndarray, static: dict[str, np.ndarray], pm_clear: np.ndarray
    ) -> float:
        incidence = np.asarray(static["incidence"], dtype=float)
        equilibrium = np.asarray(static["equilibrium_delta"], dtype=float)
        eta = incidence.T @ np.asarray(delta, dtype=float)
        eta0 = incidence.T @ equilibrium
        theta = eta - eta0
        shifted = np.cos(eta0) - np.cos(eta) - np.sin(eta0) * theta
        value = float(np.sum(self.coupling * shifted))
        if NONLINEAR_POTENTIAL in self.active_blocks:
            log_cosh = np.logaddexp(theta / self.potential_scale, -theta / self.potential_scale) - np.log(2.0)
            value += float(
                np.sum(
                    0.5 * self.potential_coefficients[0] * theta**2
                    + 0.25 * self.potential_coefficients[1] * theta**4
                    + self.potential_coefficients[2] * self.potential_scale * log_cosh
                )
            )
        dpm = coi_project_power(np.asarray(pm_clear).reshape(-1) - _pm_equilibrium(static), np.asarray(static["mass"]))
        return value - float(dpm @ np.asarray(delta))

    def storage(
        self,
        delta: np.ndarray,
        omega: np.ndarray,
        static: dict[str, np.ndarray],
        hidden: dict[str, np.ndarray] | np.ndarray | None = None,
        *,
        pm_clear: np.ndarray | None = None,
        reference_delta: np.ndarray | None = None,
    ) -> float:
        n_nodes, n_edges = self._require_fitted()
        if hidden is None:
            unpacked = {name: np.zeros(shape) for name, shape in self._hidden_shapes(n_nodes, n_edges).items()}
        elif isinstance(hidden, dict):
            unpacked = hidden
        else:
            unpacked = self._unpack_hidden(np.asarray(hidden), n_nodes, n_edges)
        clear = (
            np.asarray(static.get("Pm_clear", static["Pm_equilibrium"]), dtype=float).reshape(-1)
            if pm_clear is None
            else np.asarray(pm_clear, dtype=float).reshape(-1)
        )
        potential = self._potential_value(delta, static, clear)
        if reference_delta is not None:
            potential -= self._potential_value(reference_delta, static, clear)
        value = potential + 0.5 * float(np.sum(np.asarray(static["mass"]) * np.asarray(omega) ** 2))
        if NODE_HIGH_ORDER in self.active_blocks:
            value += 0.5 * self.node_tau * float(np.sum(unpacked["node"] ** 2))
        if EDGE_MEMORY in self.active_blocks:
            memory_states = self._split_port_state(unpacked["memory"], EDGE_MEMORY, n_edges)
            value += 0.5 * float(
                sum(
                    float(instance.time_constant) * float(np.sum(state**2))
                    for instance, state in zip(
                        self._instances(EDGE_MEMORY), memory_states, strict=True
                    )
                )
            )
        if RESONANT_MEMORY in self.active_blocks:
            q_states = self._split_port_state(unpacked["resonance_q"], RESONANT_MEMORY, n_edges)
            p_states = self._split_port_state(unpacked["resonance_p"], RESONANT_MEMORY, n_edges)
            value += 0.5 * float(
                sum(
                    float(np.sum(p_state**2))
                    + (2 * np.pi * float(instance.frequency_hz)) ** 2
                    * float(np.sum(q_state**2))
                    for instance, q_state, p_state in zip(
                        self._instances(RESONANT_MEMORY), q_states, p_states, strict=True
                    )
                )
            )
        return float(value)

    def dissipation_rate(
        self,
        delta: np.ndarray,
        omega: np.ndarray,
        hidden: dict[str, np.ndarray] | np.ndarray,
        static: dict[str, np.ndarray],
    ) -> float:
        n_nodes, n_edges = self._require_fitted()
        unpacked = hidden if isinstance(hidden, dict) else self._unpack_hidden(hidden, n_nodes, n_edges)
        _, edge_damping, nu = self._edge_forces(delta, omega, static)
        value = float(omega @ (self.node_damping * omega) + nu @ edge_damping)
        if NODE_HIGH_ORDER in self.active_blocks:
            value += float(np.sum(unpacked["node"] ** 2))
        if EDGE_MEMORY in self.active_blocks:
            value += float(np.sum(unpacked["memory"] ** 2))
        if RESONANT_MEMORY in self.active_blocks:
            p_states = self._split_port_state(
                unpacked["resonance_p"], RESONANT_MEMORY, n_edges
            )
            value += float(
                sum(
                    2
                    * float(instance.damping_ratio)
                    * (2 * np.pi * float(instance.frequency_hz))
                    * float(np.sum(state**2))
                    for instance, state in zip(
                        self._instances(RESONANT_MEMORY), p_states, strict=True
                    )
                )
            )
        return value

    def supply_identity_error(
        self,
        static: dict[str, np.ndarray],
        *,
        delta: np.ndarray | None = None,
        omega: np.ndarray | None = None,
        hidden: dict[str, np.ndarray] | None = None,
        pm: np.ndarray | None = None,
        step: float = 1e-6,
    ) -> float:
        """Numerically audit the structural storage identity at one state."""

        mass = np.asarray(static["mass"], dtype=float)
        state_delta = coi_project(
            np.asarray(static["delta0"] if delta is None else delta, dtype=float).reshape(-1),
            mass,
        )
        state_omega = coi_project(
            np.asarray(static["omega0"] if omega is None else omega, dtype=float).reshape(-1),
            mass,
        )
        state_hidden = self.initial_hidden(state_omega, static) if hidden is None else hidden
        fixed_pm = (
            np.asarray(static.get("Pm_clear", static["Pm_equilibrium"]), dtype=float).reshape(-1)
            if pm is None
            else np.asarray(pm, dtype=float).reshape(-1)
        )
        dd, dw, dh = self.vector_field_augmented(
            state_delta, state_omega, state_hidden, fixed_pm, static
        )
        packed = self._pack_hidden(state_hidden)
        packed_derivative = self._pack_hidden(dh)
        epsilon = float(step)
        forward = self.storage(
            state_delta + epsilon * dd,
            state_omega + epsilon * dw,
            static,
            packed + epsilon * packed_derivative,
            pm_clear=fixed_pm,
        )
        backward = self.storage(
            state_delta - epsilon * dd,
            state_omega - epsilon * dw,
            static,
            packed - epsilon * packed_derivative,
            pm_clear=fixed_pm,
        )
        rate = (forward - backward) / (2.0 * epsilon)
        expected = -self.dissipation_rate(
            state_delta, state_omega, state_hidden, static
        )
        return float(abs(rate - expected))

    def solve_postclear_equilibrium(
        self, static: dict[str, np.ndarray], pm_clear: np.ndarray | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        mass = np.asarray(static["mass"], dtype=float)
        basis = coi_basis(mass)
        initial = coi_project(np.asarray(static["equilibrium_delta"], dtype=float), mass)
        clear = (
            np.asarray(static.get("Pm_clear", static["Pm_equilibrium"]), dtype=float).reshape(-1)
            if pm_clear is None
            else np.asarray(pm_clear, dtype=float).reshape(-1)
        )

        def residual(coordinates: np.ndarray) -> np.ndarray:
            delta = initial + basis @ coordinates
            conservative, _, _ = self._edge_forces(delta, np.zeros_like(delta), static)
            force = clear - _pm_equilibrium(static) - np.asarray(static["incidence"]) @ conservative
            return basis.T @ force

        result = least_squares(residual, np.zeros(basis.shape[1]), max_nfev=2000)
        equilibrium = coi_project(initial + basis @ result.x, mass)
        return equilibrium, {
            "success": bool(result.success),
            "residual_inf_norm": float(np.linalg.norm(residual(result.x), ord=np.inf)),
            "iterations": int(result.nfev),
        }

    def storage_hessian(
        self, delta: np.ndarray, static: dict[str, np.ndarray]
    ) -> np.ndarray:
        incidence = np.asarray(static["incidence"], dtype=float)
        equilibrium = np.asarray(static["equilibrium_delta"], dtype=float)
        eta = incidence.T @ np.asarray(delta)
        eta0 = incidence.T @ equilibrium
        theta = eta - eta0
        stiffness = self.coupling * np.cos(eta)
        if NONLINEAR_POTENTIAL in self.active_blocks:
            stiffness = stiffness + (
                self.potential_coefficients[0]
                + 3 * self.potential_coefficients[1] * theta**2
                + self.potential_coefficients[2]
                / self.potential_scale
                / np.cosh(theta / self.potential_scale) ** 2
            )
        return incidence @ np.diag(stiffness) @ incidence.T

    def critical_energy_level(
        self,
        static: dict[str, np.ndarray],
        *,
        pm_clear: np.ndarray | None = None,
        angle_limit: float = np.pi / 2,
    ) -> dict[str, Any]:
        mass = np.asarray(static["mass"], dtype=float)
        incidence = np.asarray(static["incidence"], dtype=float)
        basis = coi_basis(mass)
        equilibrium, equilibrium_audit = self.solve_postclear_equilibrium(static, pm_clear)
        clear = (
            np.asarray(static.get("Pm_clear", static["Pm_equilibrium"]), dtype=float).reshape(-1)
            if pm_clear is None
            else np.asarray(pm_clear, dtype=float).reshape(-1)
        )

        def objective(coordinates: np.ndarray) -> float:
            delta = equilibrium + basis @ coordinates
            return self.storage(
                delta, np.zeros_like(delta), static, pm_clear=clear, reference_delta=equilibrium
            )

        candidates = []
        eta_equilibrium = incidence.T @ equilibrium
        for edge in range(incidence.shape[1]):
            direction = incidence[:, edge] @ basis
            if np.linalg.norm(direction) <= 1e-12:
                continue
            for sign in (-1.0, 1.0):
                target = sign * float(angle_limit)
                constraint = {
                    "type": "eq",
                    "fun": lambda value, edge=edge, target=target: float(
                        incidence[:, edge] @ (equilibrium + basis @ value) - target
                    ),
                }
                distance = target - eta_equilibrium[edge]
                start = distance * direction / max(float(direction @ direction), 1e-12)
                result = minimize(objective, start, method="SLSQP", constraints=[constraint])
                if result.success and np.isfinite(result.fun):
                    candidates.append({"kind": "angle_boundary", "edge": edge, "energy": float(result.fun)})

        def stationary(coordinates: np.ndarray) -> np.ndarray:
            delta = equilibrium + basis @ coordinates
            conservative, _, _ = self._edge_forces(delta, np.zeros_like(delta), static)
            force = clear - _pm_equilibrium(static) - incidence @ conservative
            return basis.T @ force

        for edge in range(incidence.shape[1]):
            direction = basis.T @ incidence[:, edge]
            if np.linalg.norm(direction) <= 1e-12:
                continue
            for sign in (-1.0, 1.0):
                start = sign * np.pi * direction / max(float(direction @ direction), 1e-12)
                result = root(stationary, start)
                if not result.success or np.linalg.norm(stationary(result.x), ord=np.inf) > 1e-7:
                    continue
                delta = coi_project(equilibrium + basis @ result.x, mass)
                hessian = basis.T @ self.storage_hessian(delta, static) @ basis
                energy = objective(result.x)
                if energy > 1e-12 and np.min(np.linalg.eigvalsh(hessian)) < -1e-8:
                    candidates.append({"kind": "adjacent_saddle", "edge": edge, "energy": float(energy)})
        positive = [row for row in candidates if row["energy"] > 0.0]
        level = min((row["energy"] for row in positive), default=float("nan"))
        return {
            "certified": bool(np.isfinite(level) and equilibrium_audit["residual_inf_norm"] <= 1e-7),
            "critical_energy": float(level),
            "equilibrium_delta": equilibrium.tolist(),
            "equilibrium_audit": equilibrium_audit,
            "candidates": candidates,
            "angle_limit": float(angle_limit),
            "sample_based": False,
            "boundary_and_adjacent_saddle_search": True,
        }

    def stability_margin(
        self,
        static: dict[str, np.ndarray],
        *,
        pm_clear: np.ndarray | None = None,
        angle_limit: float = np.pi / 2,
    ) -> dict[str, Any]:
        certificate = self.critical_energy_level(static, pm_clear=pm_clear, angle_limit=angle_limit)
        equilibrium = np.asarray(certificate["equilibrium_delta"], dtype=float)
        omega0 = np.asarray(static["omega0"], dtype=float).reshape(-1)
        hidden = self.initial_hidden(omega0, static)
        clear = (
            np.asarray(static.get("Pm_clear", static["Pm_equilibrium"]), dtype=float).reshape(-1)
            if pm_clear is None
            else np.asarray(pm_clear, dtype=float).reshape(-1)
        )
        clearing = self.storage(
            np.asarray(static["delta0"], dtype=float).reshape(-1),
            omega0,
            static,
            hidden,
            pm_clear=clear,
            reference_delta=equilibrium,
        )
        margin = float(certificate["critical_energy"] - clearing) if certificate["certified"] else float("nan")
        return {
            **certificate,
            "storage_at_clearing": float(clearing),
            "margin": margin,
            "predicted_stable": bool(np.isfinite(margin) and margin > 0.0),
            "integration_free": True,
        }

    def linearize(
        self, static: dict[str, np.ndarray], *, pm: np.ndarray | None = None, relative_step: float = 1e-6
    ) -> tuple[np.ndarray, dict[str, Any]]:
        mass = np.asarray(static["mass"], dtype=float)
        basis = coi_basis(mass)
        equilibrium = np.asarray(static["equilibrium_delta"], dtype=float)
        zero_speed = np.zeros_like(mass)
        hidden = self._pack_hidden(self.initial_hidden(zero_speed, static))
        quotient = basis.shape[1]
        input_pm = _pm_equilibrium(static) if pm is None else np.asarray(pm)
        state0 = np.zeros(2 * quotient + len(hidden))

        def field(state: np.ndarray) -> np.ndarray:
            delta = equilibrium + basis @ state[:quotient]
            omega = basis @ state[quotient : 2 * quotient]
            dd, dw, dh = self._flat_field(delta, omega, state[2 * quotient :], input_pm, static)
            return np.concatenate((basis.T @ dd, basis.T @ dw, dh))

        residual = field(state0)
        jacobian = np.empty((len(state0), len(state0)), dtype=float)
        for column in range(len(state0)):
            step = float(relative_step) * max(1.0, abs(state0[column]))
            offset = np.zeros_like(state0)
            offset[column] = step
            jacobian[:, column] = (field(state0 + offset) - field(state0 - offset)) / (2 * step)
        return jacobian, {
            "equilibrium_residual_inf_norm": float(np.linalg.norm(residual, ord=np.inf)),
            "state_dimension": len(state0),
            "quotient_dimension": int(quotient),
        }

    def _modal_energy_hessian(
        self, static: dict[str, np.ndarray]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Storage Hessian in the exact quotient/hidden coordinates of ``linearize``."""

        mass = np.asarray(static["mass"], dtype=float)
        basis = coi_basis(mass)
        quotient = basis.shape[1]
        _, n_edges = self._require_fitted()
        hidden_shapes = self._hidden_shapes(len(mass), n_edges)
        hidden_size = sum(int(np.prod(shape)) for shape in hidden_shapes.values())
        hessian = np.zeros((2 * quotient + hidden_size, 2 * quotient + hidden_size))
        equilibrium = np.asarray(static["equilibrium_delta"], dtype=float)
        potential = self.storage_hessian(equilibrium, static)
        hessian[:quotient, :quotient] = basis.T @ potential @ basis
        hessian[quotient : 2 * quotient, quotient : 2 * quotient] = (
            basis.T @ np.diag(mass) @ basis
        )
        cursor = 2 * quotient
        if NODE_HIGH_ORDER in self.active_blocks:
            size = hidden_shapes["node"][0]
            hessian[cursor : cursor + size, cursor : cursor + size] = self.node_tau * np.eye(size)
            cursor += size
        if EDGE_MEMORY in self.active_blocks:
            for index, instance in enumerate(self._instances(EDGE_MEMORY)):
                size = self._memory_factor(index, n_edges).shape[1]
                hessian[cursor : cursor + size, cursor : cursor + size] = (
                    float(instance.time_constant) * np.eye(size)
                )
                cursor += size
        if RESONANT_MEMORY in self.active_blocks:
            for index, instance in enumerate(self._instances(RESONANT_MEMORY)):
                size = self._resonant_factor(index, n_edges).shape[1]
                angular = 2 * np.pi * float(instance.frequency_hz)
                hessian[cursor : cursor + size, cursor : cursor + size] = angular**2 * np.eye(size)
                cursor += size
            for index, _instance in enumerate(self._instances(RESONANT_MEMORY)):
                size = self._resonant_factor(index, n_edges).shape[1]
                hessian[cursor : cursor + size, cursor : cursor + size] = np.eye(size)
                cursor += size
        values = np.linalg.eigvalsh(0.5 * (hessian + hessian.T))
        return hessian, {
            "energy_hessian_coordinate_system": "quotient_delta, quotient_speed, hidden_storage",
            "energy_hessian_min_eigenvalue": float(np.min(values)) if values.size else float("nan"),
            "energy_hessian_positive_definite": bool(values.size and np.all(values > 0.0)),
        }

    @staticmethod
    def _matrix_rank_diagnostics(matrix: np.ndarray) -> tuple[int, float]:
        singular_values = np.linalg.svd(np.asarray(matrix), compute_uv=False)
        smax = float(singular_values[0]) if singular_values.size else 0.0
        tolerance = float(max(matrix.shape) * np.finfo(float).eps * smax)
        return int(np.count_nonzero(singular_values > tolerance)), tolerance

    def modal_spectrum(
        self,
        static: dict[str, np.ndarray],
        *,
        min_frequency_hz: float = 0.05,
        max_frequency_hz: float = 5.0,
        minimum_speed_visibility: float = 1e-3,
    ) -> dict[str, Any]:
        """Separate all internal poles from modes reportable by the equations.

        A reciprocal hidden coordinate may have a mathematically valid pole but
        no observable speed component.  Such a pole is an implementation
        detail, not evidence for an oscillation in the reported swing model.
        """

        if not np.isfinite(minimum_speed_visibility) or minimum_speed_visibility <= 0.0:
            raise ValueError("minimum_speed_visibility must be finite and positive")

        def refused(reason: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
            return {
                "internal_poles": [],
                "equation_reportable_modes": [],
                "diagnostics": {
                    "status": "refused",
                    "refusal_reason": reason,
                    "modal_diagnostics_complete": False,
                    "paper_support_evaluated": False,
                    **(extra or {}),
                },
            }

        jacobian, diagnostics = self.linearize(static)
        if not np.isfinite(jacobian).all() or not np.isfinite(
            diagnostics["equilibrium_residual_inf_norm"]
        ):
            return refused("nonfinite_linearization", diagnostics)
        try:
            values, vectors = np.linalg.eig(jacobian)
            energy_hessian, energy_diagnostics = self._modal_energy_hessian(static)
            inverse_vectors = np.linalg.pinv(vectors)
            eigenvector_condition = float(np.linalg.cond(vectors))
        except (FloatingPointError, np.linalg.LinAlgError, ValueError) as error:
            return refused("incomplete_modal_diagnostics", {"error": str(error), **diagnostics})
        if not (
            np.isfinite(values).all()
            and np.isfinite(vectors).all()
            and np.isfinite(energy_hessian).all()
            and np.isfinite(inverse_vectors).all()
            and np.isfinite(eigenvector_condition)
        ):
            return refused("nonfinite_modal_diagnostics", diagnostics)
        mass = np.asarray(static["mass"], dtype=float)
        basis = coi_basis(mass)
        quotient = basis.shape[1]
        speed_metric = basis.T @ np.diag(mass) @ basis
        input_projector = np.eye(len(mass)) - np.outer(mass, np.ones(len(mass))) / np.sum(mass)
        input_matrix = np.zeros((jacobian.shape[0], len(mass)))
        input_matrix[quotient : 2 * quotient] = basis.T @ (input_projector / mass[:, None])
        output_matrix = np.zeros((len(mass), jacobian.shape[0]))
        output_matrix[:, quotient : 2 * quotient] = basis
        inverse_vectors = np.linalg.pinv(vectors)
        eigenvector_condition = float(np.linalg.cond(vectors))
        energy_metric_valid = bool(energy_diagnostics["energy_hessian_positive_definite"])
        equilibrium_ok = bool(diagnostics["equilibrium_residual_inf_norm"] <= 1e-7)
        internal_poles: list[dict[str, Any]] = []
        reportable: list[dict[str, Any]] = []
        diagnostics_complete = True
        for index, value in enumerate(values):
            frequency = abs(float(value.imag)) / (2 * np.pi)
            magnitude = abs(value)
            damping = -float(value.real) / magnitude if magnitude > 0 else float("nan")
            vector = vectors[:, index]
            speed_shape = basis @ vector[quotient : 2 * quotient]
            speed_coordinates = vector[quotient : 2 * quotient]
            speed_energy = float(np.real(np.vdot(speed_coordinates, speed_metric @ speed_coordinates)))
            total_energy = float(np.real(np.vdot(vector, energy_hessian @ vector)))
            energy_normalization_available = bool(
                energy_metric_valid and total_energy > 0.0 and speed_energy >= 0.0
            )
            visibility = (
                float(np.sqrt(speed_energy / total_energy)) if energy_normalization_available else 0.0
            )
            residue = (
                output_matrix
                @ vector[:, None]
                @ inverse_vectors[index : index + 1]
                @ input_matrix
            )
            try:
                residue_rank, residue_tolerance = self._matrix_rank_diagnostics(residue)
            except (FloatingPointError, np.linalg.LinAlgError, ValueError):
                residue_rank, residue_tolerance = 0, float("nan")
            pole_diagnostics_complete = bool(
                np.isfinite(
                    [frequency, damping, speed_energy, total_energy, visibility, residue_tolerance]
                ).all()
                and np.isfinite(residue).all()
            )
            diagnostics_complete = diagnostics_complete and pole_diagnostics_complete
            pole = {
                "eigenvalue_real": float(value.real),
                "eigenvalue_imag": float(value.imag),
                "frequency_hz": frequency,
                "damping_ratio": damping,
                "decay_rate": float(value.real),
                "positive_imaginary": bool(value.imag > 1e-8),
                "speed_visibility": visibility,
                "speed_energy": speed_energy,
                "total_modal_energy": total_energy,
                "energy_normalization_available": energy_normalization_available,
                "input_output_residue_rank": residue_rank,
                "input_output_residue_tolerance": residue_tolerance,
                "modal_diagnostics_complete": pole_diagnostics_complete,
            }
            internal_poles.append(pole)
            if (
                value.imag <= 1e-8
                or not min_frequency_hz <= frequency <= max_frequency_hz
                or visibility < minimum_speed_visibility
                or not equilibrium_ok
                or not energy_normalization_available
                or residue_rank <= 0
                or not pole_diagnostics_complete
            ):
                continue
            # Visibility is intentionally tested before any shape normalization:
            # an almost-zero speed component must never become a unit shape.
            shape = speed_shape / np.sqrt(total_energy)
            reportable.append(
                {
                    **pole,
                    "shape_real": np.real(shape).tolist(),
                    "shape_imag": np.imag(shape).tolist(),
                    "linearization": diagnostics,
                }
            )
        if not diagnostics_complete:
            return refused(
                "incomplete_modal_diagnostics",
                {
                    **diagnostics,
                    **energy_diagnostics,
                    "internal_pole_count": len(internal_poles),
                },
            )
        spectrum_diagnostics = {
            **diagnostics,
            "equilibrium_residual_gate": equilibrium_ok,
            "maximum_equilibrium_residual_inf_norm": 1e-7,
            "minimum_speed_visibility": float(minimum_speed_visibility),
            "modal_diagnostics_complete": True,
            **energy_diagnostics,
            "input_output_residue_rank_method": "right_eigenvector_residue_with_pseudoinverse_left_rows",
            "eigenvector_condition_number": eigenvector_condition,
            "paper_support_evaluated": False,
            "internal_pole_count": len(internal_poles),
            "equation_reportable_mode_count": len(reportable),
            "port_realizations": self._port_realization_diagnostics(
                np.asarray(static["incidence"], dtype=float).shape[1]
            ),
        }
        return {
            "internal_poles": internal_poles,
            "equation_reportable_modes": sorted(
                reportable, key=lambda row: (row["frequency_hz"], row["damping_ratio"])
            ),
            "diagnostics": spectrum_diagnostics,
        }

    def modes(
        self,
        static: dict[str, np.ndarray],
        *,
        min_frequency_hz: float = 0.05,
        max_frequency_hz: float = 5.0,
    ) -> list[dict[str, Any]]:
        """Return only equation-reportable positive-imaginary modes.

        This method applies the linear equation, equilibrium and visibility
        gates only.  It does not establish paper-level finite-window support.
        """

        return self.modal_spectrum(
            static,
            min_frequency_hz=min_frequency_hz,
            max_frequency_hz=max_frequency_hz,
        )["equation_reportable_modes"]

    @staticmethod
    def _modal_noninferior(
        reference: Sequence[dict[str, Any]],
        candidate: Sequence[dict[str, Any]],
        mass: np.ndarray,
        required_modes: Sequence[dict[str, Any]] | None = None,
    ) -> bool:
        if required_modes is not None and not required_modes:
            return True
        if not reference:
            return not required_modes
        if required_modes is None and len(candidate) < len(reference):
            return False

        protected = reference if required_modes is None else required_modes

        def best_errors(
            empirical: dict[str, Any], model_modes: Sequence[dict[str, Any]]
        ) -> tuple[float, float, float]:
            if not model_modes:
                return float("inf"), float("inf"), 0.0
            empirical_frequency = float(empirical["frequency_hz"])
            empirical_damping = float(empirical["damping_ratio"])
            empirical_shape = np.asarray(empirical["shape_real"]) + 1j * np.asarray(
                empirical["shape_imag"]
            )
            choices: list[tuple[float, float, float]] = []
            for mode in model_modes:
                relative = abs(float(mode["frequency_hz"]) - empirical_frequency) / max(
                    empirical_frequency, 1e-12
                )
                damping = abs(float(mode["damping_ratio"]) - empirical_damping)
                model_shape = np.asarray(mode["shape_real"]) + 1j * np.asarray(
                    mode["shape_imag"]
                )
                numerator = abs(np.vdot(empirical_shape * mass, model_shape)) ** 2
                denominator = float(
                    np.real(np.vdot(empirical_shape * mass, empirical_shape))
                    * np.real(np.vdot(model_shape * mass, model_shape))
                )
                mac = numerator / max(denominator, np.finfo(float).eps)
                choices.append((relative, damping, float(mac)))
            return min(
                choices,
                key=lambda value: value[0] / 0.05 + value[1] / 0.05 + (1.0 - value[2]) / 0.2,
            )

        for mode in protected:
            if required_modes is None:
                empirical = mode
                reference_errors = (0.0, 0.0, 1.0)
            else:
                empirical = mode
                reference_errors = best_errors(empirical, reference)
            candidate_errors = best_errors(empirical, candidate)
            if required_modes is None:
                limits = (0.05, 0.05, 0.8)
            else:
                limits = (
                    reference_errors[0] + 1e-8,
                    reference_errors[1] + 1e-8,
                    reference_errors[2] - 1e-8,
                )
            if (
                candidate_errors[0] > limits[0]
                or candidate_errors[1] > limits[1]
                or candidate_errors[2] < limits[2]
            ):
                return False
        return True

    def _remove_zero_port_instances(self) -> list[str]:
        """Drop dynamic-port instances whose realized port has rank zero.

        A zero reciprocal factor contributes neither force nor hidden dynamics,
        so retaining it only creates a misleading candidate for modal
        protection.  This is intentionally an unconditional structural
        cleanup, rather than a deletion trial.
        """

        _, n_edges = self._require_fitted()
        removed = {
            instance.instance_id
            for family in (EDGE_MEMORY, RESONANT_MEMORY)
            for index, instance in enumerate(self._instances(family))
            if self._port_realization(index, family, n_edges).zero_port
        }
        if not removed:
            return []

        memory_instances = self._instances(EDGE_MEMORY)
        resonant_instances = self._instances(RESONANT_MEMORY)
        memory_keep = [instance.instance_id not in removed for instance in memory_instances]
        resonant_keep = [instance.instance_id not in removed for instance in resonant_instances]
        if len(self.memory_factors) == len(memory_instances):
            self.memory_factors = [
                factor for factor, keep in zip(self.memory_factors, memory_keep, strict=True) if keep
            ]
        if self.memory_gains is not None and len(self.memory_gains) == len(memory_instances):
            self.memory_gains = self.memory_gains[np.asarray(memory_keep, dtype=bool)]
        if len(self.resonant_factors) == len(resonant_instances):
            self.resonant_factors = [
                factor
                for factor, keep in zip(self.resonant_factors, resonant_keep, strict=True)
                if keep
            ]
        if self.resonant_gains is not None and len(self.resonant_gains) == len(resonant_instances):
            self.resonant_gains = self.resonant_gains[np.asarray(resonant_keep, dtype=bool)]
        self.active_instances = tuple(
            instance for instance in self.active_instances if instance.instance_id not in removed
        )
        self.active_blocks = tuple(
            family
            for family in OPTIONAL_BLOCKS
            if any(instance.family == family for instance in self.active_instances)
        )
        self._refresh_parameter_diagnostics()
        return sorted(removed)

    def prune(
        self,
        train: list[dict[str, Any]],
        val: list[dict[str, Any]],
        run_context: RunContext,
        *,
        tolerance: float = 0.02,
        closed_loop_refine_cases: int = 2,
        closed_loop_refine_evaluations: int = 8,
        bootstrap_draws: int = 200,
        seed: int = 324,
        required_modal_modes: Sequence[dict[str, Any]] | None = None,
    ) -> PrunableEnergyModel:
        if self.node_damping is None:
            self.fit(train, val, run_context)
        current: PrunableEnergyModel = self
        trace: list[dict[str, Any]] = []
        while current.active_instances:
            zero_port_removed = current._remove_zero_port_instances()
            if zero_port_removed:
                trace.append(
                    {
                        "action": "automatic_zero_port_removal",
                        "removed_instances": zero_port_removed,
                        "reason": "realized_port_rank_zero",
                        "modal_protection_bypassed": True,
                    }
                )
                continue
            frozen_ridge = float(current.selected_ridge or run_context.ridge_values[0])
            pruning_context = RunContext(
                **{**run_context.__dict__, "ridge_values": (frozen_ridge,)}
            )
            reference_summary = current.validation_summary(val, pruning_context)
            static = train[0]["model_inputs"]["static"]
            mass = np.asarray(static["mass"], dtype=float)
            reference_modes = current.modes(static)
            trials = []
            for instance in current.active_instances:
                subset = tuple(
                    value for value in current.active_instances if value.instance_id != instance.instance_id
                )
                candidate = PrunableEnergyModel(
                    block_instances=subset,
                    memory_taus=current.memory_taus,
                    resonances=current.resonances,
                    node_tau=current.node_tau,
                    potential_scale=current.potential_scale,
                    max_internal_step=current.max_internal_step,
                    enforce_identifiability=current.enforce_identifiability,
                    minimum_equation_parameter_ratio=current.minimum_equation_parameter_ratio,
                    maximum_condition_number=current.maximum_condition_number,
                )
                fit_error = None
                try:
                    candidate.fit(train, val, pruning_context)
                    candidate.refine_closed_loop(
                        val,
                        pruning_context,
                        maximum_cases=int(closed_loop_refine_cases),
                        maximum_evaluations=int(closed_loop_refine_evaluations),
                    )
                    summary = candidate.validation_summary(val, pruning_context)
                    physics = candidate.physics_audit(static)
                    modes = candidate.modes(static)
                    modal_noninferior = self._modal_noninferior(
                        reference_modes,
                        modes,
                        mass,
                        required_modal_modes,
                    )
                    degradation = self.source_stratified_degradation(
                        current,
                        candidate,
                        val,
                        pruning_context,
                        tolerance=float(tolerance),
                        bootstrap_draws=int(bootstrap_draws),
                        seed=int(seed),
                    )
                    admissible = bool(
                        degradation["passes"]
                        and physics["status"] == "pass"
                        and modal_noninferior
                    )
                except (ValueError, RuntimeError) as error:
                    fit_error = str(error)
                    summary = {
                        "standardized_loss": float("inf"),
                        "failure_rate": 1.0,
                        "horizon_standardized_loss": {},
                    }
                    physics = {"status": "fail", "error": fit_error}
                    modal_noninferior = False
                    degradation = {
                        "passes": False,
                        "maximum_point_degradation": float("inf"),
                        "maximum_bootstrap_upper": float("inf"),
                        "rows": [],
                        "failures": [{"error": type(error).__name__}],
                    }
                    admissible = False
                trials.append(
                    {
                        "removed_instance": instance.instance_id,
                        "removed_family": instance.family,
                        "candidate": candidate,
                        "summary": summary,
                        "physics": physics,
                        "modal_noninferior": modal_noninferior,
                        "source_degradation": degradation,
                        "fit_error": fit_error,
                        "admissible": admissible,
                    }
                )
            admissible = [row for row in trials if row["admissible"]]
            trace.append(
                {
                    "active_before": [row.instance_id for row in current.active_instances],
                    "frozen_ridge": frozen_ridge,
                    "reference_loss": reference_summary["standardized_loss"],
                    "trials": [
                        {
                            key: value
                            for key, value in row.items()
                            if key != "candidate"
                        }
                        for row in trials
                    ],
                }
            )
            if not admissible:
                break
            chosen = min(
                admissible,
                key=lambda row: (
                    row["source_degradation"]["maximum_bootstrap_upper"],
                    row["removed_instance"],
                ),
            )
            trace[-1]["accepted_removal"] = chosen["removed_instance"]
            trace[-1]["accepted_family"] = chosen["removed_family"]
            current = chosen["candidate"]
        current.pruning_trace = trace
        return current

    def physics_audit(
        self, static: dict[str, np.ndarray] | None = None
    ) -> dict[str, Any]:
        if self.node_damping is None:
            return {"status": "not_fitted"}
        arrays = [
            self.node_damping,
            self.coupling,
            self.potential_coefficients,
            self.edge_damping_coefficients,
            self.node_high_order_gain,
            self.memory_gains,
            self.resonant_gains,
        ]
        finite = all(np.isfinite(value).all() for value in arrays)
        nonnegative = all(np.all(value >= -1e-12) for value in arrays)
        result: dict[str, Any] = {
            "status": "pass" if finite and nonnegative else "fail",
            "finite_parameters": finite,
            "nonnegative_parameters": nonnegative,
            "reciprocal_dynamic_ports": True,
            "energy_supply_identity_by_construction": True,
            "memory_residual_operator": "positive-semidefinite edge operator",
            "memory_connection_factor": "nonnegative square-root factor",
            "active_blocks": list(self.active_blocks),
        }
        if static is not None:
            basis = coi_basis(np.asarray(static["mass"], dtype=float))
            hessian = basis.T @ self.storage_hessian(
                np.asarray(static["equilibrium_delta"], dtype=float), static
            ) @ basis
            minimum = float(np.min(np.linalg.eigvalsh(0.5 * (hessian + hessian.T))))
            result["storage_hessian_min_eigenvalue"] = minimum
            result["storage_curvature_positive"] = bool(minimum > 1e-8)
            if minimum <= 1e-8:
                result["status"] = "fail"
            postclear, equilibrium_audit = self.solve_postclear_equilibrium(static)
            postclear_hessian = basis.T @ self.storage_hessian(postclear, static) @ basis
            result["postclear_equilibrium"] = equilibrium_audit
            result["postclear_storage_hessian_min_eigenvalue"] = float(
                np.min(np.linalg.eigvalsh(0.5 * (postclear_hessian + postclear_hessian.T)))
            )
        return result

    def save(self, path: str | Path) -> None:
        self._require_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            "node_damping": self.node_damping,
            "coupling": self.coupling,
            "potential_coefficients": self.potential_coefficients,
            "edge_damping_coefficients": self.edge_damping_coefficients,
            "node_high_order_gain": self.node_high_order_gain,
            "memory_gains": self.memory_gains,
            "resonant_gains": self.resonant_gains,
        }
        arrays.update(
            {
                f"memory_factor_{index}": self._memory_factor(index, len(self.coupling))
                for index in range(len(self._instances(EDGE_MEMORY)))
            }
        )
        arrays.update(
            {
                f"resonant_factor_{index}": self._resonant_factor(index, len(self.coupling))
                for index in range(len(self._instances(RESONANT_MEMORY)))
            }
        )
        np.savez_compressed(path, **arrays)
        path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "format_version": 3,
                    "method": self.method_name,
                    "active_blocks": list(self.active_blocks),
                    "active_block_instances": [row.to_dict() for row in self.active_instances],
                    "memory_taus": list(self.memory_taus),
                    "resonances": [list(row) for row in self.resonances],
                    "node_tau": self.node_tau,
                    "potential_scale": self.potential_scale,
                    "max_internal_step": self.max_internal_step,
                    "enforce_identifiability": self.enforce_identifiability,
                    "minimum_equation_parameter_ratio": self.minimum_equation_parameter_ratio,
                    "maximum_condition_number": self.maximum_condition_number,
                    "selected_ridge": self.selected_ridge,
                    "diagnostics": self.diagnostics,
                    "pruning_trace": self.pruning_trace,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> PrunableEnergyModel:
        path = Path(path)
        metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        model = cls(
            active_blocks=metadata["active_blocks"],
            block_instances=(
                metadata.get("active_block_instances")
                if int(metadata.get("format_version", 1)) >= 2
                else None
            ),
            memory_taus=metadata["memory_taus"],
            resonances=metadata["resonances"],
            node_tau=metadata["node_tau"],
            potential_scale=metadata["potential_scale"],
            max_internal_step=metadata["max_internal_step"],
            enforce_identifiability=bool(metadata.get("enforce_identifiability", False)),
            minimum_equation_parameter_ratio=float(
                metadata.get("minimum_equation_parameter_ratio", 3.0)
            ),
            maximum_condition_number=float(metadata.get("maximum_condition_number", 1e8)),
        )
        with np.load(path) as data:
            for name in (
                "node_damping",
                "coupling",
                "potential_coefficients",
                "edge_damping_coefficients",
                "node_high_order_gain",
                "memory_gains",
                "resonant_gains",
            ):
                setattr(model, name, np.asarray(data[name], dtype=float))
            model.memory_factors = []
            for index in range(len(model._instances(EDGE_MEMORY))):
                key = f"memory_factor_{index}"
                raw = (
                    np.asarray(data[key], dtype=float)
                    if key in data.files
                    else np.diag(np.sqrt(np.maximum(model.memory_gains[index], 0.0)))
                )
                model.memory_factors.append(_minimal_port_factor(raw).factor)
            model.resonant_factors = []
            for index in range(len(model._instances(RESONANT_MEMORY))):
                key = f"resonant_factor_{index}"
                raw = (
                    np.asarray(data[key], dtype=float)
                    if key in data.files
                    else np.diag(np.sqrt(np.maximum(model.resonant_gains[index], 0.0)))
                )
                model.resonant_factors.append(_minimal_port_factor(raw).factor)
        model.selected_ridge = metadata.get("selected_ridge")
        model.diagnostics = metadata.get("diagnostics", {})
        model.pruning_trace = metadata.get("pruning_trace", [])
        # Version-1/2 checkpoints kept a proposed port rank but may have
        # stored redundant columns.  Preserve that metadata and recompute the
        # realized rank from the loaded operator.
        model._refresh_parameter_diagnostics()
        return model
