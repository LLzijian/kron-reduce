"""Deterministic measurement-noise draws for paired TASK3 history tests.

Noise is added to the raw coordinates before the normal COI projection.  The
future trajectory is never modified by :func:`inject_raw_history`.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from kron_reduce.physics.coordinates import coi_project

ROSTERS = frozenset({"val", "test", "blind_v2"})
SCENARIOS = (
    "clean",
    "gauss_pmu_steady",
    "gauss_dse_dynamic",
    "gauss_dse_dynamic_pm",
    "gauss_snr30",
    "gauss_snr20",
    "nongauss_laplace",
    "nongauss_student_t3",
    "outlier_random5pct_10sigma",
    "outlier_endpoint_10sigma",
)
_SEED_PREFIX = "esrom_task3_noise_v1"
_SHAPE = (11, 10)


@dataclass(frozen=True)
class NoiseDraw:
    """One paired disturbance in raw angle, speed, and mechanical power."""

    eps_delta: np.ndarray
    eps_omega: np.ndarray
    eps_pm: np.ndarray
    endpoint_index: int | None = None

    @property
    def clean(self) -> bool:
        return not np.any(self.eps_delta) and not np.any(self.eps_omega) and not np.any(self.eps_pm)


@dataclass(frozen=True)
class NoiseBank:
    """Noise draws and the prescribed cohort-level byte digest."""

    draws: Mapping[str, NoiseDraw]
    sha256: str


def _validate_key(roster: str, scenario: str, rep: int, canonical_id: str) -> None:
    if roster not in ROSTERS:
        raise ValueError(f"unknown TASK3 noise roster: {roster}")
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown TASK3 noise scenario: {scenario}")
    if isinstance(rep, bool) or not isinstance(rep, (int, np.integer)):
        raise TypeError("rep must be an integer")
    if scenario == "clean":
        if int(rep) != 0:
            raise ValueError("clean uses rep=0")
    elif not 0 <= int(rep) <= 4:
        raise ValueError("noisy scenarios use reps 0 through 4")
    if (not isinstance(canonical_id, str) or canonical_id.count("/") != 1
            or not all(canonical_id.split("/"))):
        raise ValueError("canonical_id must be source-qualified")


def draw_task3_noise(roster: str, scenario: str, rep: int,
                     canonical_id: str) -> NoiseDraw:
    """Draw one deterministic paired noise sample using the frozen protocol."""
    _validate_key(roster, scenario, rep, canonical_id)
    zero_delta = np.zeros(_SHAPE, dtype=np.float64)
    zero_omega = np.zeros(_SHAPE, dtype=np.float64)
    zero_pm = np.zeros(10, dtype=np.float64)
    if scenario == "clean":
        return NoiseDraw(zero_delta, zero_omega, zero_pm)

    key = f"{_SEED_PREFIX}|{roster}|{scenario}|rep{int(rep)}|{canonical_id}".encode()
    seed = int.from_bytes(hashlib.sha256(key).digest()[:16], "little")
    rng = np.random.default_rng(np.random.PCG64(seed))
    endpoint_index = None

    if scenario in {"gauss_pmu_steady", "outlier_random5pct_10sigma",
                    "outlier_endpoint_10sigma"}:
        eps_delta = rng.normal(0.0, 0.0033, _SHAPE)
        eps_omega = rng.normal(0.0, 0.010, _SHAPE)
        eps_pm = zero_pm.copy()
        if scenario == "outlier_random5pct_10sigma":
            mask_d = rng.random(_SHAPE) < 0.05
            sign_d = rng.choice(np.array([-1.0, 1.0]), size=_SHAPE)
            mask_w = rng.random(_SHAPE) < 0.05
            sign_w = rng.choice(np.array([-1.0, 1.0]), size=_SHAPE)
            eps_delta += mask_d * sign_d * (10.0 * 0.010)
            eps_omega += mask_w * sign_w * (10.0 * 0.050)
        elif scenario == "outlier_endpoint_10sigma":
            endpoint_index = int(rng.choice(np.array([0, 10])))
            gen_idx = int(rng.integers(0, 10))
            sign_d = float(rng.choice(np.array([-1.0, 1.0])))
            sign_w = float(rng.choice(np.array([-1.0, 1.0])))
            eps_delta[endpoint_index, gen_idx] += sign_d * (10.0 * 0.010)
            eps_omega[endpoint_index, gen_idx] += sign_w * (10.0 * 0.050)
    elif scenario in {"gauss_dse_dynamic", "gauss_dse_dynamic_pm"}:
        eps_delta = rng.normal(0.0, 0.010, _SHAPE)
        eps_omega = rng.normal(0.0, 0.050, _SHAPE)
        eps_pm = (rng.normal(0.0, 0.010, (10,))
                  if scenario == "gauss_dse_dynamic_pm" else zero_pm.copy())
    elif scenario in {"gauss_snr30", "gauss_snr20"}:
        snr_db = 30.0 if scenario == "gauss_snr30" else 20.0
        scale = 10.0 ** (-snr_db / 20.0)
        eps_delta = rng.normal(0.0, scale * 0.476753, _SHAPE)
        eps_omega = rng.normal(0.0, scale * 1.003367, _SHAPE)
        eps_pm = zero_pm.copy()
    elif scenario == "nongauss_laplace":
        eps_delta = rng.laplace(0.0, 0.010 / np.sqrt(2.0), _SHAPE)
        eps_omega = rng.laplace(0.0, 0.050 / np.sqrt(2.0), _SHAPE)
        eps_pm = zero_pm.copy()
    elif scenario == "nongauss_student_t3":
        eps_delta = rng.standard_t(3.0, _SHAPE) * (0.010 / np.sqrt(3.0))
        eps_omega = rng.standard_t(3.0, _SHAPE) * (0.050 / np.sqrt(3.0))
        eps_pm = zero_pm.copy()
    else:  # guarded by _validate_key; protects future edits to SCENARIOS.
        raise ValueError(f"unsupported TASK3 noise scenario: {scenario}")

    for value in (eps_delta, eps_omega, eps_pm):
        value.flags.writeable = False
    return NoiseDraw(eps_delta, eps_omega, eps_pm, endpoint_index)


def inject_raw_history(raw_x: np.ndarray, raw_p0: np.ndarray, mass: np.ndarray,
                       draw: NoiseDraw) -> tuple[np.ndarray, np.ndarray]:
    """Copy raw values and inject noise only in the first 11 frames.

    The caller then applies its existing COI projection to the returned arrays.
    For a clean draw, the copies retain the loader's original values exactly.
    """
    x = np.array(raw_x, dtype=np.float64, copy=True)
    p0 = np.array(raw_p0, dtype=np.float64, copy=True)
    mass = np.asarray(mass, dtype=np.float64).reshape(-1)
    if x.ndim != 2 or x.shape[1] != 20 or x.shape[0] < 11:
        raise ValueError("raw_x must have shape (at least 11, 20)")
    if p0.shape != (10,) or mass.shape != (10,):
        raise ValueError("raw_p0 and mass must each have shape (10,)")
    if draw.eps_delta.shape != _SHAPE or draw.eps_omega.shape != _SHAPE or draw.eps_pm.shape != (10,):
        raise ValueError("noise draw has invalid dimensions")
    if draw.clean:
        return x, p0
    x[:11, :10] += coi_project(draw.eps_delta, mass)
    x[:11, 10:] += draw.eps_omega
    p0 += draw.eps_pm
    return x, p0


def generate_noise_bank(roster: str, scenario: str, rep: int,
                        canonical_ids: Iterable[str]) -> NoiseBank:
    """Generate a paired cohort in sorted-ID order and hash its raw draws."""
    ids = list(canonical_ids)
    if len(ids) != len(set(ids)):
        raise ValueError("canonical IDs must be unique")
    ids.sort()
    digest = hashlib.sha256()
    draws: dict[str, NoiseDraw] = {}
    for canonical_id in ids:
        draw = draw_task3_noise(roster, scenario, rep, canonical_id)
        draws[canonical_id] = draw
        digest.update(
            draw.eps_delta.astype("<f8", copy=False).tobytes()
            + draw.eps_omega.astype("<f8", copy=False).tobytes()
            + draw.eps_pm.astype("<f8", copy=False).tobytes()
        )
    return NoiseBank(draws, digest.hexdigest())

