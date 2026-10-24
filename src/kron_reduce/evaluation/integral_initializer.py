"""A14 fixed, causal integral initializer arithmetic.

No filesystem access, model loading, fitting, RHS, or rollout occurs here.
The caller supplies one fixed closure, static priors, and exactly 11 observed
q/v samples.  Owner keys bind caches; they never enter a numerical equation.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np
from scipy.linalg import expm as _expm


class ArrayRecord(dict):
    """Named numerical records, compatible with the frozen A13 outputs."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


class OperatorBudgetError(RuntimeError):
    """The next operator was refused before dispatch."""


class OperatorJournalError(RuntimeError):
    """A journal failure requires the caller to stop the campaign."""


class KernelNumericalError(FloatingPointError):
    """A calculated numerical output is invalid; retain the failed slot."""


@dataclass
class Ledger:
    """Actual operator attempts and returns, with optional caller-set limits.

    ``event_sink`` may durably save each transition. An attempted transition
    reaches the sink before the numerical callable. Sink errors propagate;
    the caller must stop rather than retry that operation.
    """

    limits: Mapping[str, int] | None = None
    event_sink: Callable[[dict], None] | None = None
    context: str = ""
    attempted: Counter = field(default_factory=Counter)
    dispatched: Counter = field(default_factory=Counter)
    returned: Counter = field(default_factory=Counter)
    failed: Counter = field(default_factory=Counter)
    entries: list[dict] = field(default_factory=list)
    poisoned: bool = False
    poison_reason: str | None = None

    def __post_init__(self):
        if self.limits is not None:
            for key, value in self.limits.items():
                if key not in {"expm", "svd", "ls", "force"} or type(value) is not int or value < 0:
                    raise ValueError("operator limits require known categories and nonnegative ints")
            self.limits = MappingProxyType(dict(self.limits))

    @property
    def counts(self):
        return self.returned

    def _emit(self, entry, transition):
        if self.event_sink is not None:
            try:
                self.event_sink({**entry, "transition": transition})
            except Exception as exc:
                self.poisoned = True
                self.poison_reason = f"journal {transition}: {type(exc).__name__}: {exc}"
                raise OperatorJournalError(self.poison_reason) from exc

    def call(self, category: str, function: Callable[[], Any], *, kind: str, shape=None):
        if self.poisoned:
            raise OperatorJournalError(self.poison_reason or "operator ledger is poisoned")
        if category not in {"expm", "svd", "ls", "force"}:
            raise ValueError("unregistered operator category")
        if self.limits is not None and self.attempted[category] >= self.limits.get(category, 0):
            self.poisoned = True
            self.poison_reason = f"{category} budget exhausted before {kind}"
            raise OperatorBudgetError(f"{category} budget exhausted before {kind}")
        entry = {
            "index": len(self.entries), "context": self.context,
            "category": category, "kind": kind,
            "shape": None if shape is None else list(shape),
            "attempted": True, "dispatched": False, "returned": False, "error": None,
        }
        self.entries.append(entry)
        self.attempted[category] += 1
        self._emit(entry, "attempted")
        self.dispatched[category] += 1
        entry["dispatched"] = True
        try:
            result = function()
        except Exception as exc:
            self.failed[category] += 1
            entry["error"] = f"{type(exc).__name__}: {exc}"
            self._emit(entry, "failed")
            raise
        self.returned[category] += 1
        entry["returned"] = True
        self._emit(entry, "returned")
        return result

    def snapshot(self):
        return {
            "attempted": dict(self.attempted), "dispatched": dict(self.dispatched),
            "returned": dict(self.returned),
            "counts": dict(self.returned), "failed": dict(self.failed),
            "entries": [dict(row) for row in self.entries],
            "poisoned": self.poisoned, "poison_reason": self.poison_reason,
        }


def _finite(value, name, shape=None):
    out = np.asarray(value, dtype=np.float64)
    if (shape is not None and out.shape != shape) or not np.isfinite(out).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return out


def _finite_output(value, name, shape=None):
    """Keep calculated-output failures separate from input/cache rejection."""
    try:
        return _finite(value, name, shape)
    except (TypeError, ValueError) as exc:
        raise KernelNumericalError(str(exc)) from exc


def _parameters(closure):
    shapes = {
        "mass": (10,), "edge_stiffness": (45,), "damping": (10, 10),
        "coupling": (10, 2), "latent_skew": (2, 2), "latent_dissipation": (2, 2),
    }
    out = ArrayRecord({name: _finite(getattr(closure, name), name, shape)
                       for name, shape in shapes.items()})
    if np.any(out.mass <= 0) or closure.backbone != "sine":
        raise ValueError("A14 requires the fixed sine closure with positive mass")
    out["backbone"] = "sine"
    return out


def _hash_value(value):
    digest = hashlib.sha256()

    def update(item):
        if isinstance(item, np.ndarray):
            digest.update(b"array:")
            digest.update(json.dumps([item.dtype.str, list(item.shape)]).encode())
            digest.update(np.ascontiguousarray(item).tobytes())
        elif isinstance(item, Mapping):
            digest.update(b"mapping:")
            for key in sorted(item):
                digest.update(json.dumps(key).encode())
                update(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(b"sequence:")
            for part in item:
                update(part)
        else:
            digest.update(json.dumps(item, allow_nan=False, sort_keys=True).encode())
        digest.update(b";")

    update(value)
    return digest.hexdigest()


def _freeze(value):
    if isinstance(value, np.ndarray):
        result = value.copy()
        result.flags.writeable = False
        return result
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class StaticCache:
    owner_key: str
    model_sha256: str
    payload_sha256: str
    data: Mapping[str, Any]


def spectral_norm_two_columns(matrix):
    """A13's scaled 2x2 Gram formula; no eigensolver or extra SVD."""
    a = _finite(matrix, "two-column matrix")
    if a.ndim != 2 or a.shape[1] != 2:
        raise ValueError("the fixed Gram formula requires two columns")
    scale = float(np.max(np.abs(a)))
    if scale == 0:
        return 0.0
    scaled = a / scale
    gram = scaled.T @ scaled
    disc = float(np.hypot(gram[0, 0] - gram[1, 1], 2.0 * gram[0, 1]))
    eigenvalue = max(0.0, 0.5 * (float(np.trace(gram)) + disc))
    return float(scale * np.sqrt(eigenvalue))


def _decompose(matrix, *, ledger, kind, natural_scale=0.0, partial=None):
    matrix = _finite_output(matrix, "SVD matrix")
    U, singular, Vh = ledger.call(
        "svd", lambda: np.linalg.svd(matrix, full_matrices=False),
        kind=kind, shape=matrix.shape,
    )
    if partial is not None:
        partial["svd_return"] = {"U": U, "singular_values": singular, "Vh": Vh}
    for value in (U, singular, Vh):
        _finite_output(value, "SVD return")
    tol = float(64 * np.finfo(np.float64).eps * max(matrix.shape)
                * max(float(singular[0]), float(natural_scale)))
    keep = singular > tol
    rank = int(np.sum(keep))
    sigma_min = float(singular[-1])
    full = rank == matrix.shape[1]
    result = ArrayRecord(
        U=U, s=singular, singular_values=singular, Vh=Vh, keep=keep,
        rank=rank, threshold=tol, tol=tol, sigma_min=sigma_min,
        condition=float(singular[0] / sigma_min) if full else None,
        inverse_sigma_min=float(1 / sigma_min) if sigma_min > 0 else None,
        natural_norm=float(natural_scale), unresolved=not full, shape=matrix.shape,
    )
    if partial is not None:
        partial["svd_diagnostics"] = result
    for key, value in result.items():
        if isinstance(value, float):
            _finite_output(value, f"SVD diagnostic {key}")
    return result


def prepare_static(closure, velocity_scale, *, owner_key: str, ledger: Ledger) -> StaticCache:
    """Exactly two SVDs per owner: PC and [PC; .01 PC A], no other operator."""
    if not isinstance(owner_key, str) or not owner_key:
        raise ValueError("a nonempty fixed model/seed owner_key is required")
    par = _parameters(closure)
    sv = _finite(velocity_scale, "velocity scale", (10,))
    if np.any(sv <= 0):
        raise ValueError("velocity scales must be positive")
    model_sha = _hash_value({"model": par, "velocity_scale": sv})
    partial = {"static_owner_key": owner_key, "static_model_sha256": model_sha,
               "parameters": par, "velocity_scale": sv,
               "status": "incomplete_numerical_failure"}
    try:
        return _prepare_static(par, sv, owner_key, ledger, model_sha, partial)
    except (np.linalg.LinAlgError, FloatingPointError, OverflowError) as error:
        error.partial_record = partial
        raise


def _prepare_static(par, sv, owner_key, ledger, model_sha, partial):
    Q = np.eye(10) - np.outer(np.ones(10), par.mass) / float(np.sum(par.mass))
    P = Q.T.copy()
    A = par.latent_skew - par.latent_dissipation
    PC = P @ par.coupling
    S = 1.0 / (par.mass * sv)
    partial.update(P=P, Q=Q, A=A, PC=PC, S=S)
    scaled_port = S[:, None] * PC
    partial["scaled_port"] = scaled_port
    natural = spectral_norm_two_columns(_finite_output(scaled_port, "scaled port"))
    partial["natural_norm"] = natural
    _finite_output(natural, "scaled port spectral norm")
    partial["PC_decomposition"] = {}
    port_svd = _decompose(PC, ledger=ledger, kind="PC", partial=partial["PC_decomposition"])
    partial["C_svd"] = port_svd
    O = np.vstack((PC, 0.01 * PC @ A))
    partial.update(O=O, O_decomposition={})
    visible_svd = _decompose(O, ledger=ledger, kind="O", partial=partial["O_decomposition"])
    partial["O_svd"] = visible_svd
    active = port_svd.U[:, port_svd.keep]
    projector = active @ active.T
    incidence = np.zeros((10, 45), dtype=np.float64)
    edge = 0
    for i in range(10):
        for j in range(i + 1, 10):
            incidence[i, edge], incidence[j, edge] = 1.0, -1.0
            edge += 1
    data = _freeze({
        "parameters": par, "velocity_scale": sv, "P": P, "Q": Q, "A": A,
        "PC": PC, "S": S, "natural_norm": natural, "O": O,
        "C_svd": port_svd, "O_svd": visible_svd, "forceproj": projector,
        "force_residual_projector": np.eye(10) - projector, "incidence": incidence,
    })
    return StaticCache(owner_key, model_sha, _hash_value(data), data)


def _validate_cache(cache, closure, owner_key):
    if not isinstance(cache, StaticCache) or owner_key != cache.owner_key:
        raise ValueError("static cache owner mismatch")
    if _hash_value(cache.data) != cache.payload_sha256:
        raise ValueError("static cache payload was changed")
    par = _parameters(closure)
    if _hash_value({"model": par, "velocity_scale": cache.data["velocity_scale"]}) != (
        cache.model_sha256
    ):
        raise ValueError("static cache model fingerprint mismatch")
    return par


def _matrix_system(time, cache, ledger, partial):
    clock = _finite(time, "history time", (11,))
    clock = clock - clock[0]
    h = np.diff(clock)
    if not np.all(h > 0) or not np.isfinite(h).all():
        raise ValueError("history intervals must be finite and positive")
    T = float(clock[-1])
    data = cache.data
    c = np.sqrt(h / T) / h
    aug = np.zeros((8, 8))
    aug[:2, :2] = data["A"]
    aug[:2, 2:4] = aug[2:4, 4:6] = aug[4:6, 6:8] = np.eye(2)
    blocks, prefixes, L_blocks = [], [np.eye(2)], []
    matrix_partial = {
        "time": clock, "h": h, "T": T, "c": c, "S": data["S"],
        "A": data["A"], "PC": data["PC"], "augmented_matrix": aug,
        "exponential_returns": [], "blocks": blocks,
        "phi_prefix": prefixes, "L_blocks": L_blocks,
    }
    partial["matrix_system"] = matrix_partial
    for width in h:
        exponential = ledger.call("expm", lambda width=width: _expm(width * aug),
                                  kind="integral_4r", shape=(8, 8))
        matrix_partial["exponential_returns"].append(exponential)
        _finite_output(exponential, "expm return", (8, 8))
        block = ArrayRecord(Phi=exponential[:2, :2].copy(), F1=exponential[:2, 2:4].copy(),
                            F2=exponential[:2, 4:6].copy(), F3=exponential[:2, 6:8].copy())
        blocks.append(block)
        L_blocks.append(data["PC"] @ prefixes[-1] @ block.F1)
        prefixes.append(block.Phi @ prefixes[-1])
    raw = np.stack(L_blocks)
    scaled = c[:, None, None] * data["S"][None, :, None] * raw
    L, Ltilde = raw.reshape(100, 2), scaled.reshape(100, 2)
    matrix_partial.update(L=L, Ltilde=Ltilde, L_blocks_stacked=raw, Ltilde_blocks=scaled)
    svd = _decompose(Ltilde, ledger=ledger, kind="L", natural_scale=data["natural_norm"],
                     partial=matrix_partial)
    matrix_partial["svd"] = svd
    return ArrayRecord(
        L=L, Ltilde=Ltilde, L_blocks=raw, Ltilde_blocks=scaled,
        c=c, S=data["S"], h=h, time=clock, T=T,
        mass=data["parameters"]["mass"], A=data["A"], C=data["parameters"]["coupling"],
        D=data["parameters"]["damping"], P=data["P"], Q=data["Q"], P_used=data["P"],
        PC=data["PC"], blocks=blocks, phi_prefix=np.stack(prefixes), PhiT=prefixes[-1].copy(),
        forceproj=data["forceproj"], force_residual_projector=data["force_residual_projector"],
        C_svd=data["C_svd"], O_svd=data["O_svd"], svd=svd,
        U=svd.U, s=svd.s, singular_values=svd.s, Vh=svd.Vh,
        rank=svd.rank, tol=svd.tol, natural_norm=svd.natural_norm,
        sigma_min=svd.sigma_min, condition=svd.condition,
        inverse_sigma_min=svd.inverse_sigma_min, unresolved=svd.unresolved,
        static_owner_key=cache.owner_key, static_model_sha256=cache.model_sha256,
        static_payload_sha256=cache.payload_sha256,
    )


def _gauss_rule(order):
    if order == 2:
        return np.array([-1.0, 1.0]) / np.sqrt(3.0), np.ones(2)
    if order == 4:
        root = np.sqrt(6.0 / 5.0)
        outer, inner = np.sqrt((3.0 + 2.0 * root) / 7.0), np.sqrt((3.0 - 2.0 * root) / 7.0)
        wo, wi = (18.0 - np.sqrt(30.0)) / 36.0, (18.0 + np.sqrt(30.0)) / 36.0
        return np.array([-outer, -inner, inner, outer]), np.array([wo, wi, wi, wo])
    raise ValueError("only the frozen Gauss orders 2/4 are permitted")


def _integrate_force(q0, q1, h, equilibrium, order, cache, ledger, partial):
    nodes, weights = _gauss_rule(order)
    B, k = cache.data["incidence"], cache.data["parameters"]["edge_stiffness"]
    out = np.zeros(10)
    record = {"order": order, "h": float(h), "q_nodes": [], "force_returns": [],
              "accumulator": out}
    partial["force_quadrature"].append(record)
    for node, weight in zip(nodes, weights):
        q = q0 + (0.5 * (node + 1.0)) * (q1 - q0)
        record["q_nodes"].append(q)

        def calculate(q=q):
            edge, reference = B.T @ q, B.T @ equilibrium
            return B @ (k * (np.sin(edge + reference) - np.sin(reference)))

        force = ledger.call("force", calculate, kind=f"gauss{order}", shape=(10,))
        record["force_returns"].append(force)
        out += (0.5 * h * weight) * _finite_output(force, "force return", (10,))
    return out


def _scale_blocks(system, raw):
    return (system.c[:, None] * system.S[None, :] * raw.reshape(10, 10)).reshape(100)


def _solve(svd, rhs, ledger, order, partial):
    if svd.unresolved:
        raise ValueError("a rank-deficient system must not produce a candidate state")

    def calculate():
        return svd.Vh.T @ ((svd.U.T @ rhs) / svd.s)

    out = ledger.call("ls", calculate, kind=f"gauss{order}", shape=(100,))
    partial["ls_return"] = out
    return _finite_output(out, "least-squares return", (2,))


def build(q, v, time, closure, p, eq, *, cache: StaticCache, owner_key: str,
          ledger: Ledger) -> ArrayRecord:
    """One slot: 10 expm, 1 L-SVD, 60 force, 2 LS if rank is two.

    A rank-deficient slot retains its matrices, both right-hand sides, force
    residuals, and kinematics. It makes zero LS calls and returns no candidate
    state. Neither an encoder fallback nor any rollout is performed here.
    Numerical exceptions carry ``partial_record`` with already calculated
    values; recording never replays an operator. Input/cache errors remain
    ValueError and do not masquerade as local numerical failures.
    """
    par = _validate_cache(cache, closure, owner_key)
    q, v = _finite(q, "history q", (11, 10)), _finite(v, "history v", (11, 10))
    clock = _finite(time, "history time", (11,))
    power, equilibrium = _finite(p, "constant power", (10,)), _finite(eq, "equilibrium", (10,))
    partial = {"static_owner_key": owner_key, "static_model_sha256": cache.model_sha256,
               "status": "incomplete_numerical_failure"}
    try:
        return _build(q, v, clock, par, power, equilibrium, cache, ledger, partial)
    except (np.linalg.LinAlgError, FloatingPointError, OverflowError) as error:
        error.partial_record = partial
        raise


def _build(q, v, clock, par, power, equilibrium, cache, ledger, partial):
    system = _matrix_system(clock, cache, ledger, partial)
    w, w_nodes, w_integrals = np.zeros(2), [np.zeros(2)], []
    g_by_order = {2: [], 4: []}
    partial.update(system=system, w_nodes=w_nodes, w_integrals=w_integrals,
                   g_by_order=g_by_order, force_quadrature=[], right_hand_sides={})
    for j, h in enumerate(system.h):
        block = system.blocks[j]
        u = par.coupling.T @ v[j]
        d = par.coupling.T @ (v[j + 1] - v[j]) / h
        w_integrals.append(block.F1 @ w + block.F2 @ u + block.F3 @ d)
        w = block.Phi @ w + block.F1 @ u + block.F2 @ d
        w_nodes.append(w.copy())
        base = h * power - h * par.damping @ (v[j] + v[j + 1]) / 2.0
        momentum = par.mass * (v[j + 1] - v[j])
        for order in (2, 4):
            force = _integrate_force(q[j], q[j + 1], h, equilibrium, order, cache, ledger, partial)
            g_by_order[order].append(system.P @ (base - force) - momentum)
    w_integrals = _finite_output(np.stack(w_integrals), "hidden input integrals", (10, 2))
    correction = w_integrals @ system.PC.T
    result = ArrayRecord(system)
    partial["result"] = result
    result.update(
        system=system, matrix_system=system, wT=w.copy(), w_nodes=np.stack(w_nodes),
        w_integrals=w_integrals,
        kin=np.diff(q, axis=0) - system.h[:, None] * (v[:-1] + v[1:]) / 2.0,
        status="unresolved" if system.unresolved else "complete",
        candidate_zh=None, interval_count=10,
    )
    for order in (2, 4):
        rhs_partial = {}
        partial["right_hand_sides"][str(order)] = rhs_partial
        g = _finite_output(np.stack(g_by_order[order]), "integrated momentum balance", (10, 10))
        rhs_partial["g"] = g
        b = _finite_output(g - correction, "hidden right-hand side", (10, 10))
        rhs_partial["b"] = b
        btilde = _finite_output(_scale_blocks(system, b), "scaled right-hand side", (100,))
        rhs_partial["btilde"] = btilde
        force_residual = g @ system.force_residual_projector.T
        rhs_partial["force_residual"] = force_residual
        z = None if system.unresolved else _solve(system.svd, btilde, ledger, order, rhs_partial)
        rhs_partial["zh"] = z
        result.update({
            f"g{order}": g.reshape(100), f"ghat{order}": g.reshape(100),
            f"g{order}_blocks": g, f"b{order}": b.reshape(100), f"b{order}_blocks": b,
            f"btilde{order}": btilde, f"zh{order}": z,
            f"zTobs{order}": None if z is None else system.PhiT @ z + w,
            f"dynamic_residual{order}": None if z is None else b.reshape(100) - system.L @ z,
            f"dynamic_residual_scaled{order}": None if z is None else btilde - system.Ltilde @ z,
            f"force_residual{order}": force_residual,
            f"force_residual_scaled{order}": force_residual * system.S[None, :],
            f"force_residual_average_force{order}": force_residual / system.h[:, None],
            f"hidden_energy{order}": None if z is None else float(0.5 * np.dot(z, z)),
        })
    result["candidate_zh"] = None if system.unresolved else result.zh4.copy()
    for key, value in result.items():
        if isinstance(value, np.ndarray):
            _finite_output(value, f"result {key}")
        elif isinstance(value, float) and not np.isfinite(value):
            raise KernelNumericalError(f"nonfinite result {key}")
    return result
