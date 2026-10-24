"""Fixed integral-initializer arithmetic; no data access, RHS, or rollout.

This scratch module implements the frozen synthetic protocol.  ``build`` uses
only the explicitly supplied visible history.  ``force_projection`` is a test
hook for S02, not an option for a future real-data initializer.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable

import numpy as np
from scipy.linalg import expm as _expm


class ArrayRecord(dict):
    """Dictionary results also support attribute access for runner convenience."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def as_dict(self) -> dict:
        return dict(self)


MatrixSystem = ArrayRecord
KernelResult = ArrayRecord


@dataclass
class Ledger:
    """Count actual operators, including failed attempts, without running RHS.

    ``counts`` records successfully returned calls, ``attempted`` records calls
    before dispatch, and ``entries`` preserves order.  Category totals are
    ``expm``, ``svd``, ``ls``, and ``force``.  Additional keys such as
    ``expm_size_8`` and ``svd_kind_L`` are breakdowns, not extra calls.
    """

    counts: Counter = field(default_factory=Counter)
    attempted: Counter = field(default_factory=Counter)
    failed: Counter = field(default_factory=Counter)
    entries: list[dict] = field(default_factory=list)

    @staticmethod
    def _keys(category: str, kind: str, size: int | None) -> list[str]:
        keys = [category, f"{category}_kind_{kind}"]
        if size is not None:
            keys.append(f"{category}_size_{size}")
        return keys

    def call(
        self, category: str, function: Callable[[], Any], *,
        kind: str = "default", size: int | None = None, shape: tuple | None = None,
    ) -> Any:
        keys = self._keys(category, kind, size)
        self.attempted.update(keys)
        entry = {
            "index": len(self.entries), "category": category, "kind": kind,
            "size": size, "shape": None if shape is None else list(shape),
            "attempted": True, "returned": False, "error": None,
        }
        self.entries.append(entry)
        try:
            result = function()
        except Exception as exc:
            self.failed.update(keys)
            entry["error"] = f"{type(exc).__name__}: {exc}"
            raise
        self.counts.update(keys)
        entry["returned"] = True
        return result

    def expm(self, matrix: np.ndarray, kind: str = "matrix") -> np.ndarray:
        a = _finite_array(matrix, "exponential matrix", ndim=2)
        if a.shape[0] != a.shape[1]:
            raise ValueError("exponential matrix must be square")
        out = self.call("expm", lambda: _expm(a), kind=kind,
                        size=a.shape[0], shape=a.shape)
        return _finite_array(out, "exponential result", ndim=2)

    def svd(self, matrix: np.ndarray, kind: str = "L") -> tuple:
        a = _finite_array(matrix, "SVD matrix", ndim=2)
        out = self.call("svd", lambda: np.linalg.svd(a, full_matrices=False),
                        kind=kind, shape=a.shape)
        for value in out:
            if not np.isfinite(value).all():
                raise FloatingPointError("nonfinite SVD result")
        return out

    def add(self, kind: str, size: int | None = None, *,
            category: str | None = None, count: int = 1) -> None:
        """Register already executed external operators; never dispatch them.

        Prefer ``expm``/``svd`` wrappers.  For an external observer, for example,
        use ``add('observer', 11, category='expm', count=11)`` only after its
        successful calls have been independently counted by the runner.
        """
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError("external count must be a nonnegative integer")
        category = kind if category is None else category
        for _ in range(count):
            self.call(category, lambda: None, kind=kind, size=size)
            self.entries[-1]["external_already_executed"] = True

    def snapshot(self) -> dict:
        return {
            "counts": dict(self.counts),
            "attempted": dict(self.attempted), "returned": dict(self.counts),
            "failed": dict(self.failed), "entries": [dict(x) for x in self.entries],
        }

    def as_dict(self) -> dict:
        return self.snapshot()


def _finite_array(value: Any, name: str, *, ndim: int | None = None) -> np.ndarray:
    out = np.asarray(value, dtype=np.float64)
    if (ndim is not None and out.ndim != ndim) or not np.isfinite(out).all():
        raise ValueError(f"{name} must be finite with ndim={ndim}")
    return out


def _parameters(closure: Any) -> ArrayRecord:
    mass = _finite_array(closure.mass, "mass", ndim=1)
    if mass.size < 2 or np.any(mass <= 0):
        raise ValueError("positive masses at at least two nodes are required")
    n = mass.size
    coupling = _finite_array(closure.coupling, "coupling", ndim=2)
    if coupling.shape != (n, 2):
        raise ValueError("this frozen kernel requires exactly two hidden states")
    damping = _finite_array(closure.damping, "damping", ndim=2)
    skew = _finite_array(closure.latent_skew, "latent_skew", ndim=2)
    dissipation = _finite_array(closure.latent_dissipation, "latent_dissipation", ndim=2)
    if damping.shape != (n, n) or skew.shape != (2, 2) or dissipation.shape != (2, 2):
        raise ValueError("closure matrix dimensions disagree")
    total = float(np.sum(mass))
    Q = np.eye(n) - np.outer(np.ones(n), mass) / total
    P = Q.T.copy()
    return ArrayRecord(m=mass, n=n, r=2, C=coupling, D=damping,
                       A=skew - dissipation, P=P, Q=Q)


def spectral_norm_two_columns(matrix: np.ndarray) -> float:
    """Spectral norm via a scaled 2x2 Gram eigenvalue; no hidden SVD/eig call."""
    a = _finite_array(matrix, "two-column matrix", ndim=2)
    if a.shape[1] != 2:
        raise ValueError("the closed-form norm requires exactly two columns")
    scale = float(np.max(np.abs(a))) if a.size else 0.0
    if scale == 0.0:
        return 0.0
    z = a / scale
    gram = z.T @ z
    disc = float(np.hypot(gram[0, 0] - gram[1, 1], 2.0 * gram[0, 1]))
    eigenvalue = max(0.0, 0.5 * (float(np.trace(gram)) + disc))
    return float(scale * np.sqrt(eigenvalue))


def decompose(matrix: np.ndarray, *, ledger: Ledger, kind: str = "L",
              natural_scale: float = 0.0) -> ArrayRecord:
    a = _finite_array(matrix, "design", ndim=2)
    if min(a.shape) < 1 or not np.isfinite(natural_scale) or natural_scale < 0:
        raise ValueError("invalid decomposition dimensions or natural scale")
    U, singular, Vh = ledger.svd(a, kind=kind)
    sigma1 = float(singular[0])
    tol = float(64 * np.finfo(np.float64).eps * max(a.shape)
                * max(sigma1, float(natural_scale)))
    keep = singular > tol
    rank = int(np.sum(keep))
    full = rank == a.shape[1]
    sigma_min = float(singular[-1]) if len(singular) >= a.shape[1] else 0.0
    return ArrayRecord(
        U=U, s=singular, singular_values=singular, Vh=Vh,
        rank=rank, tol=tol, threshold=tol, keep=keep,
        shape=a.shape, sigma_min=sigma_min,
        condition=float(sigma1 / sigma_min) if full else float("inf"),
        inverse_sigma_min=float(1.0 / sigma_min) if sigma_min > 0 else float("inf"),
        unresolved=not full, natural_norm=float(natural_scale),
    )


def solve_from_svd(system_or_svd: ArrayRecord, rhs: np.ndarray, *,
                   ledger: Ledger, kind: str = "rhs") -> np.ndarray:
    """One vector RHS and one counted LS solve, reusing a saved thin SVD."""
    svd = system_or_svd["svd"] if "svd" in system_or_svd else system_or_svd
    b = _finite_array(rhs, "least-squares right hand side", ndim=1)
    if b.shape != (svd["U"].shape[0],):
        raise ValueError("least-squares right hand side has the wrong shape")

    def solve() -> np.ndarray:
        keep = svd["keep"]
        if not np.any(keep):
            return np.zeros(svd["Vh"].shape[1], dtype=np.float64)
        return svd["Vh"][keep].T @ ((svd["U"][:, keep].T @ b) / svd["s"][keep])

    out = ledger.call("ls", solve, kind=kind, shape=b.shape)
    return _finite_array(out, "least-squares solution", ndim=1)


def _velocity_scales(scales: Any, n: int) -> np.ndarray:
    scale = _finite_array(scales, "velocity scales")
    if scale.ndim == 0:
        scale = np.full(n, float(scale))
    if scale.shape != (n,) or np.any(scale <= 0):
        raise ValueError("velocity scales must be a positive scalar or node vector")
    return scale


def build_matrix_only(time: np.ndarray, closure: Any, scales: Any = 1.0, *,
                      ledger: Ledger,
                      force_projection: np.ndarray | None = None) -> MatrixSystem:
    """J block exponentials, L/C SVDs, zero force calls and zero LS solves."""
    par = _parameters(closure)
    clock = _finite_array(time, "history time", ndim=1)
    if len(clock) < 2 or np.any(np.diff(clock) <= 0):
        raise ValueError("history needs at least two strictly increasing times")
    clock = clock - clock[0]
    h = np.diff(clock)
    T = float(clock[-1])
    P_used = par.P if force_projection is None else _finite_array(
        force_projection, "test-only force projection", ndim=2)
    if P_used.shape != (par.n, par.n):
        raise ValueError("test force projection must be node-square")
    sv = _velocity_scales(scales, par.n)
    S = 1.0 / (par.m * sv)
    c = np.sqrt(h / T) / h
    PC = P_used @ par.C
    natural = spectral_norm_two_columns(S[:, None] * PC)
    ledger.add("gram_norm_2x2", category="arithmetic")

    r = par.r
    aug = np.zeros((4 * r, 4 * r), dtype=np.float64)
    aug[:r, :r] = par.A
    aug[:r, r:2*r] = np.eye(r)
    aug[r:2*r, 2*r:3*r] = np.eye(r)
    aug[2*r:3*r, 3*r:4*r] = np.eye(r)
    blocks = []
    prefixes = [np.eye(r)]
    L_blocks = []
    for width in h:
        exponential = ledger.expm(width * aug, kind="integral_4r")
        block = ArrayRecord(Phi=exponential[:r, :r].copy(),
                            F1=exponential[:r, r:2*r].copy(),
                            F2=exponential[:r, 2*r:3*r].copy(),
                            F3=exponential[:r, 3*r:4*r].copy())
        blocks.append(block)
        L_blocks.append(PC @ prefixes[-1] @ block.F1)
        prefixes.append(block.Phi @ prefixes[-1])
    raw = np.stack(L_blocks)
    scaled = c[:, None, None] * S[None, :, None] * raw
    L = raw.reshape(-1, r)
    Ltilde = scaled.reshape(-1, r)
    svd = decompose(Ltilde, ledger=ledger, kind="L", natural_scale=natural)
    C_svd = decompose(PC, ledger=ledger, kind="C")
    active = C_svd.U[:, C_svd.keep]
    forceproj = active @ active.T
    return MatrixSystem(
        L=L, Ltilde=Ltilde, L_blocks=raw, Ltilde_blocks=scaled,
        c=c, S=S, h=h, time=clock, T=T, mass=par.m.copy(),
        A=par.A.copy(), C=par.C.copy(), D=par.D.copy(),
        P=par.P.copy(), Q=par.Q.copy(), P_used=P_used.copy(), PC=PC,
        blocks=blocks, phi_prefix=np.stack(prefixes), PhiT=prefixes[-1].copy(),
        forceproj=forceproj, force_residual_projector=np.eye(par.n) - forceproj,
        C_svd=C_svd, svd=svd, U=svd.U, s=svd.s,
        singular_values=svd.s, Vh=svd.Vh,
        rank=svd.rank, tol=svd.tol, natural_norm=natural,
        sigma_min=svd.sigma_min, condition=svd.condition,
        inverse_sigma_min=svd.inverse_sigma_min, unresolved=svd.unresolved,
        test_force_projection=force_projection is not None,
    )


# Short alias for callers that prefer the noun used in the protocol.
build_matrices = build_matrix_only


@lru_cache(maxsize=4)
def _incidence(n: int) -> np.ndarray:
    out = np.zeros((n, n * (n - 1) // 2), dtype=np.float64)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            out[i, k], out[j, k] = 1.0, -1.0
            k += 1
    out.flags.writeable = False
    return out


def force_only(closure: Any, q: np.ndarray, eq: np.ndarray, *,
               ledger: Ledger, kind: str = "quadrature") -> np.ndarray:
    """Count/evaluate only F(q), with the production incidence/sign convention."""
    mass = _finite_array(closure.mass, "mass", ndim=1)
    q = _finite_array(q, "force angle", ndim=1)
    eq = _finite_array(eq, "equilibrium angle", ndim=1)
    n = len(mass)
    if q.shape != (n,) or eq.shape != (n,):
        raise ValueError("force angles must be node-aligned")
    k = _finite_array(closure.edge_stiffness, "edge stiffness", ndim=1)
    if k.shape != (n * (n - 1) // 2,):
        raise ValueError("edge stiffness shape differs from the complete graph")
    backbone = closure.backbone
    if backbone not in ("sine", "linear"):
        raise ValueError("unsupported backbone")

    def calculate() -> np.ndarray:
        B = _incidence(n)
        edge = B.T @ q
        reference = B.T @ eq
        edge_force = edge if backbone == "linear" else (
            np.sin(edge + reference) - np.sin(reference))
        return B @ (k * edge_force)

    result = ledger.call("force", calculate, kind=kind, shape=q.shape)
    return _finite_array(result, "model force", ndim=1)


def gauss_rule(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form nodes/weights on [-1, 1]; no polynomial eigensolver."""
    if order == 2:
        return np.array([-1.0, 1.0]) / np.sqrt(3.0), np.ones(2)
    if order == 4:
        root = np.sqrt(6.0 / 5.0)
        outer = np.sqrt((3.0 + 2.0 * root) / 7.0)
        inner = np.sqrt((3.0 - 2.0 * root) / 7.0)
        w_outer = (18.0 - np.sqrt(30.0)) / 36.0
        w_inner = (18.0 + np.sqrt(30.0)) / 36.0
        return (np.array([-outer, -inner, inner, outer]),
                np.array([w_outer, w_inner, w_inner, w_outer]))
    raise ValueError("only the frozen Gauss orders 2 and 4 are permitted")


def integrate_force(q0: np.ndarray, q1: np.ndarray, h: float, closure: Any,
                    eq: np.ndarray, order: int, *, ledger: Ledger) -> np.ndarray:
    """Integrate force on the declared linear q interpolation at fixed nodes."""
    q0 = _finite_array(q0, "left angle", ndim=1)
    q1 = _finite_array(q1, "right angle", ndim=1)
    if q0.shape != q1.shape or not np.isfinite(h) or h <= 0:
        raise ValueError("invalid quadrature interval")
    nodes, weights = gauss_rule(order)
    out = np.zeros_like(q0)
    for node, weight in zip(nodes, weights):
        q = q0 + (0.5 * (node + 1.0)) * (q1 - q0)
        out += (0.5 * h * weight) * force_only(
            closure, q, eq, ledger=ledger, kind=f"gauss{order}")
    return out


def scale_blocks(system: MatrixSystem, raw: np.ndarray) -> np.ndarray:
    """Apply the frozen acceleration scaling to one stacked vector RHS."""
    a = _finite_array(raw, "block right hand side")
    shape = (len(system.h), len(system.S))
    if a.shape not in (shape, (shape[0] * shape[1],)):
        raise ValueError("right hand side does not match the interval/node count")
    return (system.c[:, None] * system.S[None, :] * a.reshape(shape)).reshape(-1)


def build(q: np.ndarray, v: np.ndarray, time: np.ndarray, closure: Any,
          p: np.ndarray, eq: np.ndarray, scales: Any = 1.0, *, ledger: Ledger,
          force_projection: np.ndarray | None = None) -> KernelResult:
    """Fixed complete build: J expm, 6J force, two SVD, and two LS calls.

    ``L/b2/b4/g2/g4`` are unscaled stacked quantities; ``Ltilde/btilde2/
    btilde4`` are the corresponding scaled system. ``ghat2/ghat4`` alias the
    quadrature approximations g2/g4.  ``forceproj`` projects onto range(PC).
    """
    par = _parameters(closure)
    q = _finite_array(q, "history q", ndim=2)
    v = _finite_array(v, "history v", ndim=2)
    clock = _finite_array(time, "history time", ndim=1)
    power = _finite_array(p, "constant power", ndim=1)
    equilibrium = _finite_array(eq, "equilibrium", ndim=1)
    if q.shape != v.shape or q.shape != (len(clock), par.n):
        raise ValueError("visible history must be time-by-node aligned")
    if power.shape != (par.n,) or equilibrium.shape != (par.n,):
        raise ValueError("power and equilibrium must be node-aligned")
    system = build_matrix_only(clock, closure, scales, ledger=ledger,
                               force_projection=force_projection)
    J = len(system.h)
    w = np.zeros(par.r)
    w_nodes = [w.copy()]
    w_integrals = []
    g_by_order = {2: [], 4: []}
    for j, h in enumerate(system.h):
        block = system.blocks[j]
        u = par.C.T @ v[j]
        d = par.C.T @ (v[j + 1] - v[j]) / h
        integral_w = block.F1 @ w + block.F2 @ u + block.F3 @ d
        w_integrals.append(integral_w)
        w = block.Phi @ w + block.F1 @ u + block.F2 @ d
        w_nodes.append(w.copy())
        base = h * power - h * par.D @ (v[j] + v[j + 1]) / 2.0
        momentum = par.m * (v[j + 1] - v[j])
        for order in (2, 4):
            force = integrate_force(q[j], q[j + 1], h, closure, equilibrium,
                                    order, ledger=ledger)
            g_by_order[order].append(system.P_used @ (base - force) - momentum)
    w_integrals = np.stack(w_integrals)
    forcing_correction = w_integrals @ system.PC.T
    result = KernelResult(system)
    result.update(
        system=system, matrix_system=system, wT=w.copy(), w_nodes=np.stack(w_nodes),
        w_integrals=w_integrals,
        kin=np.diff(q, axis=0) - system.h[:, None] * (v[:-1] + v[1:]) / 2.0,
        status="unresolved" if system.unresolved else "complete",
    )
    for order in (2, 4):
        g = np.stack(g_by_order[order])
        b = g - forcing_correction
        btilde = scale_blocks(system, b)
        z = solve_from_svd(system, btilde, ledger=ledger, kind=f"gauss{order}")
        dyn = b.reshape(-1) - system.L @ z
        force_residual = g @ system.force_residual_projector.T
        result.update({
            f"g{order}": g.reshape(-1), f"ghat{order}": g.reshape(-1),
            f"g{order}_blocks": g, f"b{order}": b.reshape(-1),
            f"b{order}_blocks": b, f"btilde{order}": btilde,
            f"zh{order}": z, f"zTobs{order}": system.PhiT @ z + w,
            f"dynamic_residual{order}": dyn,
            f"dynamic_residual_scaled{order}": btilde - system.Ltilde @ z,
            f"force_residual{order}": force_residual,
            f"force_residual_scaled{order}": force_residual * system.S[None, :],
            f"hidden_energy{order}": float(0.5 * np.dot(z, z)),
        })
    result["interval_count"] = J
    return result


# Runner-facing aliases; they do not execute or count additional operators.
quadrature_force = integrate_force
force_value = force_only
