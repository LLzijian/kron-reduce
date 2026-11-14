"""Estimate physical coefficients from observed integral momentum balances."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.linalg import cho_solve, cholesky, solve_triangular
from scipy.optimize import lsq_linear, minimize
from scipy.signal import lfilter

DT = .01
GAUSS_NODES, GAUSS_WEIGHTS = np.polynomial.legendre.leggauss(8)
GAUSS_NODES = .5*(1+GAUSS_NODES)
GAUSS_WEIGHTS *= .5


class MomentumBudgetReached(Exception):
    pass


def interpolate_angles(q, v):
    u = GAUSS_NODES[None, :, None]
    q0, q1 = q[:-1, None], q[1:, None]
    return ((1+2*u)*(1-u)**2*q0 + u*u*(3-2*u)*q1
            + DT*u*(1-u)*((1-u)*v[:-1, None]-u*v[1:, None]))


def memory_terms(q_nodes, times, tau):
    """Convolution of interpolated angle displacement and its tau derivative."""
    lag = DT*(1-GAUSS_NODES[:, None])
    filter_weight = DT*GAUSS_WEIGHTS[:, None]*np.exp(-lag/tau)
    impulse = np.sum(q_nodes*filter_weight, axis=1)
    derivative = np.sum(q_nodes*(filter_weight*lag/tau**2), axis=1)
    decay = np.exp(-DT/tau)
    h = np.zeros((len(times), len(tau)))
    dh = np.zeros_like(h)
    for i, a in enumerate(decay):
        h[1:, i] = lfilter([1.], [1., -a], impulse[:, i])
        dh[1:, i] = lfilter([1.], [1., -a], derivative[:, i]+a*DT/tau[i]**2*h[:-1, i])
    ratio = times[:, None]/tau
    rise = -np.expm1(-ratio)
    return h, dh, tau*rise, rise-ratio*np.exp(-ratio)


@dataclass
class MomentumEvent:
    q_nodes: np.ndarray
    times: np.ndarray
    keep: np.ndarray
    rhs: np.ndarray
    fixed: np.ndarray
    weight: float


def time_integral(values):
    return np.concatenate((np.zeros_like(values[:1, 0]), np.cumsum(DT*np.einsum("tne,n->te", values, GAUSS_WEIGHTS), axis=0)))


class MomentumProblem:
    """Shared bounded coefficients with event-specific nuisance moments eliminated by QR."""

    def __init__(self, model, events, target_scale, *, start_frame=0):
        if start_frame not in (0, 10):
            raise ValueError("momentum fit uses the record start or the TASK3 forecast origin")
        n = model.n_nodes
        mass, edges, basis, projection, anchor = (getattr(model, key).detach().cpu().numpy() for key in
            ("mass", "incidence", "damping_basis", "power_projection", "stiffness"))
        self.harmonic = model.harmonic_strength is not None
        self.modulated = model.modulated_damping_strength is not None
        scale = np.asarray(target_scale, dtype=float)
        if scale.shape != (2*n,) or not np.isfinite(scale).all() or np.any(scale <= 0) or not events:
            raise ValueError("finite positive train scales and nonempty train events required")
        self.scale = 1/(mass*scale[n:])
        self.wp = projection*self.scale[:, None]
        self.events = []
        self.zero_information_ids = []
        self.k_count = edges.shape[1]
        self.d_count = basis.shape[1]
        self.linear_count = edges.shape[1]+basis.shape[1]+n+int(self.harmonic)+int(self.modulated)
        self.upper = np.array([np.inf]*(self.linear_count-int(self.harmonic)-int(self.modulated))
                              + [2.]*(int(self.harmonic)+int(self.modulated)))
        self.lower = np.array([1e-10]*(self.linear_count-int(self.harmonic)-int(self.modulated))
                              + [0.]*(int(self.harmonic)+int(self.modulated)))
        for event in events:
            x = np.asarray(event["x"], dtype=float)[:811]
            if (x.ndim != 2 or x.shape[1] != 2*n or len(x) < 12 or not np.isfinite(x).all()):
                raise ValueError("momentum fit needs finite observed 100 Hz train records")
            x = x[start_frame:]
            q, v = x[:, :n], x[:, n:]
            power = np.asarray(event["p0"], dtype=float)
            equilibrium = np.asarray(event["equilibrium"], dtype=float)
            if (power.shape != (n,) or equilibrium.shape != (n,) or not np.isfinite(power).all()
                    or not np.isfinite(equilibrium).all() or abs(power.sum()) > 1e-10
                    or np.max(np.abs(q@mass)) > 1e-10 or np.max(np.abs(v@mass)) > 1e-10):
                raise ValueError("momentum fit requires balanced power and mass-COI observations")
            times = DT*np.arange(len(x))
            keep = np.unique(np.r_[np.arange(5, len(x), 5), len(x)-1])
            if len(keep) == 1:
                # At one elapsed time the free initial port force spans the whole COI force space.
                self.zero_information_ids.append(event["id"])
                continue
            nodes = interpolate_angles(q, v)
            ref = equilibrium@edges
            edge = nodes@edges
            sine = time_integral(np.sin(edge+ref)-np.sin(ref))[keep]
            linear = [sine[:, None, :]*(self.wp@edges), ((q-q[0])@basis)[keep, None, :]*(self.wp@basis)]
            if self.harmonic:
                harmonic = time_integral(np.sin(2*(edge+ref))-np.sin(2*ref))[keep]
                linear.append(((harmonic*anchor)@(self.wp@edges).T)[..., None])
            if self.modulated:
                eta = q@edges
                primitive = eta/2-np.sin(2*eta)/4
                term = (8./edges.shape[1])*((primitive[keep]-primitive[0])@(self.wp@edges).T)
                linear.append(term[..., None])
            weight = 1/np.sqrt(len(keep)*len(events)*n)
            rhs = (times[keep, None]*(power@self.wp.T)-(v[keep]-v[0])*mass*self.scale)*weight
            self.events.append(MomentumEvent(nodes-q[0], times, keep, rhs.reshape(-1),
                                             np.concatenate(linear, axis=2)*weight, weight))

    def solve(self, log_tau, budget):
        budget.spend("momentum_objectives")
        tau = np.exp(log_tau)
        gram = np.zeros((self.linear_count, self.linear_count))
        cross = np.zeros(self.linear_count)
        blocks = []
        for event in self.events:
            h, dh, e, de = (data[event.keep] for data in memory_terms(event.q_nodes, event.times, tau))
            weight = event.weight
            nuisance = (e[:, None, :]*self.wp*weight).reshape(-1, len(tau))
            memory = h[:, None, :]*self.wp*weight
            l = self.k_count+self.d_count
            design = np.concatenate((event.fixed[..., :l], memory, event.fixed[..., l:]), axis=2).reshape(-1, self.linear_count)
            budget.spend("momentum_qr")
            q, r = np.linalg.qr(nuisance, mode="reduced")
            budget.spend("momentum_nuisance_svd")
            singular = np.linalg.svd(r, compute_uv=False)
            if singular[-1] <= 1e-10*singular[0]:
                raise FloatingPointError("rank deficient nuisance momentum block")
            qa, qy = q.T@design, q.T@event.rhs
            reduced, target = design-q@qa, event.rhs-q@qy
            gram += reduced.T@reduced
            cross += reduced.T@target
            blocks.append((event, design, qa, qy, nuisance, r, dh, de))
        col = np.sqrt(gram.diagonal())
        if np.any(col <= 1e-14):
            raise FloatingPointError("unsupported shared momentum coefficient")
        budget.spend("momentum_linear_solves")
        normalized_gram = gram/col[:, None]/col
        budget.spend("momentum_design_spectrum")
        eigenvalues = np.linalg.eigvalsh(normalized_gram)
        if eigenvalues[0] <= 1e-12*eigenvalues[-1]:
            raise FloatingPointError("ill-conditioned shared momentum design")
        factor = cholesky(normalized_gram, lower=False)
        unbounded = cho_solve((factor, False), cross/col)
        lower, upper = self.lower*col, self.upper*col
        if np.any(unbounded < lower) or np.any(unbounded > upper):
            target = solve_triangular(factor.T, cross/col, lower=True)
            bounded = lsq_linear(factor, target, bounds=(lower, upper), method="bvls", tol=1e-12, max_iter=200)
            if not bounded.success:
                raise FloatingPointError("bounded physical coefficient solve failed")
            linear = bounded.x/col
        else:
            linear = unbounded/col
        qcoord = linear*col
        kkt = normalized_gram@qcoord-cross/col
        kkt[(qcoord <= lower+1e-10) & (kkt >= 0)] = 0
        kkt[(qcoord >= upper-1e-10) & (kkt <= 0)] = 0
        if np.linalg.norm(kkt, ord=np.inf) > 1e-9*max(1., np.linalg.norm(cross/col, ord=np.inf)):
            raise FloatingPointError("bounded momentum solve stationarity failed")
        error = 0.
        gradient = np.zeros_like(tau)
        s = linear[self.k_count+self.d_count:self.k_count+self.d_count+len(tau)]
        for event, design, qa, qy, nuisance, r, dh, de in blocks:
            budget.spend("momentum_nuisance_solves")
            w = solve_triangular(r, qy-qa@linear)
            residual = design@linear+nuisance@w-event.rhs
            error += residual@residual
            gradient += 2*np.sum((residual.reshape(-1, len(tau))@self.wp)*(dh*s+de*w), axis=0)*event.weight*tau
        if not (np.isfinite(error) and np.isfinite(gradient).all() and np.isfinite(linear).all()):
            raise FloatingPointError("nonfinite momentum objective")
        return float(error), gradient, linear, float(np.sqrt(eigenvalues[-1]/eigenvalues[0]))


def set_physical(model, log_tau, linear):
    n = model.n_nodes
    k, d = model.stiffness.numel(), model.damping_coefficients.numel()
    fraction = (log_tau-np.log(.02))/np.log(5./.02)
    with torch.no_grad():
        model.edge_log_factor.copy_(model._tensor(np.log(linear[:k]))-model.stiffness.log())
        model.damping_log_factor.copy_(model._tensor(np.log(linear[k:k+d]))-model.damping_coefficients.log())
        model.port_parameter.copy_(model._tensor(.5*np.log(linear[k+d:k+d+n])))
        model.rate_parameter.copy_(model._tensor(np.log(fraction/(1-fraction))))
        for parameter, value in zip((p for p in (model.harmonic_strength, model.modulated_damping_strength)
                                     if p is not None), linear[k+d+n:], strict=True):
            parameter.fill_(float(value))


def momentum_fit(model, events, target_scale, budget, *, max_calls, monitor, start_frame=0):
    """Use train-only momentum to propose checkpoints and val causal score to select one."""
    problem = MomentumProblem(model, events, target_scale, start_frame=start_frame)
    budget.spend("fits")
    bounds = [(np.log(.02001), np.log(4.999))]*model.n_nodes
    initial = np.log(model.tau.detach().cpu().numpy())
    trace, checks, seen = [], [], {}
    best = accepted = None
    base = None
    iterations = 0

    def loss(x):
        nonlocal base
        key = x.tobytes()
        if key in seen:
            return seen[key][3], seen[key][4]
        if len(trace) == max_calls:
            raise MomentumBudgetReached
        value, gradient, linear, condition = problem.solve(x, budget)
        if base is None:
            base = max(value, 1e-16)
        trace.append({"call": len(trace)+1, "train_momentum_mse": value,
                      "gradient_norm": float(np.linalg.norm(gradient)), "design_condition": condition})
        seen[key] = (x.copy(), linear, len(trace), value/base, gradient/base)
        return seen[key][3], seen[key][4]

    def check(state):
        nonlocal best
        set_physical(model, *state[:2])
        value, events = monitor(model)
        checks.append({"iteration": iterations, "call": state[2], "val_nmse": value, "val_event_nmse": events,
                       "log_tau": state[0].tolist(), "physical_linear_coefficients": state[1].tolist()})
        if value is not None and (best is None or value < best[0]):
            best = (value, iterations, state)

    def checkpoint(x):
        nonlocal iterations, accepted
        iterations += 1
        accepted = seen[x.tobytes()]
        if iterations % 10 == 0:
            check(accepted)

    initial = np.clip(initial, bounds[0][0], bounds[0][1])
    failure = None
    converged = False
    try:
        loss(initial)
        accepted = seen[initial.tobytes()]
        check(accepted)
        result = minimize(loss, initial, jac=True, method="L-BFGS-B", bounds=bounds, callback=checkpoint,
                          options={"maxiter": 100, "maxfun": max_calls, "ftol": 1e-13, "gtol": 1e-9, "maxls": 20})
        accepted = seen[result.x.tobytes()]
        converged = bool(result.success)
        reason = str(result.message)
    except MomentumBudgetReached:
        reason = "objective_call_cap"
    except (FloatingPointError, np.linalg.LinAlgError) as error:
        reason = "numerical_failure"
        failure = str(error)
    if accepted is not None and (not checks or checks[-1]["call"] != accepted[2]):
        check(accepted)
    if best is not None:
        set_physical(model, *best[2][:2])
    return {"completed_iterations": iterations, "selected_iteration": best[1] if best else None,
            "selected_val_nmse": best[0] if best else None, "converged": converged, "failure": failure,
            "stop_reason": reason, "train_trace": trace, "val_trace": checks,
            "zero_information_ids": problem.zero_information_ids}
