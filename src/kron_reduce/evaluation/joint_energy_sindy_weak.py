"""Train-only integral initialization for the always-active visible energy blocks."""
from __future__ import annotations

import numpy as np
from scipy.optimize import lsq_linear


def initialize(model, events, budget, *, frames):
    mass, incidence, projection = (t.detach().cpu().numpy() for t in
                                    (model.mass, model.incidence, model.power_projection))
    matrix, targets = [], []
    count = 0
    for event in events:
        budget.spend("weak_cases")
        path = np.asarray(event["x"])[10:11+frames]
        q, v = path[:, :10], path[:, 10:20]
        count += 1
        if len(path) < 2:
            continue
        ref = np.asarray(event["equilibrium"])@incidence
        restoring = np.sin(q@incidence+ref)-np.sin(ref)
        columns = np.concatenate((restoring[:, None, :]*(projection@incidence)[None, :, :],
                                  (v@projection)[:, None, :]*projection[None, :, :]), axis=2)
        impulse = .005*(columns[1:]+columns[:-1])
        starts = np.arange(0, len(impulse), 5)
        ends = np.minimum(starts+5, len(impulse))
        intervals = np.add.reduceat(impulse, starts, axis=0)
        applied = projection@np.asarray(event["p0"])
        rhs = (ends-starts)[:, None]*.01*applied-mass*(v[ends]-v[starts])
        weight = np.sqrt(len(starts))
        matrix.append((intervals/mass[None, :, None]/weight).reshape(-1, columns.shape[2]))
        targets.append((rhs/mass/weight).ravel())
    if not matrix:
        raise ValueError("the integral seed requires visible train trajectory intervals")
    design, response = np.concatenate(matrix), np.concatenate(targets)
    prior = np.concatenate((model.effective_stiffness.detach().cpu().numpy(), .1*mass))
    regularization = .005*np.maximum(np.linalg.norm(design, axis=0), 1e-6)
    budget.spend("weak_solves")
    result = lsq_linear(np.vstack((design, np.diag(regularization))),
                        np.concatenate((response, regularization*prior)), bounds=(1e-6, 100.*prior),
                        tol=1e-8, max_iter=100)
    if not result.success or not np.isfinite(result.x).all():
        raise FloatingPointError("nonnegative integral initialization failed")
    model.set_visible_seed(result.x[:incidence.shape[1]], result.x[incidence.shape[1]:])
    return {"allocated_train": count, "usable_for_weak_seed": len(matrix), "component_rows": len(response),
            "stiffness_min": float(result.x[:incidence.shape[1]].min()),
            "stiffness_max": float(result.x[:incidence.shape[1]].max()),
            "residual_rms": float(np.sqrt(np.mean((design@result.x-response)**2)))}
