"""Fixed Gaussian prior/noise information control; no parameters are learned."""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve


def condition_initial_state(design, residual, prior, noise, *, spend):
    """Condition a zero-mean initial Gaussian on independent noisy frames."""
    o, y, c, r = [np.asarray(x, dtype=np.float64) for x in (design, residual, prior, noise)]
    if (o.ndim != 3 or y.ndim != 3 or y.shape[1:] != o.shape[:2] or not len(y)
            or c.shape != (o.shape[2], o.shape[2]) or r.shape != (o.shape[1], o.shape[1])
            or min(o.shape) < 1 or not all(np.isfinite(x).all() for x in (o, y, c, r))
            or not np.allclose(c, c.T, rtol=1e-12, atol=1e-14)
            or not np.allclose(r, r.T, rtol=1e-12, atol=1e-14)):
        raise ValueError("invalid finite Gaussian observation problem")
    frames, observed, latent = o.shape
    spend("cholesky", 2)
    prior_factor = cho_factor(c, lower=True)
    noise_factor = cho_factor(r, lower=True)
    spend("linear_solves", 3)
    prior_precision = cho_solve(prior_factor, np.eye(latent))
    rinv_o = cho_solve(noise_factor, o.transpose(1, 0, 2).reshape(observed, -1))
    rinv_o = rinv_o.reshape(observed, frames, latent).transpose(1, 0, 2)
    rinv_y = cho_solve(noise_factor, y.transpose(2, 0, 1).reshape(observed, -1))
    rinv_y = rinv_y.reshape(observed, len(y), frames).transpose(1, 2, 0)
    spend("matrix_products", 2*frames)
    precision = prior_precision + np.einsum("tir,tis->rs", o, rinv_o)
    rhs = np.einsum("tir,bti->rb", o, rinv_y)
    spend("cholesky")
    precision_factor = cho_factor(precision, lower=True)
    spend("linear_solves")
    solved = cho_solve(precision_factor, np.c_[np.eye(latent), rhs])
    covariance, mean = solved[:, :latent], solved[:, latent:].T
    spend("matrix_products", 8)
    prior_residual = c@prior_precision-np.eye(latent)
    inverse_residual = precision@covariance-np.eye(latent)
    mean_residual = precision@mean.T-rhs
    prior_error = float(np.max(np.abs(prior_residual)))
    inverse_error = float(np.max(np.abs(inverse_residual)))
    mean_error = float(np.linalg.norm(mean_residual)/max(1., np.linalg.norm(rhs)))
    factors = [np.tril(f[0]) for f in (prior_factor, noise_factor, precision_factor)]
    factor_error = max(float(np.linalg.norm(f@f.T-a)/max(1., np.linalg.norm(a)))
                       for f, a in zip(factors, (c, r, precision), strict=True))
    oflat = o.transpose(1, 0, 2).reshape(observed, -1)
    yflat = y.transpose(2, 0, 1).reshape(observed, -1)
    roflat = rinv_o.transpose(1, 0, 2).reshape(observed, -1)
    ryflat = rinv_y.transpose(2, 0, 1).reshape(observed, -1)
    equations = ((c, prior_precision, np.eye(latent), prior_residual),
                 (r, roflat, oflat, r@roflat-oflat), (r, ryflat, yflat, r@ryflat-yflat),
                 (precision, solved, np.c_[np.eye(latent), rhs], np.c_[inverse_residual, mean_residual]))
    backward_error = max(float(np.linalg.norm(e)/max(1e-300, np.linalg.norm(a)*np.linalg.norm(x)+np.linalg.norm(b)))
                         for a, x, b, e in equations)
    symmetry_error = float(np.linalg.norm(covariance-covariance.T)/max(1., np.linalg.norm(covariance)))
    if (not all(np.isfinite(x).all() for x in (solved, precision, rhs, *factors))
            or max(prior_error, inverse_error) > 1e-8
            or max(factor_error, backward_error, symmetry_error) > 1e-12):
        raise FloatingPointError("Gaussian posterior linear-system accuracy failed")
    return {"mean": mean, "covariance": covariance, "precision": precision, "rhs": rhs,
            "prior": c, "noise": r, "prior_precision": prior_precision,
            "prior_cholesky": factors[0], "noise_cholesky": factors[1], "precision_cholesky": factors[2],
            "noise_inverse_design": rinv_o, "noise_inverse_residual": rinv_y,
            "prior_inverse_error": prior_error, "posterior_inverse_error": inverse_error,
            "posterior_mean_error": mean_error, "factor_reconstruction_error": factor_error,
            "solve_backward_error": backward_error, "covariance_symmetry_error": symmetry_error}
