"""Two-state Gaussian hidden Markov model, fitted by Baum-Welch.

Leonard Baum (of Baum-Welch) was one of Simons' first partners. Here the HMM reads the
equity basket's daily returns and outputs P(turbulent state | data up to today).

Only the *filtered* probability is used. The smoothed probability (forward-backward) uses
future data and is the most common source of look-ahead bias in regime-switching backtests.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

_SQ2PI = math.sqrt(2.0 * math.pi)


def _pdf(x: np.ndarray, mu: float, sd: float) -> np.ndarray:
    z = (x - mu) / sd
    return np.exp(-0.5 * z * z) / (sd * _SQ2PI)


def forward(x: np.ndarray, p: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Scaled forward pass. Returns filtered probs (T,2), scale factors, emission densities."""
    b0, b1 = _pdf(x, p["mu"][0], p["sd"][0]), _pdf(x, p["mu"][1], p["sd"][1])
    (a00, a01), (a10, a11) = p["A"]
    T = len(x)
    f0, f1, c = np.empty(T), np.empty(T), np.empty(T)
    u0, u1 = p["pi"][0] * b0[0], p["pi"][1] * b1[0]
    for t in range(T):
        if t:
            u0 = (f0[t - 1] * a00 + f1[t - 1] * a10) * b0[t]
            u1 = (f0[t - 1] * a01 + f1[t - 1] * a11) * b1[t]
        s = u0 + u1 + 1e-300
        c[t], f0[t], f1[t] = s, u0 / s, u1 / s
    return np.column_stack([f0, f1]), c, b0, b1


def fit(x: np.ndarray, init: dict | None = None, n_iter: int = 200, tol: float = 1e-7) -> dict:
    x = np.asarray(x, float)
    if init is None:
        s = x.std() + 1e-12
        init = {"pi": np.array([0.5, 0.5]), "A": np.array([[0.98, 0.02], [0.05, 0.95]]),
                "mu": np.array([x.mean(), x.mean()]), "sd": np.array([0.7 * s, 1.8 * s])}
    p = {k: np.array(v, float) for k, v in init.items()}
    T, ll_old = len(x), -np.inf
    for _ in range(n_iter):
        alpha, c, b0, b1 = forward(x, p)
        A = p["A"]
        beta = np.ones((T, 2))
        for t in range(T - 2, -1, -1):
            v0, v1 = b0[t + 1] * beta[t + 1, 0], b1[t + 1] * beta[t + 1, 1]
            beta[t, 0] = (A[0, 0] * v0 + A[0, 1] * v1) / c[t + 1]
            beta[t, 1] = (A[1, 0] * v0 + A[1, 1] * v1) / c[t + 1]
        gamma = alpha * beta
        gamma /= gamma.sum(1, keepdims=True)
        nb = np.column_stack([b0[1:], b1[1:]]) * beta[1:] / c[1:, None]
        xi = A * (alpha[:-1].T @ nb)
        p["A"] = xi / xi.sum(1, keepdims=True)
        p["pi"] = gamma[0]
        g = gamma.sum(0) + 1e-12
        p["mu"] = (gamma * x[:, None]).sum(0) / g
        p["sd"] = np.maximum(np.sqrt((gamma * (x[:, None] - p["mu"]) ** 2).sum(0) / g), 1e-5)
        ll = np.log(c).sum()
        if abs(ll - ll_old) < tol * max(1.0, abs(ll)):
            break
        ll_old = ll
    if p["sd"][0] > p["sd"][1]:  # label state 1 as the turbulent (high-variance) one
        p = {"pi": p["pi"][::-1], "A": p["A"][::-1, ::-1], "mu": p["mu"][::-1], "sd": p["sd"][::-1]}
    return p


def turbulence(x: pd.Series, cfg) -> pd.Series:
    """Filtered P(turbulent) for each day, refitting every hmm_refit_every days on a trailing
    window. The refit schedule is anchored at a fixed index so the series is invariant to
    how much future data exists (checked in tests/test_no_lookahead.py)."""
    v = x.fillna(0.0).to_numpy(float)
    T = len(v)
    out = np.full(T, np.nan)
    params = None
    for R in range(cfg.hmm_min_obs, T, cfg.hmm_refit_every):
        lo = max(0, R - cfg.hmm_window + 1)
        params = fit(v[lo:R + 1], init=params)
        hi = min(T, R + cfg.hmm_refit_every)
        alpha, *_ = forward(v[lo:hi], params)
        out[R:hi] = alpha[R - lo:, 1]
    return pd.Series(out, index=x.index, name="p_turbulent")
