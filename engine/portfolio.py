"""Forecasts -> long-only target weights (no shorting, no leverage: ISA rules).

v1 "absolute": fractional Kelly mean-variance with box, class and budget constraints. The model
decides everything, including how much to hold in cash. Kept as the research record.

v2 "core_tilt": a passive core (60% equity ETFs equal weight, 40% gilts) is always held and the
model may only tilt between assets around it. Forecasts are demeaned across assets first, so a
tilt expresses a relative view, never a market-timing call (v1 showed no timing skill). Tilts
are fully funded, long-only, and shrunk to fit a tracking-error budget and the class caps.

Both are solved by accelerated projected gradient (FISTA). Weak forecasts give small tilts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ewma_cov(R: np.ndarray, halflife: float) -> np.ndarray:
    w = 0.5 ** (np.arange(len(R))[::-1] / halflife)
    w /= w.sum()
    Rc = np.nan_to_num(R)
    return (Rc * w[:, None]).T @ Rc


def project(v: np.ndarray, caps: np.ndarray, budget: float) -> np.ndarray:
    """Euclidean projection onto {0 <= w <= caps, sum(w) <= budget}."""
    w = np.clip(v, 0.0, caps)
    if w.sum() <= budget:
        return w
    lo, hi = 0.0, float(np.max(v))
    for _ in range(60):
        th = 0.5 * (lo + hi)
        if np.clip(v - th, 0.0, caps).sum() > budget:
            lo = th
        else:
            hi = th
    return np.clip(v - hi, 0.0, caps)


def project_sum(v: np.ndarray, caps: np.ndarray, budget: float) -> np.ndarray:
    """Exact Euclidean projection onto {0 <= w <= caps, sum(w) = budget}.

    sum(clip(v - lam, 0, caps)) is piecewise linear and non-increasing in lam, with kinks at
    v and v - caps, so the right lam is found by checking the kinks and interpolating."""
    if budget >= caps.sum():
        return caps.astype(float).copy()
    if budget <= 0:
        return np.zeros_like(v, dtype=float)
    bp = np.sort(np.r_[v, v - caps])
    s = np.clip(v[None, :] - bp[:, None], 0.0, caps).sum(axis=1)
    k = int(np.flatnonzero(s >= budget)[-1])
    lam = bp[k] + (s[k] - budget) * (bp[k + 1] - bp[k]) / (s[k] - s[k + 1])
    return np.clip(v - lam, 0.0, caps)


def solve(mu, S, gamma, caps, budget=1.0, w0=None, iters=1000, tol=1e-9):
    """max mu'w - gamma/2 w'Sw  s.t. box + budget (FISTA)."""
    step = 1.0 / (gamma * np.linalg.eigvalsh(S)[-1] + 1e-12)
    w = project(np.zeros_like(mu) if w0 is None else w0, caps, budget)
    z, t = w.copy(), 1.0
    for _ in range(iters):
        w_new = project(z + step * (mu - gamma * (S @ z)), caps, budget)
        t_new = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
        z = w_new + ((t - 1) / t_new) * (w_new - w)
        if np.max(np.abs(w_new - w)) < tol:
            return w_new
        w, t = w_new, t_new
    return w


def solve_tilt(mu, S, gamma, core, caps, w0=None, iters=1000, tol=1e-9):
    """max mu'w - gamma/2 (w - core)'S(w - core)  s.t. 0 <= w <= caps, sum(w) = sum(core)."""
    budget = float(core.sum())
    step = 1.0 / (gamma * np.linalg.eigvalsh(S)[-1] + 1e-12)
    w = project_sum(core if w0 is None else w0, caps, budget)
    z, t = w.copy(), 1.0
    for _ in range(iters):
        w_new = project_sum(z + step * (mu - gamma * (S @ (z - core))), caps, budget)
        t_new = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
        z = w_new + ((t - 1) / t_new) * (w_new - w)
        if np.max(np.abs(w_new - w)) < tol:
            return w_new
        w, t = w_new, t_new
    return w


def core_weights(tradable: pd.DataFrame, classes: pd.Series, cfg) -> pd.DataFrame:
    """The passive core, and also the benchmark the strategy must beat: core_equity split
    equally across tradable equity ETFs, the rest in the core gilt fund."""
    trd = tradable.astype(float)
    eq = trd.loc[:, classes.reindex(trd.columns).eq("equity").to_numpy()]
    core = (cfg.core_equity * eq.div(eq.sum(axis=1).replace(0.0, np.nan), axis=0))
    core = core.reindex(columns=trd.columns).fillna(0.0)
    if cfg.core_bond in trd.columns:
        core[cfg.core_bond] += (1.0 - cfg.core_equity) * trd[cfg.core_bond]
    return core


def _cov(RT: np.ndarray, t: int, idx: np.ndarray, cfg) -> np.ndarray:
    S = ewma_cov(RT[max(0, t - cfg.cov_window + 1):t + 1, idx], cfg.cov_halflife)
    return (1 - cfg.cov_shrink) * S + cfg.cov_shrink * np.diag(np.diag(S))


def targets(pred: pd.DataFrame, sigma: pd.DataFrame, rets: pd.DataFrame, classes: pd.Series, cfg,
            core: pd.DataFrame | None = None) -> pd.DataFrame:
    """Target weights for each day. core=None runs v1 ("absolute"); a core runs v2 ("core_tilt")."""
    P, SG, RT = pred.to_numpy(float), sigma.to_numpy(float), rets.to_numpy(float)
    cls = classes.reindex(pred.columns).to_numpy()
    T, N = P.shape
    gamma = 1.0 / cfg.kelly_fraction
    if core is not None:
        return _tilt_targets(P, SG, RT, cls, core.reindex(index=pred.index, columns=pred.columns).to_numpy(float),
                             gamma, cfg, pred)
    W = np.zeros((T, N))
    prev = np.zeros(N)
    for t in range(T):
        idx = np.flatnonzero(np.isfinite(P[t]) & np.isfinite(SG[t]))
        if len(idx) == 0:
            continue
        S = _cov(RT, t, idx, cfg)
        mu = P[t, idx] * SG[t, idx] / np.sqrt(cfg.horizon)          # expected daily return
        w = solve(mu, S, gamma, np.full(len(idx), cfg.max_weight), 1.0, w0=prev[idx])
        for c, cap in cfg.class_caps.items():
            m = cls[idx] == c
            if w[m].sum() > cap:
                w[m] *= cap / w[m].sum()
        vol = np.sqrt(max(w @ S @ w, 0.0) * 252)
        if vol > cfg.vol_cap:
            w *= cfg.vol_cap / vol
        W[t, idx] = w
        prev = W[t]
    return pd.DataFrame(W, index=pred.index, columns=pred.columns)


def _tilt_targets(P, SG, RT, cls, C, gamma, cfg, like: pd.DataFrame) -> pd.DataFrame:
    T, N = P.shape
    W = np.nan_to_num(C).copy()                 # no forecast -> hold the core
    prev = W[0].copy()
    for t in range(T):
        c_all = np.nan_to_num(C[t])
        idx = np.flatnonzero(np.isfinite(P[t]) & np.isfinite(SG[t]))
        if len(idx) < 2 or c_all[idx].sum() <= 0:
            prev = W[t]
            continue
        c = c_all[idx]
        S = _cov(RT, t, idx, cfg)
        p = P[t, idx] - P[t, idx].mean()        # relative views only: no market timing
        mu = p * SG[t, idx] / np.sqrt(cfg.horizon)
        caps = np.maximum(cfg.max_weight, c)    # the core itself is always feasible
        w0 = project_sum(prev[idx], caps, float(c.sum()))
        x = solve_tilt(mu, S, gamma, c, caps, w0=w0)
        d = x - c
        s = 1.0
        te = np.sqrt(max(d @ S @ d, 0.0) * 252)
        if te > cfg.te_cap:
            s = cfg.te_cap / te
        full_d = np.zeros(N)
        full_d[idx] = d
        for k, cap in cfg.class_caps.items():   # shrink the tilt until every class cap holds
            m = cls == k
            base, extra = c_all[m].sum(), full_d[m].sum()
            if extra > 0 and base + s * extra > cap:
                s = min(s, max(cap - base, 0.0) / extra)
        W[t] = c_all + s * full_d
        prev = W[t]
    return pd.DataFrame(W, index=like.index, columns=like.columns)
