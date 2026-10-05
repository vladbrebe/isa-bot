"""Statistics a quant actually trusts: excess-return Sharpe, deflated Sharpe, alpha vs the
passive alternative, information coefficient with honest (non-overlapping) t-stats."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis, spearmanr

EULER = 0.5772156649


def summary(r: pd.Series, cash: pd.Series) -> dict:
    r = r.dropna()
    ex = r - cash.reindex(r.index).fillna(0.0)
    eq = (1 + r).cumprod()
    yrs = len(r) / 252
    m = (1 + r).resample("ME").prod() - 1
    return {"CAGR": eq.iloc[-1] ** (1 / yrs) - 1, "Vol": r.std() * np.sqrt(252),
            "Sharpe": ex.mean() / ex.std() * np.sqrt(252), "MaxDD": (eq / eq.cummax() - 1).min(),
            "WorstMonth": m.min(), "PosMonths": (m > 0).mean()}


def psr(ex: pd.Series, sr0: float = 0.0) -> float:
    """Probabilistic Sharpe ratio (Bailey & Lopez de Prado 2012); sr0 is per-period."""
    x = ex.dropna().to_numpy()
    sr = x.mean() / x.std(ddof=1)
    g3, g4 = skew(x), kurtosis(x, fisher=False)
    return float(norm.cdf((sr - sr0) * np.sqrt(len(x) - 1) / np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr * sr)))


def dsr(ex: pd.Series, n_trials: int) -> tuple[float, float]:
    """Deflated Sharpe ratio (Bailey & Lopez de Prado 2014). Returns (DSR, the annualised
    Sharpe that the best of n_trials zero-skill strategies would reach by luck)."""
    n = max(int(n_trials), 2)
    T = ex.dropna().size
    sr0 = np.sqrt(1.0 / T) * ((1 - EULER) * norm.ppf(1 - 1 / n) + EULER * norm.ppf(1 - 1 / (n * np.e)))
    return psr(ex, sr0), sr0 * np.sqrt(252)


def alpha_vs(r: pd.Series, bench: pd.Series, cash: pd.Series, lag: int = 5) -> tuple[float, float, float]:
    """OLS of strategy excess returns on benchmark excess returns, Newey-West t-stat.
    Returns (annualised alpha, its t-stat, beta)."""
    df = pd.concat([r, bench], axis=1).dropna()
    c = cash.reindex(df.index).fillna(0.0)
    y, x = (df.iloc[:, 0] - c).to_numpy(), (df.iloc[:, 1] - c).to_numpy()
    X = np.column_stack([np.ones_like(x), x])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    Xe = X * e[:, None]
    S = Xe.T @ Xe
    for L in range(1, lag + 1):
        G = Xe[L:].T @ Xe[:-L]
        S += (1 - L / (lag + 1)) * (G + G.T)
    XtX_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(XtX_inv @ S @ XtX_inv))
    return b[0] * 252, b[0] / se[0], b[1]


def ic_stats(pred: pd.DataFrame, y: pd.DataFrame, step: int) -> tuple[float, float, int]:
    """Mean cross-sectional rank IC, t-stat on non-overlapping dates only."""
    ics = []
    for t in pred.index[::step]:
        a, b = pred.loc[t], y.loc[t]
        m = a.notna() & b.notna()
        if m.sum() >= 5:
            ics.append(spearmanr(a[m], b[m]).statistic)
    ics = np.array(ics)
    return ics.mean(), ics.mean() / ics.std(ddof=1) * np.sqrt(len(ics)), len(ics)


def ic_decay(pred: pd.DataFrame, px: pd.DataFrame, lag: int, max_h: int = 20, step: int = 5) -> pd.Series:
    """Rank IC of today's forecast against the single-day return k days after the fill:
    the signal's half-life tells you how fast to trade."""
    r = np.log(px).diff()
    out = {}
    for k in range(1, max_h + 1):
        fwd = r.shift(-(lag + k))
        vals = []
        for t in pred.index[::step]:
            a, b = pred.loc[t], fwd.loc[t]
            m = a.notna() & b.notna()
            if m.sum() >= 5:
                vals.append(spearmanr(a[m], b[m]).statistic)
        out[k] = np.nanmean(vals)
    return pd.Series(out)
