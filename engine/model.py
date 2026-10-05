"""One unified model (the Mercer/Brown principle): every signal, every asset, one regression.

Pooled ridge regression of the sigma-scaled forward return on standardised features,
refit monthly on a rolling, exponentially weighted window. Purging: a label at row t needs
prices up to t+lag+h, so training stops at R - (h + lag + embargo) when fitting at day R.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def walk_forward(feats: dict[str, pd.DataFrame], y: pd.DataFrame, cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    names = list(feats)
    X = np.stack([feats[k].to_numpy(float) for k in names], axis=2)   # T x N x F
    Y = y.to_numpy(float)
    T, N, F = X.shape
    gap = cfg.horizon + cfg.exec_lag + cfg.embargo
    pred = np.full((T, N), np.nan)
    rows = []
    for R in range(gap + cfg.min_train, T, cfg.refit_every):   # fixed anchor: no dependence on T
        hi = R - gap
        lo = max(0, hi - cfg.train_window + 1)
        Xt, Yt = X[lo:hi + 1], Y[lo:hi + 1]
        wt = np.broadcast_to((0.5 ** ((hi - np.arange(lo, hi + 1)) / cfg.train_halflife))[:, None], Yt.shape)
        m = np.isfinite(Yt) & np.isfinite(Xt).all(axis=2)
        if m.any(axis=1).sum() < cfg.min_train:               # need min_train *days* of history
            continue
        x, yy, w = Xt[m], Yt[m], wt[m]
        sd = np.sqrt((w[:, None] * x ** 2).sum(0) / w.sum())
        sd[sd == 0] = 1.0
        Z = np.column_stack([np.ones(len(yy)), x / sd])
        P = np.diag([0.0] + [cfg.ridge * w.sum()] * F)      # intercept unpenalised
        b = np.linalg.solve(Z.T @ (Z * w[:, None]) + P, Z.T @ (w * yy))
        end = min(T, R + cfg.refit_every)
        pred[R:end] = b[0] + (X[R:end] / sd) @ b[1:]
        rows.append([y.index[R], *b])                          # coefficients per 1 sd of feature
    coefs = pd.DataFrame(rows, columns=["date", "intercept", *names]).set_index("date")
    return pd.DataFrame(pred, index=y.index, columns=y.columns), coefs
