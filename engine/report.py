"""One-page tear sheet."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def tear_sheet(curves: dict[str, pd.Series], cash: pd.Series, weights: pd.DataFrame, classes: pd.Series,
               coefs: pd.DataFrame, decay: pd.Series, path: str) -> None:
    fig, ax = plt.subplots(3, 2, figsize=(13, 13))
    colors = ["#1f3a5f", "#c0504d", "#9bbb59", "#8064a2", "#f79646"]
    for (name, r), c in zip(curves.items(), colors):
        eq = (1 + r).cumprod()
        ax[0, 0].plot(eq.index, eq, label=name, color=c, lw=1.6 if c == colors[0] else 1.1)
        ax[0, 1].plot(eq.index, eq / eq.cummax() - 1, color=c, lw=1.0, label=name)
        ex = r - cash.reindex(r.index).fillna(0)
        rs = ex.rolling(252).mean() / ex.rolling(252).std() * np.sqrt(252)
        ax[1, 0].plot(rs.index, rs, color=c, lw=1.0, label=name)
    ax[0, 0].set_yscale("log"); ax[0, 0].set_title("Growth of £1 (net of costs, log scale)"); ax[0, 0].legend(fontsize=8)
    ax[0, 1].set_title("Drawdown"); ax[0, 1].legend(fontsize=8)
    ax[1, 0].axhline(0, color="k", lw=0.5); ax[1, 0].set_title("Rolling 1-year Sharpe (excess of cash)")

    w = weights.loc[next(iter(curves.values())).index].T.groupby(classes).sum().T.resample("ME").mean()
    w["cash"] = (1 - w.sum(axis=1)).clip(lower=0)
    ax[1, 1].stackplot(w.index, w.T.values, labels=w.columns, alpha=0.85)
    ax[1, 1].set_ylim(0, 1); ax[1, 1].legend(fontsize=8, loc="upper left"); ax[1, 1].set_title("Model v2 allocation by asset class")

    cf = coefs.drop(columns="intercept")
    for col in cf.columns:
        ax[2, 0].plot(cf.index, cf[col], lw=1.0, label=col)
    ax[2, 0].axhline(0, color="k", lw=0.5); ax[2, 0].legend(fontsize=7, ncol=3)
    ax[2, 0].set_title("Unified model: coefficient per 1 sd of each feature, by refit")

    ax[2, 1].bar(decay.index, decay.values, color="#1f3a5f")
    ax[2, 1].axhline(0, color="k", lw=0.5)
    ax[2, 1].set_title("Signal decay: rank IC vs single-day return k days after fill")
    ax[2, 1].set_xlabel("k (trading days)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
