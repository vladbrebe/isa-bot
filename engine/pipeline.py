"""The whole research stack as one function. The backtest and the live bot both call this,
so production is the research code, not a re-implementation of it."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import features, model, portfolio, regime


MODES = ("core", "core_tilt", "absolute")


def book(mode: str, pred, sigma, rets, classes, core, cfg) -> pd.DataFrame:
    """Target weights for one of the three books (see config.Config.mode)."""
    if mode == "core":
        return core.copy()
    if mode == "core_tilt":
        return portfolio.targets(pred, sigma, rets, classes, cfg, core=core)
    if mode == "absolute":
        return portfolio.targets(pred, sigma, rets, classes, cfg)
    raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")


def run(px: pd.DataFrame, classes: pd.Series, cfg) -> dict:
    lr = np.log(px).diff()
    p_turb = regime.turbulence(lr.loc[:, classes.eq("equity")].mean(axis=1), cfg)
    feats, sigma, tradable = features.compute(px, classes, p_turb, cfg)
    y = features.label(px, sigma, cfg)
    pred, coefs = model.walk_forward(feats, y, cfg)
    pred = pred.where(tradable)
    rets = px.pct_change()
    core = portfolio.core_weights(tradable, classes, cfg)
    tgt = book(cfg.mode, pred, sigma, rets, classes, core, cfg)
    return {"features": feats, "sigma": sigma, "label": y, "pred": pred, "coefs": coefs,
            "p_turb": p_turb, "target": tgt, "core": core, "rets": rets, "tradable": tradable}
