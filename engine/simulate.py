"""Execution simulator. Same rules as the live bot: partial adjustment toward target
(Garleanu & Pedersen 2013), a no-trade band, per-instrument costs, idle cash earning the
cash proxy, and a drawdown governor with hysteresis.

Timing: target row t is decided on close t and filled at close t + exec_lag.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rebalance(h: np.ndarray, tgt: np.ndarray, tau: float, band: float, budget: float = 1.0) -> np.ndarray:
    """One day's trade rule, shared with the live bot: close tau of the gap to target, but never
    trade less than band; leave a weight alone once it is within band of target. Never short,
    never borrow (weights may not sum above budget).

    Bug fix (rerun as trial 6): the first version dropped any step smaller than band, so a gap
    under band / tau (2.9%) was never closed. Starting from cash that left ~18% uninvested."""
    gap = tgt - h
    dw = tau * gap
    dw = np.where(np.abs(dw) < band, np.sign(gap) * np.minimum(np.abs(gap), band), dw)
    dw[np.abs(gap) < band] = 0.0
    new = np.maximum(h + dw, 0.0)
    if new.sum() > budget:
        buys = np.clip(new - h, 0, None)
        excess = new.sum() - budget
        new = new - buys * min(1.0, excess / max(buys.sum(), 1e-12))
    return new


def governed(tgt: np.ndarray, core: np.ndarray | None, k: float) -> np.ndarray:
    """Drawdown governor. v1 scales the whole book; v2 scales the tilts and keeps the core."""
    return k * tgt if core is None else core + k * (tgt - core)


def run(target: pd.DataFrame, rets: pd.DataFrame, cash_ret: pd.Series, buy_bps: np.ndarray,
        sell_bps: np.ndarray, cfg, trade_rate: float | None = None, band: float | None = None,
        governor: bool = True, core: pd.DataFrame | None = None) -> dict:
    tau = cfg.trade_rate if trade_rate is None else trade_rate
    band = cfg.no_trade_band if band is None else band
    TG = target.to_numpy(float)
    CG = None if core is None else core.reindex(index=target.index, columns=target.columns).to_numpy(float)
    R = np.nan_to_num(rets.to_numpy(float))
    C = np.nan_to_num(cash_ret.reindex(rets.index).to_numpy(float))
    bb, sb = np.asarray(buy_bps) / 1e4, np.asarray(sell_bps) / 1e4
    T, N = TG.shape
    lag = cfg.exec_lag
    h = np.zeros(N)
    nav, peak, derisk = 1.0, 1.0, False
    net, gross, cost, turn, scale = (np.zeros(T) for _ in range(5))
    H = np.zeros((T, N))
    for d in range(1, T):
        g = h @ R[d] + (1.0 - h.sum()) * C[d]          # P&L of positions held since close d-1
        h = h * (1.0 + R[d]) / (1.0 + g) if (1.0 + g) > 0 else h * 0
        nav_pre = nav * (1.0 + g)
        tgt = TG[d - lag] if d - lag >= 0 else np.zeros(N)
        k = 1.0
        if governor:
            dd = 1.0 - nav / peak                          # known at decision time (close d-1)
            derisk = dd > cfg.dd_soft or (derisk and dd > cfg.dd_soft_release)
            k = 0.5 if derisk else 1.0
        if k != 1.0:
            tgt = governed(tgt, None if CG is None or d - lag < 0 else CG[d - lag], k)
        new = rebalance(h, tgt, tau, band)
        dw = new - h
        c = np.clip(dw, 0, None) @ bb + np.clip(-dw, 0, None) @ sb
        h = new
        nav = nav_pre * (1.0 - c)
        peak = max(peak, nav)
        net[d], gross[d], cost[d], turn[d], scale[d] = (1 + g) * (1 - c) - 1, g, c, np.abs(dw).sum() / 2, k
        H[d] = h
    idx = rets.index
    return {"net": pd.Series(net, idx), "gross": pd.Series(gross, idx), "cost": pd.Series(cost, idx),
            "turnover": pd.Series(turn, idx), "risk_scale": pd.Series(scale, idx),
            "weights": pd.DataFrame(H, idx, rets.columns)}
