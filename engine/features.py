"""Features: many weak, economically motivated signals. None is meant to work alone.

All features at row t use prices up to the close of t only. Each is roughly unit-scale and
clipped at +/-4 so a single bad day can't dominate the fit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

FEATURES = ["trend_fast", "trend_med", "trend_slow", "tsmom_12m", "reversal_5d",
            "xs_momentum", "vol_ratio", "regime_x_beta", "tom_x_beta"]


def daily_vol(lr: pd.DataFrame, cfg) -> pd.DataFrame:
    return lr.ewm(halflife=cfg.vol_halflife, min_periods=60).std().clip(lower=5e-4)


def _rank_gauss(df: pd.DataFrame) -> pd.DataFrame:
    n = df.notna().sum(axis=1)
    u = df.rank(axis=1).sub(0.5).div(n, axis=0)
    return pd.DataFrame(norm.ppf(u.to_numpy(float)), index=df.index, columns=df.columns)


def _ewmac(logp: pd.DataFrame, sigma: pd.DataFrame, fast: int, slow: int) -> pd.DataFrame:
    """Baz et al. (2015) CTA trend signal: EMA crossover in sigma units, normalised by its own
    1y dispersion, passed through a response that fades very extended trends."""
    x = (logp.ewm(span=fast, min_periods=slow).mean() - logp.ewm(span=slow, min_periods=slow).mean()) / sigma
    y = x / x.rolling(252, min_periods=126).std()
    return y * np.exp(-(y ** 2) / 4.0) / 0.89


def turn_of_month(index: pd.DatetimeIndex, horizon: int, lag: int) -> pd.Series:
    """Share of the label window's business days that fall in the turn-of-month window
    (last business day + first three). Calendar-only, so knowable in advance."""
    bd = pd.bdate_range(index[0], index[-1] + pd.offsets.BDay(horizon + lag + 30))
    df = pd.DataFrame({"m": bd.to_period("M")}, index=bd)
    k = df.groupby("m").cumcount()
    n = k.groupby(df["m"]).transform("max")
    tom = ((k < 3) | (k == n)).to_numpy(float)
    pos = bd.searchsorted(index, side="right")  # first business day after t (the fill day)
    vals = [tom[p + lag: p + lag + horizon].mean() for p in pos]  # bd[p] is the fill day t+1
    return pd.Series(vals, index=index)


def compute(px: pd.DataFrame, classes: pd.Series, p_turb: pd.Series, cfg) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Returns (features, sigma, tradable mask)."""
    logp = np.log(px)
    lr = logp.diff()
    sigma = daily_vol(lr, cfg)
    tradable = lr.notna().cumsum().ge(cfg.min_history) & px.notna()

    eq = lr.loc[:, classes.eq("equity")].mean(axis=1)
    cov = lr.mul(eq, axis=0).rolling(252, min_periods=126).mean() - \
        lr.rolling(252, min_periods=126).mean().mul(eq.rolling(252, min_periods=126).mean(), axis=0)
    beta = cov.div(eq.rolling(252, min_periods=126).var(ddof=0), axis=0)

    def ret(n):
        return (logp - logp.shift(n)) / (sigma * np.sqrt(n))

    tom = turn_of_month(px.index, cfg.horizon, cfg.exec_lag)
    f = {
        "trend_fast": _ewmac(logp, sigma, 8, 32),
        "trend_med": _ewmac(logp, sigma, 16, 64),
        "trend_slow": _ewmac(logp, sigma, 32, 128),
        "tsmom_12m": ret(252),                               # Moskowitz, Ooi, Pedersen (2012)
        "reversal_5d": -ret(5),                              # short-horizon liquidity provision
        "xs_momentum": _rank_gauss(ret(126).where(tradable)),  # relative strength across assets
        "vol_ratio": np.log(lr.rolling(20).std() / lr.rolling(252, min_periods=126).std()),  # Moreira-Muir
        "regime_x_beta": beta.mul(0.5 - p_turb, axis=0),     # turbulence hurts high-beta assets
        "tom_x_beta": beta.mul(tom, axis=0),                 # turn-of-month equity flows
    }
    f = {k: v.clip(-4, 4).where(tradable) for k, v in f.items()}
    return f, sigma, tradable


def label(px: pd.DataFrame, sigma: pd.DataFrame, cfg) -> pd.DataFrame:
    """Target: return from the fill (close t+lag) to close t+lag+h, in units of ex-ante sigma."""
    logp = np.log(px)
    fwd = logp.shift(-(cfg.exec_lag + cfg.horizon)) - logp.shift(-cfg.exec_lag)
    return (fwd / (sigma * np.sqrt(cfg.horizon))).clip(-4, 4)
