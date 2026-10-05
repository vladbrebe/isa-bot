"""Data: download, repair, align.

Renaissance's first real edge was cleaner data than anyone else. Yahoo's LSE feed has a
specific, recurring fault: a day (or a few weeks) quoted on the wrong currency line, e.g. a
USD price on a GBX line, which shows up as a jump of roughly the GBP/USD rate that later
reverses. Left in, those prints create fake trends and fake reversals.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd


def download(tickers: list[str], start: str, end: str | None = None, retries: int = 3) -> pd.DataFrame:
    """Adjusted closes (dividends reinvested), one batched yfinance call, with backoff."""
    import yfinance as yf

    last_err = None
    for attempt in range(retries):
        try:
            raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                              progress=False, threads=True)
            if raw is not None and not raw.empty:
                close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
                if not isinstance(raw.columns, pd.MultiIndex):
                    close.columns = tickers[:1]
                close = close.reindex(columns=tickers)
                close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
                if close.notna().any().all():
                    return close.sort_index()
                last_err = RuntimeError(f"missing tickers: {list(close.columns[~close.notna().any()])}")
        except Exception as e:  # network, rate limit, parsing
            last_err = e
        time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"price download failed after {retries} attempts: {last_err}")


def _repair(s: pd.Series, burn_in: int, jump: float, sigmas: float,
            max_len: int = 60, tol: float = 0.3) -> tuple[pd.Series, list[tuple]]:
    s = s.where(s > 0).copy()
    events: list[tuple] = []
    v = s.dropna()
    if len(v) <= burn_in:
        return s * np.nan, [("too_short", None, None)]
    if burn_in:
        s.loc[: v.index[burn_in - 1]] = np.nan
        events.append(("burn_in", v.index[0], v.index[burn_in - 1]))

    # Reverting jumps: big move, then an opposite move of similar size within max_len sessions
    # that lands back near the pre-jump level. Real crashes don't retrace to within noise.
    for _ in range(100):
        v = s.dropna()
        x = np.log(v.to_numpy())
        d = np.diff(x)
        sd = pd.Series(d).rolling(60, min_periods=20).std().shift(1).bfill().to_numpy()
        hit = False
        for i in np.flatnonzero(np.abs(d) > np.maximum(jump, sigmas * sd)):
            for j in range(i + 1, min(len(d), i + 1 + max_len)):
                if np.sign(d[j]) == -np.sign(d[i]) and abs(d[i] + d[j]) < tol * abs(d[i]):
                    bad = v.index[i + 1: j + 1]
                    s.loc[bad] = np.nan
                    events.append(("reverting_jump", bad[0], bad[-1]))
                    hit = True
                    break
            if hit:
                break
        if not hit:
            break

    # Permanent x100 unit switches (GBP <-> GBX): rescale the earlier history.
    v = s.dropna()
    d = np.diff(np.log(v.to_numpy()))
    for i in np.flatnonzero(np.abs(np.abs(d) - np.log(100.0)) < 0.1):
        s.loc[: v.index[i]] = s.loc[: v.index[i]] * np.exp(d[i])
        events.append(("unit_switch", v.index[i], v.index[i + 1]))
    return s, events


def clean(close: pd.DataFrame, cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Repair vendor errors, align on one calendar, forward-fill short gaps only."""
    out, log = {}, []
    for col in close.columns:
        s, ev = _repair(close[col], cfg.burn_in, cfg.bad_jump, cfg.bad_sigmas)
        out[col] = s
        log += [(col, *e) for e in ev]
    px = pd.DataFrame(out).dropna(how="all").ffill(limit=5)
    return px, pd.DataFrame(log, columns=["ticker", "fix", "from", "to"])


def last_print_suspect(px: pd.DataFrame, cfg) -> list[str]:
    """Causal guard for live trading: the newest print can't be checked for reversal yet,
    so quarantine any asset whose last move is implausibly large for its own volatility."""
    lr = np.log(px).diff()
    sd = lr.iloc[:-1].tail(60).std()
    last = lr.iloc[-1].abs()
    return list(last[(last > np.maximum(cfg.bad_jump, cfg.bad_sigmas * sd))].index)
