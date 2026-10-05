"""Research run: walk-forward backtest, honest statistics, tear sheet.

    python backtest.py            # uses cached prices if present
    python backtest.py --refresh  # re-download
"""
from __future__ import annotations

import argparse
import os
from dataclasses import replace

import numpy as np
import pandas as pd

from config import BENCH_EQUITY, CASH_PROXY, N_TRIALS, UNIVERSE, Config, costs_bps
from engine import data, metrics, pipeline, report, simulate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--out", default="output")
    args = ap.parse_args()
    cfg, tickers = Config(), list(UNIVERSE)
    os.makedirs("data", exist_ok=True)
    os.makedirs(args.out, exist_ok=True)

    cache = "data/prices.pkl"
    if os.path.exists(cache) and not args.refresh:
        raw = pd.read_pickle(cache)
    else:
        raw = data.download(tickers + [CASH_PROXY], cfg.start)
        raw.to_pickle(cache)
    px, fixes = data.clean(raw[tickers], cfg)
    cash = raw[CASH_PROXY].reindex(px.index).ffill().pct_change().clip(-0.01, 0.01).fillna(0.0)
    classes = pd.Series({k: v.cls for k, v in UNIVERSE.items()})
    bad = fixes[fixes.fix != "burn_in"]
    print(f"Data: {px.shape[1]} ETFs, {px.index[0].date()} to {px.index[-1].date()}; "
          f"repaired {len(bad)} bad-print episodes in {bad.ticker.nunique()} tickers")

    out = pipeline.run(px, classes, cfg)
    core = out["core"]
    buy = np.array([costs_bps(t)[0] for t in tickers])
    sell = np.array([costs_bps(t)[1] for t in tickers])

    def sim(target, **kw):
        return simulate.run(target, out["rets"], cash, buy, sell, cfg, **kw)

    def core_for(mode):  # the governor shrinks tilts around the core, or the whole v1 book
        return None if mode == "absolute" else core

    # Scored from the first day the model has a forecast; nothing before it counts for anyone.
    start = out["pred"].index[int(np.argmax(out["pred"].notna().any(axis=1).to_numpy()))]
    oos = slice(start + pd.Timedelta(days=1), None)
    trd = out["tradable"]

    # Passive and naive alternatives, same costs and fill timing. A model must beat these.
    spx = pd.DataFrame(0.0, index=px.index, columns=tickers)
    spx[BENCH_EQUITY] = trd[BENCH_EQUITY].astype(float)
    ma = (px > px.rolling(200).mean()) & trd
    trend_v1 = ma.astype(float) / len(tickers)
    nets = {name: sim(t, trade_rate=1.0, band=0.02, governor=False)["net"]
            for name, t in [("Passive 60/40", core), ("S&P 500 only", spx)]}
    nets["MA-200 trend (first bot idea)"] = sim(trend_v1, trade_rate=1.0, band=0.005, governor=False)["net"]

    # The three books under the bot's own trading rules. The research question is v2 vs 60/40.
    label = {"absolute": "Model v1, absolute (trials 1-4)", "core_tilt": "Model v2, core + tilt (trials 5-6)",
             "core": "Passive core, bot's trading rules"}
    books, res = {}, {}
    for mode in ("absolute", "core_tilt", "core"):
        books[mode] = out["target"] if mode == cfg.mode else pipeline.book(
            mode, out["pred"], out["sigma"], out["rets"], classes, core, replace(cfg, mode=mode))
        res[mode] = sim(books[mode], core=core_for(mode))
        nets[label[mode] + (" <- bot runs this" if mode == cfg.mode else "")] = res[mode]["net"]
    v2 = "core_tilt"
    nets["Model v2, no DD governor"] = sim(books[v2], governor=False, core=core)["net"]
    for mult in (0.0, 2.0):
        nets[f"Model v2, costs x{mult:g}"] = simulate.run(books[v2], out["rets"], cash, buy * mult, sell * mult,
                                                          cfg, core=core)["net"]

    curves = {k: v.loc[oos] for k, v in nets.items()}
    table = pd.DataFrame({k: metrics.summary(v, cash) for k, v in curves.items()}).T
    pd.set_option("display.width", 160)
    fmt = table.copy()
    for c in ["CAGR", "Vol", "MaxDD", "WorstMonth", "PosMonths"]:
        fmt[c] = (table[c] * 100).map("{:.1f}%".format)
    fmt["Sharpe"] = table["Sharpe"].map("{:.2f}".format)
    first = curves["Passive 60/40"].index
    print(f"\nOut-of-sample {first[0].date()} to {first[-1].date()} "
          f"(every number below is walk-forward; nothing was fitted on the period it is scored on)\n")
    print(fmt.to_string())

    s = res[v2]["net"].loc[oos]
    ex = s - cash.reindex(s.index).fillna(0)
    d, sr_luck = metrics.dsr(ex, N_TRIALS)
    a, at, beta = metrics.alpha_vs(s, curves["Passive 60/40"], cash)
    ic, ict, nic = metrics.ic_stats(out["pred"].loc[oos], out["label"].loc[oos], cfg.horizon)
    act = (s - curves["Passive 60/40"]).dropna()
    w = res[v2]["weights"].loc[oos]
    print(f"\nModel v2 against the passive 60/40 it tilts around:")
    print(f"Deflated Sharpe (N_TRIALS={N_TRIALS}): P(true Sharpe > {sr_luck:.2f} luck threshold) = {d:.2f}")
    print(f"Alpha vs passive 60/40: {a * 100:.2f}%/yr, Newey-West t = {at:.2f}, beta = {beta:.2f}")
    print(f"Active return vs 60/40: {act.mean() * 252 * 100:.2f}%/yr, tracking error {act.std() * np.sqrt(252) * 100:.2f}%,"
          f" information ratio {act.mean() / act.std() * np.sqrt(252):.2f}")
    print(f"Forecast rank IC: {ic:.3f} (t = {ict:.2f} on {nic} non-overlapping dates)")
    for mode in ("core_tilt", cfg.mode) if cfg.mode != "core_tilt" else ("core_tilt",):
        r = res[mode]
        print(f"{label[mode]}: avg invested {r['weights'].loc[oos].sum(axis=1).mean() * 100:.0f}%"
              f" | turnover {r['turnover'].loc[oos].mean() * 252 * 100:.0f}%/yr"
              f" | cost drag {r['cost'].loc[oos].sum() / (len(s) / 252) * 1e4:.0f} bp/yr"
              + (f" | governor active {(r['risk_scale'].loc[oos] < 1).mean() * 100:.0f}% of days" if mode != "core" else ""))
    cf = out["coefs"].drop(columns="intercept")
    print("\nCoefficient per 1 sd (mean over refits, share of refits with that sign):")
    print("  " + ", ".join(f"{k} {v:+.3f} ({(np.sign(cf[k]) == np.sign(v)).mean() * 100:.0f}%)"
                           for k, v in cf.mean().items()))

    bot = [k for k in curves if k.endswith("<- bot runs this")][0]
    plot = {bot.replace(" <- bot runs this", " (bot)"): curves[bot]}
    for k in ("Passive 60/40", "S&P 500 only", label["core_tilt"], label["absolute"]):
        if k in curves and len(plot) < 5:
            plot[k] = curves[k]
    decay = metrics.ic_decay(out["pred"].loc[oos], px, cfg.exec_lag)
    report.tear_sheet(plot, cash, res[v2]["weights"], classes, out["coefs"], decay,
                      os.path.join(args.out, "tearsheet.png"))
    table.to_csv(os.path.join(args.out, "summary.csv"))
    print(f"\nTear sheet: {args.out}/tearsheet.png")


if __name__ == "__main__":
    main()
