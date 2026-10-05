"""Daily live run. GitHub Actions calls this once each weekday during London trading hours.

Safety, in the order it is checked:
  1. T212_ENV is "demo" (paper money) unless it is set to "live".
  2. No order is sent unless TRADING_ENABLED=true. That repository variable is the kill switch.
     Without it, or with DRY_RUN=true, the bot works out and logs the trades it would make.
  3. Circuit breaker: if the unit value falls more than dd_hard below its peak, trading stops
     until someone re-runs the workflow with reset_halt. With dd_hard_action="flatten" it sells
     everything once instead of only stopping.
  4. It fails closed: unmapped tickers, stale prices, a closed market, or manual orders still
     pending mean no orders. Anything unexpected stops the run and turns it red.

Holdings outside the bot's 15 ETFs are left alone, unless a run is started with
SELL_OTHER_HOLDINGS=true (the workflow's "sell_other_holdings" box). That sells them all once,
which is how the bot takes over an account from an older bot.

    python live.py --map    # check the Yahoo -> Trading 212 ticker mapping, place nothing
    python live.py          # one run, settings from environment variables
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from config import T212_TICKERS, UNIVERSE, Config
from engine import data, pipeline
from engine import execution as ex
from engine.broker import T212Error
from engine.simulate import governed

STATE_DIR = "state"


class Report:
    """Prints as it goes and keeps a markdown copy for the GitHub run summary."""

    def __init__(self):
        self.lines: list[str] = []

    def __call__(self, line: str = "") -> None:
        print(line, flush=True)
        self.lines.append(line)

    def flush_summary(self) -> None:
        path = os.environ.get("GITHUB_STEP_SUMMARY")
        if path:
            with open(path, "a") as f:
                f.write("\n".join(self.lines) + "\n")


def flag(name: str, default: bool = False) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


def load_state(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_state(path: str, st: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f, indent=2, sort_keys=True, default=str)
    os.replace(tmp, path)


def append_run_log(path: str, row: dict) -> None:
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def yahoo_prices(tickers: list[str], cfg, today: pd.Timestamp) -> tuple[pd.DataFrame, pd.Series]:
    """(completed daily closes before today, latest price per ticker for sizing orders)."""
    raw = data.download(tickers, cfg.start)
    return raw.loc[raw.index < today], raw.ffill().iloc[-1]


def wait_until_gone(broker, ids: set, timeout_s: float, report) -> set:
    """Poll pending orders until none of ids is still pending. Returns any still pending."""
    deadline = time.monotonic() + timeout_s
    left = set(ids)
    while left and time.monotonic() < deadline:
        left &= {o.get("id") for o in broker.pending_orders()}
        if left:
            time.sleep(5)
    if left:
        report(f"- still pending after {timeout_s:.0f}s: {sorted(left)}")
    return left


def place(broker, order: ex.Order, price: float, cfg, report) -> dict | None:
    """Send one market order. A 400 means it was rejected before reaching the market, so a
    fractional quantity is retried once as whole shares. Other errors propagate."""
    try:
        r = broker.market_order(order.ticker, order.quantity)
    except T212Error as e:
        if e.status != 400:
            raise
        whole = float(np.trunc(order.quantity))
        if whole == order.quantity or abs(whole) * price < cfg.min_order_gbp:
            report(f"- {order.ticker}: rejected, nothing placed ({e.body[:160]})")
            return None
        report(f"- {order.ticker}: {order.quantity:g} rejected ({e.body[:120]}); retrying as {whole:g} whole shares")
        try:
            r = broker.market_order(order.ticker, whole)
        except T212Error as e2:
            if e2.status != 400:
                raise
            report(f"- {order.ticker}: rejected again, nothing placed ({e2.body[:160]})")
            return None
        order.quantity, order.value = whole, whole * price
    report(f"- {'BUY ' if order.quantity > 0 else 'SELL'} {abs(order.quantity):g} {order.ticker} "
           f"(~£{abs(order.value):,.2f}) -> order {r.get('id')} {r.get('status')}")
    return r


def run(broker, cfg, *, env: str, trading_enabled: bool, dry_run: bool, reset_halt: bool,
        state_path: str, prices=yahoo_prices, now: datetime | None = None, report=None,
        sell_other: bool = False, once_per_day: bool = False) -> dict:
    report = report or Report()
    now = now or datetime.now(timezone.utc)
    today = pd.Timestamp(now.astimezone(ex.LONDON).date())
    tickers = list(UNIVERSE)
    classes = pd.Series({k: v.cls for k, v in UNIVERSE.items()})
    st = load_state(state_path)
    send = trading_enabled and not dry_run
    report(f"## ISA bot run {now:%Y-%m-%d %H:%M} UTC, {env.upper()} account, book: {cfg.mode}")
    report(f"Orders: {'ENABLED' if send else 'off (dry run)' if trading_enabled else 'off (TRADING_ENABLED is not true)'}")

    if reset_halt and st.get("halted"):
        report(f"Halt cleared by reset_halt (was: {st.get('halt_reason')}). Drawdown baseline restarts today.")
        st.update(halted=False, halt_reason=None, flattened=False, peak_unit_value=None, derisk=False)

    # A scheduled run stops here if an earlier one already traded today. The workflow has backup
    # slots because GitHub sometimes starts a scheduled run hours late or not at all.
    day = str(today.date())
    if once_per_day and st.get("traded_day") == day:
        report(f"Already traded today ({day}): this scheduled run is a backup, nothing to do.")
        return st

    # 1. Account, cash flows and guardrails -----------------------------------------------------
    acct = broker.account_summary()
    if acct.get("currency", "GBP") != "GBP":
        raise RuntimeError(f"account currency is {acct.get('currency')}; this bot assumes a GBP ISA")
    nav = float(acct["totalValue"])
    cash_free = float(acct["cash"]["availableToTrade"])
    if nav <= 0:
        report("Account value is zero: nothing to do.")
        return st
    first = not st.get("units")
    since = now - timedelta(days=3) if first else datetime.fromisoformat(st["last_txn_time"]) - timedelta(days=3)
    seen = set(st.get("seen_refs", []))
    since_utc = since.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    flows, refs, notes = ex.net_flows(broker.transactions(since_utc), seen)
    if first:
        flows = 0.0                  # deposits before the first run are already in the opening value
    st = ex.update_performance(st, nav, flows, cfg)
    st.update(seen_refs=(list(seen) + refs)[-500:], last_txn_time=now.isoformat(timespec="seconds"))
    for n in notes:
        report(f"Note: {n}")
    report(f"Account £{nav:,.2f} (free cash £{cash_free:,.2f}); net deposits since last run £{flows:,.2f}")
    report(f"Unit value {st['unit_value']:.4f}, drawdown {st['drawdown']:.1%}"
           f"{', de-risked: tilts halved' if st['derisk'] and cfg.mode != 'core' else ''}"
           f"{'  HALTED: ' + st['halt_reason'] if st.get('halted') else ''}")

    # 2. Instruments and positions ---------------------------------------------------------------
    instruments = broker.instruments()
    tmap, by_t, problems = ex.map_tickers(instruments, tickers, T212_TICKERS)
    inv = {t: y for y, t in tmap.items()}
    col = {y: i for i, y in enumerate(tickers)}
    N = len(tickers)
    value, qty, t212_px = np.zeros(N), np.zeros(N), np.full(N, np.nan)
    other_value, notes, other_sells = 0.0, [], []
    for p in broker.positions():
        inst = p.get("instrument") or {}
        t = inst.get("ticker") or p.get("ticker")
        cur = (p.get("walletImpact") or {}).get("currentValue")
        price = ex.price_in_pounds(float(p.get("currentPrice") or 0.0),
                                   inst.get("currency") or by_t.get(t, {}).get("currencyCode", "GBP"))
        q = float(p.get("quantity") or 0.0)
        cur = float(cur) if cur is not None else q * price
        if p.get("quantityInPies"):
            notes.append(f"{t} has shares in a pie; the bot can't trade those")
        if t in inv:
            i = col[inv[t]]
            value[i], t212_px[i] = cur, price
            qty[i] = float(p.get("quantityAvailableForTrading", q) or 0.0)
        elif not sell_other:
            other_value += cur
            notes.append(f"{t} (£{cur:,.2f}) is outside the bot's 15 ETFs; left alone")
        else:
            avail = float(p.get("quantityAvailableForTrading", q) or 0.0)
            if avail <= 0 or cur < 1.0:
                other_value += cur
                notes.append(f"{t} (£{cur:,.2f}) is outside the bot's 15 ETFs but can't be sold (pie or under £1)")
            else:
                share = avail / q if q > 0 else 1.0
                other_sells.append(ex.Order(t, t, -avail, -cur * share, "switch-over"))
                other_value += cur * (1.0 - share)
                notes.append(f"{t} (£{cur * share:,.2f}) is outside the bot's 15 ETFs; selling it (sell_other_holdings)")
    for n in notes:
        report(f"Note: {n}")

    # 3. Prices and the model ---------------------------------------------------------------------
    model_raw, last_raw = prices(tickers, cfg, today)
    px, fixes = data.clean(model_raw[tickers], cfg)
    last_close = px.index[-1]
    stale = (today - last_close).days > cfg.max_data_age_days
    suspects = set(data.last_print_suspect(px, cfg))
    recent = fixes[(fixes.fix == "unit_switch") & (pd.to_datetime(fixes["to"]) >= last_close - pd.Timedelta(days=30))]
    out = pipeline.run(px, classes, cfg)
    target = np.nan_to_num(out["target"].iloc[-1].to_numpy(float))
    core = np.nan_to_num(out["core"].iloc[-1].to_numpy(float))
    if st["derisk"]:
        target = governed(target, None if cfg.mode == "absolute" else core, 0.5)

    yahoo_px = np.array([ex.price_in_pounds(float(last_raw[y]), "GBX" if UNIVERSE[y].unit == "GBX" else "GBP")
                         for y in tickers])
    close_px = np.array([ex.price_in_pounds(float(px[y].iloc[-1]), "GBX" if UNIVERSE[y].unit == "GBX" else "GBP")
                         for y in tickers])
    with np.errstate(divide="ignore", invalid="ignore"):
        jumpy = ~np.isfinite(yahoo_px) | (np.abs(np.log(yahoo_px / close_px)) > cfg.bad_jump)
        # Held positions are sized at Trading 212's own price. If it disagrees with Yahoo by more
        # than 25%, one of them has the units wrong (pounds vs pence): don't trade that asset.
        clash = np.isfinite(t212_px) & (t212_px > 0) & (np.abs(np.log(t212_px / close_px)) > np.log(1.25))
    price = np.where(np.isfinite(t212_px) & (t212_px > 0), t212_px, np.where(jumpy, close_px, yahoo_px))
    bad_price = suspects | set(recent.ticker) | {y for i, y in enumerate(tickers) if clash[i]}
    frozen = np.array([y in bad_price or not np.isfinite(price[i]) or price[i] <= 0 or y not in tmap
                       for i, y in enumerate(tickers)])
    for y in sorted(bad_price):
        report(f"Note: {y} not traded today: its prices disagree or its latest Yahoo print looks wrong")

    # 4. What would we trade? ---------------------------------------------------------------------
    blockers = []
    if stale:
        blockers.append(f"newest close is {last_close.date()}, older than {cfg.max_data_age_days} days")
    needed = [y for y in tickers if y not in tmap and (target[col[y]] > 0 or value[col[y]] > 0)]
    if needed:
        blockers.append("can't match these to Trading 212: " + "; ".join(p for p in problems if p.split(":")[0] in needed))
    flatten = bool(st.get("halted")) and cfg.dd_hard_action == "flatten" and not st.get("flattened")
    if st.get("halted") and not flatten:
        blockers.append("circuit breaker is on (re-run the workflow with reset_halt to clear it)")
    sched = by_t.get(tmap.get("VUSA.L") or next(iter(tmap.values()), ""), {}).get("workingScheduleId")
    is_open, why = ex.market_open(broker.exchanges(), sched, now)
    if not is_open:
        blockers.append(f"London market not open ({why})")

    if flatten:
        h, new_w, trade = value / nav, np.zeros(N), -value
        report("Circuit breaker action is 'flatten': selling every position once.")
    else:
        h, new_w, trade = ex.plan_trades(tickers, value, nav, target, cfg, frozen, other_value)
    tbl = pd.DataFrame({"class": classes, "target": target, "now": h, "after": new_w, "trade_gbp": trade},
                       index=tickers)
    report(f"\nModel uses closes up to {last_close.date()}. Weights (target, now, after today's trades):\n")
    report("| ETF | class | target | now | after | trade £ |\n|---|---|---|---|---|---|")
    for y, r in tbl.iterrows():
        if r.target > 0 or r.now > 0 or r.trade_gbp != 0:
            report(f"| {y} | {r['class']} | {r.target:.1%} | {r.now:.1%} | {r.after:.1%} | {r.trade_gbp:+,.2f} |")
    report(f"| cash | | {max(0, 1 - target.sum()):.1%} | {cash_free / nav:.1%} | | |")
    for o in other_sells:
        report(f"\nAlso selling {o.ticker}: all {abs(o.quantity):g} shares (~£{abs(o.value):,.2f}), to switch over.")
    turnover = (np.abs(trade).sum() + sum(abs(o.value) for o in other_sells)) / 2 / nav

    placed, failures, done = [], [], False
    if blockers:
        report("\nNo orders today: " + " | ".join(blockers))
    elif not np.any(trade) and not other_sells:
        report("\nNo trades needed today (everything is inside the no-trade band).")
        done = send
    elif not send:
        report("\nDry run: the trades above were not sent.")
    else:
        # 5. Execute: clear our own stale orders, sell, wait, buy with the cash actually free ----
        pending = broker.pending_orders()
        manual = [o for o in pending if o.get("initiatedFrom") != "API"]
        if manual:
            report(f"\nNo orders today: {len(manual)} order(s) placed outside the bot are pending.")
        else:
            report("\nOrders:")
            done = True
            for o in pending:
                broker.cancel_order(o["id"])
                report(f"- cancelled stale bot order {o.get('id')} ({o.get('ticker')})")
            if pending and wait_until_gone(broker, {o.get("id") for o in pending}, 60, report):
                raise RuntimeError("stale orders would not cancel; stopping before trading")
            sells = ex.sell_orders(tickers, tmap, trade, new_w, qty, price, cfg)
            sell_ids = set()
            for o in other_sells + sells:
                each = abs(o.value / o.quantity) if o in other_sells else price[col[o.yahoo]]
                r = place(broker, o, each, cfg, report)
                if r:
                    placed.append(o)
                    sell_ids.add(r.get("id"))
                else:
                    failures.append(o.ticker)
            if sell_ids:
                wait_until_gone(broker, sell_ids, cfg.sell_fill_timeout_s, report)
            cash_now = (float(broker.account_summary()["cash"]["availableToTrade"]) if sells or other_sells
                        else cash_free)
            if not flatten:
                for o in ex.buy_orders(tickers, tmap, trade, cash_now, nav, price, cfg):
                    if place(broker, o, price[col[o.yahoo]], cfg, report):
                        placed.append(o)
                    else:
                        failures.append(o.ticker)
            if flatten:
                st["flattened"] = True

    # 6. Record -----------------------------------------------------------------------------------
    if done:
        st["traded_day"] = day
    st.update(last_run=now.isoformat(timespec="seconds"), env=env, mode=cfg.mode,
              last_orders=[o.__dict__ for o in placed], last_close=str(last_close.date()))
    save_state(state_path, st)
    append_run_log(os.path.join(os.path.dirname(state_path), f"{env}_runs.csv"), {
        "time_utc": now.isoformat(timespec="seconds"), "mode": cfg.mode, "account_gbp": round(nav, 2),
        "net_deposits_gbp": round(flows, 2), "unit_value": round(st["unit_value"], 6),
        "drawdown": round(st["drawdown"], 4), "derisk": st["derisk"], "halted": bool(st.get("halted")),
        "orders_sent": len(placed), "turnover": round(float(turnover), 4) if placed else 0.0,
        "blocked": " | ".join(blockers)})
    if failures:
        raise RuntimeError(f"some orders were rejected: {failures}")
    return st


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", action="store_true", help="print the ticker mapping and exit")
    args = ap.parse_args()
    env = os.environ.get("T212_ENV", "demo").strip().lower() or "demo"
    try:
        return _main(args, env)
    except T212Error as e:
        if e.status in (401, 403):
            print(f"\nTrading 212 refused the API key ({e.status} on {e.path}). Check that T212_API_KEY and "
                  f"T212_API_SECRET are right, that the key was made in your {env} account (T212_ENV={env}), "
                  "that it has the permissions listed in the README, and that it isn't IP-restricted.")
            return 1
        raise


def _main(args, env: str) -> int:
    from engine.broker import Trading212

    broker = Trading212(os.environ.get("T212_API_KEY", ""), os.environ.get("T212_API_SECRET", ""), env)
    if args.map:
        tmap, by_t, problems = ex.map_tickers(broker.instruments(), list(UNIVERSE), T212_TICKERS)
        for y in UNIVERSE:
            t = tmap.get(y)
            i = by_t.get(t, {})
            print(f"{y:8s} -> {t or '??':12s} {i.get('name', '')} [{i.get('currencyCode', '')}, {i.get('type', '')}]")
        for p in problems:
            print("PROBLEM:", p)
        return 1 if problems else 0
    report = Report()
    try:
        run(broker, Config(), env=env, trading_enabled=flag("TRADING_ENABLED"), dry_run=flag("DRY_RUN"),
            reset_halt=flag("RESET_HALT"), state_path=os.path.join(STATE_DIR, f"{env}.json"), report=report,
            sell_other=flag("SELL_OTHER_HOLDINGS"), once_per_day=flag("SCHEDULED"))
    finally:
        report.flush_summary()
    return 0


if __name__ == "__main__":
    sys.exit(main())
