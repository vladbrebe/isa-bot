"""Live-trading logic with no network access, so all of it is unit-tested.

  map_tickers        Yahoo ticker (VUSA.L) -> Trading 212 ticker (e.g. VUSAl_EQ), strictly
  update_performance a unitised performance index, so deposits and withdrawals are not
                     mistaken for gains or losses by the drawdown guardrails
  plan_trades        the simulator's trade rule applied to the real account, in pounds
  sell_orders / buy_orders   pounds -> share quantities, sells first, buys capped by cash
  market_open        is the London Stock Exchange in its regular session right now?
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, time as dtime, timezone
from zoneinfo import ZoneInfo

import numpy as np

from .simulate import rebalance

LONDON = ZoneInfo("Europe/London")
POUND_CODES = {"GBP"}
PENCE_CODES = {"GBX", "GBp"}


# ---- instruments ------------------------------------------------------------------------------
def map_tickers(instruments: list[dict], yahoo: list[str], overrides: dict[str, str]) -> tuple[dict, dict, list]:
    """Returns ({yahoo: t212 ticker}, {t212 ticker: instrument}, problems).

    A Yahoo LSE ticker "XXXX.L" matches the Trading 212 instrument whose shortName is XXXX,
    whose ticker ends in "l_EQ" (Trading 212's suffix for London lines) and which trades in
    pounds or pence. Anything ambiguous or missing is reported, never guessed."""
    by_ticker = {i.get("ticker"): i for i in instruments if i.get("ticker")}
    out, problems = {}, []
    for y in yahoo:
        if y in overrides:
            t = overrides[y]
            if t in by_ticker:
                out[y] = t
            else:
                problems.append(f"{y}: override {t} is not in Trading 212's instrument list")
            continue
        root = y.split(".")[0].upper()
        cands = [i for i in instruments
                 if str(i.get("ticker", "")).endswith("l_EQ")
                 and str(i.get("currencyCode", "")) in POUND_CODES | PENCE_CODES
                 and (str(i.get("shortName", "")).upper() == root
                      or str(i.get("ticker", ""))[:-4].upper() == root)]
        if len(cands) == 1:
            out[y] = cands[0]["ticker"]
        elif not cands:
            problems.append(f"{y}: no London-listed GBP/GBX instrument with short name {root}")
        else:
            problems.append(f"{y}: several matches {[c['ticker'] for c in cands]}; set one in T212_TICKERS")
    return out, by_ticker, problems


def price_in_pounds(price: float, currency: str) -> float:
    return price / 100.0 if currency in PENCE_CODES else price


# ---- performance and guardrails --------------------------------------------------------------
def net_flows(transactions: list[dict], seen: set[str]) -> tuple[float, list[str], list[str]]:
    """Money in minus money out among transactions not seen before.
    Fees and interest are performance, not flows. Returns (flow, new refs, notes)."""
    flow, refs, notes = 0.0, [], []
    for tx in transactions:
        ref = str(tx.get("reference") or f"{tx.get('dateTime')}|{tx.get('type')}|{tx.get('amount')}")
        if ref in seen or ref in refs:
            continue
        refs.append(ref)
        kind, amt = tx.get("type"), float(tx.get("amount") or 0.0)
        if kind == "DEPOSIT":
            flow += abs(amt)
        elif kind == "WITHDRAW":
            flow -= abs(amt)
        elif kind == "TRANSFER":
            flow += amt
            notes.append(f"transfer of {amt:+.2f} counted as a cash flow")
    return flow, refs, notes


def update_performance(state: dict, nav: float, flows: float, cfg) -> dict:
    """Unit-price accounting, as a fund does it: deposits buy units at the current unit value and
    withdrawals redeem them, so only investment gains and losses move the unit value. The
    drawdown guardrails read the unit value, not the account balance."""
    st = dict(state)
    units = float(st.get("units") or 0.0)
    if units <= 0 or nav <= 0:
        units, uv = (nav if nav > 0 else 0.0), 1.0
    else:
        uv = (nav - flows) / units                   # value per unit before today's flows
        if uv <= 0:
            raise RuntimeError(f"unit value would be {uv:.4f}: account value {nav}, flows {flows}")
        units += flows / uv
    peak = max(float(st.get("peak_unit_value") or uv), uv)
    dd = 1.0 - uv / peak
    derisk = dd > cfg.dd_soft or (bool(st.get("derisk")) and dd > cfg.dd_soft_release)
    st.update(units=units, unit_value=uv, peak_unit_value=peak, drawdown=dd, derisk=derisk)
    if dd > cfg.dd_hard and not st.get("halted"):
        st.update(halted=True, halt_reason=f"drawdown {dd:.1%} passed the {cfg.dd_hard:.0%} circuit breaker")
    return st


# ---- orders ---------------------------------------------------------------------------------
@dataclass
class Order:
    yahoo: str
    ticker: str
    quantity: float      # signed: + buy, - sell
    value: float         # estimated pounds, signed
    note: str = ""


def _round_down(x: float, decimals: int) -> float:
    f = 10 ** decimals
    return math.floor(abs(x) * f + 1e-9) / f * (1 if x >= 0 else -1)


def plan_trades(tickers: list[str], value: np.ndarray, nav: float, target: np.ndarray, cfg,
                frozen: np.ndarray, other_value: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The backtest's trade rule on the real account. Returns (current weights, new weights,
    trade in pounds). Frozen assets are not traded. Trades below min_order_gbp are dropped."""
    h = value / nav
    budget = max(0.0, 1.0 - other_value / nav)      # holdings outside the universe are left alone
    new = rebalance(h, np.nan_to_num(target), cfg.trade_rate, cfg.no_trade_band, budget)
    new = np.where(frozen, h, new)
    dw = new - h
    turnover = np.abs(dw).sum() / 2
    if turnover > cfg.max_turnover_per_run:
        dw *= cfg.max_turnover_per_run / turnover
    trade = dw * nav
    trade[np.abs(trade) < cfg.min_order_gbp] = 0.0
    return h, h + trade / nav, trade


def sell_orders(tickers, tmap, trade, new_w, qty_avail, price_gbp, cfg) -> list[Order]:
    out = []
    for i, y in enumerate(tickers):
        if trade[i] >= 0:
            continue
        held = float(qty_avail[i])
        if held <= 0:
            continue
        if new_w[i] <= 1e-4:                                     # exiting: sell all, leave no dust
            q, note = -held, "exit"
        else:
            q, note = _round_down(trade[i] / price_gbp[i], cfg.qty_decimals), ""
            q = max(q, -held)
        if q != 0 and abs(q) * price_gbp[i] >= cfg.min_order_gbp * 0.999:
            out.append(Order(y, tmap[y], q, q * price_gbp[i], note))
    return out


def buy_orders(tickers, tmap, trade, cash_free, nav, price_gbp, cfg) -> list[Order]:
    """Buys scaled down together if they would need more than the free cash minus a buffer."""
    want = np.clip(trade, 0.0, None)
    room = max(0.0, cash_free - cfg.cash_buffer * nav)
    if want.sum() > room:
        want = want * (room / want.sum()) if want.sum() > 0 else want
    out = []
    for i, y in enumerate(tickers):
        if want[i] < cfg.min_order_gbp:
            continue
        q = _round_down(want[i] / price_gbp[i], cfg.qty_decimals)
        if q > 0 and q * price_gbp[i] >= cfg.min_order_gbp * 0.999:
            out.append(Order(y, tmap[y], q, q * price_gbp[i]))
    return out


# ---- market hours ---------------------------------------------------------------------------
def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))


def market_open(exchanges: list[dict], schedule_id, now: datetime) -> tuple[bool, str]:
    """True during the regular session of the instrument's working schedule. Falls back to
    London clock hours (Mon-Fri 08:00-16:30) if the schedule can't be found."""
    now = now.astimezone(timezone.utc)
    for ex in exchanges or []:
        for ws in ex.get("workingSchedules") or []:
            if ws.get("id") != schedule_id:
                continue
            events = sorted(((_parse(e["date"]), e.get("type")) for e in ws.get("timeEvents") or []
                             if e.get("date")), key=lambda x: x[0])
            past = [e for e in events if e[0] <= now]
            if past:
                kind = past[-1][1]
                return kind in ("OPEN", "BREAK_END"), f"{ex.get('name', 'exchange')}: last event {kind}"
    local = now.astimezone(LONDON)
    is_open = local.weekday() < 5 and dtime(8, 0) <= local.time() < dtime(16, 30)
    return is_open, "schedule not found, used London clock hours (bank holidays not known)"
