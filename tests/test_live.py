"""The live bot against a fake Trading 212 account and synthetic prices: no network, no keys."""
import json
import os
import sys
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import live  # noqa: E402
from config import UNIVERSE, Config  # noqa: E402
from engine import execution as ex  # noqa: E402
from engine import pipeline  # noqa: E402
from engine.broker import T212Error  # noqa: E402

NOW = datetime(2026, 10, 5, 13, 20, tzinfo=timezone.utc)      # a Monday, 14:20 in London
TICKERS = list(UNIVERSE)


def t212(y):
    return y.split(".")[0] + "l_EQ"


class FakeT212:
    """Fills market orders instantly at a fixed price; enforces cash and whole-share rules."""

    def __init__(self, cash=10_000.0, whole_shares_only=(), now=NOW):
        self.cash, self.qty, self.orders, self.pending, self.txs = cash, {}, [], [], []
        self.whole_only = set(whole_shares_only)
        self.price = {t212(y): (900.0 if a.unit == "GBX" else 45.0) for y, a in UNIVERSE.items()}
        self.ccy = {t212(y): a.unit for y, a in UNIVERSE.items()}
        self.now = now
        self.cancelled = []

    def gbp(self, t):
        return self.price[t] / 100 if self.ccy[t] == "GBX" else self.price[t]

    def account_summary(self):
        inv = sum(q * self.gbp(t) for t, q in self.qty.items())
        return {"currency": "GBP", "totalValue": self.cash + inv,
                "cash": {"availableToTrade": self.cash, "reservedForOrders": 0, "inPies": 0}}

    def positions(self):
        return [{"instrument": {"ticker": t, "currency": self.ccy[t]}, "quantity": q,
                 "quantityAvailableForTrading": q, "currentPrice": self.price[t],
                 "walletImpact": {"currentValue": q * self.gbp(t)}} for t, q in self.qty.items() if q > 0]

    def pending_orders(self):
        return list(self.pending)

    def cancel_order(self, oid):
        self.cancelled.append(oid)
        self.pending = [o for o in self.pending if o["id"] != oid]

    def instruments(self):
        out = [{"ticker": t212(y), "shortName": y.split(".")[0], "currencyCode": a.unit, "type": "ETF",
                "workingScheduleId": 7, "name": y} for y, a in UNIVERSE.items()]
        out.append({"ticker": "VUSAa_EQ", "shortName": "VUSA", "currencyCode": "EUR", "type": "ETF"})
        return out

    def exchanges(self):
        days = [self.now.date() + timedelta(days=k) for k in range(-3, 15)]
        ev = [e for d in days if d.weekday() < 5 for e in (
            {"date": f"{d.isoformat()}T07:00:00Z", "type": "OPEN"}, {"date": f"{d.isoformat()}T15:30:00Z", "type": "CLOSE"})]
        return [{"name": "London Stock Exchange", "workingSchedules": [{"id": 7, "timeEvents": ev}]}]

    def transactions(self, since_iso):
        return list(self.txs)

    def market_order(self, ticker, quantity):
        if ticker in self.whole_only and quantity != int(quantity):
            raise T212Error(400, '{"code":"FractionalNotAllowed"}', "/api/v0/equity/orders/market")
        if round(quantity, 2) != quantity:
            raise T212Error(400, '{"code":"QuantityPrecisionMismatch"}', "/api/v0/equity/orders/market")
        value = quantity * self.gbp(ticker)
        if value > self.cash + 1e-9:
            raise T212Error(400, '{"code":"InsufficientFreeForStocksBuy"}', "/api/v0/equity/orders/market")
        if quantity < 0 and -quantity > self.qty.get(ticker, 0) + 1e-9:
            raise T212Error(400, '{"code":"SellingEquityNotOwned"}', "/api/v0/equity/orders/market")
        self.cash -= value
        self.qty[ticker] = self.qty.get(ticker, 0) + quantity
        self.orders.append((ticker, quantity))
        return {"id": len(self.orders), "status": "FILLED"}


def synthetic_prices(seed=5, end="2026-10-30", T=1520):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=end, periods=T)
    r = rng.normal(0.0002, 0.009, (T, len(TICKERS)))
    lvl = np.array([800.0 if UNIVERSE[y].unit == "GBX" else 40.0 for y in TICKERS])
    px = pd.DataFrame(lvl * np.exp(np.cumsum(r, 0)), index=idx, columns=TICKERS)
    px = px / px.iloc[-1] * np.array([900.0 if UNIVERSE[y].unit == "GBX" else 45.0 for y in TICKERS])

    def prices(tickers, cfg, today):     # completed closes before today, plus today's live price
        return px.loc[px.index < today, tickers], px.loc[px.index <= today, tickers].iloc[-1]
    return prices


@pytest.fixture
def cfg():
    return replace(Config(), min_train=300, hmm_min_obs=300, train_window=600)


@pytest.fixture(autouse=True)
def fast_pipeline(monkeypatch):
    """The model is tested elsewhere; here, compute it once per price history."""
    cache, real = {}, pipeline.run

    def cached(px, classes, cfg):
        key = (len(px), str(px.index[-1]), cfg.mode)
        if key not in cache:
            cache[key] = real(px, classes, cfg)
        return cache[key]
    monkeypatch.setattr(live.pipeline, "run", cached)


def go(broker, cfg, tmp_path, now=NOW, **kw):
    args = dict(env="demo", trading_enabled=True, dry_run=False, reset_halt=False)
    args.update(kw)
    return live.run(broker, cfg, state_path=str(tmp_path / "demo.json"), prices=synthetic_prices(),
                    now=now, report=lambda *a: None, **args)


# ---- pure logic -------------------------------------------------------------------------------
def test_map_tickers():
    inst = FakeT212().instruments()
    tmap, by_t, problems = ex.map_tickers(inst, TICKERS, {})
    assert not problems and tmap["VUSA.L"] == "VUSAl_EQ" and len(tmap) == len(TICKERS)
    tmap, _, problems = ex.map_tickers([i for i in inst if i["shortName"] != "SGLN"], TICKERS, {})
    assert "SGLN.L" not in tmap and any(p.startswith("SGLN.L") for p in problems)
    tmap, _, problems = ex.map_tickers(inst, TICKERS, {"ISF.L": "NOPE_EQ"})
    assert "ISF.L" not in tmap and "override" in problems[0]
    dup = inst + [{"ticker": "VUSAl_EQ2", "shortName": "VUSA", "currencyCode": "GBP"}]
    _, _, problems = ex.map_tickers(dup + [{"ticker": "XVUSAl_EQ", "shortName": "VUSA", "currencyCode": "GBX"}],
                                    TICKERS, {})
    assert any("several matches" in p for p in problems)


def test_unitised_performance_ignores_deposits(cfg):
    st = ex.update_performance({}, 10_000, 0.0, cfg)
    assert st["unit_value"] == 1.0 and st["drawdown"] == 0
    st = ex.update_performance(st, 15_000, 5_000, cfg)          # a deposit is not a gain
    assert abs(st["unit_value"] - 1.0) < 1e-12
    st = ex.update_performance(st, 13_500, 0.0, cfg)            # a real 10% loss
    assert abs(st["drawdown"] - 0.10) < 1e-12 and not st["derisk"]
    st = ex.update_performance(st, 6_750, -6_750, cfg)          # withdrawing half is not a loss
    assert abs(st["drawdown"] - 0.10) < 1e-12
    st = ex.update_performance(st, 6_000, 0.0, cfg)
    assert st["derisk"] and not st.get("halted")
    st = ex.update_performance(st, 4_800, 0.0, cfg)
    assert st["halted"] and "circuit breaker" in st["halt_reason"]


def test_net_flows_dedupes():
    txs = [{"reference": "a", "type": "DEPOSIT", "amount": 500}, {"reference": "b", "type": "WITHDRAW", "amount": 200},
           {"reference": "c", "type": "INTEREST_ON_FREE_CASH", "amount": 3}, {"reference": "d", "type": "FEE", "amount": -1}]
    flow, refs, _ = ex.net_flows(txs, set())
    assert flow == 300 and refs == ["a", "b", "c", "d"]
    assert ex.net_flows(txs, set(refs))[0] == 0


def test_buys_fit_cash_and_rounding(cfg):
    n = len(TICKERS)
    target = np.full(n, 1.0 / n)
    h, new_w, trade = ex.plan_trades(TICKERS, np.zeros(n), 10_000, target, cfg, np.zeros(n, bool))
    tmap = {y: t212(y) for y in TICKERS}
    price = np.full(n, 37.37)
    buys = ex.buy_orders(TICKERS, tmap, trade, 2_000, 10_000, price, cfg)
    assert sum(o.value for o in buys) <= 2_000 - 0.01 * 10_000 + 1e-9
    assert all(round(o.quantity, 2) == o.quantity and o.value >= cfg.min_order_gbp for o in buys)
    assert abs(trade.sum() / 10_000 - cfg.trade_rate) < 1e-9     # moves 35% of the way, as in the backtest


def test_market_open():
    fake = FakeT212()
    assert ex.market_open(fake.exchanges(), 7, NOW)[0]
    assert not ex.market_open(fake.exchanges(), 7, NOW.replace(hour=19))[0]
    assert not ex.market_open([], None, datetime(2026, 10, 4, 12, tzinfo=timezone.utc))[0]   # Sunday


# ---- end to end -------------------------------------------------------------------------------
def test_first_runs_build_the_core(cfg, tmp_path):
    b = FakeT212()
    go(b, cfg, tmp_path)
    assert b.orders and all(q > 0 for _, q in b.orders)                    # from cash: buys only
    v = b.account_summary()
    assert abs(v["totalValue"] - 10_000) < 1e-6 and 0.30 < 1 - v["cash"]["availableToTrade"] / 10_000 < 0.40
    st = json.loads((tmp_path / "demo.json").read_text())
    assert st["unit_value"] == 1.0 and st["last_orders"]
    for day in range(1, 4):
        go(b, cfg, tmp_path, now=NOW + timedelta(days=day))
    w = {t: q * b.gbp(t) / 10_000 for t, q in b.qty.items()}
    assert abs(w["IGLTl_EQ"] - 0.4 * (1 - 0.65 ** 4)) < 0.02                # partial adjustment
    assert set(w) <= {t212(y) for y in TICKERS if UNIVERSE[y].cls == "equity"} | {"IGLTl_EQ"}
    assert (tmp_path / "demo_runs.csv").read_text().count("\n") == 5


def test_kill_switch_and_dry_run_send_nothing(cfg, tmp_path):
    b = FakeT212()
    go(b, cfg, tmp_path, trading_enabled=False)
    go(b, cfg, tmp_path, dry_run=True)
    assert not b.orders


def test_closed_market_and_manual_orders_block(cfg, tmp_path):
    b = FakeT212()
    go(b, cfg, tmp_path, now=NOW.replace(hour=18))
    assert not b.orders
    b.pending = [{"id": 99, "ticker": "ISFl_EQ", "initiatedFrom": "WEB"}]
    go(b, cfg, tmp_path)
    assert not b.orders
    b.pending = [{"id": 98, "ticker": "ISFl_EQ", "initiatedFrom": "API"}]   # our own stale order
    go(b, cfg, tmp_path)
    assert b.cancelled == [98] and b.orders


def test_circuit_breaker_halts_until_reset(cfg, tmp_path):
    b = FakeT212()
    go(b, cfg, tmp_path)
    st = json.loads((tmp_path / "demo.json").read_text())
    st["peak_unit_value"] = 1.5                                   # pretend we are 33% below the peak
    (tmp_path / "demo.json").write_text(json.dumps(st))
    n = len(b.orders)
    go(b, cfg, tmp_path, now=NOW + timedelta(days=1))
    assert len(b.orders) == n and json.loads((tmp_path / "demo.json").read_text())["halted"]
    go(b, cfg, tmp_path, now=NOW + timedelta(days=2), reset_halt=True)
    st = json.loads((tmp_path / "demo.json").read_text())
    assert not st["halted"] and len(b.orders) > n


def test_flatten_sells_everything_once(cfg, tmp_path):
    b = FakeT212()
    go(b, cfg, tmp_path)
    st = json.loads((tmp_path / "demo.json").read_text())
    st["peak_unit_value"] = 1.5
    (tmp_path / "demo.json").write_text(json.dumps(st))
    go(b, replace(cfg, dd_hard_action="flatten"), tmp_path, now=NOW + timedelta(days=1))
    assert all(abs(q) < 1e-9 for q in b.qty.values())
    n = len(b.orders)
    go(b, replace(cfg, dd_hard_action="flatten"), tmp_path, now=NOW + timedelta(days=2))
    assert len(b.orders) == n


def test_whole_share_fallback(cfg, tmp_path):
    b = FakeT212(whole_shares_only={"IGLTl_EQ"})
    go(b, cfg, tmp_path)
    q = b.qty["IGLTl_EQ"]
    assert q == int(q) and q > 0


def test_old_bot_holdings_left_alone_unless_switching_over(cfg, tmp_path):
    b = FakeT212(cash=300.0)
    b.price["OLDl_EQ"], b.ccy["OLDl_EQ"], b.qty["OLDl_EQ"] = 20.0, "GBP", 10.0    # £200 an old bot bought
    go(b, cfg, tmp_path)
    assert b.qty["OLDl_EQ"] == 10.0 and b.orders                       # left alone; spare cash still invested
    go(b, cfg, tmp_path, now=NOW + timedelta(days=1), sell_other=True, dry_run=True)
    assert b.qty["OLDl_EQ"] == 10.0                                    # a dry run sends nothing
    n = len(b.orders)
    go(b, cfg, tmp_path, now=NOW + timedelta(days=2), sell_other=True)
    assert b.qty["OLDl_EQ"] == 0 and b.orders[n] == ("OLDl_EQ", -10.0)  # sold first, in full
    assert all(q > 0 for t, q in b.orders[n + 1:])                      # then the core is bought
    assert abs(b.account_summary()["totalValue"] - 500.0) < 1e-6

def test_backup_runs_skip_once_traded_today(cfg, tmp_path):
    b = FakeT212()
    go(b, cfg, tmp_path, now=NOW.replace(hour=6), once_per_day=True)  # before the open: nothing
    assert not b.orders
    go(b, cfg, tmp_path, once_per_day=True)  # a backup during market hours trades
    n = len(b.orders)
    assert n
    go(b, cfg, tmp_path, now=NOW + timedelta(minutes=50), once_per_day=True)  # later backup skips
    assert len(b.orders) == n
    go(b, cfg, tmp_path, now=NOW + timedelta(days=1), once_per_day=True)  # next day trades again
    assert len(b.orders) > n
