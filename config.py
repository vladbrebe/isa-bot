"""Every tunable number lives here. Research changes happen in this file, never inside engine/.

Each parameter was set a priori from first principles, not tuned on the backtest.
If you change one and re-run, add 1 to N_TRIALS so the deflated Sharpe stays honest.
"""
from dataclasses import dataclass, field
from typing import NamedTuple


class Asset(NamedTuple):
    cls: str             # equity | bond | gold | property
    unit: str            # "GBP" or "GBX" (pence): how Yahoo quotes it
    stamp: bool = False  # True for UK company shares: 0.5% SDRT on every buy
    fx: bool = False     # True if not a GBP line: Trading 212 FX fee each way


# GBP/GBX lines of LSE ETFs: no stamp duty (ETFs/ETCs are exempt) and no FX fee in a GBP ISA.
# A daily-turnover book on UK *shares* pays 0.5% per buy, which no short-horizon edge survives.
UNIVERSE: dict[str, Asset] = {
    "ISF.L":  Asset("equity", "GBX"),     # FTSE 100
    "MIDD.L": Asset("equity", "GBX"),     # FTSE 250
    "VUSA.L": Asset("equity", "GBP"),     # S&P 500
    "EQQQ.L": Asset("equity", "GBX"),     # Nasdaq 100
    "IEUX.L": Asset("equity", "GBX"),     # Europe ex UK
    "VJPN.L": Asset("equity", "GBP"),     # Japan
    "VAPX.L": Asset("equity", "GBP"),     # Asia Pacific ex Japan
    "VFEM.L": Asset("equity", "GBP"),     # Emerging markets
    "IGLT.L": Asset("bond", "GBP"),       # UK gilts
    "IGLS.L": Asset("bond", "GBP"),       # short gilts 0-5y
    "INXG.L": Asset("bond", "GBP"),       # index-linked gilts
    "SLXX.L": Asset("bond", "GBP"),       # sterling corporate bonds
    "SGLN.L": Asset("gold", "GBX"),       # physical gold
    "IUKP.L": Asset("property", "GBX"),   # UK property
    "IWDP.L": Asset("property", "GBX"),   # global property
}
# Trading 212 ticker for any Yahoo ticker the automatic match gets wrong or can't find.
# Run `python live.py --map` against your account to check, e.g. {"VUSA.L": "VUSAl_EQ"}.
T212_TICKERS: dict[str, str] = {}

CASH_PROXY = "ERNS.L"   # ultrashort sterling bond fund: proxies interest on idle ISA cash (backtest only)
BENCH_EQUITY = "VUSA.L"
BENCH_GILTS = "IGLT.L"

HALF_SPREAD_BPS = {"equity": 6.0, "bond": 5.0, "gold": 8.0, "property": 12.0}  # one-way, incl. slippage
STAMP_BPS = 50.0
FX_BPS = 15.0
N_TRIALS = 6   # configurations evaluated so far: v1 runs incl. bug-fix reruns, v2, v2 after the trade-rule fix


@dataclass(frozen=True)
class Config:
    start: str = "2009-01-01"
    # data hygiene
    burn_in: int = 60              # drop the first 60 sessions after a listing (launch prints are unreliable)
    bad_jump: float = 0.15         # log-move that triggers a bad-print check
    bad_sigmas: float = 10.0       # ...and must also exceed this many trailing daily sigmas
    min_history: int = 260         # clean sessions needed before an asset is tradable
    # risk
    vol_halflife: int = 30
    cov_window: int = 252
    cov_halflife: int = 63
    cov_shrink: float = 0.30       # shrink covariance 30% toward its diagonal
    # model
    horizon: int = 5               # label: next 5-day return, in units of ex-ante sigma
    exec_lag: int = 1              # decide on close t, fill at close t+1 (conservative)
    embargo: int = 5               # extra purge gap between training labels and the fit date
    train_window: int = 1260       # 5y rolling window
    train_halflife: int = 756      # older samples decay with a 3y half-life
    min_train: int = 504
    refit_every: int = 21
    ridge: float = 0.30            # penalty as a fraction of total sample weight
    # regime model (2-state Gaussian HMM, Baum-Welch)
    hmm_window: int = 1500
    hmm_min_obs: int = 500
    hmm_refit_every: int = 63
    # portfolio: which book the live bot runs (the backtest always reports all three)
    # Out-of-sample 2013-04 to 2026-10, after costs (trial 6 = rerun after the trade-rule fix):
    # "absolute" (v1, trials 1-4): the model decides everything, including how much to invest.
    #     2.9%/yr, Sharpe 0.16, against 6.3% and 0.53 for the passive 60/40 benchmark.
    # "core_tilt" (v2, trials 5-6): the passive core is always held and the model only tilts
    #     between assets inside a tracking-error budget. Registered before it was run.
    #     5.7%/yr, Sharpe 0.44. Before costs 6.9% and 0.56, but its tilts turn the portfolio over
    #     about 9 times a year and the costs (1.2%/yr) more than eat the gain.
    # "core" (default): hold the passive core and rebalance it (6.4%/yr, Sharpe 0.53). Neither
    #     model beat it, so the bot runs this until a new idea beats it out of sample.
    mode: str = "core"
    core_equity: float = 0.60      # core = 60% equity ETFs (equal weight) + 40% gilts: the bar to beat
    core_bond: str = "IGLT.L"
    te_cap: float = 0.03           # v2: annualised ex-ante tracking-error ceiling around the core
    kelly_fraction: float = 0.5    # half Kelly: risk aversion gamma = 1 / kelly_fraction
    vol_cap: float = 0.10          # v1 only: annualised ex-ante volatility ceiling
    max_weight: float = 0.25
    class_caps: dict = field(default_factory=lambda: {"equity": 0.80, "bond": 0.70, "gold": 0.20, "property": 0.20})
    trade_rate: float = 0.35       # Garleanu-Pedersen: close 35% of the gap to target per day
    no_trade_band: float = 0.01    # no trade smaller than 1% of NAV; a weight within 1% of target is left alone
    # guardrails
    dd_soft: float = 0.10          # halve the model's risk while drawdown exceeds 10% (released below 5%);
    dd_soft_release: float = 0.05  # in v2 this shrinks the tilts only, never the passive core
    dd_hard: float = 0.25          # live only: circuit breaker. The core alone fell 22% in 2022, so 20% would
    dd_hard_action: str = "halt"   # fire on an ordinary bear market. "halt" freezes trading until a human
                                   # resets it; "flatten" also sells everything (locks in the loss)
    max_turnover_per_run: float = 0.75
    min_order_gbp: float = 5.0
    # live execution (live.py)
    qty_decimals: int = 2          # share quantities are rounded down to this many decimal places
    cash_buffer: float = 0.01      # keep 1% of the account in cash so a market buy never bounces
    sell_fill_timeout_s: int = 180 # wait this long for sells to fill before buying
    max_data_age_days: int = 5     # refuse to trade if the newest close is older than this


def costs_bps(ticker: str) -> tuple[float, float]:
    """(buy, sell) one-way cost in basis points."""
    a = UNIVERSE[ticker]
    half = HALF_SPREAD_BPS[a.cls]
    buy = half + (STAMP_BPS if a.stamp else 0.0) + (FX_BPS if a.fx else 0.0)
    sell = half + (FX_BPS if a.fx else 0.0)
    return buy, sell
