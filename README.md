# ISA Quant

An automated trading bot for a Trading 212 Stocks and Shares ISA, run daily by GitHub Actions.
It trades London-listed ETFs (UK, US, European, Japanese, Asian and emerging-market shares,
gilts, corporate bonds, gold and property), which avoids stamp duty and FX fees.

**The honest headline:** two forecasting models were built and tested properly, and neither beat
simply holding a 60/40 portfolio. So the bot ships set to hold that passive 60/40 core and keep it
rebalanced. Switching it to a model is one line in `config.py` (see "Changing what it runs").

## Results

Walk-forward backtest, 15 April 2013 to 1 October 2026, after spreads and slippage. Nothing was
fitted on the period it was scored on.

| | Return/yr | Volatility | Sharpe | Worst fall |
|---|---|---|---|---|
| **Passive 60/40 core, as the bot trades it** | **6.4%** | 9.0% | **0.53** | -21.8% |
| Passive 60/40 benchmark | 6.3% | 8.9% | 0.53 | -21.9% |
| Model v2: 60/40 core plus model tilts | 5.7% | 9.5% | 0.44 | -20.5% |
| Model v1: model decides everything | 2.9% | 8.6% | 0.16 | -16.5% |
| 200-day moving-average trend (the first bot idea) | 2.9% | 5.8% | 0.20 | -11.3% |
| S&P 500 only | 14.9% | 15.1% | 0.88 | -25.5% |

What the numbers say:

- **v1** tried to time the market (how much to hold in cash) and had no skill at it.
- **v2** was designed after that and registered before it was run: always hold the 60/40, and let
  the model only shift weight between assets within a 3% tracking-error budget. Before costs it
  did slightly better than the core (6.9% a year, Sharpe 0.56), but its tilts turn the portfolio
  over about 9 times a year, and costs of 1.2% a year more than wiped that out. Its forecasts
  have a rank correlation of 0.025 with what happened next (t = 1.6): real enough to show before
  costs, too weak to pay for the trading.
- **S&P 500 only** did best, but 2013 to 2026 was an exceptional run for US shares, it had the
  deepest fall, and choosing it now with hindsight would be exactly the kind of overfitting this
  project avoids.

Run `python backtest.py` for the full report and `output/tearsheet.png` for the charts.

## How it works

```
config.py            every setting, with the reasoning and the research record
backtest.py          walk-forward research run: tables, honest statistics, tear sheet
live.py              the daily live run (what GitHub Actions calls)
switch_over.py       one-command move from an older bot, using your own GitHub login
engine/data.py       Yahoo prices, with repair of Yahoo's known bad London prints
engine/regime.py     2-state hidden Markov model of calm vs turbulent markets
engine/features.py   9 trend, momentum, reversal, volatility and calendar signals
engine/model.py      one pooled ridge regression, refitted monthly, purged of look-ahead
engine/portfolio.py  long-only weights: the passive core, v2 tilts, or v1
engine/simulate.py   costs, fills and the trade rule shared by the backtest and the bot
engine/execution.py  live logic: ticker matching, deposit-proof drawdown, order sizing
engine/broker.py     Trading 212 API client
tests/               look-ahead tests and a full fake-account test of the live bot
```

The backtest and the live bot call the same code, so what was tested is what trades. Each day
the bot moves 35% of the way from its current holdings to the target and skips any change smaller
than 1% of the account, the same rule the backtest used.


## Safety switches

- **Kill switch:** set `TRADING_ENABLED` to `false`. The bot keeps running and logging but sends
  nothing. (Phone: github.com in the browser, the GitHub app can't edit variables.)
- **Circuit breaker:** if the account falls 25% from its peak, trading stops until you run the
  workflow with "reset_halt" ticked. Deposits and withdrawals don't count as gains or losses: the
  bot tracks a unit price like a fund does. The core alone fell 22% in 2022, which is why the line
  isn't 20%. Set `dd_hard_action = "flatten"` in `config.py` if you'd rather it sold everything.
- **Fails closed:** stale or suspicious prices, an unmatched ticker, a closed market or a rejected
  order mean no orders and a red run. GitHub emails you when a run fails.
- **Never leverages or shorts:** weights never go below zero or above the cash available.
- **Other holdings:** anything outside its 15 ETFs is left alone, except on a run started with
  "sell_other_holdings" ticked, which sells it once.

## Day to day

- Each run's summary (Actions tab, click a run) shows the account, drawdown, targets and orders.
- `state/<env>.json` holds the bot's memory and `state/<env>_runs.csv` one line per run.
- GitHub sometimes starts scheduled runs late or skips one. That's harmless: the next run
  catches up.
- It uses about 3 to 5 minutes of GitHub Actions a day, well inside the free 2,000 a month.


