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

## Set it up (about 20 minutes)

1. **Create a private GitHub repository** and push this folder to it. Keep it private: the bot
   commits your account value to `state/` every day.
2. **Make a demo API key.** In the Trading 212 app, switch to your Practice (demo) account, go to
   Settings, API (Beta), and generate a key. Tick these permissions: account, portfolio,
   orders:read, orders:execute, metadata, history:transactions. Don't restrict it by IP address,
   because GitHub's servers change IP every run.
3. **Add the key to GitHub.** In the repository: Settings, Secrets and variables, Actions.
   - Secrets: `T212_API_KEY` and `T212_API_SECRET`
   - Variables: `T212_ENV` = `demo`, and `TRADING_ENABLED` = `false` for now
4. **Check the ticker mapping.** Actions tab, Trade, Run workflow, tick "Only check the ticker
   mapping". The log should list all 15 ETFs against Trading 212 tickers ending in `l_EQ`. If any
   show `??`, add the right ticker to `T212_TICKERS` in `config.py`.
5. **Do a dry run.** Run the workflow again with "dry run" ticked. The run summary shows the
   account, the target weights and the exact trades it would make.
6. **Let it trade the demo account.** Set `TRADING_ENABLED` to `true`. From then on it runs every
   weekday at 13:17 UTC (14:17 in summer). Leave it on demo for a few weeks and check the runs.

## Going live

1. In the app, switch to your real ISA and generate a key with the same permissions.
2. Replace the two secrets with the live key and set `T212_ENV` to `live`.
3. `TRADING_ENABLED` stays `true`. The next weekday run trades real money.

Starting from cash it builds the portfolio over about 8 trading days (35% of the gap each day). Don't
use pies or trade by hand in this ISA while the bot runs: it leaves other holdings alone, but it
won't send orders while a manual order is pending.

## Replacing an older bot

**Easiest:** on your PC, from the unzipped folder, run `python switch_over.py`. It uses your own
GitHub login through the GitHub CLI (install commands at the top of the file). It disables the old
bot's workflows, creates a private `isa-bot` repository, stores your Trading 212 key as encrypted
secrets with trading off, runs the ticker check and a dry run, then stops so you can check.
`python switch_over.py --go-live` (weekdays, 08:00 to 16:30 UK time) switches trading on.

By hand instead:

1. **Stop the old bot first.** In its repository: Actions, select its workflow, the "..." menu,
   Disable workflow (or delete its `.yml` file). Two bots on one account would fight, and this one
   cancels any pending API orders it finds, including the old bot's.
2. Put this code in its place (same repository with the old files deleted, or a new private one).
3. Make sure the key was generated with the **Stocks ISA** selected in the app (each account has its
   own key) and has the permissions listed above. Store it as `T212_API_KEY` and `T212_API_SECRET`,
   then set `T212_ENV` = `live` and `TRADING_ENABLED` = `false`.
4. Run the workflow with "check the ticker mapping" ticked, then with "dry run" and
   "sell_other_holdings" both ticked. The summary lists what it would sell and buy on your real
   account without sending anything.
5. Set `TRADING_ENABLED` = `true` and run it once more with only "sell_other_holdings" ticked. It
   sells whatever the old bot held outside these 15 ETFs, keeps any of these ETFs it already holds,
   and starts building the 60/40. After that it runs on its own every weekday.

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

## Changing what it runs

In `config.py`, `mode` picks the book: `"core"` (default, passive 60/40), `"core_tilt"` (v2) or
`"absolute"` (v1). Everything else, including the guardrails, stays the same.

Research rule: every time you change a model setting and rerun the backtest, add 1 to
`N_TRIALS`. The deflated Sharpe ratio uses it to discount the luck of having tried many ideas.
A new idea only replaces the core if it beats it out of sample after costs.

## Run it locally

```
pip install -r requirements.txt
python -m pytest -q          # 20 tests, no internet or API key needed
python backtest.py           # downloads prices the first time (about 2 minutes)
```

## Know the limits

- Market risk remains: a 60/40 portfolio can fall 20% or more, and past results don't
  guarantee future ones. This is software, not financial advice.
- Prices come from Yahoo Finance, which is free but unofficial. The bot repairs known faults
  and refuses to trade on suspicious prints.
- Trading 212's API is in beta and can change. Market orders aren't idempotent there, so the bot
  never resends an order that might have gone through.
- Order quantities are rounded down to 2 decimal places. If an ETF only allows whole shares, the
  bot retries with whole shares.
