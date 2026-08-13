# StockGPT Enhanced

A ground-up rewrite of [StockGPT](https://github.com/rjganatra/StockGPT), the NSE market-intelligence
bot: same design intent, same score-band language, same "share the dashboard, gate the watchlist"
model -- rebuilt clean, with the bugs found during a deep read of the original fixed at the root
rather than patched, and one new flagship feature: a real strategy backtester that answers "if I'd
entered every time I saw X, what would my win rate actually have been?"

This is a separate codebase. It does not touch, import from, or write to the original repo. Every
module here was written from an understanding of *why* the original behaves the way it does (read in
full, plus its git history), not copied from it.

## Dashboard tab parity

Worth being upfront about: the first version of this dashboard shipped with 8 tabs. The original has
11. That gap wasn't found by re-reading code -- it was found by opening the original's *live* deployed
app side by side with this one and counting. Three tabs were missing outright (Heatmap, Sectors, Model
Performance / Signal Tracker), and several others were thinner than the original (Opportunities was
missing its Swing Candidates section, Watchlist had no Remove button, Fundamentals had no ranked
tables, Movers & Changes just dumped a raw table instead of breaking it into improvers/droppers/alerts,
Range Bound had no sub-sections, and Stock Explorer had no reason breakdown). All of that is fixed now
-- the dashboard has all 11 of the original's tabs plus 2 new ones (Signal Performance, Strategy Lab),
verified end to end against real data (see "Verified against real data" below). The scoring/pipeline
logic underneath was never the problem; the dashboard's UI surface was where corners got cut on the
first pass, and now isn't.

One catalog worth calling out specifically: `src/stockgpt/signal_catalog.py` reproduces the original's
~12 built-in signal types (`app/performance/signal_performance.py::detect_signals`) as real `Strategy`
objects, so their win rate runs through the same rigorous backtest engine as any Strategy Lab query --
exact fixed-day holding periods, rising-edge signal dedup, open-trade exclusion -- instead of the
original's looser "compare to today's price, bucket by however many days happened to pass" method. One
signal type, "Top Final Conviction" (top 25 by score that day), didn't carry over: it's a ranking, not
a per-row threshold, and the query-string format every Strategy uses can't express "top N of the day."
It's covered in spirit by "High Conviction" (`final_score >= 65`), which stays meaningful regardless of
how the universe size changes -- documented in the module rather than silently dropped.

## What's actually different from the original

**The risk-penalty bug.** The original's `score_engine.py` computed a combined technical+fundamental
risk penalty correctly, then did `pd.concat([df, risk_df], axis=1)` followed by
`df.loc[:, ~df.columns.duplicated()]`. `df` already had an older, technical-only `risk_penalty` column
from an earlier pipeline stage, and `duplicated()` keeps the *first* occurrence by default -- so the
newly computed, fundamentals-aware penalty was silently discarded on every run, while its accompanying
`risk_reasons` text (no naming collision) survived untouched. Verified live: a stock with debt/equity
275 and -2.7% net margin correctly said so in its risk reasons, while its actual `risk_penalty` --
the number subtracted from the score -- was 0. `src/stockgpt/scoring.py` has exactly one
`compute_risk_penalty()` function, and its output is assigned directly (`df["risk_penalty"] = ...`),
never concatenated against a same-named column that could shadow it.

**Hardcoded filter bounds.** You'd already caught and fixed one instance of this yourself (a Rs.1.3L
price ceiling silently excluding MRF at Rs.1.5L) before this rebuild started. Rather than re-fix that
one slider, `dashboard/app.py` has exactly one slider constructor, `adaptive_slider()`, used for every
filter in the app. It reads `min()`/`max()` off the live dataset on every render. There is no second
place a hardcoded bound could hide.

**Missing fundamental data read as "safe".** yfinance's `financialData` block (ROE, ROA, current/quick
ratio, operating & free cash flow) is all-or-nothing for roughly half of NSE tickers -- verified
against the original's live data, nullness across those six fields was 92-97% correlated, i.e. one
missing upstream block, not six independent failures. The original's pipeline later did `fillna(0)` on
several of these on the way to the final CSV. A 0 debt-to-equity reads as "no debt" to every downstream
risk check, so an unfetched ratio scored as *safe*. Here every ratio keeps `NaN` when unknown, plus a
`<field>_known` companion column end to end, and a fundamentals section with too little coverage picks
up a small explicit risk penalty instead of a free pass (`RiskPenalties.low_data_coverage_penalty`).

**The WebApp's combo strategy builder was dead on arrival, and the git history explains exactly how.**
`webapp/index.html`'s inline JS has `comboBuilderBox.classList.toggle("hidden", action !== );` and
`if (action === ) {` -- both missing their right-hand operand, a `SyntaxError` that breaks the entire
`<script>` block. `git log -S` on the original repo traces this to commit `ef96e813`, titled (not
making this up) "Repair combo strategy builder WebApp option." A follow-up commit, `840a8094`, "Fix
duplicate combo strategy payload handler," touched a different function and didn't fix it. Two scoring
engines existed in the original too (`scan_52w.py`'s v1: trend/momentum/reversal/volume scores, and
`score_engine.py`'s v2: differently-weighted technical score) -- both shipped to the CSV, v1 never read
by anything downstream. This rebuild has one technical score, computed once (`scanner.py`), and no
WebApp/Telegram layer yet at all (see "What's deliberately not here yet" below).

**Two scoring engines became one, everywhere.** Wherever the original accumulated near-duplicate
columns across rewrites (`conviction_score` vs `score` vs `final_conviction_score`; `risk_penalty`
computed twice), `src/stockgpt/schema.py` now has exactly one canonical name per concept. Importing a
name from `schema` instead of typing a raw string turns a typo into an `ImportError` instead of a
silent `KeyError -> .get(..., 0)` fallback three modules later -- which is exactly the bug class that
caused the risk-penalty issue above.

**What's kept, deliberately unchanged, because the original design was already right:**
- Range-bound bands sized to each stock's own 10th/90th percentile close price, not a fixed rupee
  level -- a Rs.50 stock and a Rs.1.5L stock each get bands scaled to their own price automatically.
- Sector-aware fundamental scoring: a bank's 8x leverage isn't "very high debt" the way a
  consumer-goods company's would be, because it's structural to how banks fund their balance sheet.
  Kept the idea, generalized to fewer, clearer per-bucket rules (`fundamentals.py::_sector_adjustment_row`).
- The watchlist's write-gating. The dashboard is meant to be shared; the watchlist isn't. Every write
  path calls one function, `watchlist.has_write_access()`, which fails closed on an empty secret.
- The score-band ladder (renamed from letter grades to Avoid / Weak / Neutral / Watchlist / Strong /
  High Conviction -- same six-tier structure, without implying more precision than a heuristic score
  actually has).

## The new thing: Strategy Lab

The original could tell you a stock's score today. It couldn't tell you whether that score has
historically meant anything. `src/stockgpt/backtest/` adds a real backtester:

- Write an entry rule as a pandas query against any scan column: `final_score >= 65 and score_band in
  ['Strong', 'High Conviction']`, or `technical_score >= 70 and range_status == 'Accumulation Zone'`,
  or anything else the daily snapshot has a column for.
- Exit either on a fixed holding period (5D/15D/30D/60D, configurable) or a condition (hold until the
  entry condition stops being true, or a separate exit rule fires) -- both modes, selectable per
  strategy.
- Get back win rate, average/median return, best/worst trade, and sample size, computed from your
  actual historical daily snapshots -- not eyeballed, not simulated.

Two correctness details that matter for trusting the numbers: a condition that stays true for 40
consecutive days counts as **one** entry (rising-edge detection), not 40 inflated signals; and a trade
still open at the end of your data is excluded from win-rate math and marked `is_open`, never silently
dropped or scored as a fake 0%. Below `BacktestDefaults.min_signals_for_confidence` (10) closed trades,
the summary carries a `low_sample_warning` flag rather than presenting a thin sample with false
confidence.

Try it in the dashboard's **Strategy Lab** tab, or directly. One honest caveat on the "Save this
strategy" button: it writes to a local JSON file (`data/backtest/saved_strategies.json`), which is
durable if you're running the dashboard locally or self-hosted (and will get picked up next time
`daily_pipeline.yml` commits `data/` back to the repo), but is **not** durable on Streamlit Community
Cloud, where the filesystem resets on redeploy. If you want saved strategies to survive redeploys on
Cloud, the watchlist's `GitHubWatchlistStore` pattern in `watchlist.py` is the template to follow --
same idea, committing through the GitHub API instead of local disk. Not built yet because it wasn't
asked for; flagging it here since it's the kind of gap that's easy to hit only after you've saved a
strategy and lost it.

```python
from stockgpt.backtest import Strategy, ExitMode, run_backtest, summarize
from stockgpt.backtest.engine import load_history_panel

panel = load_history_panel("data/history")
strategy = Strategy(
    name="Strong conviction",
    entry_query="final_score >= 65 and score_band in ['Strong', 'High Conviction']",
    exit_mode=ExitMode.FIXED_HOLDING,
    fixed_holding_days=(7, 15, 30, 60),
)
trades = run_backtest(panel, strategy)
print(summarize(trades, strategy.name))
```

## Verified against real data, not just synthetic tests

`scripts/verify_against_real_data.py` rebuilds this pipeline's output for all 66 real NSE trading days
captured by the *original* bot (`data/history/*/latest_scan.csv` in the original repo -- used purely as
a real-data fixture here, no original code involved). Worth knowing what it found:

- Day one of that history has 30 rows (a manual watchlist); every day after has 1,500-2,200+ rows --
  the original bot switched to scanning the full NSE universe starting day two. The verification script
  restricts to the ~1,874 symbols present on at least 60 of the 66 days so relative-strength history is
  apples-to-apples.
- Backtesting the literal example from the original ask -- `final_score >= 65 and score_band in
  ['Strong', 'High Conviction']` -- produces **zero** signals in this specific fixture, for an honest,
  traceable reason: relative-strength scoring has 40 of its 100 points gated on outperformance vs.
  Nifty (no benchmark series exists in this fixture) and another 10 gated on a 6-month return window
  (126 trading days -- longer than the fixture's 66). Both stay `None`, per design, rather than being
  faked as 0. The backtester correctly reports "no historical evidence" instead of inventing a number.
- Backtesting `technical_score >= 65` alone (unaffected by that gap) on the same real data produces
  27,384 real trade-legs and win rates of 50-58% across 5D/15D/30D/60D holding periods, with realistic
  best/worst trades (including a couple of real ~90-95% smallcap drawdowns the backtester correctly
  captured rather than smoothed over).
- The dashboard itself was run end to end against this real data via Streamlit's `AppTest` harness --
  all 12 tabs render (including Signal Performance running all 11 catalog signals through the backtest
  engine live), the Strategy Lab tab runs a live backtest through the actual UI and reproduces the same
  numbers -- zero exceptions, zero warnings.

Run it yourself against a fresh checkout of the original repo:

```bash
python scripts/verify_against_real_data.py /path/to/StockGPT-main
```

Sample output from that run is committed under `examples/` so the dashboard and Strategy Lab work
immediately after cloning this repo, before you've run the pipeline once.

## What's deliberately not here yet

Telegram bot commands and the Mini App WebApp -- by design, this pass is core pipeline + fixed scoring
+ dashboard + backtester, verified end to end, first. The Telegram/WebApp layer is substantial (the
original's `telegram_bot_commands.py` alone is 3,300+ lines across several UX rewrites) and layers
cleanly on top of everything here once it's solid. Nothing about this codebase's design blocks adding
it later.

## Project layout

```
src/stockgpt/
  schema.py          canonical column names -- import these, never type raw strings
  config.py           every weight/threshold/window, named and documented, in one place
  scanner.py           technical score (RSI, SMA, volume, price location)
  universe.py           NSE universe fetch + fallback list
  fundamentals.py        fetch + sector-aware fundamental scoring
  relative_strength.py    1M/3M/6M returns, vs-Nifty, sector rank
  range_bound.py           adaptive support/resistance bands
  scoring.py                risk penalty + final blended score + band classification (the bug fix)
  history.py                  daily snapshot save/diff
  watchlist.py                 GitHub-backed watchlist store, secret-gated writes
  signal_catalog.py             the original's ~12 signal types as backtestable Strategy objects
  alerts.py                      detection (new High Conviction / saved-strategy matches / sharp
                                   watchlist changes) + Telegram/email delivery
  backtest/
    strategy.py                  Strategy definition (entry/exit rules)
    engine.py                     signal detection + trade simulation
    metrics.py                     win rate / return summary stats

dashboard/app.py     Streamlit dashboard, 12 tabs: Overview (incl. a live custom-filter query box),
                       Heatmap, Opportunities, Sectors, Stock Explorer, History, Watchlist (incl.
                       sector concentration check), Fundamentals, Movers & Changes, Range Bound,
                       Signal Performance, Strategy Lab (incl. a parameter sweep) -- full parity
                       with the original's 11 tabs plus the 2 new backtest-powered ones

scripts/
  run_daily_pipeline.py       universe -> scan -> relative strength -> range bound -> merge
                                fundamentals -> final score -> save + history/change tracking
  run_weekly_fundamentals.py    fetch + score fundamentals separately (Yahoo throttles bulk .info)
  run_alerts.py                  runs after the daily pipeline; reads scan/changes/saved
                                   strategies/watchlist and sends a summary via Telegram/email
  verify_against_real_data.py    the real-data verification described above

tests/                 54 unit tests: scoring (incl. a regression test encoding the original's
                          risk-penalty bug as an assertion), backtest engine correctness (hand-computed
                          expected returns), fundamentals, scanner, watchlist access control,
                          signal catalog validity, alert detection + delivery (mocked network)
examples/                real 66-day sample dataset (see above) so the app works on first clone
.github/workflows/         daily_pipeline.yml (includes the alerts step), weekly_fundamentals.yml, tests.yml
```

## Setup

1. Create a new GitHub repo and push this codebase to it.
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.streamlit/secrets.toml` (for local dashboard runs) and/or set the same
   values as repository secrets in GitHub Actions / your Streamlit Cloud app settings. Everything is
   optional except that an unset `WATCHLIST_SECRET` makes the watchlist tab read-only everywhere,
   fail-closed, which is the intended default.
4. `python -m pytest tests/ -q` to confirm everything passes (54 tests).
5. `streamlit run dashboard/app.py` to run the dashboard locally against the sample data in `examples/`
   (copy `examples/sample_scan_latest.csv` to `data/scans/latest_scan.csv` and
   `examples/sample_history/*` to `data/history/` first, or just run `run_daily_pipeline.py` to fetch
   live data).
6. Enable GitHub Actions on the new repo -- `daily_pipeline.yml` and `weekly_fundamentals.yml` will
   start populating `data/` on their schedules (or trigger them manually via `workflow_dispatch`).
7. Deploy `dashboard/app.py` on Streamlit Community Cloud (or anywhere else that runs Streamlit) for a
   shareable link, same as the original.

## Alerts

`scripts/run_alerts.py` runs automatically as the last step of `daily_pipeline.yml`, after data is
committed. It checks three things: symbols that newly crossed into the High Conviction band since the
last snapshot, any saved Strategy Lab strategy whose entry condition matches today's scan (a live
check, not a backtest), and sharp score/band/risk changes on symbols already in the watchlist. If
nothing triggers, it does nothing. If something does, it's sent via Telegram and/or email -- whichever
you've configured as repository secrets (see `.env.example`); configuring neither still computes and
prints the alert in the workflow log, it just isn't delivered anywhere. A delivery failure on one
channel doesn't block the other, and never fails the pipeline job itself -- the data commit already
succeeded by the time alerts run.
