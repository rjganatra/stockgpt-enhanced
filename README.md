---
title: StockGPT Enhanced
emoji: 📈
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.61.1
app_file: dashboard/app.py
pinned: false
---

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

Three correctness details that matter for trusting the numbers: a condition that stays true for 40
consecutive days counts as **one** entry (rising-edge detection), not 40 inflated signals; a trade
still open at the end of your data is excluded from win-rate math and marked `is_open`, never silently
dropped or scored as a fake 0%; and a trade whose holding window contains a demerger-sized price jump
(`src/stockgpt/backtest/corporate_actions.py`) is excluded entirely, not counted at face value -- a
demerger's price effect is a real, permanent step-change in the raw data this pipeline has no way to
retroactively adjust for, so a return computed straight through one is meaningless, not just noisy. This
is a price-based heuristic (no free, reliable feed of actual NSE corporate-action events exists), not a
certainty: it flags any (symbol, date) whose `day_change_pct` diverges from that day's cross-sectional
median move by more than `BacktestDefaults.price_jump_threshold_pct` (35 points, market-relative so a
genuine broad-market crash day doesn't get every symbol flagged). Below
`BacktestDefaults.min_signals_for_confidence` (10) closed trades, the summary carries a
`low_sample_warning` flag rather than presenting a thin sample with false confidence.

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

## Beyond parity: five more ways to read the same data

Once feature parity and the core backtester were solid, five additions built on top of the existing
scan/history data and backtest engine -- no new data sources:

- **Score history** (Stock Explorer tab) -- every score component for one symbol, plotted across every
  daily snapshot on record, so a climb into High Conviction (or a quiet slide out of it) shows up as a
  trend instead of a single day's number.
- **Sector rotation** (Sectors tab, `src/stockgpt/sector_rotation.py`) -- compares each sector's average
  `final_score` over the most recent stretch of history against the stretch right before it, surfacing
  sectors gaining or losing conviction before it's obvious from one day's snapshot.
- **Leaderboard** (new tab) -- every saved Strategy Lab strategy plus every built-in preset, backtested
  and ranked in one pass by average return (confident, well-sampled results ranked above thin ones,
  never excluded outright).
- **Walk-forward validation** (Strategy Lab tab addition, `src/stockgpt/backtest/walkforward.py`) --
  the existing parameter sweep picks the best threshold across all of history at once, which risks
  finding what happened to work in hindsight. This splits history into an earlier training window and a
  later testing window, picks the best threshold using only the training window, then checks whether
  that same fixed threshold still performs on the unseen test window -- the standard way to tell a real
  edge from curve-fitting.
- **Portfolio (top-K) backtest** (new tab, `src/stockgpt/backtest/portfolio.py`) -- `run_backtest`
  treats every matching signal as its own independent trade, which doesn't reflect a real portfolio on a
  day where 40 stocks cross the threshold at once. This filters to only the top-K highest-ranked signals
  per entry day before simulating trades. Documented honestly in the module and the UI: it's top-K
  *signal selection*, not a full capital-tracked equity curve (no position sizing, no overlapping-capital
  modeling) -- a narrower, still-useful question about whether being selective would have helped.

One performance note worth recording: the backtest engine originally re-evaluated each strategy's entry
query once per symbol in a Python loop (`sdf.eval(...)` inside a `groupby`), which is fine for a single
Strategy Lab run but multiplies badly once a single button click runs several strategies back to back
(Leaderboard) or the same sweep twice (walk-forward's train + test). Switched to evaluating the query
once across the whole panel and slicing the resulting mask per symbol -- identical row-for-row (these are
row-wise boolean expressions, no cross-row aggregation), about 20x faster on the real production history
panel (~1,800 symbols): a single strategy backtest went from ~27s to ~1.5s.

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
  all 14 tabs render (including Signal Performance running all 11 catalog signals through the backtest
  engine live, and the Leaderboard/walk-forward/portfolio-backtest sections added in a later pass), the
  Strategy Lab tab runs a live backtest through the actual UI and reproduces the same numbers -- zero
  exceptions, zero warnings.

Run it yourself against a fresh checkout of the original repo:

```bash
python scripts/verify_against_real_data.py /path/to/StockGPT-main
```

Sample output from that run is committed under `examples/` so the dashboard and Strategy Lab work
immediately after cloning this repo, before you've run the pipeline once.

## Historical backfill

Every backtest feature above is only as good as how much real history it has to run against -- and a
freshly-deployed instance only accumulates one day of real history per day, which is why most strategies
start out flagged `low_sample_warning`. `scripts/backfill_history.py` fixes that by reconstructing years
of past `data/history/YYYY-MM-DD/scan.csv` snapshots directly, using the exact same scoring engine the
live pipeline uses.

This works cleanly because `scanner.py`, `relative_strength.py`, `range_bound.py`, and
`scoring.compute_final_scores` are all pure functions of whatever price-history window they're handed --
none of them assume "today," they just look at the last row of whatever DataFrame they're given. None of
that code changed for this. The one genuinely new problem is fundamentals: Yahoo's `.info` block (what
`fundamentals.py` normally fetches) is a live snapshot with no "as of last year" mode -- there's no way to
ask "what was this company's ROE a year ago" directly. `src/stockgpt/backfill.py` solves that by pulling
each symbol's raw filed financial statements (balance sheet, income statement, cash flow -- which Yahoo
does expose historically, several periods back) and deriving the same ratios `fundamentals.py` computes
from `.info` (ROE, ROA, margins, growth, debt/equity, current/quick ratio, cash, debt, operating/free cash
flow) directly from those statement line items, one set of ratios per real filing period, forward-filled
day by day until the next filing -- so a trading day's fundamentals are always the most recently *actually
filed* numbers as of that day, never a number from the future. Verified via `tests/test_backfill.py`
against a synthetic multi-period statement with a hand-checked no-lookahead assertion: the day before a
new filing still shows the OLD numbers, the filing's exact date is the first day the NEW numbers appear.

Two honest limits, by design:
- Valuation ratios that mix price with a per-share statement figure (PE, forward PE, price-to-book,
  dividend yield) and beta are not derived historically -- forward PE is a forward-looking analyst
  estimate with no historical record at all, and the others would need a separate shares-outstanding
  history this pass doesn't fetch. These fall back to today's current value, held constant, for the whole
  backfilled window.
- If a symbol's statements are empty or unusable on Yahoo (thin coverage, delisted, etc.), its *entire*
  fundamentals row falls back to today's current values held constant across the backfill window, rather
  than leaving it with zero fundamentals data. Every fallback is visible in the run summary
  (`derived_fundamentals` vs `fallback_fundamentals` counts) and in the code as
  `FUNDAMENTALS_SOURCE_CURRENT_FALLBACK` -- never silently blended with the real per-period numbers.

Also worth stating plainly: this backfills whatever symbols are in *today's* universe. Any company that
delisted, merged, or was removed from the index during the backfill window is invisible to it -- only
today's survivors get a multi-year history. That's a real survivorship bias in any backtest run against
this data. A bias-free backfill would need a point-in-time universe snapshot per year, which is a
separate, bigger project than this pass.

Run the pilot first -- the 50 most liquid NSE names (Nifty 50), most likely to have complete data on
both fronts, so you can check the real historical-fundamentals derivation actually works before spending
hours on the full universe:

```bash
python scripts/backfill_history.py --pilot --years 3
```

It prints a summary (`derived_fundamentals` / `fallback_fundamentals` split, days written) at the end.
Once that looks right, the recommended next step is `--nifty500` -- a real, live-fetched Nifty 500
constituent list (via `universe.fetch_nifty500_symbols()`, same niftyindices.com source `fetch_sector_map()`
already uses), not a hardcoded ticker list:

```bash
python scripts/backfill_history.py --nifty500 --years 3
```

This gives a meaningfully bigger backtest sample than the 50-symbol pilot (~500 vs 50 symbols) while
staying well short of the memory footprint the full ~2,000-symbol universe would need once this data is
loaded into the deployed dashboard. If the live fetch fails (network hiccup, or niftyindices.com changes
its CSV format), the script falls back to the 50-symbol pilot list automatically rather than crashing.

Beyond that, the full universe is available with the same command minus `--pilot`/`--nifty500`. It's safe
to re-run any of these after a rate-limit interruption -- already-written days are skipped by default, not
re-fetched (`--no-resume` forces a clean overwrite instead). The full universe can genuinely take hours for
~2,000 symbols x 3+ years of price history and statements; Yahoo can throttle bulk requests partway
through, which is exactly what the resumability is for.

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
  universe.py           NSE universe fetch + fallback list; sector base map now tries Nifty
                          Total Market (~750 symbols) before falling back to Nifty 500 (~500) --
                          same trusted niftyindices.com domain/format, just broader coverage
  fundamentals.py        fetch + sector-aware fundamental scoring
  relative_strength.py    1M/3M/6M returns, vs-Nifty, sector rank
  range_bound.py           adaptive support/resistance bands
  scoring.py                risk penalty + final blended score + band classification (the bug fix)
  history.py                  daily snapshot save/diff
  watchlist.py                 GitHub-backed watchlist store, secret-gated writes
  signal_catalog.py             the original's ~12 signal types as backtestable Strategy objects
  alerts.py                      detection (new High Conviction / saved-strategy matches / sharp
                                   watchlist changes) + Telegram/email delivery
  sector_rotation.py              recent-vs-prior-window average final_score per sector
  backfill.py                      reconstructs years of historical scan.csv snapshots by deriving
                                     point-in-time fundamentals from filed statements (see "Historical
                                     backfill" above)
  backtest/
    strategy.py                  Strategy definition (entry/exit rules)
    engine.py                     signal detection + trade simulation
    metrics.py                     win rate / return summary stats
    walkforward.py                 train/test date split + walk-forward threshold sweep
    portfolio.py                    top-K per-day signal-selection backtest
    corporate_actions.py             price-jump heuristic (demergers etc.) -- excludes contaminated trades

dashboard/app.py     Streamlit dashboard, 14 tabs: Overview (incl. a live custom-filter query box),
                       Heatmap, Opportunities, Sectors (incl. sector rotation), Stock Explorer (incl.
                       score history chart), History, Watchlist (incl. sector concentration check),
                       Fundamentals, Movers & Changes, Range Bound, Signal Performance, Strategy Lab
                       (incl. a parameter sweep + walk-forward validation), Leaderboard, Portfolio
                       Backtest -- full parity with the original's 11 tabs plus 4 new backtest-powered
                       ones and this pass's 2 new tabs (Leaderboard, Portfolio Backtest)

scripts/
  run_daily_pipeline.py       universe -> scan -> relative strength -> range bound -> merge
                                fundamentals -> final score -> save + history/change tracking
  run_weekly_fundamentals.py    fetch + score fundamentals separately (Yahoo throttles bulk .info)
  run_alerts.py                  runs after the daily pipeline; reads scan/changes/saved
                                   strategies/watchlist and sends a summary via Telegram/email
  verify_against_real_data.py    the real-data verification described above
  backfill_history.py             multi-year historical backfill (see "Historical backfill" above)

tests/                 111 unit tests (plus a separate tests/js/ suite for the browser-side backtest
                          engine port, cross-verified against this Python engine on identical data --
                          see below): scoring (incl. a regression test encoding the original's
                          risk-penalty bug as an assertion), backtest engine correctness (hand-computed
                          expected returns), fundamentals, scanner, watchlist access control,
                          signal catalog validity, alert detection + delivery (mocked network),
                          sector-source fallback chain, sector rotation, walk-forward validation
                          (synthetic panels with a known generalizing vs. overfit outcome),
                          top-K portfolio backtest, historical backfill (ratio derivation from raw
                          statements, no-lookahead as-of merge, fallback triggering), corporate-action
                          price-jump detection and exclusion (market-relative flagging, and a
                          mid-holding-window trade actually gets excluded, not just entry/exit days)
tests/js/               dashboard/static/{query_parser,backtest_engine,panel_loader}.js unit + cross-
                          verification tests (Node's built-in test runner, no added dependency) --
                          gen_fixture.py generates a synthetic panel (including a deliberate demerger-
                          sized price jump) through the REAL Python engine, crossverify.node.js runs
                          the same panel through the JS port and diffs every number; panel_loader and
                          browser_build.test.js additionally cover the data-shaping helpers and the
                          actual concatenated-into-one-script browser build (see widget_strategy_lab.py)
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
4. `python -m pytest tests/ -q` to confirm everything passes (111 tests).
5. `streamlit run dashboard/app.py` to run the dashboard locally against the sample data in `examples/`
   (copy `examples/sample_scan_latest.csv` to `data/scans/latest_scan.csv` and
   `examples/sample_history/*` to `data/history/` first, or just run `run_daily_pipeline.py` to fetch
   live data).
6. Enable GitHub Actions on the new repo -- `daily_pipeline.yml` and `weekly_fundamentals.yml` will
   start populating `data/` on their schedules (or trigger them manually via `workflow_dispatch`).
7. Deploy `dashboard/app.py` somewhere that runs Streamlit for a shareable link. Streamlit Community
   Cloud is the simplest option but caps memory at ~1GB, which the full historical backfill (see below)
   comfortably exceeds. Hugging Face Spaces' free CPU tier gives 16GB instead -- see the next section.

## Deploying on Hugging Face Spaces (recommended once you've run the historical backfill)

This repo's `README.md` already has the YAML metadata block Hugging Face Spaces looks for at the very
top (`sdk: streamlit`, `app_file: dashboard/app.py`, etc.) -- no extra config file needed.

1. Create a free account at [huggingface.co](https://huggingface.co) if you don't have one.
2. Create a new Space: choose the **Streamlit** SDK and the **free CPU basic** hardware tier (2 vCPU /
   16GB RAM).
3. Connect it to this GitHub repo. Hugging Face Spaces can either host its own git remote (you push
   directly to it) or sync from GitHub automatically via a GitHub Action -- either way, once connected,
   pushes to `main` redeploy the Space the same way pushes redeploy a Streamlit Community Cloud app.
4. Add any secrets (`WATCHLIST_SECRET`, `TELEGRAM_BOT_TOKEN`, etc.) under the Space's **Settings ->
   Variables and secrets** -- same idea as Streamlit Cloud's secrets.toml, different place to paste
   them. The app reads them via `st.secrets` / `os.environ` either way, no code changes needed.
5. The Space gets a `huggingface.co/spaces/<you>/<space-name>` URL. Free-tier Spaces sleep after a
   period with no visitors, same as Streamlit Community Cloud's 12-hour sleep -- the memory ceiling is
   what actually changes, not the sleep behavior.

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
