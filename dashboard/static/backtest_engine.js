/**
 * JavaScript port of src/stockgpt/backtest/{engine,metrics,walkforward,portfolio}.py.
 *
 * This exists so Strategy Lab, the parameter sweep, walk-forward validation,
 * and the portfolio backtest -- the four tools that let a visitor type an
 * arbitrary condition and see a real historical answer -- can run entirely
 * in the visitor's own browser against a downloaded historical panel,
 * instead of needing a live Python server to hold the full multi-year
 * dataset in RAM to answer them. See scripts/export_dashboard_data.py's
 * module docstring for why the OTHER backtest-powered tabs (Leaderboard,
 * Signal Performance) do NOT need this -- they're fixed, precomputed daily
 * server-side instead.
 *
 * Every function here is a deliberate line-for-line mirror of its Python
 * counterpart, not a reimplementation from the description -- see the
 * matching comment in each Python source file for the reasoning behind a
 * given piece of logic; it isn't repeated here. tests/js/*.test.js
 * cross-checks this file's output against the real Python engine's output
 * on identical inputs, not just "does it run".
 *
 * Panel shape used throughout: `{ columns: string[], data: { [col]:
 * any[] } }` -- one flat array per column, all the same length, index-
 * aligned (row i's symbol is data.symbol[i], its scan_date is
 * data.scan_date[i], etc.). scan_date values are "YYYY-MM-DD" strings,
 * which sort correctly with plain string comparison -- no Date parsing
 * needed anywhere in this file. Missing numeric values must be JS `NaN`;
 * missing string/categorical values must be JS `null` -- see
 * query_parser.js's module comment for why that specific pair of
 * conventions is what makes pandas-equivalent comparison semantics fall
 * out "for free" from plain JS operators.
 */

/* global require, module */
// Deliberately NOT named `evalQueryMask`/`QueryParseError` here: in the
// browser build, widget_strategy_lab.py concatenates this file with
// query_parser.js into ONE inline <script>, so they share a single
// top-level scope. query_parser.js already declares `function
// evalQueryMask` and `class QueryParseError` at that same top level --
// redeclaring either name here (even as `var`, which normally coexists
// fine with a `function` of the same name) throws a SyntaxError for the
// `class`-declared one ("Identifier 'QueryParseError' has already been
// declared"), since class declarations can't share a name with a second
// var/let/const/class/function anywhere in the same scope. Using unique
// local names sidesteps the collision entirely while still resolving to
// the exact same function/class either way (Node's require(), or the
// window properties query_parser.js's own browser-build branch attaches).
const _qp = (typeof require !== "undefined") ? require("./query_parser.js") : window;
const _evalQueryMask = _qp.evalQueryMask;
const _QueryParseError = _qp.QueryParseError;

const MIN_SIGNALS_FOR_CONFIDENCE = 10; // BACKTEST_DEFAULTS.min_signals_for_confidence
const PRICE_JUMP_THRESHOLD_PCT = 35.0; // BACKTEST_DEFAULTS.price_jump_threshold_pct

/**
 * Line-for-line port of corporate_actions.py::flag_price_jumps -- see that
 * file's module docstring for the full reasoning (demergers/other
 * corporate actions produce a real, permanent price discontinuity that
 * current_price/day_change_pct don't get adjusted for, so a trade whose
 * holding window spans one gets a contaminated return). Returns a
 * Uint8Array aligned with `panel`'s row order (1 = flagged), not a
 * per-symbol array -- callers index it the same way they already index
 * panel.data.current_price/scan_date, via a global row number.
 */
function computePriceJumpFlags(panel, thresholdPct = PRICE_JUMP_THRESHOLD_PCT) {
  const n = panel.data[panel.columns[0]] ? panel.data[panel.columns[0]].length : 0;
  const flags = new Uint8Array(n);
  const dayChange = panel.data.day_change_pct;
  if (!dayChange) return flags; // column missing -> nothing flagged, matches the Python fallback
  const dates = panel.data.scan_date;

  // Cross-sectional median day_change_pct per date -- a cheap proxy for
  // "how the market moved that day" computed straight from the panel
  // already in memory, no external index feed needed. NaN values are
  // excluded from the median itself (matches pandas' skipna=True default),
  // not coerced to 0.
  const byDate = new Map();
  for (let i = 0; i < n; i++) {
    const v = dayChange[i];
    if (Number.isNaN(v)) continue;
    const d = dates[i];
    if (!byDate.has(d)) byDate.set(d, []);
    byDate.get(d).push(v);
  }
  const medianByDate = new Map();
  for (const [d, vals] of byDate) {
    vals.sort((a, b) => a - b);
    const mid = Math.floor(vals.length / 2);
    medianByDate.set(d, vals.length % 2 === 0 ? (vals[mid - 1] + vals[mid]) / 2 : vals[mid]);
  }

  for (let i = 0; i < n; i++) {
    const v = dayChange[i];
    // Missing day_change_pct (a symbol's first-ever row in the panel --
    // new listing, or start of this data's coverage) is flagged as
    // "unsafe", the same deliberate choice corporate_actions.py makes, not
    // silently treated as "no jump happened".
    if (Number.isNaN(v)) { flags[i] = 1; continue; }
    const market = medianByDate.has(dates[i]) ? medianByDate.get(dates[i]) : 0;
    flags[i] = Math.abs(v - market) >= thresholdPct ? 1 : 0;
  }
  return flags;
}

function pyRound(value, decimals = 0) {
  // Matches Python's built-in round() (banker's / round-half-to-even)
  // closely enough for 2-3 decimal places of a price-derived percentage --
  // see this file's module comment on the cross-verification harness that
  // checks this against real Python output with a small tolerance, same
  // philosophy the existing Python test suite already uses
  // (pytest.approx(..., abs=0.01)) rather than demanding bit-exact floats.
  if (!Number.isFinite(value)) return value;
  const factor = Math.pow(10, decimals);
  const scaled = value * factor;
  const floor = Math.floor(scaled);
  const diff = scaled - floor;
  let rounded;
  if (Math.abs(diff - 0.5) < 1e-9) {
    rounded = (floor % 2 === 0) ? floor : floor + 1;
  } else {
    rounded = Math.round(scaled);
  }
  return rounded / factor;
}

function mean(arr) {
  if (arr.length === 0) return NaN;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function median(arr) {
  if (arr.length === 0) return NaN;
  const sorted = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
}

/** Builds symbol -> row-indices-into-panel-arrays, each sorted ascending
 * by scan_date -- the JS equivalent of `panel.groupby(SYMBOL)` followed by
 * `.sort_values(SCAN_DATE)` per group. Computed once per backtest run, not
 * once per strategy, since it doesn't depend on the strategy at all. */
function groupBySymbol(panel) {
  const symbols = panel.data.symbol;
  const dates = panel.data.scan_date;
  const groups = new Map();
  for (let i = 0; i < symbols.length; i++) {
    const sym = symbols[i];
    if (!groups.has(sym)) groups.set(sym, []);
    groups.get(sym).push(i);
  }
  for (const indices of groups.values()) {
    indices.sort((a, b) => (dates[a] < dates[b] ? -1 : dates[a] > dates[b] ? 1 : 0));
  }
  return groups;
}

/** Rising-edge entry detection: `mask` values indexed by POSITION within
 * `indices` (0..indices.length-1), already date-sorted -- returns the
 * positions where the mask transitions false->true, so a condition that
 * stays true for 40 consecutive days counts as ONE entry, matching
 * engine.py::_entry_dates exactly. */
function entryPositions(indices, maskAtPosition) {
  const positions = [];
  let prev = false;
  for (let pos = 0; pos < indices.length; pos++) {
    const val = !!maskAtPosition(pos);
    if (val && !prev) positions.push(pos);
    prev = val;
  }
  return positions;
}

function runFixedHolding(symbol, indices, panel, positions, holdingDays, jumpFlags) {
  const trades = [];
  const price = panel.data.current_price;
  const dates = panel.data.scan_date;
  const n = indices.length;
  for (const pos of positions) {
    if (jumpFlags && jumpFlags[indices[pos]]) continue; // don't even open on a flagged day
    const entryRow = indices[pos];
    const entryDate = dates[entryRow];
    const entryPrice = Number(price[entryRow]);
    if (!(entryPrice > 0)) continue;
    for (const days of holdingDays) {
      const exitPos = pos + days;
      const label = `${days}D`;
      if (exitPos < n) {
        // Excluded, not just re-priced, if ANY day from entry through exit
        // (inclusive) was flagged -- see computePriceJumpFlags's comment
        // and corporate_actions.py's module docstring for why a clean
        // entry/exit price pair doesn't mean the return between them is.
        let spansJump = false;
        if (jumpFlags) {
          for (let k = pos; k <= exitPos; k++) {
            if (jumpFlags[indices[k]]) { spansJump = true; break; }
          }
        }
        if (spansJump) continue;
        const exitRow = indices[exitPos];
        const exitPrice = Number(price[exitRow]);
        if (!(exitPrice > 0)) continue;
        const ret = pyRound(((exitPrice - entryPrice) / entryPrice) * 100, 2);
        trades.push({
          symbol, entryDate, entryPrice, exitDate: dates[exitRow], exitPrice,
          holdingDays: days, returnPct: ret, isOpen: false, horizonLabel: label,
        });
      } else {
        trades.push({
          symbol, entryDate, entryPrice, exitDate: null, exitPrice: null,
          holdingDays: null, returnPct: null, isOpen: true, horizonLabel: label,
        });
      }
    }
  }
  return trades;
}

function runConditionExit(symbol, indices, panel, positions, entryMaskAtPosition, exitMaskAtPosition, jumpFlags) {
  const trades = [];
  const price = panel.data.current_price;
  const dates = panel.data.scan_date;
  const n = indices.length;
  // stop_mask = exit_mask if provided else ~entry_mask.fillna(False) --
  // here masks are already plain booleans (no NaN state to fill), so "stop
  // when the entry condition is no longer true" is simply `!entryMask`.
  const stopAtPosition = exitMaskAtPosition || ((pos) => !entryMaskAtPosition(pos));

  for (const pos of positions) {
    if (jumpFlags && jumpFlags[indices[pos]]) continue;
    const entryRow = indices[pos];
    const entryDate = dates[entryRow];
    const entryPrice = Number(price[entryRow]);
    if (!(entryPrice > 0)) continue;

    let exitPos = null;
    for (let j = pos + 1; j < n; j++) {
      if (stopAtPosition(j)) { exitPos = j; break; }
    }
    if (exitPos === null) {
      trades.push({
        symbol, entryDate, entryPrice, exitDate: null, exitPrice: null,
        holdingDays: null, returnPct: null, isOpen: true, horizonLabel: "condition_exit",
      });
      continue;
    }
    let spansJump = false;
    if (jumpFlags) {
      for (let k = pos; k <= exitPos; k++) {
        if (jumpFlags[indices[k]]) { spansJump = true; break; }
      }
    }
    if (spansJump) continue;
    const exitRow = indices[exitPos];
    const exitPrice = Number(price[exitRow]);
    if (!(exitPrice > 0)) continue;
    const holdingDays = exitPos - pos;
    const ret = pyRound(((exitPrice - entryPrice) / entryPrice) * 100, 2);
    trades.push({
      symbol, entryDate, entryPrice, exitDate: dates[exitRow], exitPrice,
      holdingDays, returnPct: ret, isOpen: false, horizonLabel: "condition_exit",
    });
  }
  return trades;
}

const EXIT_MODE = { FIXED_HOLDING: "fixed_holding", CONDITION_EXIT: "condition_exit" };

/**
 * strategy: { name, entryQuery, exitMode, fixedHoldingDays: number[],
 * exitQuery: string|null }
 * Throws QueryParseError (message meant to be shown to the person who
 * typed the query) on a bad entry/exit query -- same up-front-validation
 * contract as run_backtest in engine.py.
 */
function runBacktest(panel, strategy) {
  const n = panel.data[panel.columns[0]] ? panel.data[panel.columns[0]].length : 0;
  if (n === 0) return [];

  const fullEntryMask = _evalQueryMask(panel, strategy.entryQuery);
  const fullExitMask = strategy.exitQuery ? _evalQueryMask(panel, strategy.exitQuery) : null;
  // Strategy-independent, computed once per call -- see computePriceJumpFlags.
  const jumpFlags = computePriceJumpFlags(panel);

  const groups = groupBySymbol(panel);
  const allTrades = [];
  for (const [symbol, indices] of groups) {
    const entryAtPos = (pos) => !!fullEntryMask[indices[pos]];
    const positions = entryPositions(indices, entryAtPos);
    if (positions.length === 0) continue;

    if (strategy.exitMode === EXIT_MODE.FIXED_HOLDING) {
      allTrades.push(...runFixedHolding(symbol, indices, panel, positions, strategy.fixedHoldingDays, jumpFlags));
    } else {
      const exitAtPos = fullExitMask ? (pos) => !!fullExitMask[indices[pos]] : null;
      allTrades.push(...runConditionExit(symbol, indices, panel, positions, entryAtPos, exitAtPos, jumpFlags));
    }
  }
  return allTrades;
}

/** One summary row per horizon_label -- mirrors metrics.py::summarize exactly,
 * including the low_sample_warning threshold and the "closed empty -> all
 * zeros but still a row, not a skipped row" behavior. */
function summarize(trades, strategyName, winThresholdPct = 0.0) {
  if (trades.length === 0) return [];

  const byHorizon = new Map();
  for (const t of trades) {
    if (!byHorizon.has(t.horizonLabel)) byHorizon.set(t.horizonLabel, []);
    byHorizon.get(t.horizonLabel).push(t);
  }

  const summaries = [];
  for (const [horizon, group] of byHorizon) {
    const closed = group.filter((t) => !t.isOpen);
    const openCount = group.filter((t) => t.isOpen).length;
    const total = group.length;

    if (closed.length === 0) {
      summaries.push({
        strategyName, horizonLabel: horizon, totalSignals: total, closedTrades: 0,
        openTrades: openCount, winRatePct: 0.0, avgReturnPct: 0.0, medianReturnPct: 0.0,
        bestReturnPct: 0.0, worstReturnPct: 0.0, avgHoldingDays: 0.0, lowSampleWarning: true,
      });
      continue;
    }

    const returns = closed.map((t) => t.returnPct);
    const holding = closed.map((t) => t.holdingDays);
    const wins = returns.filter((r) => r > winThresholdPct).length;

    summaries.push({
      strategyName,
      horizonLabel: horizon,
      totalSignals: total,
      closedTrades: closed.length,
      openTrades: openCount,
      winRatePct: pyRound((wins / closed.length) * 100, 2),
      avgReturnPct: pyRound(mean(returns), 2),
      medianReturnPct: pyRound(median(returns), 2),
      bestReturnPct: pyRound(Math.max(...returns), 2),
      worstReturnPct: pyRound(Math.min(...returns), 2),
      avgHoldingDays: pyRound(mean(holding), 1),
      lowSampleWarning: closed.length < MIN_SIGNALS_FOR_CONFIDENCE,
    });
  }
  return summaries;
}

/** Builds a new columnar panel containing only the given global row indices
 * -- the JS equivalent of pandas boolean-indexing a DataFrame. */
function subPanel(panel, rowIndices) {
  const data = {};
  for (const col of panel.columns) {
    const src = panel.data[col];
    data[col] = rowIndices.map((i) => src[i]);
  }
  return { columns: panel.columns, data };
}

/** Splits by distinct scan_date (not by row) -- earliest splitPct% of
 * DAYS go to train, the rest to test, mirroring walkforward.py::
 * split_panel_by_date exactly, including the >=2-distinct-dates guard and
 * the [1, len(dates)-1] clamp so neither side can end up empty. */
function splitPanelByDate(panel, splitPct = 70.0) {
  const dates = panel.data.scan_date;
  if (dates.length === 0) return { train: subPanel(panel, []), test: subPanel(panel, []) };

  const distinctDates = [...new Set(dates)].sort();
  if (distinctDates.length < 2) return { train: subPanel(panel, []), test: subPanel(panel, []) };

  let splitIdx = Math.round((distinctDates.length * splitPct) / 100);
  splitIdx = Math.max(1, Math.min(distinctDates.length - 1, splitIdx));
  const trainDates = new Set(distinctDates.slice(0, splitIdx));

  const trainIdx = [];
  const testIdx = [];
  for (let i = 0; i < dates.length; i++) {
    (trainDates.has(dates[i]) ? trainIdx : testIdx).push(i);
  }
  return { train: subPanel(panel, trainIdx), test: subPanel(panel, testIdx) };
}

/** Mirrors walkforward.py::walk_forward_sweep exactly: sweep `thresholds`
 * on the TRAIN window, pick the best-by-avg_return_pct threshold per
 * horizon (preferring thresholds with enough closed trades to be
 * confident, same tie-break as the Python version), then report that same
 * fixed threshold's real performance on the TEST window it was never
 * selected against. */
function walkForwardSweep(panel, template, thresholds, holdingDays, splitPct = 70.0, winThresholdPct = 0.0) {
  const { train, test } = splitPanelByDate(panel, splitPct);
  const trainLen = train.data[panel.columns[0]] ? train.data[panel.columns[0]].length : 0;
  const testLen = test.data[panel.columns[0]] ? test.data[panel.columns[0]].length : 0;
  if (trainLen === 0 || testLen === 0) return [];

  const trainRows = []; // { horizonLabel, threshold, ...summaryRow }
  for (const t of thresholds) {
    const query = template.replace("{t}", formatThreshold(t));
    const strategy = { name: `t=${t}`, entryQuery: query, exitMode: EXIT_MODE.FIXED_HOLDING, fixedHoldingDays: holdingDays };
    let trades;
    try {
      trades = runBacktest(train, strategy);
    } catch (e) {
      if (e instanceof _QueryParseError) continue;
      throw e;
    }
    if (trades.length === 0) continue;
    for (const row of summarize(trades, `t=${t}`, winThresholdPct)) {
      trainRows.push({ ...row, threshold: t });
    }
  }
  if (trainRows.length === 0) return [];

  const byHorizon = new Map();
  for (const row of trainRows) {
    if (!byHorizon.has(row.horizonLabel)) byHorizon.set(row.horizonLabel, []);
    byHorizon.get(row.horizonLabel).push(row);
  }

  const results = [];
  for (const [horizon, group] of byHorizon) {
    const confident = group.filter((r) => r.closedTrades >= MIN_SIGNALS_FOR_CONFIDENCE);
    const candidates = confident.length > 0 ? confident : group;
    // Python: candidates.sort_values("avg_return_pct", ascending=False).iloc[0]
    // -- a plain descending sort by avg_return_pct, first element wins;
    // ties keep original (thus threshold-ascending) order since JS Array
    // sort is stable, matching pandas' stable sort here too.
    const best = [...candidates].sort((a, b) => b.avgReturnPct - a.avgReturnPct)[0];
    const bestThreshold = best.threshold;

    const testQuery = template.replace("{t}", formatThreshold(bestThreshold));
    const testStrategy = { name: `t=${bestThreshold}`, entryQuery: testQuery, exitMode: EXIT_MODE.FIXED_HOLDING, fixedHoldingDays: holdingDays };
    let testTrades = [];
    try {
      testTrades = runBacktest(test, testStrategy);
    } catch (e) {
      if (!(e instanceof _QueryParseError)) throw e;
    }

    let testRow = null;
    if (testTrades.length > 0) {
      const testSummary = summarize(testTrades, `t=${bestThreshold}`, winThresholdPct);
      testRow = testSummary.find((r) => r.horizonLabel === horizon) || null;
    }

    results.push({
      horizonLabel: horizon,
      bestThreshold,
      trainAvgReturnPct: best.avgReturnPct,
      trainClosedTrades: best.closedTrades,
      testAvgReturnPct: testRow ? testRow.avgReturnPct : null,
      testClosedTrades: testRow ? testRow.closedTrades : 0,
      testWinRatePct: testRow ? testRow.winRatePct : null,
      testLowSampleWarning: testRow ? testRow.lowSampleWarning : true,
    });
  }
  return results;
}

// Python's str.format inserts a plain str(t) -- for a float like 65.0 that
// prints "65.0", for 65 (already-int threshold) it prints "65". thresholds
// in this app are always generated as floats (sweep_min/step are floats),
// so mirror that by always keeping one decimal unless more precision was
// explicitly entered.
function formatThreshold(t) {
  return String(t);
}

/** Mirrors portfolio.py::run_topk_backtest exactly: same rising-edge entry
 * detection as runBacktest, fixed-holding exit only, but entries are
 * bucketed by their actual calendar entry date across ALL symbols first,
 * and only the top_k highest-`rankColumn` entries per day are kept. */
function runTopKBacktest(panel, strategy, topK, rankColumn = "final_score") {
  const n = panel.data[panel.columns[0]] ? panel.data[panel.columns[0]].length : 0;
  if (n === 0) return [];
  if (strategy.exitMode !== EXIT_MODE.FIXED_HOLDING) {
    throw new Error("Top-K portfolio backtest only supports fixed-holding exits for now.");
  }
  if (topK < 1) throw new Error("topK must be at least 1.");

  const fullEntryMask = _evalQueryMask(panel, strategy.entryQuery);
  const jumpFlags = computePriceJumpFlags(panel);
  const groups = groupBySymbol(panel);
  const rankArr = panel.data[rankColumn];
  const dates = panel.data.scan_date;

  const byDate = new Map(); // entryDate -> [{symbol, indices, pos, rankValue}]
  for (const [symbol, indices] of groups) {
    const entryAtPos = (pos) => !!fullEntryMask[indices[pos]];
    const positions = entryPositions(indices, entryAtPos);
    for (const pos of positions) {
      const row = indices[pos];
      const entryDate = dates[row];
      const rankValue = rankArr ? rankArr[row] : -Infinity;
      if (!byDate.has(entryDate)) byDate.set(entryDate, []);
      byDate.get(entryDate).push({ symbol, indices, pos, rankValue });
    }
  }

  const kept = [];
  for (const candidates of byDate.values()) {
    const ranked = [...candidates].sort((a, b) => {
      const av = Number.isNaN(a.rankValue) ? -Infinity : a.rankValue;
      const bv = Number.isNaN(b.rankValue) ? -Infinity : b.rankValue;
      return bv - av; // descending, stable (matches pandas' stable sort)
    });
    kept.push(...ranked.slice(0, topK));
  }

  const allTrades = [];
  for (const { symbol, indices, pos } of kept) {
    allTrades.push(...runFixedHolding(symbol, indices, panel, [pos], strategy.fixedHoldingDays, jumpFlags));
  }
  return allTrades;
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    EXIT_MODE, pyRound, mean, median, groupBySymbol, entryPositions,
    runFixedHolding, runConditionExit, runBacktest, summarize,
    subPanel, splitPanelByDate, walkForwardSweep, runTopKBacktest, formatThreshold,
    computePriceJumpFlags, PRICE_JUMP_THRESHOLD_PCT,
  };
}
