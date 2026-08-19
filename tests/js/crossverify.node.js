/**
 * Runs the JS backtest engine against the EXACT panel dumped by
 * tests/js/gen_and_run_python.py (via the outputs helper, see that
 * script's own comment) and diffs the result against python_results.json
 * -- the real Python engine's output on identical input. This is the
 * actual correctness check for the JS port: not "does it run", but "does
 * it produce the same numbers as the engine it's replacing, on the same
 * data, including NaN/missing-value edge cases, condition-exit,
 * walk-forward, and top-K portfolio logic".
 *
 * Usage: node tests/js/crossverify.node.js <path-to-xverify-dir>
 */
const fs = require("fs");
const path = require("path");

const {
  runBacktest, summarize, walkForwardSweep, splitPanelByDate, runTopKBacktest, EXIT_MODE,
} = require("../../dashboard/static/backtest_engine.js");

const xverifyDir = process.argv[2] || path.join(__dirname, "fixtures");

const panel = JSON.parse(fs.readFileSync(path.join(xverifyDir, "panel.json"), "utf8"));
const pyOut = JSON.parse(fs.readFileSync(path.join(xverifyDir, "python_results.json"), "utf8"));

let failures = 0;
let checks = 0;

function approxEqual(a, b, tol = 0.011) {
  if (a === null || b === null) return a === b;
  if (typeof a === "number" && typeof b === "number") {
    if (Number.isNaN(a) && Number.isNaN(b)) return true;
    return Math.abs(a - b) <= tol;
  }
  return a === b;
}

function jsRowToComparable(row) {
  return {
    horizon_label: row.horizonLabel,
    total_signals: row.totalSignals,
    closed_trades: row.closedTrades,
    open_trades: row.openTrades,
    win_rate_pct: row.winRatePct,
    avg_return_pct: row.avgReturnPct,
    median_return_pct: row.medianReturnPct,
    best_return_pct: row.bestReturnPct,
    worst_return_pct: row.worstReturnPct,
    avg_holding_days: row.avgHoldingDays,
    low_sample_warning: row.lowSampleWarning,
  };
}

function compareRows(label, jsRows, pyRows) {
  const jsSorted = [...jsRows].map(jsRowToComparable).sort((a, b) => a.horizon_label.localeCompare(b.horizon_label));
  const pySorted = [...pyRows].sort((a, b) => a.horizon_label.localeCompare(b.horizon_label));

  checks++;
  if (jsSorted.length !== pySorted.length) {
    failures++;
    console.error(`[FAIL] ${label}: row count mismatch JS=${jsSorted.length} PY=${pySorted.length}`);
    return;
  }
  for (let i = 0; i < jsSorted.length; i++) {
    const j = jsSorted[i], p = pySorted[i];
    if (j.horizon_label !== p.horizon_label) {
      failures++;
      console.error(`[FAIL] ${label}[${i}]: horizon_label JS=${j.horizon_label} PY=${p.horizon_label}`);
      continue;
    }
    const fields = ["total_signals", "closed_trades", "open_trades", "win_rate_pct", "avg_return_pct",
      "median_return_pct", "best_return_pct", "worst_return_pct", "avg_holding_days", "low_sample_warning"];
    for (const f of fields) {
      if (!approxEqual(j[f], p[f])) {
        failures++;
        console.error(`[FAIL] ${label}[${j.horizon_label}].${f}: JS=${j[f]} PY=${p[f]}`);
      }
    }
  }
  if (failures === 0 || true) {
    console.log(`[ok]   ${label}: ${jsSorted.length} horizon row(s) checked`);
  }
}

function strategyFromPy(name, entryQuery, exitMode, fixedHoldingDays, exitQuery) {
  return { name, entryQuery, exitMode, fixedHoldingDays: fixedHoldingDays || [], exitQuery: exitQuery || null };
}

// --- fixed_basic ---
{
  const strat = strategyFromPy("fixed_basic", "final_score >= 65", EXIT_MODE.FIXED_HOLDING, [3, 7, 15]);
  const trades = runBacktest(panel, strat);
  compareRows("fixed_basic", summarize(trades, strat.name), pyOut.results.fixed_basic);
}

// --- fixed_and_or_not ---
{
  const strat = strategyFromPy("fixed_and_or_not", "final_score >= 60 and not (sector == 'Energy')", EXIT_MODE.FIXED_HOLDING, [5]);
  const trades = runBacktest(panel, strat);
  compareRows("fixed_and_or_not", summarize(trades, strat.name), pyOut.results.fixed_and_or_not);
}

// --- fixed_in_list ---
{
  const strat = strategyFromPy("fixed_in_list", "score_band in ['Strong', 'High Conviction']", EXIT_MODE.FIXED_HOLDING, [5, 10]);
  const trades = runBacktest(panel, strat);
  compareRows("fixed_in_list", summarize(trades, strat.name), pyOut.results.fixed_in_list);
}

// --- fixed_risk_nan (exercises NaN comparison semantics) ---
{
  const strat = strategyFromPy("fixed_risk_nan", "risk_penalty <= 12", EXIT_MODE.FIXED_HOLDING, [5]);
  const trades = runBacktest(panel, strat);
  compareRows("fixed_risk_nan", summarize(trades, strat.name), pyOut.results.fixed_risk_nan);
}

// --- condition_exit_basic ---
{
  const strat = strategyFromPy("condition_exit_basic", "final_score >= 65", EXIT_MODE.CONDITION_EXIT, []);
  const trades = runBacktest(panel, strat);
  compareRows("condition_exit_basic", summarize(trades, strat.name), pyOut.results.condition_exit_basic);
}

// --- condition_exit_explicit ---
{
  const strat = strategyFromPy("condition_exit_explicit", "rsi <= 40", EXIT_MODE.CONDITION_EXIT, [], "rsi >= 60");
  const trades = runBacktest(panel, strat);
  compareRows("condition_exit_explicit", summarize(trades, strat.name), pyOut.results.condition_exit_explicit);
}

// --- str_contains ---
{
  const strat = strategyFromPy("str_contains", "sector.str.contains('Tech')", EXIT_MODE.FIXED_HOLDING, [5]);
  const trades = runBacktest(panel, strat);
  compareRows("str_contains", summarize(trades, strat.name), pyOut.results.str_contains);
}

// --- top-K portfolio ---
{
  const strat = strategyFromPy("topk", "final_score >= 55", EXIT_MODE.FIXED_HOLDING, [5, 10]);
  const trades = runTopKBacktest(panel, strat, 2);
  compareRows("topk", summarize(trades, "topk"), pyOut.results.topk);
}

// --- split_panel_by_date ---
{
  const { train, test } = splitPanelByDate(panel, 65.0);
  const trainDays = new Set(train.data.scan_date).size;
  const testDays = new Set(test.data.scan_date).size;
  checks++;
  if (trainDays !== pyOut.results._split_check.train_days || testDays !== pyOut.results._split_check.test_days) {
    failures++;
    console.error(`[FAIL] split_panel_by_date: JS train=${trainDays} test=${testDays} PY train=${pyOut.results._split_check.train_days} test=${pyOut.results._split_check.test_days}`);
  } else {
    console.log(`[ok]   split_panel_by_date: train=${trainDays} test=${testDays}`);
  }
}

// --- walk_forward_sweep ---
{
  const wfRows = walkForwardSweep(panel, "final_score >= {t}", [50.0, 55.0, 60.0, 65.0, 70.0], [5, 10], 65.0);
  const jsSorted = [...wfRows].sort((a, b) => a.horizonLabel.localeCompare(b.horizonLabel));
  const pySorted = [...pyOut.results.walk_forward].sort((a, b) => a.horizon_label.localeCompare(b.horizon_label));
  checks++;
  if (jsSorted.length !== pySorted.length) {
    failures++;
    console.error(`[FAIL] walk_forward: row count JS=${jsSorted.length} PY=${pySorted.length}`);
  } else {
    for (let i = 0; i < jsSorted.length; i++) {
      const j = jsSorted[i], p = pySorted[i];
      const pairs = [
        ["bestThreshold", "best_threshold"], ["trainAvgReturnPct", "train_avg_return_pct"],
        ["trainClosedTrades", "train_closed_trades"], ["testAvgReturnPct", "test_avg_return_pct"],
        ["testClosedTrades", "test_closed_trades"], ["testWinRatePct", "test_win_rate_pct"],
        ["testLowSampleWarning", "test_low_sample_warning"],
      ];
      for (const [jk, pk] of pairs) {
        if (!approxEqual(j[jk], p[pk])) {
          failures++;
          console.error(`[FAIL] walk_forward[${j.horizonLabel}].${jk}: JS=${j[jk]} PY=${p[pk]}`);
        }
      }
    }
    console.log(`[ok]   walk_forward_sweep: ${jsSorted.length} horizon row(s) checked`);
  }
}

console.log(`\n${checks - failures > 0 ? "" : ""}Checks: ${checks}, failing rows/fields logged above: ${failures}`);
if (failures > 0) {
  console.error(`\nCROSS-VERIFICATION FAILED: ${failures} mismatch(es).`);
  process.exit(1);
} else {
  console.log("\nCROSS-VERIFICATION PASSED: JS engine output matches Python engine output exactly (within 0.011 rounding tolerance).");
}
