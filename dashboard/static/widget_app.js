/**
 * Browser-side UI for Strategy Lab, the parameter sweep, walk-forward
 * validation, and the portfolio backtest -- the four ad-hoc, visitor-typed
 * backtest tools that run entirely client-side now (see backtest_engine.js
 * and panel_loader.js's module comments for why). This file owns the parts
 * that genuinely need a real browser and can't be unit-tested in Node:
 * IndexedDB caching, fetch(), and DOM rendering. The data-shaping logic it
 * calls into (panel_loader.js) and the backtest math itself
 * (backtest_engine.js, query_parser.js) ARE unit-tested/cross-verified
 * against the Python engine -- see tests/js/. This file's own wiring is
 * verified by hand against a real deployed page (see this project's commit
 * history for that verification note), the same way any DOM/fetch code
 * ultimately has to be.
 *
 * Expects three globals to already be defined by the time this runs (the
 * Python-side widget assembler concatenates query_parser.js,
 * backtest_engine.js, and panel_loader.js before this file -- see
 * dashboard/widget_strategy_lab.py):
 *   evalQueryMask, QueryParseError      (query_parser.js)
 *   runBacktest, summarize, splitPanelByDate, walkForwardSweep,
 *   runTopKBacktest, EXIT_MODE          (backtest_engine.js)
 *   quartersInRange, rawShardToPanel, concatPanels  (panel_loader.js)
 *
 * Also expects a `window.STOCKGPT_CONFIG` object injected by the Python
 * assembler: { rawBase, meta: {days, symbols, date_from, date_to},
 * presetStrategies: [...] }.
 */
(function () {
  "use strict";

  const CONFIG = window.STOCKGPT_CONFIG;
  const DB_NAME = "stockgpt_strategy_lab_v1";
  const STORE_NAME = "quarters";

  let panel = null; // the currently-loaded combined panel
  let loadedQuarters = [];
  let allQuarters = [];

  // --- IndexedDB (persistent, cross-session cache for closed/immutable
  // quarters -- see the design note in this project's history on why this
  // is what makes "close the tab and reopen" NOT re-download anything) ---
  function openDb() {
    return new Promise((resolve, reject) => {
      const req = indexedDB.open(DB_NAME, 1);
      req.onupgradeneeded = () => { req.result.createObjectStore(STORE_NAME); };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
  }

  function idbGet(db, key) {
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, "readonly");
      const req = tx.objectStore(STORE_NAME).get(key);
      req.onsuccess = () => resolve(req.result || null);
      req.onerror = () => reject(req.error);
    });
  }

  function idbPut(db, key, value) {
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, "readwrite");
      tx.objectStore(STORE_NAME).put(value, key);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  }

  async function fetchQuarter(quarter) {
    const url = `${CONFIG.rawBase}/strategy_lab/${quarter}.json`;
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`Failed to fetch ${quarter} (HTTP ${resp.status})`);
    return resp.json();
  }

  // The CURRENT (still-growing) quarter is never cached indefinitely --
  // it's re-fetched fresh every widget load, same reasoning
  // scripts/trim_history_symbols.py's --latest-only docstring gives for
  // never treating a not-yet-closed period as immutable. Every OTHER
  // quarter is closed and permanent, so it's fetched once, ever, per
  // browser.
  function isCurrentQuarter(quarter) {
    const [y, qStr] = quarter.split("-Q");
    const q = parseInt(qStr, 10);
    const now = new Date();
    const nowQuarter = Math.floor(now.getMonth() / 3) + 1;
    return parseInt(y, 10) === now.getFullYear() && q === nowQuarter;
  }

  async function loadQuartersCached(db, quarters, onProgress) {
    const raws = [];
    for (const q of quarters) {
      let raw = isCurrentQuarter(q) ? null : await idbGet(db, q);
      if (!raw) {
        raw = await fetchQuarter(q);
        if (!isCurrentQuarter(q)) await idbPut(db, q, raw);
      }
      raws.push(raw);
      if (onProgress) onProgress(q);
    }
    return raws;
  }

  function setStatus(text) {
    document.getElementById("sl-status").textContent = text;
  }

  async function ensureDefaultPanel() {
    setStatus("Loading trailing 1 year of history...");
    allQuarters = quartersInRange(CONFIG.meta.date_from, CONFIG.meta.date_to);
    const defaultQuarters = allQuarters.slice(-4);
    const db = await openDb();
    try {
      const raws = await loadQuartersCached(db, defaultQuarters, (q) => setStatus(`Loading ${q}...`));
      loadedQuarters = defaultQuarters;
      panel = concatPanels(raws.map(rawShardToPanel));
      const rowCount = panel.data[panel.columns[0]] ? panel.data[panel.columns[0]].length : 0;
      setStatus(`Ready -- ${loadedQuarters.length} quarter(s) loaded (${rowCount.toLocaleString()} rows, `
        + `${loadedQuarters[0]} to ${loadedQuarters[loadedQuarters.length - 1]}). `
        + `Cached in this browser -- won't re-download on reload.`);
      setButtonsEnabled(true);
    } catch (e) {
      setStatus(`Couldn't load historical data: ${e.message}. Try reloading the page.`);
    }
  }

  async function loadFullHistory() {
    const remaining = allQuarters.filter((q) => !loadedQuarters.includes(q));
    if (remaining.length === 0) return;
    const totalMbEstimate = remaining.length * 8; // rough, matches this project's own measured ~8MB/quarter
    const ok = window.confirm(
      `This downloads the remaining ${remaining.length} quarter(s) of history (~${totalMbEstimate}MB). `
      + `Recommended on desktop/wifi, not mobile data. Continue?`
    );
    if (!ok) return;
    setStatus(`Loading ${remaining.length} more quarter(s)...`);
    const db = await openDb();
    try {
      const raws = await loadQuartersCached(db, remaining, (q) => setStatus(`Loading ${q}...`));
      loadedQuarters = allQuarters.slice();
      panel = concatPanels(loadedQuarters.map((q, i) => {
        const idx = remaining.indexOf(q);
        return idx >= 0 ? rawShardToPanel(raws[idx]) : null;
      }).filter(Boolean).length === remaining.length
        ? [...loadedQuarters.slice(0, loadedQuarters.length - remaining.length).map(() => null), ...raws].filter(Boolean)
        : raws); // fallback path below is the real one used
      // Simpler and correct: just refetch-concat everything we now have
      // cached, in order, rather than trying to splice partial arrays.
      const db2 = db;
      const allRaws = await loadQuartersCached(db2, allQuarters, null);
      panel = concatPanels(allRaws.map(rawShardToPanel));
      const rowCount = panel.data[panel.columns[0]] ? panel.data[panel.columns[0]].length : 0;
      setStatus(`Ready -- full history loaded (${loadedQuarters.length} quarters, ${rowCount.toLocaleString()} rows).`);
      document.getElementById("sl-load-full-btn").style.display = "none";
    } catch (e) {
      setStatus(`Couldn't load full history: ${e.message}. Still using the ${loadedQuarters.length}-quarter default.`);
    }
  }

  function setButtonsEnabled(enabled) {
    document.querySelectorAll("button.sl-run-btn").forEach((b) => { b.disabled = !enabled; });
  }

  // --- Result rendering ---
  function renderTable(containerId, rows, columns) {
    const container = document.getElementById(containerId);
    if (!rows || rows.length === 0) {
      container.innerHTML = "<p class='sl-empty'>No historical signals matched.</p>";
      return;
    }
    const cols = columns || Object.keys(rows[0]);
    let html = "<table class='sl-table'><thead><tr>";
    for (const c of cols) html += `<th>${c}</th>`;
    html += "</tr></thead><tbody>";
    for (const row of rows) {
      html += "<tr>";
      for (const c of cols) {
        const v = row[c];
        html += `<td>${v === null || v === undefined ? "-" : v}</td>`;
      }
      html += "</tr>";
    }
    html += "</tbody></table>";
    container.innerHTML = html;
  }

  const SUMMARY_COLS = ["horizonLabel", "totalSignals", "closedTrades", "openTrades",
    "winRatePct", "avgReturnPct", "medianReturnPct", "bestReturnPct", "worstReturnPct",
    "avgHoldingDays", "lowSampleWarning"];

  function parseDays(text) {
    return text.split(",").map((s) => parseInt(s.trim(), 10)).filter((n) => !Number.isNaN(n));
  }

  // --- Strategy Lab ---
  function runStrategyLab() {
    const entryQuery = document.getElementById("sl-entry-query").value;
    const exitMode = document.getElementById("sl-exit-mode").value;
    const winThreshold = parseFloat(document.getElementById("sl-win-threshold").value) || 0;
    const strategy = {
      name: "custom", entryQuery, exitMode,
      fixedHoldingDays: exitMode === "fixed_holding" ? parseDays(document.getElementById("sl-holding-days").value) : [],
      exitQuery: exitMode === "condition_exit" ? (document.getElementById("sl-exit-query").value || null) : null,
    };
    try {
      const trades = runBacktest(panel, strategy);
      if (trades.length === 0) {
        document.getElementById("sl-results").innerHTML = "<p class='sl-empty'>No historical signals matched this entry condition.</p>";
        return;
      }
      renderTable("sl-results", summarize(trades, "custom", winThreshold), SUMMARY_COLS);
    } catch (e) {
      document.getElementById("sl-results").innerHTML = `<p class='sl-error'>${e.message}</p>`;
    }
  }

  // Strategy Lab has no way to write back to saved_strategies.json --
  // st.components.v1.html is one-way (Python -> iframe), with no channel
  // back out, unlike a full custom Streamlit component (Streamlit.
  // setComponentValue()), which is a much bigger build than this migration
  // warranted. This is the honest substitute: show the Strategy.to_dict()-
  // shaped JSON in a read-only box so the visitor can copy it and paste it
  // into data/backtest/saved_strategies.json (or PRESET_STRATEGIES in
  // src/stockgpt/backtest/strategy.py) by hand.
  function showStrategyConfig() {
    const config = {
      name: document.getElementById("sl-name").value || "My strategy",
      entry_query: document.getElementById("sl-entry-query").value,
      exit_mode: document.getElementById("sl-exit-mode").value,
      fixed_holding_days: document.getElementById("sl-exit-mode").value === "fixed_holding"
        ? parseDays(document.getElementById("sl-holding-days").value) : [],
      exit_query: document.getElementById("sl-exit-mode").value === "condition_exit"
        ? (document.getElementById("sl-exit-query").value || null) : null,
      win_return_threshold_pct: parseFloat(document.getElementById("sl-win-threshold").value) || 0,
      description: "",
    };
    const box = document.getElementById("sl-config-json");
    box.style.display = "block";
    box.value = JSON.stringify(config, null, 2);
    box.focus();
    box.select();
  }

  // --- Parameter sweep ---
  function runSweep() {
    const template = document.getElementById("sweep-template").value;
    const min = parseFloat(document.getElementById("sweep-min").value);
    const max = parseFloat(document.getElementById("sweep-max").value);
    const step = parseFloat(document.getElementById("sweep-step").value) || 1;
    const holdingDays = parseDays(document.getElementById("sweep-holding-days").value);
    const winThreshold = parseFloat(document.getElementById("sweep-win-threshold").value) || 0;

    const thresholds = [];
    for (let v = min; v <= max + 1e-9; v += step) thresholds.push(Math.round(v * 1e4) / 1e4);

    const rows = [];
    const errors = [];
    for (const t of thresholds) {
      let query;
      try { query = template.replace("{t}", String(t)); } catch (e) { errors.push(`t=${t}: ${e.message}`); continue; }
      const strat = { name: `t=${t}`, entryQuery: query, exitMode: EXIT_MODE.FIXED_HOLDING, fixedHoldingDays: holdingDays };
      let trades;
      try { trades = runBacktest(panel, strat); } catch (e) { errors.push(`t=${t}: ${e.message}`); continue; }
      if (trades.length === 0) continue;
      for (const row of summarize(trades, `t=${t}`, winThreshold)) rows.push({ threshold: t, ...row });
    }
    if (rows.length === 0) {
      document.getElementById("sweep-results").innerHTML = "<p class='sl-empty'>No threshold in this range produced any historical signal.</p>";
      return;
    }
    rows.sort((a, b) => a.horizonLabel.localeCompare(b.horizonLabel) || b.avgReturnPct - a.avgReturnPct);
    renderTable("sweep-results", rows, ["threshold", "horizonLabel", "closedTrades", "winRatePct", "avgReturnPct", "medianReturnPct", "lowSampleWarning"]);
  }

  // --- Walk-forward validation ---
  function runWalkForward() {
    const template = document.getElementById("wf-template").value;
    const min = parseFloat(document.getElementById("wf-min").value);
    const max = parseFloat(document.getElementById("wf-max").value);
    const step = parseFloat(document.getElementById("wf-step").value) || 1;
    const holdingDays = parseDays(document.getElementById("wf-holding-days").value);
    const splitPct = parseFloat(document.getElementById("wf-split-pct").value) || 70;

    const thresholds = [];
    for (let v = min; v <= max + 1e-9; v += step) thresholds.push(Math.round(v * 1e4) / 1e4);

    try {
      const rows = walkForwardSweep(panel, template, thresholds, holdingDays, splitPct);
      if (rows.length === 0) {
        document.getElementById("wf-results").innerHTML = "<p class='sl-empty'>No threshold produced any historical signal on the training window.</p>";
        return;
      }
      renderTable("wf-results", rows, ["horizonLabel", "bestThreshold", "trainAvgReturnPct", "trainClosedTrades",
        "testAvgReturnPct", "testClosedTrades", "testWinRatePct", "testLowSampleWarning"]);
    } catch (e) {
      document.getElementById("wf-results").innerHTML = `<p class='sl-error'>${e.message}</p>`;
    }
  }

  // --- Portfolio backtest ---
  function runPortfolio() {
    const entryQuery = document.getElementById("pf-entry-query").value;
    const holdingDays = parseDays(document.getElementById("pf-holding-days").value);
    const topK = parseInt(document.getElementById("pf-top-k").value, 10) || 5;
    const winThreshold = parseFloat(document.getElementById("pf-win-threshold").value) || 0;
    const strategy = { name: "portfolio", entryQuery, exitMode: EXIT_MODE.FIXED_HOLDING, fixedHoldingDays: holdingDays };
    try {
      const baseline = runBacktest(panel, strategy);
      const topk = runTopKBacktest(panel, strategy, topK);
      if (baseline.length === 0) {
        document.getElementById("pf-results").innerHTML = "<p class='sl-empty'>No historical signals matched this entry condition.</p>";
        return;
      }
      const rows = [
        ...summarize(baseline, "Every signal", winThreshold),
        ...summarize(topk, `Top ${topK}/day`, winThreshold),
      ].sort((a, b) => a.horizonLabel.localeCompare(b.horizonLabel) || a.strategyName.localeCompare(b.strategyName));
      renderTable("pf-results", rows, ["strategyName", ...SUMMARY_COLS]);
    } catch (e) {
      document.getElementById("pf-results").innerHTML = `<p class='sl-error'>${e.message}</p>`;
    }
  }

  function populatePresetDropdown() {
    const select = document.getElementById("sl-preset");
    for (const strat of CONFIG.presetStrategies) {
      const opt = document.createElement("option");
      opt.value = strat.name;
      opt.textContent = strat.name;
      select.appendChild(opt);
    }
    select.addEventListener("change", () => {
      const strat = CONFIG.presetStrategies.find((s) => s.name === select.value);
      if (!strat) return;
      document.getElementById("sl-name").value = strat.name;
      document.getElementById("sl-entry-query").value = strat.entry_query;
      document.getElementById("sl-exit-mode").value = strat.exit_mode;
      document.getElementById("sl-holding-days").value = (strat.fixed_holding_days || []).join(",");
      document.getElementById("sl-exit-query").value = strat.exit_query || "";
      if (strat.win_return_threshold_pct !== undefined) {
        document.getElementById("sl-win-threshold").value = strat.win_return_threshold_pct;
      }
    });
  }

  function wireButtons() {
    document.getElementById("sl-run-btn").addEventListener("click", runStrategyLab);
    document.getElementById("sl-config-btn").addEventListener("click", showStrategyConfig);
    document.getElementById("sweep-run-btn").addEventListener("click", runSweep);
    document.getElementById("wf-run-btn").addEventListener("click", runWalkForward);
    document.getElementById("pf-run-btn").addEventListener("click", runPortfolio);
    document.getElementById("sl-load-full-btn").addEventListener("click", loadFullHistory);
  }

  function init() {
    setButtonsEnabled(false);
    populatePresetDropdown();
    wireButtons();
    ensureDefaultPanel();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
