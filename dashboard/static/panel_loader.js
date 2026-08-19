/**
 * Pure data-shaping helpers for turning the quarterly shard files
 * scripts/export_dashboard_data.py::export_strategy_lab_shards writes into
 * the panel shape backtest_engine.js expects. Deliberately separated from
 * any fetch()/IndexedDB/DOM code (see widget_app.js) so this file has no
 * dependency on running inside a browser -- it's plain data transformation,
 * unit-tested directly in Node (tests/js/panel_loader.test.js), which a
 * fetch()-and-DOM-heavy file can't be without a real browser.
 */

/** Every "YYYY-Qn" quarter key from dateFrom through dateTo inclusive
 * (both "YYYY-MM-DD" strings), in chronological order. */
function quartersInRange(dateFrom, dateTo) {
  const [fy, fm] = dateFrom.split("-").map(Number);
  const [ty, tm] = dateTo.split("-").map(Number);
  const fq = Math.floor((fm - 1) / 3);
  const tq = Math.floor((tm - 1) / 3);
  const quarters = [];
  let y = fy, q = fq;
  while (y < ty || (y === ty && q <= tq)) {
    quarters.push(`${y}-Q${q + 1}`);
    q++;
    if (q > 3) { q = 0; y++; }
  }
  return quarters;
}

/** A raw shard's data has JSON `null` for every missing value (JSON can't
 * encode NaN). backtest_engine.js and query_parser.js require missing
 * NUMERIC values to be JS `NaN` and missing STRING/categorical values to
 * stay `null` -- see query_parser.js's module comment for why that split
 * convention is what makes comparison semantics match pandas/numpy for
 * free. `scan_date` and `symbol` are always strings; every other column in
 * a shard is numeric (the *_reasons text columns were already dropped at
 * export time, and no other string columns are part of this export's
 * schema) -- STRING_COLUMNS lists the ones that stay as strings/null;
 * everything else gets the null->NaN conversion. */
const STRING_COLUMNS = new Set(["symbol", "scan_date", "sector", "score_band"]);

function rawShardToPanel(raw) {
  const data = {};
  for (const col of raw.columns) {
    const arr = raw.data[col];
    data[col] = STRING_COLUMNS.has(col) ? arr : arr.map((v) => (v === null ? NaN : v));
  }
  return { columns: raw.columns, data };
}

/** Concatenates already-converted panels (same column set, different
 * quarters) into one combined panel -- order of the input array becomes
 * row order, so callers should pass quarters chronologically for a
 * panel that's already date-ordered without a separate sort pass. */
function concatPanels(panels) {
  if (panels.length === 0) return { columns: [], data: {} };
  const columns = panels[0].columns;
  const data = {};
  for (const col of columns) {
    data[col] = [].concat(...panels.map((p) => p.data[col]));
  }
  return { columns, data };
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { quartersInRange, rawShardToPanel, concatPanels, STRING_COLUMNS };
}
