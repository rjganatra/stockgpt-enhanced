/** Uses Node's built-in test runner (node --test) -- no extra dependency,
 * consistent with this repo's preference for a small, dependency-light JS
 * surface (see backtest_engine.js's own module comment). */
const test = require("node:test");
const assert = require("node:assert/strict");

const { quartersInRange, rawShardToPanel, concatPanels } = require("../../dashboard/static/panel_loader.js");

test("quartersInRange: single quarter", () => {
  assert.deepEqual(quartersInRange("2026-05-01", "2026-06-15"), ["2026-Q2"]);
});

test("quartersInRange: spans a year boundary", () => {
  assert.deepEqual(
    quartersInRange("2025-10-01", "2026-02-01"),
    ["2025-Q4", "2026-Q1"],
  );
});

test("quartersInRange: exact quarter-start/end boundaries", () => {
  assert.deepEqual(
    quartersInRange("2026-01-01", "2026-12-31"),
    ["2026-Q1", "2026-Q2", "2026-Q3", "2026-Q4"],
  );
});

test("quartersInRange: multi-year span", () => {
  const result = quartersInRange("2023-08-16", "2026-08-19");
  assert.equal(result[0], "2023-Q3");
  assert.equal(result[result.length - 1], "2026-Q3");
  assert.equal(result.length, 13); // matches the real export's actual quarter count
});

test("rawShardToPanel: converts null to NaN for numeric columns, keeps null for string columns", () => {
  const raw = {
    columns: ["symbol", "scan_date", "sector", "final_score", "risk_penalty"],
    data: {
      symbol: ["AAA", "BBB"],
      scan_date: ["2026-01-01", "2026-01-01"],
      sector: ["Technology", null],
      final_score: [65.5, null],
      risk_penalty: [null, 10.0],
    },
  };
  const panel = rawShardToPanel(raw);
  assert.equal(panel.data.sector[1], null, "missing string stays null");
  assert.ok(Number.isNaN(panel.data.final_score[1]), "missing number becomes NaN");
  assert.ok(Number.isNaN(panel.data.risk_penalty[0]), "missing number becomes NaN");
  assert.equal(panel.data.final_score[0], 65.5, "present number is untouched");
  assert.equal(panel.data.symbol[0], "AAA");
});

test("concatPanels: preserves order and combines all rows", () => {
  const p1 = { columns: ["symbol", "final_score"], data: { symbol: ["AAA"], final_score: [10] } };
  const p2 = { columns: ["symbol", "final_score"], data: { symbol: ["BBB", "CCC"], final_score: [20, 30] } };
  const combined = concatPanels([p1, p2]);
  assert.deepEqual(combined.data.symbol, ["AAA", "BBB", "CCC"]);
  assert.deepEqual(combined.data.final_score, [10, 20, 30]);
});

test("concatPanels: empty input returns empty panel, not a crash", () => {
  const combined = concatPanels([]);
  assert.deepEqual(combined.columns, []);
});
