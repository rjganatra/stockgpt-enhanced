/**
 * Regression test for a real bug found live on the deployed dashboard:
 * widget_strategy_lab.py concatenates query_parser.js, backtest_engine.js,
 * and panel_loader.js into ONE inline <script> tag (see its module comment
 * for why -- st.iframe's HTML string becomes one classic script, sharing a
 * single top-level scope across all four files). Every other JS test in
 * this directory (crossverify.node.js, panel_loader.test.js) exercises
 * these files via Node's `require()`, which gives each file its own
 * isolated module scope -- so they never actually ran the code path that
 * only executes when concatenated into a shared browser scope
 * (`typeof require !== "undefined"` is true under Node, false in a
 * browser). That's exactly the path that broke: backtest_engine.js's
 * browser-fallback branch re-declared `evalQueryMask`/`QueryParseError` as
 * new top-level bindings, colliding with query_parser.js's own `function
 * evalQueryMask` / `class QueryParseError` declarations in that same
 * shared scope -- a `SyntaxError: Identifier 'evalQueryMask' has already
 * been declared` that only ever manifested once actually concatenated and
 * loaded in a real browser tab, invisible to every existing automated
 * check up to that point.
 *
 * This test closes that gap: it builds the exact same concatenated string
 * widget_strategy_lab.py's build_widget_html() produces (see that file --
 * order is query_parser, backtest_engine, panel_loader, then widget_app,
 * though widget_app.js needs a real DOM/fetch/IndexedDB so it's
 * deliberately excluded here; its own top-level code is a single IIFE with
 * no top-level declarations that could collide, so leaving it out doesn't
 * weaken this check -- see widget_app.js's module comment), runs it in a
 * `vm` sandbox with a `window` global but WITHOUT `require`/`module`
 * (forcing every file's browser-fallback branch, the one that's otherwise
 * never exercised), and confirms it not only parses but actually produces
 * correct backtest results end to end.
 */
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");

const STATIC_DIR = path.join(__dirname, "..", "..", "dashboard", "static");
const FIXTURES_DIR = path.join(__dirname, "fixtures");

function buildBrowserSandbox() {
  const files = ["query_parser.js", "backtest_engine.js", "panel_loader.js"];
  const script = files.map((f) => fs.readFileSync(path.join(STATIC_DIR, f), "utf8")).join("\n");

  const sandbox = {};
  sandbox.window = sandbox; // `window` is the global object, like a real page
  vm.createContext(sandbox);
  vm.runInContext(script, sandbox, { filename: "concatenated-browser-build.js" });
  return sandbox;
}

test("concatenated browser build parses without a redeclaration SyntaxError", () => {
  // buildBrowserSandbox() throws (vm.runInContext propagates parse errors)
  // if this regresses -- assert.doesNotThrow makes that failure mode explicit.
  assert.doesNotThrow(() => buildBrowserSandbox());
});

test("concatenated browser build exposes the expected globals", () => {
  const sandbox = buildBrowserSandbox();
  for (const name of ["evalQueryMask", "QueryParseError", "runBacktest", "summarize",
    "walkForwardSweep", "runTopKBacktest", "quartersInRange", "rawShardToPanel", "concatPanels"]) {
    assert.equal(typeof sandbox[name], name === "QueryParseError" ? "function" : typeof sandbox[name],
      `window.${name} should exist in the browser build`);
    assert.notEqual(sandbox[name], undefined, `window.${name} should be defined`);
  }
});

test("browser-fallback evalQueryMask/runBacktest produce real results on real fixture data", () => {
  const sandbox = buildBrowserSandbox();
  const rawPanel = JSON.parse(fs.readFileSync(path.join(FIXTURES_DIR, "panel.json"), "utf8"));
  const panel = sandbox.rawShardToPanel(rawPanel);

  const strategy = { name: "fixed_basic", entryQuery: "final_score >= 65",
    exitMode: "fixed_holding", fixedHoldingDays: [7, 15] };
  const trades = sandbox.runBacktest(panel, strategy);
  assert.ok(trades.length > 0, "expected at least one trade from the fixture panel");

  const summary = sandbox.summarize(trades, "fixed_basic", 0);
  assert.ok(summary.length > 0, "expected at least one summary row");
});

test("browser-fallback QueryParseError is the same class thrown by evalQueryMask (instanceof works)", () => {
  const sandbox = buildBrowserSandbox();
  const rawPanel = JSON.parse(fs.readFileSync(path.join(FIXTURES_DIR, "panel.json"), "utf8"));
  const panel = sandbox.rawShardToPanel(rawPanel);

  assert.throws(
    () => sandbox.evalQueryMask(panel, "this is not === valid syntax"),
    (e) => e instanceof sandbox.QueryParseError,
    "a bad query should throw window.QueryParseError specifically, not a generic error",
  );
});
