"""Assembles the client-side Strategy Lab / parameter sweep / walk-forward /
portfolio backtest widget: one self-contained HTML document embedded into
the dashboard via ``st.components.v1.html``.

Why this exists: these four tools all let a visitor type an arbitrary
pandas-query-style condition at read time. That can't be precomputed by the
daily GitHub Actions export (scripts/export_dashboard_data.py) the way the
other tabs (Sectors, Signal Performance, Leaderboard, ...) are, and running
it against the full history panel server-side is exactly the RAM pattern
that used to crash the free Streamlit Cloud tier. So the computation moved
into the visitor's own browser: dashboard/static/{query_parser,
backtest_engine}.js is a from-scratch port of
src/stockgpt/backtest/{engine,metrics,walkforward,portfolio}.py, cross-
verified against the real Python engine in CI (tests/js/crossverify.node.js,
see .github/workflows/tests.yml). dashboard/static/panel_loader.js turns the
quarterly JSON shards scripts/export_dashboard_data.py::export_strategy_lab_shards
writes into the panel shape the engine expects (unit-tested,
tests/js/panel_loader.test.js). dashboard/static/widget_app.js is the
DOM/fetch/IndexedDB glue that ties it all together in a real browser.

This module's only job is composing those four JS files plus a small CSS
block and a JSON config (``window.STOCKGPT_CONFIG``) into one HTML string.
"""

from __future__ import annotations

import json
from pathlib import Path

from stockgpt.backtest.strategy import PRESET_STRATEGIES, Strategy

DASHBOARD_DIR = Path(__file__).parent
STATIC_DIR = DASHBOARD_DIR / "static"
DATA_DIR = Path("data")
EXPORT_DIR = DATA_DIR / "exports"
STRATEGIES_PATH = DATA_DIR / "backtest" / "saved_strategies.json"

# This repo's own GitHub owner/name -- used to build the raw.githubusercontent.com
# base URL the browser fetches quarterly shards from. Confirmed via `git remote -v`
# against the actual deployed repo rather than assumed; not read from a runtime
# secret because anonymous visitors to the public dashboard have none, and this
# needs to work for them.
GITHUB_REPO = "rjganatra/stockgpt-enhanced"
GITHUB_BRANCH = "main"

# Order matters: widget_app.js's own module comment documents this exact
# dependency order (query_parser -> backtest_engine -> panel_loader -> widget_app).
_JS_FILES = ["query_parser.js", "backtest_engine.js", "panel_loader.js", "widget_app.js"]

_CSS = """
<style>
  /* Hardcoded to match this dashboard's actual deployed theme (dark --
     see .streamlit/config.toml / the live screenshots), not adaptive to
     the visitor's Streamlit theme setting. st.iframe's srcdoc iframe has
     no access to the parent page's CSS variables or theme, so "adapt to
     whatever theme the visitor has" would need a postMessage handshake
     this embed doesn't have -- out of scope. Explicit dark colors here
     instead of relying on default/inherited backgrounds is also the fix
     for the real bug this replaces: the previous version set dark text
     (#262730) with no background color declared, which is fine on a
     plain white page but unreadable/low-contrast once the iframe's actual
     background turned out not to be reliably white in the deployed app. */
  :root {
    --sl-bg: #0e1117;
    --sl-bg-secondary: #262730;
    --sl-border: #3b3f4a;
    --sl-text: #fafafa;
    --sl-text-dim: #b0b3bd;
    --sl-accent: #ff4b4b;
  }
  html, body { background: var(--sl-bg); }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         margin: 0; padding: 12px; color: var(--sl-text); }
  .sl-section { margin-bottom: 28px; }
  .sl-status-bar { background: var(--sl-bg-secondary); border: 1px solid var(--sl-border);
                    border-radius: 6px; padding: 10px 14px; margin-bottom: 16px;
                    font-size: 0.9em; display: flex; align-items: center;
                    justify-content: space-between; gap: 12px; color: var(--sl-text); }
  .sl-status-bar span { flex: 1; }
  h2 { font-size: 1.15em; margin: 0 0 6px 0; color: var(--sl-text); }
  h3 { font-size: 1.0em; margin: 18px 0 6px 0; color: var(--sl-text); }
  p.sl-caption { color: var(--sl-text-dim); font-size: 0.88em; margin: 4px 0 10px 0; }
  label { display: block; font-size: 0.85em; font-weight: 600; margin: 10px 0 4px 0;
          color: var(--sl-text); }
  input[type=text], input[type=number], textarea, select {
    width: 100%; box-sizing: border-box; padding: 6px 8px; font-size: 0.9em;
    border: 1px solid var(--sl-border); border-radius: 4px; font-family: inherit;
    background: var(--sl-bg-secondary); color: var(--sl-text);
  }
  input[readonly] { color: var(--sl-text-dim); }
  textarea { min-height: 56px; font-family: monospace; }
  select option { background: var(--sl-bg-secondary); color: var(--sl-text); }
  .sl-row { display: flex; gap: 12px; }
  .sl-row > div { flex: 1; }
  button { margin-top: 12px; padding: 8px 16px; font-size: 0.9em; font-weight: 600;
           border: none; border-radius: 4px; cursor: pointer; background: var(--sl-accent);
           color: #ffffff; }
  button:disabled { background: var(--sl-border); color: var(--sl-text-dim); cursor: not-allowed; }
  button.sl-secondary { background: var(--sl-bg-secondary); color: var(--sl-text);
                         border: 1px solid var(--sl-border); }
  table.sl-table { border-collapse: collapse; width: 100%; margin-top: 12px; font-size: 0.85em;
                    color: var(--sl-text); }
  table.sl-table th, table.sl-table td { border: 1px solid var(--sl-border); padding: 5px 8px;
                                          text-align: left; }
  table.sl-table th { background: var(--sl-bg-secondary); color: var(--sl-text); }
  table.sl-table td { background: var(--sl-bg); }
  p.sl-empty { color: var(--sl-text-dim); font-style: italic; }
  p.sl-error { color: #ff6b6b; }
  hr { border: none; border-top: 1px solid var(--sl-border); margin: 24px 0; }
</style>
"""


def _load_saved_strategies_raw() -> list[dict]:
    if not STRATEGIES_PATH.exists():
        return []
    try:
        return json.loads(STRATEGIES_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return []


def _preset_strategies_json() -> str:
    """Merged preset + saved strategies, snake_case dicts (Strategy.to_dict()
    schema) -- matches what widget_app.js's populatePresetDropdown() expects
    to read from CONFIG.presetStrategies. Saved strategies override presets
    on name collision, same precedence app.py's Strategy Lab tab used
    server-side (all_strategies dict built preset-first then .update()'d)."""
    merged: dict[str, dict] = {s.name: s.to_dict() for s in PRESET_STRATEGIES}
    for raw in _load_saved_strategies_raw():
        try:
            merged[raw["name"]] = Strategy.from_dict(raw).to_dict()
        except (KeyError, ValueError):
            continue
    return json.dumps(list(merged.values()))


def _meta() -> dict:
    path = EXPORT_DIR / "meta.json"
    if not path.exists():
        return {"date_from": None, "date_to": None, "days": 0, "symbols": 0}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {"date_from": None, "date_to": None, "days": 0, "symbols": 0}


def build_widget_html() -> str:
    """Returns the full HTML document to pass to
    ``st.components.v1.html(html, height=..., scrolling=True)``."""
    js_blobs = [(STATIC_DIR / name).read_text(encoding="utf-8") for name in _JS_FILES]
    meta = _meta()
    config = {
        "rawBase": f"https://raw.githubusercontent.com/{GITHUB_REPO}/{GITHUB_BRANCH}/data/exports",
        "meta": {
            "date_from": meta.get("date_from"),
            "date_to": meta.get("date_to"),
            "days": meta.get("days"),
            "symbols": meta.get("symbols"),
        },
        "presetStrategies": json.loads(_preset_strategies_json()),
    }

    body = f"""
<div class="sl-status-bar">
  <span id="sl-status">Initializing...</span>
  <button id="sl-load-full-btn" class="sl-secondary" type="button">Load full history</button>
</div>

<div class="sl-section">
  <h2>Strategy Lab</h2>
  <p class="sl-caption">
    Phrase a rule like <code>final_score &gt;= 65 and score_band == 'Strong'</code> and see what
    your actual historical win rate would have been. Runs entirely in this browser tab --
    nothing is sent to any server.
  </p>
  <label for="sl-preset">Start from a strategy</label>
  <select id="sl-preset"><option value="">-- New custom strategy --</option></select>
  <label for="sl-name">Strategy name</label>
  <input type="text" id="sl-name" value="My strategy" />
  <label for="sl-entry-query">Entry condition</label>
  <textarea id="sl-entry-query">final_score >= 65 and score_band == 'Strong'</textarea>
  <div class="sl-row">
    <div>
      <label for="sl-exit-mode">Exit style</label>
      <select id="sl-exit-mode">
        <option value="fixed_holding">fixed_holding</option>
        <option value="condition_exit">condition_exit</option>
      </select>
    </div>
    <div>
      <label for="sl-win-threshold">Win threshold (return % above)</label>
      <input type="number" id="sl-win-threshold" value="0" step="0.5" />
    </div>
  </div>
  <label for="sl-holding-days">Holding periods (days, comma separated)</label>
  <input type="text" id="sl-holding-days" value="7,15,30,60" />
  <label for="sl-exit-query">Exit condition (only used when exit style is condition_exit; leave blank to exit as soon as the entry condition stops matching)</label>
  <input type="text" id="sl-exit-query" value="" />
  <button id="sl-run-btn" class="sl-run-btn" type="button">Run backtest</button>
  <button id="sl-config-btn" class="sl-secondary" type="button">Get strategy as JSON to save</button>
  <p class="sl-caption">
    This widget can't write back to the server (no round trip channel through this embed), so
    "saving" a strategy means copying the JSON below and pasting it into
    <code>data/backtest/saved_strategies.json</code> (or adding it to <code>PRESET_STRATEGIES</code>
    in <code>src/stockgpt/backtest/strategy.py</code>) yourself.
  </p>
  <textarea id="sl-config-json" readonly style="display:none; min-height: 140px; font-family: monospace;"></textarea>
  <div id="sl-results"></div>
</div>

<hr />

<div class="sl-section">
  <h2>Parameter sweep</h2>
  <p class="sl-caption">
    Runs the same entry condition across a whole range of threshold values in one pass. Write
    your condition with <code>{{t}}</code> wherever the swept number goes, e.g.
    <code>final_score &gt;= {{t}} and score_band == "Strong"</code>. Fixed-holding exit only.
  </p>
  <label for="sweep-template">Entry condition template</label>
  <input type="text" id="sweep-template" value="final_score >= {{t}}" />
  <div class="sl-row">
    <div><label for="sweep-min">Sweep from</label><input type="number" id="sweep-min" value="50" step="5" /></div>
    <div><label for="sweep-max">Sweep to</label><input type="number" id="sweep-max" value="80" step="5" /></div>
    <div><label for="sweep-step">Step</label><input type="number" id="sweep-step" value="5" step="1" min="0.5" /></div>
  </div>
  <div class="sl-row">
    <div><label for="sweep-holding-days">Holding periods (days, comma separated)</label><input type="text" id="sweep-holding-days" value="15,30" /></div>
    <div><label for="sweep-win-threshold">Win threshold (return % above)</label><input type="number" id="sweep-win-threshold" value="0" step="0.5" /></div>
  </div>
  <button id="sweep-run-btn" class="sl-run-btn" type="button">Run parameter sweep</button>
  <div id="sweep-results"></div>
</div>

<hr />

<div class="sl-section">
  <h2>Walk-forward validation</h2>
  <p class="sl-caption">
    Splits history into an earlier training window and a later testing window it never saw
    during selection: picks the best threshold on training, then checks whether that SAME
    threshold still performs on testing. If training looked great but testing doesn't, that's
    the signature of curve-fitting rather than a genuine signal.
  </p>
  <label for="wf-template">Entry condition template</label>
  <input type="text" id="wf-template" value="final_score >= {{t}}" />
  <div class="sl-row">
    <div><label for="wf-min">Sweep from</label><input type="number" id="wf-min" value="50" step="5" /></div>
    <div><label for="wf-max">Sweep to</label><input type="number" id="wf-max" value="80" step="5" /></div>
    <div><label for="wf-step">Step</label><input type="number" id="wf-step" value="5" step="1" min="0.5" /></div>
  </div>
  <div class="sl-row">
    <div><label for="wf-holding-days">Holding periods (days, comma separated)</label><input type="text" id="wf-holding-days" value="15,30" /></div>
    <div><label for="wf-split-pct">Training window size (% of days)</label><input type="number" id="wf-split-pct" value="70" min="50" max="90" step="5" /></div>
  </div>
  <button id="wf-run-btn" class="sl-run-btn" type="button">Run walk-forward validation</button>
  <div id="wf-results"></div>
</div>

<hr />

<div class="sl-section">
  <h2>Portfolio backtest (top-K signal selection)</h2>
  <p class="sl-caption">
    Every matching signal is treated as an independent trade by default -- on a day where 40
    stocks all cross your threshold at once, no real portfolio buys all 40. This simulates
    picking only your top-K highest-final_score signals per day instead. Scope, honestly: this
    is top-K SELECTION, not a full capital-tracked equity curve -- no position sizing, capital
    limits, or compounding.
  </p>
  <label for="pf-entry-query">Entry condition</label>
  <textarea id="pf-entry-query">final_score >= 65 and score_band == 'Strong'</textarea>
  <div class="sl-row">
    <div><label for="pf-holding-days">Holding periods (days, comma separated)</label><input type="text" id="pf-holding-days" value="15,30" /></div>
    <div><label for="pf-top-k">Top K picks per day</label><input type="number" id="pf-top-k" value="5" min="1" step="1" /></div>
    <div><label for="pf-win-threshold">Win threshold (return % above)</label><input type="number" id="pf-win-threshold" value="0" step="0.5" /></div>
  </div>
  <button id="pf-run-btn" class="sl-run-btn" type="button">Run portfolio backtest</button>
  <div id="pf-results"></div>
</div>
"""

    script = "\n".join(js_blobs)
    config_json = json.dumps(config)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
{_CSS}
</head>
<body>
{body}
<script>
window.STOCKGPT_CONFIG = {config_json};
</script>
<script>
{script}
</script>
</body>
</html>
"""
