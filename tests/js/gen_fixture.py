#!/usr/bin/env python3
"""Regenerates tests/js/fixtures/{panel,python_results}.json -- the fixture
tests/js/crossverify.node.js diffs the JS backtest engine's output against.

This panel is synthetic, not real market data, but deliberately dense with
the edge cases that are actually hard to get right in a from-scratch port:
NaN scores (real fetch gaps), a missing/null sector, a short mid-series gap
for one symbol (a real fetch failure looks exactly like this), both open
and closed trades, condition-exit with and without an explicit exit_query,
`in [...]`, `and`/`or`/`not`, `.str.contains(...)`, a walk-forward
train/test split, and a top-K portfolio day-collision. Fixed random seed,
so re-running this always produces byte-identical fixtures -- if you change
either engine, regenerate with this script and re-run crossverify.node.js;
if the two engines still agree, the fixture files won't even change.

Usage: python tests/js/gen_fixture.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from stockgpt import schema as S
from stockgpt.backtest import (
    ExitMode, Strategy, run_backtest, summarize,
    split_panel_by_date, walk_forward_sweep, run_topk_backtest,
)

random.seed(42)
np.random.seed(42)

SYMBOLS = ["AAA", "BBB", "CCC", "DDD", "EEE"]
N_DAYS = 60
dates = pd.date_range("2026-01-01", periods=N_DAYS, freq="B")

rows = []
for sym_i, sym in enumerate(SYMBOLS):
    price = 100.0 + sym_i * 20
    sector = ["Technology", "Healthcare", None, "Technology", "Energy"][sym_i]
    for day_i, d in enumerate(dates):
        drift = np.sin((day_i + sym_i * 7) / 6.0) * 5 + (day_i * 0.15)
        price = max(1.0, price + drift + np.random.normal(0, 1.0))

        final_score = 50 + 20 * np.sin((day_i + sym_i * 5) / 8.0) + np.random.normal(0, 3)
        risk_penalty = np.nan if (sym == "EEE" or random.random() < 0.1) else max(0, np.random.normal(10, 5))
        rsi = 30 + 40 * ((day_i + sym_i * 3) % 20) / 20.0
        score_band = ("High Conviction" if final_score >= 70 else "Strong" if final_score >= 60
                      else "Neutral" if final_score >= 45 else "Weak")

        rows.append({
            S.SYMBOL: sym,
            S.SCAN_DATE: d.strftime("%Y-%m-%d"),
            S.CURRENT_PRICE: round(price, 2),
            S.FINAL_SCORE: round(final_score, 2),
            S.SCORE_BAND: score_band,
            S.RISK_PENALTY: None if pd.isna(risk_penalty) else round(risk_penalty, 2),
            S.RSI: round(rsi, 2),
            S.SECTOR: sector,
            "distance_from_low_pct": round(abs(np.random.normal(20, 15)), 2),
        })

rows = [r for r in rows if not (r[S.SYMBOL] == "CCC" and 20 <= dates.get_loc(pd.Timestamp(r[S.SCAN_DATE])) <= 22)]

panel = pd.DataFrame(rows)
panel[S.SCAN_DATE] = pd.to_datetime(panel[S.SCAN_DATE])
panel = panel.sort_values([S.SYMBOL, S.SCAN_DATE]).reset_index(drop=True)

strategies = [
    Strategy(name="fixed_basic", entry_query="final_score >= 65",
             exit_mode=ExitMode.FIXED_HOLDING, fixed_holding_days=(3, 7, 15)),
    Strategy(name="fixed_and_or_not", entry_query="final_score >= 60 and not (sector == 'Energy')",
             exit_mode=ExitMode.FIXED_HOLDING, fixed_holding_days=(5,)),
    Strategy(name="fixed_in_list", entry_query="score_band in ['Strong', 'High Conviction']",
             exit_mode=ExitMode.FIXED_HOLDING, fixed_holding_days=(5, 10)),
    Strategy(name="fixed_risk_nan", entry_query="risk_penalty <= 12",
             exit_mode=ExitMode.FIXED_HOLDING, fixed_holding_days=(5,)),
    Strategy(name="condition_exit_basic", entry_query="final_score >= 65", exit_mode=ExitMode.CONDITION_EXIT),
    Strategy(name="condition_exit_explicit", entry_query="rsi <= 40",
             exit_mode=ExitMode.CONDITION_EXIT, exit_query="rsi >= 60"),
    Strategy(name="str_contains", entry_query="sector.str.contains('Tech')",
             exit_mode=ExitMode.FIXED_HOLDING, fixed_holding_days=(5,)),
]

results: dict = {}
errors: dict = {}
for strat in strategies:
    try:
        trades = run_backtest(panel, strat)
    except ValueError as e:
        errors[strat.name] = str(e)
        continue
    summary = summarize(trades, strat.name)
    results[strat.name] = json.loads(summary.to_json(orient="records"))

topk_strategy = Strategy(name="topk", entry_query="final_score >= 55",
                          exit_mode=ExitMode.FIXED_HOLDING, fixed_holding_days=(5, 10))
topk_trades = run_topk_backtest(panel, topk_strategy, top_k=2)
results["topk"] = json.loads(summarize(topk_trades, "topk").to_json(orient="records"))

wf_result = walk_forward_sweep(panel, "final_score >= {t}", [50.0, 55.0, 60.0, 65.0, 70.0], (5, 10), split_pct=65.0)
results["walk_forward"] = json.loads(wf_result.to_json(orient="records")) if not wf_result.empty else []

train_panel, test_panel = split_panel_by_date(panel, 65.0)
results["_split_check"] = {
    "train_days": int(train_panel[S.SCAN_DATE].nunique()),
    "test_days": int(test_panel[S.SCAN_DATE].nunique()),
}

out_dir = Path(__file__).resolve().parent / "fixtures"
out_dir.mkdir(exist_ok=True)

panel_out = panel.copy()
panel_out[S.SCAN_DATE] = panel_out[S.SCAN_DATE].dt.strftime("%Y-%m-%d")
columns = list(panel_out.columns)
data = {}
for col in columns:
    series = panel_out[col]
    if pd.api.types.is_float_dtype(series) or pd.api.types.is_integer_dtype(series):
        data[col] = [None if pd.isna(v) else float(v) for v in series]
    else:
        data[col] = [None if pd.isna(v) else str(v) for v in series]

(out_dir / "panel.json").write_text(json.dumps({"columns": columns, "data": data}))
(out_dir / "python_results.json").write_text(json.dumps({"results": results, "errors": errors}, indent=2))
print(f"Wrote {out_dir}/panel.json ({len(panel_out)} rows) and python_results.json "
      f"({len(results)} strategy results, {len(errors)} errors)")
