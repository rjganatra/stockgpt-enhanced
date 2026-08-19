#!/usr/bin/env python3
"""Precomputes everything the dashboard's history-panel-dependent tabs need,
once, offline (run by the daily GitHub Actions pipeline) -- so the live
Streamlit process never has to load the full multi-year history panel into
its own RAM just to answer a page view.

This is the fix for the app's recurring memory problem, at its actual root
cause. `dashboard/app.py` used to call `load_history_panel_cached()`
UNCONDITIONALLY at the top of the script, every single rerun, for every
visitor -- Streamlit re-executes the whole script top-to-bottom on every
widget interaction, so that one line kept the entire history panel
(hundreds of MB, growing daily) resident in the server's shared memory
essentially all the time. Four tabs actually needed it for something that
doesn't change from one visitor's click to the next -- sector rotation,
per-symbol score history, signal catalog performance, and the strategy
leaderboard -- so instead of computing those live per request, this script
computes them once here and writes small, cheap-to-read files. The two
remaining history-panel consumers, Strategy Lab and Portfolio Backtest, are
NOT covered here because their whole point is a query typed by a visitor at
read time -- no precomputation can anticipate that. Those move to running
client-side in the visitor's own browser (see the JS backtest port), not to
this script.

Outputs, all under data/exports/:
  meta.json                  -- generated_at, day/symbol counts, date range
  sector_rotation.csv        -- sector_rotation.compute_sector_rotation() output
  signal_performance.csv     -- every SIGNAL_CATALOG entry backtested
  leaderboard.csv            -- every preset + saved Strategy backtested
  score_history/<SYMBOL>.csv -- one small file per symbol: scan_date + the
                                 score columns Stock Explorer's chart plots

Reads from the SERVER-SIDE history archive (data/history), independent of
whatever gets shipped to browsers for client-side backtesting -- those are
sized and shipped separately (see the Strategy Lab data export).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stockgpt import schema as S
from stockgpt import sector_rotation
from stockgpt.backtest import Strategy, run_backtest, summarize
from stockgpt.backtest.engine import load_history_panel
from stockgpt.backtest.strategy import PRESET_STRATEGIES
from stockgpt.signal_catalog import SIGNAL_CATALOG, SIGNAL_DIRECTIONS

DATA_DIR = Path("data")
EXPORT_DIR = DATA_DIR / "exports"
STRATEGIES_PATH = DATA_DIR / "backtest" / "saved_strategies.json"

# Same columns Stock Explorer's "Score History" chart plots today -- see
# dashboard/app.py's score_cols list. Keeping this file per-symbol and this
# narrow (not the full ~90-column row) is what makes it cheap to read on
# demand for just the one symbol a visitor selected, instead of loading
# every symbol's full row history to get one stock's chart.
SCORE_HISTORY_COLUMNS = [
    S.FINAL_SCORE, S.TECHNICAL_SCORE, S.SECTOR_ADJUSTED_FUNDAMENTAL_SCORE,
    S.RELATIVE_STRENGTH_SCORE, S.RISK_PENALTY,
]


def load_saved_strategies() -> list[dict]:
    if not STRATEGIES_PATH.exists():
        return []
    try:
        return json.loads(STRATEGIES_PATH.read_text())
    except Exception:
        return []


def export_sector_rotation(panel: pd.DataFrame) -> None:
    rotation_df = sector_rotation.compute_sector_rotation(panel)
    path = EXPORT_DIR / "sector_rotation.csv"
    if rotation_df.empty:
        # Still write an (empty-but-headered) file so the dashboard can
        # distinguish "ran, found nothing yet" from "never ran" -- an
        # absent file and an empty-with-headers file mean different things
        # to a reader, and only one of them is actually true here.
        pd.DataFrame(columns=[S.SECTOR, "recent_avg_final_score", "prior_avg_final_score",
                               "change", "change_pct"]).to_csv(path, index=False)
        print("Sector rotation: not enough history yet, wrote empty file.")
        return
    rotation_df.to_csv(path, index=False)
    print(f"Sector rotation: {len(rotation_df)} sectors -> {path}")


def export_score_history(panel: pd.DataFrame) -> int:
    out_dir = EXPORT_DIR / "score_history"
    out_dir.mkdir(parents=True, exist_ok=True)
    if panel.empty or S.SYMBOL not in panel.columns:
        print("Score history: empty panel, nothing written.")
        return 0

    cols = [c for c in SCORE_HISTORY_COLUMNS if c in panel.columns]
    if not cols:
        print("Score history: none of the expected score columns present, nothing written.")
        return 0

    written = 0
    # Reuse the already-loaded, already-sorted panel rather than re-reading
    # per symbol -- one groupby pass, not N re-scans of the same data.
    for symbol, sdf in panel.groupby(S.SYMBOL, sort=False):
        sym_hist = sdf[[S.SCAN_DATE] + cols].sort_values(S.SCAN_DATE)
        if sym_hist.empty:
            continue
        # Symbols can contain characters that aren't filesystem-safe on
        # some platforms in theory; NSE symbols in practice are plain
        # uppercase alphanumerics (+ the occasional '&', '-'), which are
        # safe everywhere this repo actually runs (Linux CI, Windows local).
        sym_hist.to_csv(out_dir / f"{symbol}.csv", index=False)
        written += 1
    print(f"Score history: {written} per-symbol files -> {out_dir}")
    return written


def export_signal_performance(panel: pd.DataFrame) -> None:
    path = EXPORT_DIR / "signal_performance.csv"
    if panel.empty:
        pd.DataFrame().to_csv(path, index=False)
        print("Signal performance: empty panel, wrote empty file.")
        return

    rows = []
    errors = []
    for strategy in SIGNAL_CATALOG:
        try:
            trades = run_backtest(panel, strategy)
        except ValueError as e:
            errors.append(f"{strategy.name}: {e}")
            continue
        if not trades:
            continue
        summary = summarize(trades, strategy.name)
        summary["direction"] = SIGNAL_DIRECTIONS.get(strategy.name, "Bullish")
        rows.append(summary)

    if errors:
        for e in errors:
            print(f"Signal performance error: {e}")

    if not rows:
        pd.DataFrame().to_csv(path, index=False)
        print("Signal performance: no signal produced any trades, wrote empty file.")
        return

    result = pd.concat(rows, ignore_index=True)
    result.to_csv(path, index=False)
    print(f"Signal performance: {len(result)} rows ({result['strategy_name'].nunique()} signals) -> {path}")


def export_leaderboard(panel: pd.DataFrame) -> None:
    path = EXPORT_DIR / "leaderboard.csv"
    if panel.empty:
        pd.DataFrame().to_csv(path, index=False)
        print("Leaderboard: empty panel, wrote empty file.")
        return

    saved = load_saved_strategies()
    # Saved strategies override presets of the same name -- identical merge
    # logic to the dashboard's "Start from a strategy" dropdown, so the
    # leaderboard and that dropdown never disagree about which version of a
    # same-named strategy is current.
    strategies = {s.name: s for s in PRESET_STRATEGIES}
    strategies.update({s["name"]: Strategy.from_dict(s) for s in saved})

    rows = []
    errors = []
    for name, strat in strategies.items():
        try:
            trades = run_backtest(panel, strat)
        except ValueError as e:
            errors.append(f"{name}: {e}")
            continue
        if not trades:
            continue
        rows.append(summarize(trades, name))

    if errors:
        for e in errors:
            print(f"Leaderboard error: {e}")

    if not rows:
        pd.DataFrame().to_csv(path, index=False)
        print("Leaderboard: no strategy produced any trades, wrote empty file.")
        return

    board = pd.concat(rows, ignore_index=True)
    board = board.sort_values(["low_sample_warning", "avg_return_pct"], ascending=[True, False])
    board.to_csv(path, index=False)
    print(f"Leaderboard: {len(board)} rows ({board['strategy_name'].nunique()} strategies) -> {path}")


def export_meta(panel: pd.DataFrame) -> None:
    if panel.empty:
        meta = {"generated_at": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(),
                "days": 0, "symbols": 0, "date_from": None, "date_to": None}
    else:
        dates = sorted(panel[S.SCAN_DATE].unique())
        meta = {
            "generated_at": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(),
            "days": len(dates),
            "symbols": int(panel[S.SYMBOL].nunique()) if S.SYMBOL in panel.columns else 0,
            "date_from": str(pd.Timestamp(dates[0]).date()),
            "date_to": str(pd.Timestamp(dates[-1]).date()),
        }
    (EXPORT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Meta: {meta}")


def main() -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading server-side history panel...")
    panel = load_history_panel(DATA_DIR / "history")
    print(f"Panel: {len(panel)} rows" if not panel.empty else "Panel is empty.")

    export_meta(panel)
    export_sector_rotation(panel)
    export_score_history(panel)
    export_signal_performance(panel)
    export_leaderboard(panel)
    print("Export complete.")


if __name__ == "__main__":
    main()
