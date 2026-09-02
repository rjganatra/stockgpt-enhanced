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
  strategy_lab/<YYYY>-Q<n>.json -- quarterly-sharded columnar panel for the
                                 BROWSER-SIDE JS backtest engine (Strategy
                                 Lab, sweep, walk-forward, portfolio
                                 backtest). Same server-side archive as
                                 everything else above, just reshaped and
                                 trimmed for the one consumer that has to
                                 download it instead of read it locally:
                                 the free-text *_reasons columns are
                                 dropped (a filter/backtest query never
                                 references them -- see the size
                                 measurement in this project's own history
                                 for why that alone roughly halves the
                                 download) and rows are split one file per
                                 calendar quarter so a visitor's browser
                                 only ever fetches the quarters it actually
                                 needs (last 4 by default, all of them only
                                 if they opt into the "load 3 years"
                                 button), and so each file stays well under
                                 GitHub's 100MB per-file limit regardless
                                 of how large the underlying universe gets.
                                 Old quarters that have aged out of
                                 data/history's retention window have their
                                 shard file deleted here too, so this
                                 directory never accumulates stale files
                                 with no supporting data behind them.
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

# Free-text explanation columns -- measured (on a real production day) at
# 51% of that day's file size, and never something a filter/backtest query
# meaningfully evaluates against (`.str.contains(...)` on a *_reasons field
# is technically parseable but not a real use case the app's own examples
# or column-reference caption ever demonstrate). Dropped from the
# browser-shipped Strategy Lab panel only -- the daily views still show
# full reason text from data/scans/latest_scan.csv, untouched.
STRATEGY_LAB_DROP_COLUMNS = [
    S.TECHNICAL_REASONS, "relative_strength_reasons", S.RANGE_REASONS,
    S.FUNDAMENTAL_REASONS, S.FUNDAMENTAL_RISK_REASONS,
    "sector_adjustment_reasons", S.RISK_REASONS, S.REASONS,
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


def _quarter_key(timestamp: pd.Timestamp) -> str:
    return f"{timestamp.year}-Q{(timestamp.month - 1) // 3 + 1}"


def export_strategy_lab_shards(panel: pd.DataFrame) -> dict[str, int]:
    """Returns {quarter: file_size_bytes} so export_meta() can publish real
    sizes -- the browser widget used to guess ~8MB/quarter (measured back
    when history was capped at 250 symbols); now that data/history holds
    close to the full ~2,300-symbol universe, real shard sizes are running
    60-70MB+ each, and a stale guess would badly understate the download
    the widget's own "Load Full History" confirm dialog shows the user."""
    out_dir = EXPORT_DIR / "strategy_lab"
    out_dir.mkdir(parents=True, exist_ok=True)

    if panel.empty:
        print("Strategy Lab shards: empty panel, nothing written.")
        return {}

    lab_panel = panel.drop(columns=[c for c in STRATEGY_LAB_DROP_COLUMNS if c in panel.columns])
    quarter_series = lab_panel[S.SCAN_DATE].apply(_quarter_key)
    present_quarters = set(quarter_series.unique())

    columns = list(lab_panel.columns)
    written = 0
    for quarter in sorted(present_quarters):
        qdf = lab_panel[quarter_series == quarter]
        data = {}
        for col in columns:
            series = qdf[col]
            if col == S.SCAN_DATE:
                data[col] = series.dt.strftime("%Y-%m-%d").tolist()
            elif pd.api.types.is_float_dtype(series) or pd.api.types.is_integer_dtype(series):
                # NaN isn't valid JSON -- becomes `null`, which is exactly
                # the missing-value convention query_parser.js's module
                # comment documents for numeric columns... except that
                # convention actually calls for NaN, not null, for
                # numerics (see that file's comment on why: NaN comparisons
                # already behave like pandas/numpy for free in JS, `null`
                # comparisons don't). The browser-side loader is
                # responsible for converting a `null` it reads back from
                # this JSON into `NaN` before handing the panel to the
                # backtest engine -- JSON simply has no way to encode NaN
                # directly (json.dumps(float('nan')) produces invalid JSON
                # if allow_nan=False, and the non-standard `NaN` literal
                # allow_nan=True produces isn't valid JSON either, so
                # `null` is the only genuinely portable choice here).
                data[col] = [None if pd.isna(v) else float(v) for v in series]
            else:
                data[col] = [None if pd.isna(v) else str(v) for v in series]
        path = out_dir / f"{quarter}.json"
        path.write_text(json.dumps({"columns": columns, "data": data}, separators=(",", ":")))
        written += 1

    # Delete shard files for quarters that no longer have any backing data
    # (aged out of data/history's retention window) -- otherwise a stale
    # file just sits there forever, silently wrong the moment someone
    # fetches it expecting it to reflect the current retention window.
    removed = 0
    for existing in out_dir.glob("*.json"):
        if existing.stem not in present_quarters:
            existing.unlink()
            removed += 1

    shard_sizes = {q: (out_dir / f"{q}.json").stat().st_size for q in present_quarters}
    total_bytes = sum(shard_sizes.values())
    print(f"Strategy Lab shards: {written} quarter(s) written ({total_bytes / 1e6:.1f} MB total), "
          f"{removed} stale quarter(s) removed -> {out_dir}")
    return shard_sizes


def export_meta(panel: pd.DataFrame, shard_sizes: dict[str, int] | None = None) -> None:
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
    # Real per-quarter Strategy Lab shard sizes (MB, rounded) -- lets the
    # browser widget show/estimate accurate download sizes instead of a
    # hardcoded guess that goes stale as the universe size changes. See
    # export_strategy_lab_shards()'s docstring.
    meta["quarter_sizes_mb"] = {q: round(b / 1e6, 1) for q, b in sorted((shard_sizes or {}).items())}
    (EXPORT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Meta: {meta}")


def main() -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading server-side history panel...")
    panel = load_history_panel(DATA_DIR / "history")
    print(f"Panel: {len(panel)} rows" if not panel.empty else "Panel is empty.")

    shard_sizes = export_strategy_lab_shards(panel)
    export_meta(panel, shard_sizes)
    export_sector_rotation(panel)
    export_score_history(panel)
    export_signal_performance(panel)
    export_leaderboard(panel)
    print("Export complete.")


if __name__ == "__main__":
    main()
