"""End-to-end verification against REAL historical data.

This is not a synthetic smoke test. It rebuilds the new scoring pipeline's
output for every real trading day found in the original StockGPT repo's
`data/history/*/latest_scan.csv` snapshots -- 66 real NSE trading days,
captured live by the user's own bot -- merges in the real fundamentals
snapshot from `data/fundamentals/fundamentals.csv`, runs the NEW
scoring.compute_final_scores() on every single day, and then runs the NEW
backtest engine against the exact strategy the user described by hand:

    "if I saw conviction score above 65 and it came under the Strong band,
     what would my win rate have been?"

What the real data actually looked like (found while writing this, not
assumed going in): day 1 (2026-05-14) has only 30 rows -- a manual watchlist
-- and every day after jumps to 1500-2200+ rows, i.e. the bot switched to
scanning the full NSE universe starting day 2. We keep symbols present on
>=60 of the 66 days (see restrict_to_stable_universe) so relative-strength
history is apples-to-apples, rather than mixing a stock with 2 days of
history against one with 66.

Two more real limitations, reported rather than faked:
  - range_bound needs 80 days of price history (WINDOWS.range_min_history_days)
    and this fixture only has 66 real days, so range_status/range_score stay
    empty here. That's the module correctly declining to guess, not a bug.
  - There's no real Nifty index series in this fixture, so relative-strength
    is scored on absolute 1M/3M/6M return only, not vs-Nifty outperformance
    (those columns stay None, per relative_strength.py's contract).

Usage:
    python scripts/verify_against_real_data.py /path/to/StockGPT-main
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stockgpt import schema as S
from stockgpt.backtest import ExitMode, Strategy, run_backtest, summarize
from stockgpt.fundamentals import score_fundamentals, sector_bucket
from stockgpt.relative_strength import relative_strength_score
from stockgpt.scanner import technical_score
from stockgpt.scoring import compute_final_scores

OUT_HISTORY = Path("examples/sample_history")
OUT_LATEST = Path("examples/sample_scan_latest.csv")


def build_price_panel(source_history_dir: Path) -> pd.DataFrame:
    """Stack every real daily snapshot into one long-format frame of the raw
    old-schema columns, sorted by symbol then date -- this is the raw
    material, not yet run through the new scorer."""
    frames = []
    for folder in sorted(source_history_dir.iterdir()):
        f = folder / "latest_scan.csv"
        if not f.exists():
            continue
        try:
            date = pd.Timestamp(folder.name)
        except ValueError:
            continue
        day = pd.read_csv(f)
        day[S.SCAN_DATE] = date
        frames.append(day)
    panel = pd.concat(frames, ignore_index=True)
    panel[S.SYMBOL] = panel[S.SYMBOL].astype(str).str.upper().str.strip()
    return panel.sort_values([S.SYMBOL, S.SCAN_DATE]).reset_index(drop=True)


def load_real_fundamentals(source_fund_csv: Path) -> pd.DataFrame:
    """NOTE (found during a later audit, kept here rather than silently
    fixed): `keep` below is a narrow column subset built for this script's
    one job -- verifying the backtest engine against real historical scans.
    It deliberately excludes S.ROE and most other raw ratios, and does not
    run them through fundamentals.py's `_safe_percent()` scale conversion
    (this passes `fund` straight into score_fundamentals(), unlike the real
    production path in fetch_one()). That's fine for this script's own
    purpose, but it means a `data/fundamentals/fundamentals_scored.csv`
    produced by THIS function is missing ROE entirely -- which is exactly
    what shipped as the example dataset and is why the dashboard's ROE
    slider showed no real data until a real scripts/run_weekly_fundamentals.py
    run replaces it. If you extend `keep` to include S.ROE, verify the
    source CSV's scale first (percentage vs fraction) -- score_row()'s ROE
    thresholds assume percentage scale, same convention fetch_one() enforces
    via _safe_percent()."""
    fund = pd.read_csv(source_fund_csv)
    fund[S.SYMBOL] = fund[S.SYMBOL].astype(str).str.upper().str.strip()
    fund = fund.drop_duplicates(subset=S.SYMBOL, keep="last")
    fund[S.SECTOR_BUCKET] = fund.apply(
        lambda r: sector_bucket(str(r.get("sector_yf", "")), str(r.get("industry_yf", ""))), axis=1
    )
    scored = score_fundamentals(fund)
    keep = [S.SYMBOL, S.SECTOR_BUCKET, S.FUNDAMENTAL_SCORE, S.SECTOR_ADJUSTMENT,
            S.SECTOR_ADJUSTED_FUNDAMENTAL_SCORE, S.DEBT_TO_EQUITY, S.NET_PROFIT_MARGIN,
            S.REVENUE_GROWTH, S.EARNINGS_GROWTH, S.OPERATING_CASHFLOW_CR, S.FREE_CASHFLOW_CR,
            S.FUNDAMENTAL_DATA_COVERAGE]
    return scored[keep]


def restrict_to_stable_universe(price_panel: pd.DataFrame, min_days_present: int = 60) -> pd.DataFrame:
    """The real snapshots span two eras of the old bot: a ~30-stock manual
    watchlist for the first day, then a jump to a full ~2000+ stock NSE
    universe scan from day 2 onward (visible directly in the row counts per
    date). Mixing those eras would make relative-strength history
    inconsistent (a stock with 2 days of history vs one with 66). We keep
    symbols present on at least `min_days_present` of the real trading days
    -- real data, just excluding the first-day ramp-up artifact -- rather
    than quietly padding missing days with fabricated prices."""
    n_dates = price_panel[S.SCAN_DATE].nunique()
    counts = price_panel.groupby(S.SYMBOL)[S.SCAN_DATE].nunique()
    keep = counts[counts >= min_days_present].index
    kept = price_panel[price_panel[S.SYMBOL].isin(keep)].copy()
    print(f"Stable universe: {len(keep)} symbols with >= {min_days_present}/{n_dates} real days "
          f"(dropped {price_panel[S.SYMBOL].nunique() - len(keep)} symbols only seen in the "
          f"first-day ramp-up or that were added/dropped from the universe mid-window)")
    return kept


def score_every_real_day(price_panel: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Recompute the NEW technical_score + relative_strength_score for every
    (symbol, day) using the real historical indicator values already present
    in the raw snapshots (rsi/sma50/sma200/volume_ratio/distance figures are
    themselves real numbers the old bot computed from real yfinance OHLC --
    we're feeding them through the NEW formula, not inventing new numbers),
    then runs the NEW compute_final_scores() per day.

    Important: the later snapshot files (once the old bot's pipeline grew
    fundamentals/relative-strength/range/sector-score stages) already carry
    THEIR OWN computed columns under the same names our new schema uses
    (sector_score, risk_penalty, final_conviction_score, ...). We select
    down to only the raw price/indicator columns before recomputing
    anything, so our new formulas run clean instead of silently colliding
    with -- or being shadowed by -- the old bot's own already-computed
    output columns."""
    RAW_COLS = ["symbol", "sector", "industry", "current_price", "day_change_pct",
                "distance_pct", "distance_from_high_pct", "rsi", "sma50", "sma200",
                "volume_ratio", "trend", S.SCAN_DATE]
    raw = price_panel[[c for c in RAW_COLS if c in price_panel.columns]].copy()
    if "industry" not in raw.columns:
        raw["industry"] = "Unknown"

    panel = raw.rename(columns={
        "sector": S.SECTOR, "current_price": S.CURRENT_PRICE,
        "day_change_pct": S.DAY_CHANGE_PCT, "distance_pct": S.DISTANCE_FROM_LOW_PCT,
        "distance_from_high_pct": S.DISTANCE_FROM_HIGH_PCT, "rsi": S.RSI,
        "sma50": S.SMA50, "sma200": S.SMA200, "volume_ratio": S.VOLUME_RATIO,
        "trend": S.TREND,
    }).sort_values([S.SYMBOL, S.SCAN_DATE]).reset_index(drop=True)

    # --- technical score: vectorised apply, one pass over the whole panel ---
    def _tech_row(r):
        score, reasons = technical_score(
            rsi=float(r[S.RSI]), above_sma50=bool(r[S.CURRENT_PRICE] > r[S.SMA50]),
            above_sma200=bool(r[S.CURRENT_PRICE] > r[S.SMA200]) if pd.notna(r[S.SMA200]) else None,
            distance_from_low_pct=float(r[S.DISTANCE_FROM_LOW_PCT]),
            distance_from_high_pct=float(r[S.DISTANCE_FROM_HIGH_PCT]),
            volume_ratio=float(r[S.VOLUME_RATIO]), day_change_pct=float(r[S.DAY_CHANGE_PCT]),
        )
        return pd.Series({S.TECHNICAL_SCORE: score, S.TECHNICAL_REASONS: ", ".join(reasons)})

    panel[[S.TECHNICAL_SCORE, S.TECHNICAL_REASONS]] = panel.apply(_tech_row, axis=1)

    # --- relative strength: vectorised per-symbol shift, exactly matching
    # relative_strength.calc_return's "close.iloc[-days] vs close.iloc[-1]"
    # semantics but computed for every day at once instead of a python loop ---
    g = panel.groupby(S.SYMBOL)[S.CURRENT_PRICE]
    for label, days in (("_r1m", 21), ("_r3m", 63), ("_r6m", 126)):
        past = g.shift(days)
        panel[label] = ((panel[S.CURRENT_PRICE] - past) / past.replace(0, np.nan) * 100).round(2)

    def _none_if_nan(v):
        return None if pd.isna(v) else float(v)

    def _rel_row(r):
        score, _ = relative_strength_score(
            _none_if_nan(r["_r1m"]), _none_if_nan(r["_r3m"]), _none_if_nan(r["_r6m"]),
            None, None, None,
        )
        return score

    panel[S.RELATIVE_STRENGTH_SCORE] = panel.apply(_rel_row, axis=1)
    panel[S.RETURN_1M], panel[S.RETURN_3M], panel[S.RETURN_6M] = panel["_r1m"], panel["_r3m"], panel["_r6m"]
    panel = panel.drop(columns=["_r1m", "_r3m", "_r6m"])

    panel = panel.merge(fundamentals, on=S.SYMBOL, how="left")

    scored_days = []
    for date, day in panel.groupby(S.SCAN_DATE):
        final = compute_final_scores(day.copy())
        final[S.SCAN_DATE] = date
        scored_days.append(final)

    return pd.concat(scored_days, ignore_index=True)


def main():
    source_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("../stockgpt_extract/StockGPT-main")
    price_panel = build_price_panel(source_root / "data" / "history")
    fundamentals = load_real_fundamentals(source_root / "data" / "fundamentals" / "fundamentals.csv")

    print(f"Real trading days: {price_panel[S.SCAN_DATE].nunique()}")
    print(f"Real symbols seen (any day): {price_panel[S.SYMBOL].nunique()}")

    stable = restrict_to_stable_universe(price_panel)
    scored = score_every_real_day(stable, fundamentals)

    OUT_HISTORY.mkdir(parents=True, exist_ok=True)
    for date, day in scored.groupby(S.SCAN_DATE):
        folder = OUT_HISTORY / date.strftime("%Y-%m-%d")
        folder.mkdir(exist_ok=True)
        day.drop(columns=[S.SCAN_DATE]).to_csv(folder / "scan.csv", index=False)

    latest_date = scored[S.SCAN_DATE].max()
    latest = scored[scored[S.SCAN_DATE] == latest_date].copy()
    latest[S.SCAN_TIME] = str(latest_date.date())
    latest.to_csv(OUT_LATEST, index=False)

    print(f"\nScore band distribution across all {len(scored)} (symbol, day) rows, NEW scorer:")
    print(scored[S.SCORE_BAND].value_counts().reindex(S.BAND_ORDER, fill_value=0))

    print(f"\nRisk penalty > 0 on {int((scored[S.RISK_PENALTY] > 0).sum())} of {len(scored)} rows "
          f"(mean penalty where >0: {scored.loc[scored[S.RISK_PENALTY] > 0, S.RISK_PENALTY].mean():.1f})")

    print(f"\nrelative_strength_score mean: {scored[S.RELATIVE_STRENGTH_SCORE].mean():.1f} "
          f"({(scored[S.RELATIVE_STRENGTH_SCORE] == 0).mean()*100:.0f}% of rows score exactly 0). "
          "Two real, honest reasons, not a bug: this fixture has no Nifty benchmark series, so the "
          "40 vs-Nifty points in relative_strength_score() are structurally unreachable here, and "
          "the 6M window (126 trading days) never populates because the fixture is only 66 days "
          "long -- both stay None per relative_strength.py's contract instead of being faked as 0.")

    # --- the exact scenario the user described by hand ---
    strategy_final_score = Strategy(
        name="Strong conviction, 15D hold",
        entry_query="final_score >= 65 and score_band in ['Strong', 'High Conviction']",
        exit_mode=ExitMode.FIXED_HOLDING, fixed_holding_days=(5, 15, 30),
    )
    trades = run_backtest(scored, strategy_final_score)
    summary = summarize(trades, strategy_final_score.name)
    print(f"\n=== Backtest 1 (user's literal example): '{strategy_final_score.entry_query}' ===")
    print(f"Total trade-legs generated: {len(trades)}")
    if not summary.empty:
        print(summary.to_string(index=False))
    else:
        print("Zero signals in this fixture -- consistent with the relative-strength gap above: "
              "final_score can't clear 65 when 25% of its weight is structurally starved of vs-Nifty "
              "and 6M data in this particular 66-day window. This is the backtester correctly reporting "
              "'no historical evidence' rather than inventing a number -- see summarize()'s empty-frame path.")

    # --- second strategy, tuned to what this specific short/no-benchmark
    # fixture can actually produce signals for, to prove the SAME engine
    # returns real, non-trivial win-rate numbers once there ARE signals ---
    strategy_technical = Strategy(
        name="Strong technical setup, multi-horizon",
        entry_query="technical_score >= 65",
        exit_mode=ExitMode.FIXED_HOLDING, fixed_holding_days=(5, 15, 30),
    )
    trades2 = run_backtest(scored, strategy_technical)
    summary2 = summarize(trades2, strategy_technical.name)
    print(f"\n=== Backtest 2 (proves the engine on real data): '{strategy_technical.entry_query}' ===")
    print(f"Total trade-legs generated: {len(trades2)}")
    print(summary2.to_string(index=False))


if __name__ == "__main__":
    main()
