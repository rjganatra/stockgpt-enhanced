#!/usr/bin/env python3
"""Daily pipeline: universe -> technical scan -> relative strength ->
range-bound -> merge with last known fundamentals -> final score ->
snapshot to history -> change tracking.

Fundamentals are NOT re-fetched here (see run_weekly_fundamentals.py) --
same split as the original repo, because a full fundamentals crawl is slow
and Yahoo throttles it. This script merges whatever fundamentals were most
recently fetched, and that merge always carries the coverage/known flags
forward so a stale-but-present fundamental is never confused with a
freshly-verified one on the dashboard.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stockgpt import history, range_bound, relative_strength, scanner, scoring, universe
from stockgpt import schema as S

DATA_DIR = Path("data")


def _chunk(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def download_history(symbols: list[str], period: str = "1y") -> dict[str, pd.DataFrame]:
    """Batch-download then retry failures individually, same resilience
    pattern as the original repo (Yahoo batch responses are flaky)."""
    out: dict[str, pd.DataFrame] = {}
    failed: list[str] = []

    for chunk in _chunk(symbols, 100):
        yf_symbols = [f"{s}.NS" for s in chunk]
        try:
            data = yf.download(yf_symbols, period=period, interval="1d",
                                group_by="ticker", threads=True, progress=False, timeout=25)
        except Exception:
            failed.extend(chunk)
            continue

        for symbol in chunk:
            yf_symbol = f"{symbol}.NS"
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    if yf_symbol not in data.columns.get_level_values(0):
                        failed.append(symbol)
                        continue
                    hist = data[yf_symbol].dropna(how="all")
                else:
                    hist = data.dropna(how="all")
                if hist.empty:
                    failed.append(symbol)
                    continue
                out[symbol] = hist
            except Exception:
                failed.append(symbol)
        time.sleep(1)

    for symbol in failed:
        try:
            hist = yf.Ticker(f"{symbol}.NS").history(period=period)
            if not hist.empty:
                out[symbol] = hist
        except Exception:
            pass
        time.sleep(0.2)

    return out


def main() -> None:
    print("Refreshing universe...")
    uni, used_fallback = universe.fetch_universe()
    (DATA_DIR / "universe").mkdir(parents=True, exist_ok=True)
    uni.to_csv(DATA_DIR / "universe" / "universe.csv", index=False)
    print(f"Universe: {len(uni)} symbols (fallback={used_fallback})")

    symbols = uni[S.SYMBOL].tolist()
    sector_map = dict(zip(uni[S.SYMBOL], uni[S.SECTOR]))

    print(f"Downloading price history for {len(symbols)} symbols...")
    price_history = download_history(symbols)
    print(f"Got history for {len(price_history)}/{len(symbols)} symbols")

    print("Running technical scanner...")
    scan_rows = []
    for symbol, hist in price_history.items():
        row = scanner.process_history(symbol, hist, sector_map.get(symbol, "Unknown"), "Unknown")
        if row:
            scan_rows.append(row)
    scan_df = pd.DataFrame(scan_rows)
    print(f"Technical scan: {len(scan_df)} stocks scored")

    print("Computing relative strength...")
    nifty_1m = nifty_3m = nifty_6m = None
    try:
        nifty_hist = yf.download("^NSEI", period="8mo", interval="1d", progress=False, timeout=20)
        nifty_close = nifty_hist["Close"]
        # Some yfinance versions return MultiIndex columns (ticker sub-level)
        # even for a single-symbol download -- nifty_hist["Close"] is then a
        # one-column DataFrame, not a Series, and calc_return()'s scalar
        # comparisons (`if past == 0`) raise "truth value of a Series is
        # ambiguous" on that shape. Same MultiIndex check already used in
        # download_history() above; applied here too for the one place that
        # was missing it.
        if isinstance(nifty_close, pd.DataFrame):
            nifty_close = nifty_close.iloc[:, 0]
        nifty_close = nifty_close.dropna()
        nifty_1m = relative_strength.calc_return(nifty_close, 21)
        nifty_3m = relative_strength.calc_return(nifty_close, 63)
        nifty_6m = relative_strength.calc_return(nifty_close, 126)
    except Exception as e:  # noqa: BLE001
        print(f"Nifty benchmark unavailable, continuing without it: {e}")

    rs_rows = []
    for symbol, hist in price_history.items():
        if symbol not in scan_df[S.SYMBOL].values:
            continue
        rs_rows.append(relative_strength.compute_for_symbol(
            symbol, hist["Close"], sector_map.get(symbol, "Unknown"), nifty_1m, nifty_3m, nifty_6m))
    rs_df = relative_strength.add_sector_rank(pd.DataFrame(rs_rows))

    print("Running range-bound scanner...")
    range_rows = []
    scan_lookup = scan_df.set_index(S.SYMBOL).to_dict(orient="index")
    for symbol, hist in price_history.items():
        base = scan_lookup.get(symbol, {})
        row = range_bound.analyse(symbol, hist, base.get(S.TECHNICAL_SCORE, 0), 0)
        if row:
            range_rows.append(row)
    range_df = pd.DataFrame(range_rows)

    print("Merging with latest fundamentals...")
    fundamentals_path = DATA_DIR / "fundamentals" / "fundamentals_scored.csv"
    if fundamentals_path.exists():
        fund_df = pd.read_csv(fundamentals_path)
    else:
        print("No fundamentals file yet -- run run_weekly_fundamentals.py first. Continuing with 0 coverage.")
        fund_df = pd.DataFrame(columns=[S.SYMBOL, S.SECTOR_ADJUSTED_FUNDAMENTAL_SCORE])

    merged = scan_df.merge(rs_df.drop(columns=[S.SECTOR], errors="ignore"), on=S.SYMBOL, how="left")
    merged = merged.merge(range_df, on=S.SYMBOL, how="left")
    merged = merged.merge(fund_df, on=S.SYMBOL, how="left", suffixes=("", "_fund"))

    if S.SECTOR_ADJUSTED_FUNDAMENTAL_SCORE not in merged.columns:
        merged[S.SECTOR_ADJUSTED_FUNDAMENTAL_SCORE] = 0.0
    merged[S.RELATIVE_STRENGTH_SCORE] = merged[S.RELATIVE_STRENGTH_SCORE].fillna(0.0)

    print("Computing final scores...")
    final_df = scoring.compute_final_scores(merged)
    (DATA_DIR / "scans").mkdir(parents=True, exist_ok=True)
    final_df.to_csv(DATA_DIR / "scans" / "latest_scan.csv", index=False)
    print(f"Final scan saved: {len(final_df)} rows")

    print("Saving history snapshot + change tracking...")
    today = pd.Timestamp.now(tz="Asia/Kolkata").normalize().tz_localize(None)
    history.save_snapshot(final_df, DATA_DIR / "history", today)

    prev_path = history.previous_snapshot_path(DATA_DIR / "history", today)
    if prev_path is not None:
        prev_df = pd.read_csv(prev_path)
        changes = history.compute_changes(final_df, prev_df)
        changes.to_csv(DATA_DIR / "history" / "latest_changes.csv", index=False)
        print(f"Change tracking: {len(changes)} rows compared against {prev_path}")
    else:
        print("No previous snapshot found -- skipping change tracking for today.")

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
