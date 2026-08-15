#!/usr/bin/env python3
"""Backfills years of historical daily snapshots (data/history/YYYY-MM-DD/
scan.csv) by re-running the exact same scoring engine the live daily
pipeline uses against price history and financial statements pulled from
Yahoo -- so the backtest engine, Leaderboard, walk-forward validation, and
portfolio backtest all get years of real history immediately instead of
waiting for it to accumulate one day at a time.

Run the pilot first:
    python scripts/backfill_history.py --pilot --years 3

Then, once you've checked the pilot output looks right (see README's
"Historical backfill" section for what to check), scale up:
    python scripts/backfill_history.py --years 3

This can take a long time for the full universe (~2000 symbols x 3+ years
of price history and statements) and Yahoo can throttle or block bulk
requests partway through -- it's safe to just re-run the same command if
that happens. Already-written days are skipped, not re-fetched or
recomputed (see --no-resume to force a clean re-run instead).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stockgpt import backfill, schema as S, universe as U

DATA_DIR = Path("data")

# The pilot batch: the 50 most liquid, most widely covered NSE names
# (Nifty 50 constituents as of this writing). Chosen deliberately, not
# arbitrarily -- these are the symbols most likely to have complete price
# history AND complete financial statements on Yahoo, which is exactly what
# a pilot needs to answer "does the real historical-fundamentals path
# actually work, or does everything fall back to the current-value
# approximation" before spending hours running the full ~2000-symbol
# universe against the same question.
PILOT_SYMBOLS = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "HINDUNILVR",
    "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK", "LT", "AXISBANK",
    "BAJFINANCE", "ASIANPAINT", "MARUTI", "SUNPHARMA", "TITAN",
    "ULTRACEMCO", "WIPRO", "NESTLEIND", "ADANIENT", "ADANIPORTS",
    "TATAMOTORS", "TATASTEEL", "POWERGRID", "NTPC", "HCLTECH",
    "TECHM", "M&M", "BAJAJFINSV", "ONGC", "COALINDIA", "INDUSINDBK",
    "GRASIM", "JSWSTEEL", "DRREDDY", "CIPLA", "EICHERMOT", "BRITANNIA",
    "APOLLOHOSP", "DIVISLAB", "HEROMOTOCO", "HINDALCO", "BPCL",
    "SBILIFE", "HDFCLIFE", "BAJAJ-AUTO", "UPL", "SHREECEM", "TATACONSUM",
]


def load_sector_maps() -> tuple[dict, dict, dict, dict]:
    """Reuses whatever universe + fundamentals data the live pipeline has
    already fetched -- no separate sector/fundamentals fetch needed here,
    only price history and financial statements are new."""
    uni_path = DATA_DIR / "universe" / "universe.csv"
    fund_path = DATA_DIR / "fundamentals" / "fundamentals_scored.csv"

    sector_map, industry_map, sector_bucket_map, current_fund = {}, {}, {}, {}

    if uni_path.exists():
        uni = pd.read_csv(uni_path)
        sector_map = dict(zip(uni[S.SYMBOL], uni[S.SECTOR]))

    if fund_path.exists():
        fund = pd.read_csv(fund_path)
        for _, row in fund.iterrows():
            symbol = row[S.SYMBOL]
            if "sector_yf" in row and isinstance(row["sector_yf"], str) and row["sector_yf"] != "Unknown":
                sector_map[symbol] = row["sector_yf"]
            if "industry_yf" in row and isinstance(row["industry_yf"], str):
                industry_map[symbol] = row["industry_yf"]
            if S.SECTOR_BUCKET in row:
                sector_bucket_map[symbol] = row[S.SECTOR_BUCKET]
            current_fund[symbol] = row.to_dict()
    else:
        print("Warning: no data/fundamentals/fundamentals_scored.csv found yet -- "
              "current-value fallback fundamentals will be unavailable for every "
              "symbol whose historical statements can't be derived. Run "
              "run_weekly_fundamentals.py first for the richest possible backfill.")

    return sector_map, industry_map, sector_bucket_map, current_fund


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pilot", action="store_true",
                         help=f"Backfill only the {len(PILOT_SYMBOLS)}-symbol Nifty 50 pilot batch "
                              "instead of the full universe. Start here.")
    parser.add_argument("--nifty500", action="store_true",
                         help="Backfill the real Nifty 500 constituent list (fetched live from "
                              "niftyindices.com), instead of --pilot or the full universe. Run "
                              "--pilot first to sanity-check the pipeline; this is the recommended "
                              "next scale-up step after that -- a meaningfully larger universe than "
                              "the pilot without the memory risk of the full ~2000-symbol universe.")
    parser.add_argument("--symbols", type=str, default=None,
                         help="Comma-separated symbol list to backfill instead of --pilot or the full universe.")
    parser.add_argument("--years", type=float, default=3.0, help="How many years back to backfill (default 3).")
    parser.add_argument("--output-dir", type=str, default=str(DATA_DIR / "history"),
                         help="Where to write YYYY-MM-DD/scan.csv folders (default data/history).")
    parser.add_argument("--no-resume", action="store_true",
                         help="Recompute and overwrite days that already have a scan.csv, instead of "
                              "skipping them (the default -- safe to re-run after a rate-limit interruption).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    elif args.pilot:
        symbols = PILOT_SYMBOLS
    elif args.nifty500:
        symbols = U.fetch_nifty500_symbols()
        if not symbols:
            print("Nifty 500 fetch failed (network issue or niftyindices.com format change) -- "
                  "falling back to the 50-symbol pilot list instead. Re-run --nifty500 later to retry.")
            symbols = PILOT_SYMBOLS
        else:
            print(f"Fetched {len(symbols)} real Nifty 500 symbols from niftyindices.com")
    else:
        uni_path = DATA_DIR / "universe" / "universe.csv"
        if not uni_path.exists():
            print("No data/universe/universe.csv found -- run run_daily_pipeline.py at least once first, "
                  "or pass --pilot / --symbols explicitly.")
            sys.exit(1)
        symbols = pd.read_csv(uni_path)[S.SYMBOL].tolist()

    print(f"Backfilling {len(symbols)} symbols, {args.years} years, into {args.output_dir}")
    sector_map, industry_map, sector_bucket_map, current_fund = load_sector_maps()

    summary = backfill.run_backfill(
        symbols=symbols,
        sector_map=sector_map,
        industry_map=industry_map,
        sector_bucket_map=sector_bucket_map,
        current_fundamentals_by_symbol=current_fund,
        years=args.years,
        output_dir=args.output_dir,
        skip_existing=not args.no_resume,
    )

    print()
    print("=== Backfill summary ===")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print()
    if summary.get("symbols_fetched", 0):
        derived = summary.get("derived_fundamentals", 0)
        fallback = summary.get("fallback_fundamentals", 0)
        total = derived + fallback
        if total:
            print(f"{derived}/{total} symbols got REAL historical fundamentals derived from filed "
                  f"financial statements. {fallback}/{total} fell back to today's current fundamentals "
                  "held constant across the backfill window (statements unavailable/unusable for those).")
    print("Done. Point the dashboard at this data/history folder (or copy it into your working repo) "
          "and the Leaderboard, walk-forward validation, and portfolio backtest tabs will pick it up "
          "automatically -- no other changes needed.")


if __name__ == "__main__":
    main()
