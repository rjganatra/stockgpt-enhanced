#!/usr/bin/env python3
"""Trims every data/history/YYYY-MM-DD/scan.csv down to the top-N symbols
by market cap, in place -- no network calls, no re-fetch, no recomputation
of scores or fundamentals. Purely a row filter on data that's already been
correctly backfilled.

Why this exists: Streamlit Community Cloud's free tier caps memory at
~1GB. A full 500-symbol x 3-year backfill measurably peaks around 1.4GB
when the dashboard loads it (see README's "Historical backfill" section),
which crashes the live deployed app the same way the earlier per-rerun
duplicate-copy bug did. Reducing to the top ~250 names by market cap keeps
the full 3-year *time* depth (what walk-forward validation and the
low_sample_warning fix actually need) while cutting row count, and
therefore memory, roughly in half.

This is destructive -- it overwrites data/history in place. If you want to
keep the full untrimmed dataset for local-only use, copy data/history
somewhere else first.

Usage:
    python scripts/trim_history_symbols.py --keep-top 250
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stockgpt import schema as S

DATA_DIR = Path("data")


def top_symbols_by_market_cap(keep_top: int) -> list[str]:
    """Ranks by data/fundamentals/fundamentals_scored.csv's market_cap_cr
    (today's real market cap -- the same field the dashboard's Fundamentals
    tab already surfaces). Symbols with no known market cap are dropped
    from consideration rather than guessed at; that's a small, honest gap
    consistent with how the rest of this pipeline treats missing data."""
    fund_path = DATA_DIR / "fundamentals" / "fundamentals_scored.csv"
    if not fund_path.exists():
        raise SystemExit(f"{fund_path} not found -- can't rank by market cap without it.")
    fund = pd.read_csv(fund_path)
    ranked = fund[[S.SYMBOL, "market_cap_cr"]].dropna().sort_values("market_cap_cr", ascending=False)
    return ranked[S.SYMBOL].head(keep_top).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--keep-top", type=int, default=250,
                         help="How many symbols to keep, ranked by market cap (default 250).")
    parser.add_argument("--history-dir", type=str, default=str(DATA_DIR / "history"),
                         help="Directory of YYYY-MM-DD/scan.csv folders to trim in place (default data/history).")
    parser.add_argument("--dry-run", action="store_true",
                         help="Report what would change without writing anything.")
    args = parser.parse_args()

    keep = set(top_symbols_by_market_cap(args.keep_top))
    print(f"Keeping top {len(keep)} symbols by market cap.")

    history_dir = Path(args.history_dir)
    days_touched = 0
    rows_before_total = 0
    rows_after_total = 0

    for folder in sorted(history_dir.iterdir()):
        if not folder.is_dir():
            continue
        scan_path = folder / "scan.csv"
        if not scan_path.exists():
            continue
        df = pd.read_csv(scan_path, low_memory=False)
        rows_before = len(df)
        trimmed = df[df[S.SYMBOL].isin(keep)]
        rows_after = len(trimmed)
        rows_before_total += rows_before
        rows_after_total += rows_after
        days_touched += 1
        if not args.dry_run:
            trimmed.to_csv(scan_path, index=False)

    print(f"{'Would trim' if args.dry_run else 'Trimmed'} {days_touched} day(s): "
          f"{rows_before_total} rows -> {rows_after_total} rows "
          f"({rows_before_total - rows_after_total} removed).")
    if args.dry_run:
        print("Dry run -- nothing was written. Re-run without --dry-run to apply.")


if __name__ == "__main__":
    main()
