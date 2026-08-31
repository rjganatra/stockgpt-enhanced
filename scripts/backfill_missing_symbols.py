#!/usr/bin/env python3
"""Backfills the symbols currently MISSING from data/history (everything
outside the top-250-by-market-cap set trim_history_symbols.py kept) without
touching what's already there.

Why this isn't just `backfill_history.py --symbols <missing> --output-dir
data/history`: backfill_history.py's `history.save_snapshot()` calls
`df.to_csv(path, index=False)` -- a full overwrite, not a merge. Every
day's data/history/YYYY-MM-DD/scan.csv already exists (holding the current
top-250 set), so pointing the ordinary backfill CLI straight at data/history
would either (a) skip every single day and write nothing at all
(skip_existing=True, the default -- exists() is already true for every
day), or (b) with --no-resume, silently REPLACE each day's file with ONLY
the newly-fetched missing symbols, destroying the 250 that were already
there. Neither is what "add more symbols" means. This script fetches the
missing symbols into a separate scratch directory first, then explicitly
merges each day's new rows into the real file alongside what's already
there (concat + drop_duplicates on symbol, existing rows always win on any
overlap -- there shouldn't be any, since the missing-symbol list is
computed as universe minus what's already present, but this is a
deliberate no-data-loss guard, not decoration).

Usage:
    # Step 1 (safe, writes only to --scratch-dir, never touches data/history):
    python scripts/backfill_missing_symbols.py --fetch --years 3

    # Step 2 (after eyeballing the scratch dir looks sane -- merges into
    # data/history for real):
    python scripts/backfill_missing_symbols.py --merge
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stockgpt import backfill, schema as S

DATA_DIR = Path("data")
DEFAULT_SCRATCH = DATA_DIR / "_backfill_scratch"


def load_sector_maps():
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
    return sector_map, industry_map, sector_bucket_map, current_fund


def currently_kept_symbols(keep_top: int = 250) -> set[str]:
    """The exact same ranking trim_history_symbols.py uses -- what's
    already present in every data/history/*/scan.csv day today."""
    fund_path = DATA_DIR / "fundamentals" / "fundamentals_scored.csv"
    fund = pd.read_csv(fund_path)
    ranked = fund[[S.SYMBOL, "market_cap_cr"]].dropna().sort_values("market_cap_cr", ascending=False)
    return set(ranked[S.SYMBOL].head(keep_top).tolist())


def missing_symbols(keep_top: int = 250) -> list[str]:
    uni = pd.read_csv(DATA_DIR / "universe" / "universe.csv")
    all_symbols = set(uni[S.SYMBOL].astype(str).str.strip())
    kept = currently_kept_symbols(keep_top)
    return sorted(all_symbols - kept)


def do_fetch(years: float, scratch_dir: Path, keep_top: int, limit: int | None = None, offset: int = 0) -> None:
    missing = missing_symbols(keep_top)
    print(f"{len(missing)} symbols currently missing from data/history (outside top {keep_top} by market cap).")
    if offset:
        missing = missing[offset:]
        print(f"--offset {offset}: skipping the first {offset} (for chunked/resumed runs).")
    if limit:
        missing = missing[:limit]
        print(f"--limit {limit}: only fetching the next {len(missing)} (chunk/pilot run).")
    sector_map, industry_map, sector_bucket_map, current_fund = load_sector_maps()
    summary = backfill.run_backfill(
        symbols=missing, sector_map=sector_map, industry_map=industry_map,
        sector_bucket_map=sector_bucket_map, current_fundamentals_by_symbol=current_fund,
        years=years, output_dir=scratch_dir, skip_existing=True, progress_every=20,
    )
    print("=== Fetch (to scratch dir) summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"Wrote to {scratch_dir} -- data/history itself is untouched. "
          f"Run with --merge once this output looks right.")


def do_merge(scratch_dir: Path) -> None:
    if not scratch_dir.exists():
        print(f"{scratch_dir} doesn't exist -- run --fetch first.")
        sys.exit(1)
    real_dir = DATA_DIR / "history"
    day_folders = sorted(p for p in scratch_dir.iterdir() if p.is_dir())
    # The most recent day in the scratch dir is "today" (or the latest
    # session as of whenever --fetch ran) -- yfinance's price history
    # always runs right up to the current day. That's exactly the one
    # folder the daily pipeline (.github/workflows/daily_pipeline.yml)
    # also writes and commits on every run, since it's the day still "in
    # motion". Merging into it here raced with a concurrent daily-pipeline
    # commit and produced an unresolvable git conflict on
    # data/history/<today>/scan.csv the first time this ran. Skipping the
    # single latest day sidesteps that entirely -- it costs nothing, since
    # that day's data for these symbols will be picked up the next time
    # this script runs (or once trim_history_symbols.py's --keep-top
    # policy is revisited), and every other (already-closed) day here is
    # untouched by the daily pipeline and safe to merge.
    if day_folders:
        skipped_latest = day_folders[-1]
        day_folders = day_folders[:-1]
        print(f"Skipping {skipped_latest.name} (latest/in-progress day -- owned by the daily pipeline, "
              f"not merged here to avoid a commit race).")
    merged = skipped_no_new = created = 0
    for folder in day_folders:
        new_file = folder / "scan.csv"
        if not new_file.exists():
            continue
        new_df = pd.read_csv(new_file, low_memory=False)
        if new_df.empty:
            skipped_no_new += 1
            continue
        real_file = real_dir / folder.name / "scan.csv"
        if real_file.exists():
            existing_df = pd.read_csv(real_file, low_memory=False)
            combined = pd.concat([existing_df, new_df], ignore_index=True)
            # Existing rows win on any overlap (shouldn't be any -- the
            # missing-symbol list was computed as universe minus what's
            # already kept -- but this is the actual safety guarantee,
            # not the symbol-list computation).
            combined = combined.drop_duplicates(subset=[S.SYMBOL], keep="first")
            combined.to_csv(real_file, index=False)
            merged += 1
        else:
            real_file.parent.mkdir(parents=True, exist_ok=True)
            new_df.to_csv(real_file, index=False)
            created += 1
    print(f"Merged {merged} existing day(s), created {created} new day(s), "
          f"{skipped_no_new} scratch day(s) were empty and skipped.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fetch", action="store_true", help="Fetch missing symbols into --scratch-dir.")
    parser.add_argument("--merge", action="store_true", help="Merge --scratch-dir into data/history.")
    parser.add_argument("--years", type=float, default=3.0)
    parser.add_argument("--keep-top", type=int, default=250,
                         help="Must match whatever trim_history_symbols.py was last run with.")
    parser.add_argument("--scratch-dir", type=str, default=str(DEFAULT_SCRATCH))
    parser.add_argument("--limit", type=int, default=None,
                         help="Only fetch this many missing symbols -- for a quick pilot run or one chunk.")
    parser.add_argument("--offset", type=int, default=0,
                         help="Skip this many missing symbols before applying --limit -- lets a large run "
                              "be split into sequential chunks (e.g. --offset 0 --limit 500, then "
                              "--offset 500 --limit 500, ...). Each chunk should use a distinct "
                              "--scratch-dir so chunks don't overwrite each other before merging.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.fetch:
        do_fetch(args.years, Path(args.scratch_dir), args.keep_top, args.limit, args.offset)
    elif args.merge:
        do_merge(Path(args.scratch_dir))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
