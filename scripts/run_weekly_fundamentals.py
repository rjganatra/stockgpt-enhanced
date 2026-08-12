#!/usr/bin/env python3
"""Weekly fundamentals refresh: fetch -> score -> sector-adjust -> save.

Runs separately from the daily pipeline because a full .info crawl across
the universe is slow and Yahoo throttles it hard. The daily pipeline merges
whatever this script last produced, so fundamentals can legitimately be up
to a week stale relative to same-day price data -- that's a real, known
trade-off, not an oversight, and the dashboard should say so explicitly
(see dashboard/tabs/fundamentals.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stockgpt import fundamentals as F
from stockgpt import schema as S

DATA_DIR = Path("data")


def main() -> None:
    scan_path = DATA_DIR / "scans" / "latest_scan.csv"
    universe_path = DATA_DIR / "universe" / "universe.csv"

    if scan_path.exists():
        symbols = pd.read_csv(scan_path)[S.SYMBOL].dropna().astype(str).str.upper().unique().tolist()
    elif universe_path.exists():
        symbols = pd.read_csv(universe_path)[S.SYMBOL].dropna().astype(str).str.upper().unique().tolist()
    else:
        raise SystemExit("No scan or universe file found -- run run_daily_pipeline.py first.")

    print(f"Fetching fundamentals for {len(symbols)} symbols...")
    raw = F.fetch_fundamentals(symbols)
    if raw.empty:
        raise SystemExit("No fundamentals fetched -- aborting without overwriting the last good file.")

    print(f"Fetched {len(raw)}/{len(symbols)}. Scoring...")
    scored = F.score_fundamentals(raw)

    out_dir = DATA_DIR / "fundamentals"
    out_dir.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out_dir / "fundamentals_scored.csv", index=False)

    failed = sorted(set(symbols) - set(raw[S.SYMBOL]))
    pd.DataFrame({S.SYMBOL: failed}).to_csv(out_dir / "failed_symbols.csv", index=False)

    print(f"Saved fundamentals for {len(scored)} symbols. Failed: {len(failed)}")


if __name__ == "__main__":
    main()
