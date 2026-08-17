#!/usr/bin/env python3
"""Deletes data/history/YYYY-MM-DD/ folders older than a rolling retention
window -- the automatic counterpart to the manual backfill. Without this,
data/history grows by one folder every trading day forever, and the exact
memory problem this project hit once (a dataset too big for where the
dashboard lives) would eventually happen again on its own, silently, one
day at a time, with no single command anyone ran to blame it on.

Meant to run as a step in daily_pipeline.yml, right after the day's new
snapshot is written and right before it's committed -- so every commit
both adds the new day and removes whatever just aged out, and
data/history's total size stays roughly constant indefinitely instead of
growing without bound.

Usage:
    python scripts/prune_history.py --keep-years 3
    python scripts/prune_history.py --keep-years 3 --dry-run
"""

from __future__ import annotations

import argparse
import shutil
from datetime import date, timedelta
from pathlib import Path

DATA_DIR = Path("data")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--keep-years", type=float, default=3.0,
                         help="Rolling window to keep, in years (default 3 -- matches the backfill window).")
    parser.add_argument("--history-dir", type=str, default=str(DATA_DIR / "history"),
                         help="Directory of YYYY-MM-DD folders to prune (default data/history).")
    parser.add_argument("--dry-run", action="store_true",
                         help="Report what would be deleted without deleting anything.")
    args = parser.parse_args()

    cutoff = date.today() - timedelta(days=int(args.keep_years * 365.25))
    history_dir = Path(args.history_dir)

    removed = []
    kept = 0
    for folder in sorted(history_dir.iterdir()):
        if not folder.is_dir():
            continue
        try:
            folder_date = date.fromisoformat(folder.name)
        except ValueError:
            # Not a YYYY-MM-DD folder (e.g. some other subfolder) -- leave
            # it alone, this script only ever touches dated snapshots.
            continue
        if folder_date < cutoff:
            removed.append(folder)
        else:
            kept += 1

    for folder in removed:
        if args.dry_run:
            print(f"Would remove {folder} (older than {cutoff.isoformat()})")
        else:
            shutil.rmtree(folder)

    verb = "Would remove" if args.dry_run else "Removed"
    print(f"{verb} {len(removed)} day(s) older than {cutoff.isoformat()}; kept {kept} day(s).")
    if args.dry_run and removed:
        print("Dry run -- nothing was deleted. Re-run without --dry-run to apply.")


if __name__ == "__main__":
    main()
