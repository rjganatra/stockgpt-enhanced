#!/usr/bin/env python3
"""Runs after the daily pipeline: detects new High Conviction stocks, saved
Strategy Lab matches against today's scan, and sharp changes on watchlist
symbols, then sends a summary via Telegram and/or email -- whichever has
secrets configured. Both channels are optional and independent; running with
neither configured still computes and prints the alert, it just doesn't
deliver it anywhere (useful for testing via workflow_dispatch before wiring
up real secrets).

The watchlist is read directly from the repo checkout
(data/watchlist/watchlist.csv) rather than through the GitHub API -- this
step already runs inside a checkout of the same repo/branch the watchlist
lives in, so there's no need for a second network round-trip to read a file
that's already sitting on disk.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stockgpt import alerts
from stockgpt import schema as S

DATA_DIR = Path("data")


def main() -> None:
    changes_path = DATA_DIR / "history" / "latest_changes.csv"
    scan_path = DATA_DIR / "scans" / "latest_scan.csv"
    strategies_path = DATA_DIR / "backtest" / "saved_strategies.json"
    watchlist_path = DATA_DIR / "watchlist" / "watchlist.csv"

    changes_df = pd.read_csv(changes_path) if changes_path.exists() else pd.DataFrame()
    today_df = pd.read_csv(scan_path, low_memory=False) if scan_path.exists() else pd.DataFrame()
    saved_strategies = json.loads(strategies_path.read_text()) if strategies_path.exists() else []
    watchlist_df = pd.read_csv(watchlist_path) if watchlist_path.exists() else pd.DataFrame(columns=[S.SYMBOL])

    if today_df.empty:
        print("No scan data yet -- run the daily pipeline first. Skipping alerts.")
        return

    print(f"Checking alerts against {len(today_df)} scanned stocks, "
          f"{len(changes_df)} change rows, {len(saved_strategies)} saved strategies, "
          f"{len(watchlist_df)} watchlist entries...")

    bundle = alerts.build_alert_bundle(changes_df, today_df, saved_strategies, watchlist_df)
    if bundle.is_empty():
        print("Nothing to alert on today.")
        return

    message = alerts.format_message(bundle)
    print("\n" + message + "\n")

    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if bot_token or chat_id:
        ok, err = alerts.send_telegram(bot_token, chat_id, message)
        print(f"Telegram: {'sent' if ok else f'FAILED -- {err}'}")

    smtp_host = os.environ.get("SMTP_HOST", "")
    if smtp_host:
        ok, err = alerts.send_email(
            smtp_host=smtp_host,
            smtp_port=int(os.environ.get("SMTP_PORT", "587")),
            smtp_user=os.environ.get("SMTP_USER", ""),
            smtp_password=os.environ.get("SMTP_PASSWORD", ""),
            to_addr=os.environ.get("ALERT_EMAIL_TO", ""),
            subject="StockGPT Enhanced -- Alert Summary",
            body=message,
        )
        print(f"Email: {'sent' if ok else f'FAILED -- {err}'}")

    if not (bot_token or chat_id or smtp_host):
        print("No Telegram or email secrets configured -- alert computed above but not delivered. "
              "Set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID and/or SMTP_HOST + SMTP_USER + SMTP_PASSWORD "
              "+ ALERT_EMAIL_TO as repo secrets (Settings -> Secrets and variables -> Actions) to enable delivery.")


if __name__ == "__main__":
    main()
