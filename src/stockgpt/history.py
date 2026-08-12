"""Daily snapshot storage + day-over-day change detection.

Snapshots are what the backtest engine's `load_history_panel` consumes, so
the on-disk layout is deliberately simple and stable:
    data/history/YYYY-MM-DD/scan.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import schema as S


def save_snapshot(df: pd.DataFrame, history_root: str | Path, scan_date: pd.Timestamp) -> Path:
    folder = Path(history_root) / scan_date.strftime("%Y-%m-%d")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "scan.csv"
    df.to_csv(path, index=False)
    return path


def previous_snapshot_path(history_root: str | Path, before: pd.Timestamp) -> Path | None:
    history_root = Path(history_root)
    if not history_root.exists():
        return None
    candidates = []
    for folder in history_root.iterdir():
        if not folder.is_dir():
            continue
        try:
            d = pd.Timestamp(folder.name)
        except ValueError:
            continue
        if d < before and (folder / "scan.csv").exists():
            candidates.append(d)
    if not candidates:
        return None
    latest = max(candidates)
    return history_root / latest.strftime("%Y-%m-%d") / "scan.csv"


def compute_changes(current: pd.DataFrame, previous: pd.DataFrame) -> pd.DataFrame:
    """Day-over-day score/RSI/risk/band movement per symbol."""
    keep_current = [S.SYMBOL, S.SECTOR, S.INDUSTRY, S.CURRENT_PRICE, S.RSI,
                     S.RISK_PENALTY, S.FINAL_SCORE, S.SCORE_BAND]
    keep_previous = [S.SYMBOL, S.RSI, S.RISK_PENALTY, S.FINAL_SCORE, S.SCORE_BAND]

    cur = current[[c for c in keep_current if c in current.columns]].copy()
    prev = previous[[c for c in keep_previous if c in previous.columns]].copy()
    prev = prev.rename(columns={
        S.RSI: "previous_rsi", S.RISK_PENALTY: "previous_risk_penalty",
        S.FINAL_SCORE: "previous_final_score", S.SCORE_BAND: "previous_score_band",
    })
    cur = cur.rename(columns={
        S.RSI: "current_rsi", S.RISK_PENALTY: "current_risk_penalty",
        S.FINAL_SCORE: "current_final_score", S.SCORE_BAND: "current_score_band",
    })

    merged = cur.merge(prev, on=S.SYMBOL, how="left")
    merged["score_change"] = (merged["current_final_score"] - merged["previous_final_score"]).round(2)
    merged["rsi_change"] = (merged["current_rsi"] - merged["previous_rsi"]).round(2)
    merged["risk_change"] = (merged["current_risk_penalty"] - merged["previous_risk_penalty"]).round(2)

    def signal(row) -> str:
        signals = []
        if pd.isna(row["previous_final_score"]):
            return "New stock"
        if row["score_change"] >= 10:
            signals.append("Score improved sharply")
        elif row["score_change"] <= -10:
            signals.append("Score dropped sharply")
        if row["previous_score_band"] != row["current_score_band"]:
            signals.append(f"Band changed: {row['previous_score_band']} -> {row['current_score_band']}")
        if row["risk_change"] >= 8:
            signals.append("Risk increased")
        return ", ".join(signals) if signals else "No major change"

    merged["change_signal"] = merged.apply(signal, axis=1)
    return merged.sort_values("score_change", ascending=False)
