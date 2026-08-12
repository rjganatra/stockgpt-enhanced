"""Turns a list of Trade objects into the numbers a person actually wants:
win rate, average/median return, best/worst, sample size, and a plain-
language confidence note when the sample is thin.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import pandas as pd

from ..config import BACKTEST_DEFAULTS
from .engine import Trade


@dataclass
class BacktestSummary:
    strategy_name: str
    horizon_label: str
    total_signals: int
    closed_trades: int
    open_trades: int
    win_rate_pct: float
    avg_return_pct: float
    median_return_pct: float
    best_return_pct: float
    worst_return_pct: float
    avg_holding_days: float
    low_sample_warning: bool

    def to_dict(self) -> dict:
        return asdict(self)


def summarize(trades: list[Trade], strategy_name: str,
              win_threshold_pct: float = 0.0) -> pd.DataFrame:
    """One summary row per horizon_label (e.g. one row per "7D"/"15D"/...
    for fixed-holding strategies, or a single "condition_exit" row)."""
    if not trades:
        return pd.DataFrame(columns=list(BacktestSummary.__annotations__.keys()))

    rows = [{
        "horizon_label": t.horizon_label,
        "return_pct": t.return_pct,
        "holding_days": t.holding_days,
        "is_open": t.is_open,
    } for t in trades]
    df = pd.DataFrame(rows)

    summaries = []
    for horizon, group in df.groupby("horizon_label"):
        closed = group[~group["is_open"]]
        open_count = int(group["is_open"].sum())
        total = len(group)

        if closed.empty:
            summaries.append(BacktestSummary(
                strategy_name, horizon, total, 0, open_count,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, True,
            ))
            continue

        wins = (closed["return_pct"] > win_threshold_pct).sum()
        summaries.append(BacktestSummary(
            strategy_name=strategy_name,
            horizon_label=horizon,
            total_signals=total,
            closed_trades=len(closed),
            open_trades=open_count,
            win_rate_pct=round(wins / len(closed) * 100, 2),
            avg_return_pct=round(closed["return_pct"].mean(), 2),
            median_return_pct=round(closed["return_pct"].median(), 2),
            best_return_pct=round(closed["return_pct"].max(), 2),
            worst_return_pct=round(closed["return_pct"].min(), 2),
            avg_holding_days=round(closed["holding_days"].mean(), 1),
            low_sample_warning=len(closed) < BACKTEST_DEFAULTS.min_signals_for_confidence,
        ))

    return pd.DataFrame([s.to_dict() for s in summaries])
