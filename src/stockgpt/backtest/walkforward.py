"""Walk-forward validation for the parameter sweep.

The existing sweep (dashboard's Strategy Lab tab) picks the best-performing
threshold across the ENTIRE history at once. That answers "what threshold
worked over this whole period" -- which risks just finding what happened to
work in hindsight, not what's genuinely predictive going forward. This
module answers the more honest question: split history into an earlier
TRAIN window and a later TEST window the selection process never saw, pick
the best threshold using only the train window, then check whether that
SAME fixed threshold still performs on the test window. If it does, the
edge is more likely real. If train looked great and test doesn't, that's
the signature of curve-fitting to one specific stretch of history.
"""

from __future__ import annotations

import pandas as pd

from .. import schema as S
from ..config import BACKTEST_DEFAULTS
from .engine import run_backtest
from .metrics import summarize
from .strategy import ExitMode, Strategy


def split_panel_by_date(panel: pd.DataFrame, split_pct: float = 70.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits by distinct scan_date (not by row), so train and test never
    share a day -- the earliest split_pct% of days go to train, the rest to
    test. Returns (train_panel, test_panel), both possibly empty if there
    aren't at least 2 distinct dates to split."""
    if panel.empty or S.SCAN_DATE not in panel.columns:
        return panel.iloc[0:0], panel.iloc[0:0]

    dates = sorted(panel[S.SCAN_DATE].unique())
    if len(dates) < 2:
        return panel.iloc[0:0], panel.iloc[0:0]

    split_idx = round(len(dates) * split_pct / 100)
    split_idx = max(1, min(len(dates) - 1, split_idx))
    train_dates = set(dates[:split_idx])
    test_dates = set(dates[split_idx:])
    return panel[panel[S.SCAN_DATE].isin(train_dates)], panel[panel[S.SCAN_DATE].isin(test_dates)]


def walk_forward_sweep(panel: pd.DataFrame, template: str, thresholds: list[float],
                        holding_days: tuple[int, ...], split_pct: float = 70.0,
                        win_threshold_pct: float = 0.0) -> pd.DataFrame:
    """One row per holding-period horizon: the threshold that performed best
    on the TRAIN window (by avg_return_pct, preferring thresholds that meet
    BACKTEST_DEFAULTS.min_signals_for_confidence if any do), plus that same
    fixed threshold's actual performance on the TEST window it was never
    selected against. Columns: horizon_label, best_threshold,
    train_avg_return_pct, train_closed_trades, test_avg_return_pct,
    test_closed_trades, test_win_rate_pct, test_low_sample_warning.

    Returns an empty DataFrame if there isn't enough history to split, or if
    no threshold produced any signal on the train window."""
    train_panel, test_panel = split_panel_by_date(panel, split_pct)
    if train_panel.empty or test_panel.empty:
        return pd.DataFrame()

    train_rows = []
    for t in thresholds:
        try:
            query = template.format(t=t)
        except (KeyError, IndexError):
            continue
        strategy = Strategy(name=f"t={t}", entry_query=query,
                             exit_mode=ExitMode.FIXED_HOLDING, fixed_holding_days=holding_days)
        try:
            trades = run_backtest(train_panel, strategy)
        except ValueError:
            continue
        if not trades:
            continue
        summary = summarize(trades, f"t={t}", win_threshold_pct)
        summary["threshold"] = t
        train_rows.append(summary)

    if not train_rows:
        return pd.DataFrame()

    train_df = pd.concat(train_rows, ignore_index=True)

    results = []
    for horizon, group in train_df.groupby("horizon_label"):
        confident = group[group["closed_trades"] >= BACKTEST_DEFAULTS.min_signals_for_confidence]
        candidates = confident if not confident.empty else group
        best = candidates.sort_values("avg_return_pct", ascending=False).iloc[0]
        best_threshold = best["threshold"]

        try:
            test_query = template.format(t=best_threshold)
        except (KeyError, IndexError):
            continue
        test_strategy = Strategy(name=f"t={best_threshold}", entry_query=test_query,
                                  exit_mode=ExitMode.FIXED_HOLDING, fixed_holding_days=holding_days)
        try:
            test_trades = run_backtest(test_panel, test_strategy)
        except ValueError:
            test_trades = []

        test_row = pd.DataFrame()
        if test_trades:
            test_summary = summarize(test_trades, f"t={best_threshold}", win_threshold_pct)
            test_row = test_summary[test_summary["horizon_label"] == horizon]

        results.append({
            "horizon_label": horizon,
            "best_threshold": best_threshold,
            "train_avg_return_pct": float(best["avg_return_pct"]),
            "train_closed_trades": int(best["closed_trades"]),
            "test_avg_return_pct": float(test_row["avg_return_pct"].iloc[0]) if not test_row.empty else None,
            "test_closed_trades": int(test_row["closed_trades"].iloc[0]) if not test_row.empty else 0,
            "test_win_rate_pct": float(test_row["win_rate_pct"].iloc[0]) if not test_row.empty else None,
            "test_low_sample_warning": bool(test_row["low_sample_warning"].iloc[0]) if not test_row.empty else True,
        })

    return pd.DataFrame(results)
