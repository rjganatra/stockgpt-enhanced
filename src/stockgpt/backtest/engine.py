"""Runs a Strategy against a panel of daily scan snapshots.

The panel is a single long-format DataFrame: every historical
`data/history/YYYY-MM-DD/scan.csv` stacked together with a `scan_date`
column added, one row per (symbol, date). `load_history_panel` builds this
from a directory of snapshots; `run_backtest` operates on the panel itself
so it's trivially unit-testable with a small synthetic panel (see
tests/test_backtest_engine.py) without touching disk or the network.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from .. import schema as S
from .strategy import ExitMode, Strategy


@dataclass
class Trade:
    symbol: str
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: Optional[pd.Timestamp]
    exit_price: Optional[float]
    holding_days: Optional[int]
    return_pct: Optional[float]
    is_open: bool               # True if we ran out of future data before an exit
    horizon_label: str          # e.g. "15D" or "condition_exit"


def load_history_panel(history_dir: str | Path, filename: str = "scan.csv") -> pd.DataFrame:
    """Stack every dated snapshot folder under history_dir into one panel."""
    history_dir = Path(history_dir)
    frames = []
    for folder in sorted(history_dir.iterdir()):
        if not folder.is_dir():
            continue
        file_path = folder / filename
        if not file_path.exists():
            continue
        try:
            snap_date = pd.Timestamp(folder.name)
        except ValueError:
            continue
        df = pd.read_csv(file_path, low_memory=False)
        df[S.SCAN_DATE] = snap_date
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames, ignore_index=True)
    panel[S.SYMBOL] = panel[S.SYMBOL].astype(str).str.upper().str.strip()
    return panel.sort_values([S.SYMBOL, S.SCAN_DATE]).reset_index(drop=True)


def _entry_dates(symbol_df: pd.DataFrame, mask: pd.Series) -> list[int]:
    """Row-positions (within symbol_df, already date-sorted) where `mask`
    transitions from False (or absent) to True -- a rising edge, so a stock
    that stays in-condition for 40 days counts as ONE entry, not 40."""
    mask = mask.fillna(False).to_numpy()
    prev = False
    positions = []
    for i, val in enumerate(mask):
        if val and not prev:
            positions.append(i)
        prev = val
    return positions


def _run_fixed_holding(symbol: str, sdf: pd.DataFrame, entry_positions: list[int],
                        holding_days: tuple[int, ...], strategy: Strategy) -> list[Trade]:
    trades: list[Trade] = []
    n = len(sdf)
    for pos in entry_positions:
        entry_date = sdf[S.SCAN_DATE].iloc[pos]
        entry_price = float(sdf[S.CURRENT_PRICE].iloc[pos])
        if entry_price <= 0:
            continue
        for days in holding_days:
            exit_pos = pos + days
            label = f"{days}D"
            if exit_pos < n:
                exit_row = sdf.iloc[exit_pos]
                exit_price = float(exit_row[S.CURRENT_PRICE])
                if exit_price <= 0:
                    continue
                ret = round(((exit_price - entry_price) / entry_price) * 100, 2)
                trades.append(Trade(symbol, entry_date, entry_price, exit_row[S.SCAN_DATE],
                                     exit_price, days, ret, False, label))
            else:
                trades.append(Trade(symbol, entry_date, entry_price, None, None,
                                     None, None, True, label))
    return trades


def _run_condition_exit(symbol: str, sdf: pd.DataFrame, entry_positions: list[int],
                         entry_mask: pd.Series, exit_mask: Optional[pd.Series]) -> list[Trade]:
    trades: list[Trade] = []
    n = len(sdf)
    stop_mask = exit_mask if exit_mask is not None else ~entry_mask.fillna(False)
    stop_arr = stop_mask.fillna(False).to_numpy()

    for pos in entry_positions:
        entry_date = sdf[S.SCAN_DATE].iloc[pos]
        entry_price = float(sdf[S.CURRENT_PRICE].iloc[pos])
        if entry_price <= 0:
            continue
        exit_pos = None
        for j in range(pos + 1, n):
            if stop_arr[j]:
                exit_pos = j
                break
        if exit_pos is None:
            trades.append(Trade(symbol, entry_date, entry_price, None, None,
                                 None, None, True, "condition_exit"))
            continue
        exit_row = sdf.iloc[exit_pos]
        exit_price = float(exit_row[S.CURRENT_PRICE])
        if exit_price <= 0:
            continue
        holding_days = exit_pos - pos
        ret = round(((exit_price - entry_price) / entry_price) * 100, 2)
        trades.append(Trade(symbol, entry_date, entry_price, exit_row[S.SCAN_DATE],
                             exit_price, holding_days, ret, False, "condition_exit"))
    return trades


def run_backtest(panel: pd.DataFrame, strategy: Strategy) -> list[Trade]:
    """Evaluate `strategy` against every symbol's history in `panel`.

    Raises a clear error if entry_query/exit_query reference columns that
    don't exist in the panel, rather than pandas' harder-to-read KeyError.
    """
    if panel.empty:
        return []

    # Validate the queries once, up front, against the full panel so a typo
    # in a column name fails loudly and immediately -- rather than resolving
    # silently per-symbol and only surfacing as "zero signals found".
    try:
        panel.eval(strategy.entry_query)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"entry_query failed: {e}") from e

    if strategy.exit_query:
        try:
            panel.eval(strategy.exit_query)
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"exit_query failed: {e}") from e

    # Evaluated ONCE across the whole panel and then sliced per symbol, rather
    # than calling .eval() separately inside the per-symbol loop below. Each
    # .eval() call carries a fixed parsing/compile overhead independent of
    # row count -- on a real production panel with ~1800 distinct symbols,
    # calling it 1800 times (once per symbol) cost ~27s per strategy purely
    # in that overhead, which multiplies badly once the Leaderboard tab runs
    # several strategies back to back or walk-forward validation runs the
    # same sweep twice (train + test). A single whole-panel eval is exactly
    # equivalent row-for-row (these are all row-wise boolean expressions,
    # no cross-row aggregation), just sliced by each symbol's own index
    # before being reset alongside sdf so alignment is preserved exactly.
    full_entry_mask = panel.eval(strategy.entry_query)
    full_exit_mask = panel.eval(strategy.exit_query) if strategy.exit_query else None

    all_trades: list[Trade] = []
    for symbol, sdf in panel.groupby(S.SYMBOL, sort=False):
        sdf = sdf.sort_values(S.SCAN_DATE)
        order = sdf.index
        sdf = sdf.reset_index(drop=True)
        entry_mask = full_entry_mask.loc[order].reset_index(drop=True)
        exit_mask = full_exit_mask.loc[order].reset_index(drop=True) if full_exit_mask is not None else None

        entry_positions = _entry_dates(sdf, entry_mask)
        if not entry_positions:
            continue

        if strategy.exit_mode == ExitMode.FIXED_HOLDING:
            all_trades.extend(_run_fixed_holding(symbol, sdf, entry_positions,
                                                   strategy.fixed_holding_days, strategy))
        else:
            all_trades.extend(_run_condition_exit(symbol, sdf, entry_positions, entry_mask, exit_mask))

    return all_trades
