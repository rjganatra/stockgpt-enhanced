"""Top-K signal-selection backtest.

`run_backtest` treats every matching signal as its own independent trade --
which answers "what if I'd taken every single signal", not "what if I only
ever took my highest-conviction picks". On a day where 40 stocks all cross
the entry threshold at once, no one actually buys all 40; a real portfolio
has to choose. This module simulates that choice: on each entry day, if
more symbols match than `top_k`, only the `top_k` ranked highest by
`rank_column` (final_score by default) are actually taken -- the rest are
dropped, same as running out of capital or attention that day.

Being honest about scope: this is NOT a full capital-tracked equity curve.
It doesn't model position sizing, overlapping holding periods competing for
the same capital slot, or compounding returns across trades. It answers a
narrower, still useful question -- "does filtering to just my best K picks
per day change the win rate / average return compared to taking every
signal" -- not "exactly how much money would this have made me". A true
capital-tracked portfolio simulator is a meaningfully bigger project and
wasn't asked for; flagging the gap here rather than overclaiming what this
gives you.
"""

from __future__ import annotations

import pandas as pd

from .. import schema as S
from .corporate_actions import flag_price_jumps
from .engine import Trade, _entry_dates, _run_fixed_holding
from .strategy import ExitMode, Strategy


def run_topk_backtest(panel: pd.DataFrame, strategy: Strategy, top_k: int,
                       rank_column: str = S.FINAL_SCORE) -> list[Trade]:
    """Same rising-edge entry detection as run_backtest, but entries are
    bucketed by their actual calendar entry date across ALL symbols first;
    on any day with more matching entries than `top_k`, only the `top_k`
    with the highest `rank_column` value (at the moment of entry) are kept.
    Returns a list[Trade] in the exact same shape run_backtest returns, so
    it drops straight into the existing `summarize()` function.

    Fixed-holding exit only for now -- a condition-based exit doesn't have
    a single well-defined "day the trade opened" cohort to rank within in
    the same simple way, and wasn't asked for."""
    if panel.empty:
        return []
    if strategy.exit_mode != ExitMode.FIXED_HOLDING:
        raise ValueError("Top-K portfolio backtest only supports fixed-holding exits for now.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    try:
        panel.eval(strategy.entry_query)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"entry_query failed: {e}") from e

    # Evaluated once across the whole panel and sliced per symbol -- same
    # fix, and same reasoning, as run_backtest in engine.py: per-symbol
    # .eval() calls carry fixed overhead that adds up badly across ~1800
    # real symbols.
    full_entry_mask = panel.eval(strategy.entry_query)
    # Same corporate-action flagging run_backtest applies -- see
    # corporate_actions.py. Computed once here too so a demerger-corrupted
    # trade can't sneak into the top-K results even though this path
    # doesn't call run_backtest/_run_fixed_holding's usual entry point.
    full_jump_flags = flag_price_jumps(panel)

    # Every rising-edge entry, per symbol -- identical detection to run_backtest.
    per_symbol_entries: dict[str, tuple[pd.DataFrame, list[int]]] = {}
    jump_flags_by_symbol: dict[str, "pd.Series"] = {}
    for symbol, sdf in panel.groupby(S.SYMBOL, sort=False):
        sdf = sdf.sort_values(S.SCAN_DATE)
        order = sdf.index
        sdf = sdf.reset_index(drop=True)
        entry_mask = full_entry_mask.loc[order].reset_index(drop=True)
        positions = _entry_dates(sdf, entry_mask)
        if positions:
            per_symbol_entries[symbol] = (sdf, positions)
            jump_flags_by_symbol[symbol] = full_jump_flags.loc[order].to_numpy()

    # Bucket every entry by its actual calendar date, across all symbols.
    by_date: dict = {}
    for symbol, (sdf, positions) in per_symbol_entries.items():
        for pos in positions:
            entry_date = sdf[S.SCAN_DATE].iloc[pos]
            rank_value = sdf[rank_column].iloc[pos] if rank_column in sdf.columns else float("-inf")
            by_date.setdefault(entry_date, []).append((symbol, sdf, pos, rank_value))

    # Keep only the top_k highest-ranked entries per day.
    kept: list[tuple[str, pd.DataFrame, int]] = []
    for entry_date, candidates in by_date.items():
        ranked = sorted(
            candidates,
            key=lambda c: (c[3] if pd.notna(c[3]) else float("-inf")),
            reverse=True,
        )
        kept.extend((c[0], c[1], c[2]) for c in ranked[:top_k])

    all_trades: list[Trade] = []
    for symbol, sdf, pos in kept:
        all_trades.extend(_run_fixed_holding(symbol, sdf, [pos], strategy.fixed_holding_days,
                                              strategy, jump_flags_by_symbol[symbol]))
    return all_trades
