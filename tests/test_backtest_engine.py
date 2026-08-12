"""Backtest engine correctness, verified against hand-computed expected
returns -- not just "it runs without crashing".
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stockgpt.backtest import ExitMode, Strategy, run_backtest, summarize


def _single_stock_panel():
    dates = pd.date_range("2026-01-01", periods=30, freq="D")
    score = [40] * 5 + [70] * 16 + [40] * 9
    price = [100] * 5
    p = 100
    for _ in range(16):
        p += 1
        price.append(p)
    while len(price) < 30:
        price.append(price[-1])
    return pd.DataFrame({
        "symbol": "STOCKA", "scan_date": dates, "current_price": price,
        "final_score": score,
        "score_band": ["Neutral" if s < 65 else "Strong" for s in score],
    })


def test_rising_edge_counts_as_one_entry_not_sixteen():
    panel = _single_stock_panel()
    strat = Strategy(name="t", entry_query="final_score >= 65",
                      exit_mode=ExitMode.FIXED_HOLDING, fixed_holding_days=(5,))
    trades = run_backtest(panel, strat)
    assert len(trades) == 1, "condition held true for 16 days but must count as ONE entry"


def test_fixed_holding_return_matches_hand_calculation():
    panel = _single_stock_panel()
    strat = Strategy(name="t", entry_query="final_score >= 65",
                      exit_mode=ExitMode.FIXED_HOLDING, fixed_holding_days=(5, 10))
    trades = run_backtest(panel, strat)
    by_horizon = {t.horizon_label: t for t in trades}

    # entry at index 5, price 101
    assert by_horizon["5D"].entry_price == 101.0
    assert by_horizon["5D"].exit_price == 106.0
    assert by_horizon["5D"].return_pct == pytest.approx(4.95, abs=0.01)

    assert by_horizon["10D"].exit_price == 111.0
    assert by_horizon["10D"].return_pct == pytest.approx(9.9, abs=0.01)


def test_condition_exit_holds_until_condition_flips_false():
    panel = _single_stock_panel()
    strat = Strategy(name="t", entry_query="final_score >= 65", exit_mode=ExitMode.CONDITION_EXIT)
    trades = run_backtest(panel, strat)
    assert len(trades) == 1
    trade = trades[0]
    assert trade.holding_days == 16          # true for indices 5..20 inclusive
    assert trade.exit_price == 116.0
    assert trade.return_pct == pytest.approx(14.85, abs=0.01)
    assert trade.is_open is False


def test_trade_still_open_at_end_of_panel_is_excluded_from_win_rate():
    dates = pd.date_range("2026-01-01", periods=10, freq="D")
    # condition becomes true on the LAST day -- no room for a 5-day exit.
    score = [40] * 9 + [70]
    price = [100] * 10
    panel = pd.DataFrame({
        "symbol": "STOCKB", "scan_date": dates, "current_price": price,
        "final_score": score, "score_band": ["Neutral"] * 9 + ["Strong"],
    })
    strat = Strategy(name="t", entry_query="final_score >= 65",
                      exit_mode=ExitMode.FIXED_HOLDING, fixed_holding_days=(5,))
    trades = run_backtest(panel, strat)
    assert len(trades) == 1
    assert trades[0].is_open is True
    assert trades[0].return_pct is None

    summary = summarize(trades, "t")
    row = summary.iloc[0]
    assert row["closed_trades"] == 0
    assert row["open_trades"] == 1
    assert bool(row["low_sample_warning"]) is True  # 0 closed trades -> can't be confident


def test_win_rate_calculation_is_correct_with_mixed_outcomes():
    dates = pd.date_range("2026-01-01", periods=20, freq="D")

    def make(symbol, entry_idx, prices):
        score = [40] * entry_idx + [70] + [40] * (19 - entry_idx)
        price = [100] * 20
        for i, p in enumerate(prices):
            price[entry_idx + i] = p
        return pd.DataFrame({
            "symbol": symbol, "scan_date": dates, "current_price": price,
            "final_score": score, "score_band": ["Strong" if s >= 65 else "Neutral" for s in score],
        })

    # winner: +20% after 3 days. loser: -10% after 3 days.
    winner = make("WIN", 2, [100, 105, 110, 120])
    loser = make("LOSE", 2, [100, 95, 90, 90])
    panel = pd.concat([winner, loser], ignore_index=True)

    strat = Strategy(name="t", entry_query="final_score >= 65",
                      exit_mode=ExitMode.FIXED_HOLDING, fixed_holding_days=(3,))
    trades = run_backtest(panel, strat)
    summary = summarize(trades, "t")
    row = summary.iloc[0]

    assert row["closed_trades"] == 2
    assert row["win_rate_pct"] == 50.0
    assert row["avg_return_pct"] == pytest.approx((20 + (-10)) / 2, abs=0.5)


def test_bad_entry_query_raises_clear_error_not_keyerror():
    panel = _single_stock_panel()
    strat = Strategy(name="t", entry_query="this_column_does_not_exist >= 65")
    with pytest.raises(ValueError, match="entry_query failed"):
        run_backtest(panel, strat)


def test_no_matching_signals_returns_empty_not_a_crash():
    panel = _single_stock_panel()
    strat = Strategy(name="t", entry_query="final_score >= 999")
    trades = run_backtest(panel, strat)
    assert trades == []
    summary = summarize(trades, "t")
    assert summary.empty
