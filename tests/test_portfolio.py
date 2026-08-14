"""Top-K portfolio backtest correctness -- verified against hand-picked
rankings, not just "it runs"."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stockgpt.backtest import ExitMode, Strategy
from stockgpt.backtest.metrics import summarize
from stockgpt.backtest.portfolio import run_topk_backtest


def _panel_three_symbols():
    """A, B, C all cross the entry threshold on the SAME day (index 2), with
    different final_score (A=70, B=90, C=80) and different subsequent
    returns, so top-K filtering by final_score has an unambiguous expected
    answer: keep B and C, drop A when top_k=2."""
    dates = pd.date_range("2026-01-01", periods=10, freq="D")

    def make(symbol, score_at_entry, return_pct):
        score = [40] * 10
        score[2] = score_at_entry
        price = [100.0] * 10
        price[5] = 100.0 * (1 + return_pct / 100)  # exit at pos 2+3=5
        return pd.DataFrame({
            "symbol": symbol, "scan_date": dates, "current_price": price, "final_score": score,
        })

    return pd.concat([
        make("A", 70, 5.0),
        make("B", 90, 10.0),
        make("C", 80, 8.0),
    ], ignore_index=True)


class TestRunTopkBacktest:
    def test_keeps_only_top_k_by_rank_column_on_a_crowded_day(self):
        panel = _panel_three_symbols()
        strat = Strategy(name="t", entry_query="final_score >= 65",
                          exit_mode=ExitMode.FIXED_HOLDING, fixed_holding_days=(3,))
        trades = run_topk_backtest(panel, strat, top_k=2)
        symbols = {t.symbol for t in trades}
        assert symbols == {"B", "C"}, "A has the lowest final_score and should be dropped"
        assert len(trades) == 2

    def test_top_k_larger_than_available_keeps_everyone(self):
        panel = _panel_three_symbols()
        strat = Strategy(name="t", entry_query="final_score >= 65",
                          exit_mode=ExitMode.FIXED_HOLDING, fixed_holding_days=(3,))
        trades = run_topk_backtest(panel, strat, top_k=10)
        assert {t.symbol for t in trades} == {"A", "B", "C"}

    def test_results_feed_directly_into_summarize(self):
        panel = _panel_three_symbols()
        strat = Strategy(name="t", entry_query="final_score >= 65",
                          exit_mode=ExitMode.FIXED_HOLDING, fixed_holding_days=(3,))
        trades = run_topk_backtest(panel, strat, top_k=2)
        summary = summarize(trades, "t")
        row = summary.iloc[0]
        assert row["closed_trades"] == 2
        # kept trades were B (+10%) and C (+8%) -> avg 9%
        assert row["avg_return_pct"] == pytest.approx(9.0, abs=0.01)

    def test_condition_exit_strategy_raises_clear_error(self):
        panel = _panel_three_symbols()
        strat = Strategy(name="t", entry_query="final_score >= 65", exit_mode=ExitMode.CONDITION_EXIT)
        with pytest.raises(ValueError, match="fixed-holding"):
            run_topk_backtest(panel, strat, top_k=2)

    def test_bad_entry_query_raises_clear_error(self):
        panel = _panel_three_symbols()
        strat = Strategy(name="t", entry_query="not_a_real_column >= 65",
                          exit_mode=ExitMode.FIXED_HOLDING, fixed_holding_days=(3,))
        with pytest.raises(ValueError, match="entry_query failed"):
            run_topk_backtest(panel, strat, top_k=2)

    def test_top_k_below_one_raises(self):
        panel = _panel_three_symbols()
        strat = Strategy(name="t", entry_query="final_score >= 65",
                          exit_mode=ExitMode.FIXED_HOLDING, fixed_holding_days=(3,))
        with pytest.raises(ValueError, match="top_k"):
            run_topk_backtest(panel, strat, top_k=0)

    def test_empty_panel_returns_empty_list(self):
        strat = Strategy(name="t", entry_query="final_score >= 65",
                          exit_mode=ExitMode.FIXED_HOLDING, fixed_holding_days=(3,))
        assert run_topk_backtest(pd.DataFrame(), strat, top_k=2) == []

    def test_custom_rank_column_changes_which_symbols_survive(self):
        panel = _panel_three_symbols()
        # Rank by current_price at entry instead of final_score -- all three
        # start at 100.0 at entry (index 2), so add a distinguishing column.
        panel["custom_rank"] = 0
        panel.loc[panel["symbol"] == "A", "custom_rank"] = 100
        panel.loc[panel["symbol"] == "B", "custom_rank"] = 1
        panel.loc[panel["symbol"] == "C", "custom_rank"] = 2
        strat = Strategy(name="t", entry_query="final_score >= 65",
                          exit_mode=ExitMode.FIXED_HOLDING, fixed_holding_days=(3,))
        trades = run_topk_backtest(panel, strat, top_k=1, rank_column="custom_rank")
        assert {t.symbol for t in trades} == {"A"}, "A has the highest custom_rank and should survive alone"
