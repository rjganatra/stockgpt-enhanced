"""Walk-forward validation correctness -- verified with synthetic panels
where the "true" train-vs-test outcome is known by construction, not just
"it runs without crashing"."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stockgpt.backtest import split_panel_by_date, walk_forward_sweep


def _make_panel(strong_test_prices: list[float]) -> pd.DataFrame:
    """20 days, one symbol. A 'weak' signal (score 60) fires on day 1 (train)
    and day 11 (test), always with a mediocre -6% return. A 'strong' signal
    (score 85) fires on day 5 (train, always +12%) and day 15 (test, return
    controlled by `strong_test_prices` so callers can make it generalize or
    diverge). All other days score 40 (no entry)."""
    dates = pd.date_range("2026-01-01", periods=20, freq="D")
    score = [40] * 20
    score[1] = 60
    score[5] = 85
    score[11] = 60
    score[15] = 85

    price = [100.0] * 20
    # weak-train: entry day1=100, exit day4 -> -6%
    price[2], price[3], price[4] = 98.0, 96.0, 94.0
    # strong-train: entry day5=100, exit day8 -> +12%
    price[6], price[7], price[8] = 104.0, 108.0, 112.0
    # weak-test: entry day11=100, exit day14 -> -6%
    price[12], price[13], price[14] = 98.0, 96.0, 94.0
    # strong-test: entry day15=100, exit day18 -> caller-controlled
    price[16], price[17], price[18] = strong_test_prices

    return pd.DataFrame({
        "symbol": "GEN", "scan_date": dates, "current_price": price,
        "final_score": score,
    })


class TestSplitPanelByDate:
    def test_splits_by_distinct_date_not_row_count(self):
        dates = pd.date_range("2026-01-01", periods=10, freq="D")
        panel = pd.DataFrame({"symbol": "A", "scan_date": list(dates) * 2, "final_score": [50] * 20})
        train, test = split_panel_by_date(panel, split_pct=70)
        assert set(train["scan_date"]) == set(dates[:7])
        assert set(test["scan_date"]) == set(dates[7:])

    def test_fewer_than_two_dates_returns_two_empty_frames(self):
        panel = pd.DataFrame({"symbol": "A", "scan_date": [pd.Timestamp("2026-01-01")], "final_score": [50]})
        train, test = split_panel_by_date(panel)
        assert train.empty and test.empty

    def test_empty_panel_returns_two_empty_frames(self):
        train, test = split_panel_by_date(pd.DataFrame())
        assert train.empty and test.empty


class TestWalkForwardSweep:
    def test_generalizing_threshold_performs_similarly_on_test(self):
        # Strong signal returns +12% in BOTH train and test -- a real edge.
        panel = _make_panel(strong_test_prices=[104.0, 108.0, 112.0])
        result = walk_forward_sweep(panel, "final_score >= {t}", [50, 70], (3,), split_pct=50)

        assert not result.empty
        row = result.iloc[0]
        assert row["best_threshold"] == 70  # stricter threshold has the better train return
        assert row["train_avg_return_pct"] == pytest.approx(12.0, abs=0.01)
        assert row["test_avg_return_pct"] == pytest.approx(12.0, abs=0.01)
        assert row["test_closed_trades"] == 1
        assert bool(row["test_low_sample_warning"]) is True  # only 1 closed trade, honestly flagged

    def test_overfit_threshold_diverges_on_test(self):
        # Same threshold looks great in train (+12%) but the identical
        # condition actually loses money in test (-12%) -- the walk-forward
        # result should surface that divergence rather than hide it.
        panel = _make_panel(strong_test_prices=[96.0, 92.0, 88.0])
        result = walk_forward_sweep(panel, "final_score >= {t}", [50, 70], (3,), split_pct=50)

        assert not result.empty
        row = result.iloc[0]
        assert row["best_threshold"] == 70
        assert row["train_avg_return_pct"] == pytest.approx(12.0, abs=0.01)
        assert row["test_avg_return_pct"] == pytest.approx(-12.0, abs=0.01)
        assert row["train_avg_return_pct"] > row["test_avg_return_pct"]

    def test_insufficient_history_returns_empty(self):
        panel = pd.DataFrame({
            "symbol": "A", "scan_date": [pd.Timestamp("2026-01-01")], "final_score": [80],
            "current_price": [100.0],
        })
        result = walk_forward_sweep(panel, "final_score >= {t}", [50], (3,))
        assert result.empty

    def test_no_threshold_produces_signals_returns_empty(self):
        panel = _make_panel(strong_test_prices=[104.0, 108.0, 112.0])
        result = walk_forward_sweep(panel, "final_score >= {t}", [999], (3,), split_pct=50)
        assert result.empty
