"""Price-jump corporate-action heuristic -- verified against hand-computed
expected flags and hand-computed expected trade exclusions, not just "it
runs". See src/stockgpt/backtest/corporate_actions.py's module docstring
for the reasoning this test suite is checking.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stockgpt.backtest import ExitMode, Strategy, run_backtest
from stockgpt.backtest.corporate_actions import flag_price_jumps


def _panel(rows):
    """rows: list of (symbol, date_str, current_price, day_change_pct)."""
    symbols, dates, prices, changes = zip(*rows)
    return pd.DataFrame({
        "symbol": symbols,
        "scan_date": pd.to_datetime(dates),
        "current_price": prices,
        "day_change_pct": changes,
        "final_score": [70] * len(rows),
        "score_band": ["Strong"] * len(rows),
    })


def _multi_symbol_panel_with_one_jump(jump_date="2026-01-02", jump_pct=-40.0):
    """5 calm symbols (day_change_pct ~1.0 every day, every day) plus one
    symbol (BBB) that jumps only on `jump_date`. A realistic production
    panel has hundreds of symbols per day, so one outlier barely moves the
    cross-sectional median -- this needs enough OTHER symbols per day to
    reproduce that, unlike a 1-2 symbol panel where the median gets pulled
    halfway to the very outlier it's supposed to detect (confirmed by hand:
    with 2 symbols, median of [1.0, -40.0] is -19.5, so BOTH symbols land
    ~20pts from it and the jump goes undetected -- an artifact of small n,
    not a real weakness of the market-relative approach at realistic scale)."""
    rows = []
    for date in ["2026-01-01", "2026-01-02", "2026-01-03"]:
        for i, sym in enumerate(["AAA", "CCC", "DDD", "EEE", "FFF"]):
            rows.append((sym, date, 100 + i, 1.0))
        rows.append(("BBB", date, 200, jump_pct if date == jump_date else 1.5))
    return _panel(rows)


def test_idiosyncratic_jump_is_flagged_but_calm_day_is_not():
    panel = _multi_symbol_panel_with_one_jump()
    flags = flag_price_jumps(panel, threshold_pct=35.0)
    flags_by_row = dict(zip(zip(panel["symbol"], panel["scan_date"]), flags))

    assert flags_by_row[("BBB", pd.Timestamp("2026-01-02"))] is True
    assert flags_by_row[("AAA", pd.Timestamp("2026-01-01"))] is False
    assert flags_by_row[("AAA", pd.Timestamp("2026-01-02"))] is False
    assert flags_by_row[("BBB", pd.Timestamp("2026-01-01"))] is False
    assert flags_by_row[("BBB", pd.Timestamp("2026-01-03"))] is False


def test_broad_market_move_day_is_not_flagged_even_though_raw_change_is_large():
    # Every symbol drops ~30% together (a real market-wide crash day) --
    # market-relative deviation is ~0 for all of them, so nothing should
    # be flagged even though the raw day_change_pct is well past a naive
    # fixed threshold.
    panel = _panel([
        ("AAA", "2026-01-01", 100, -30.0), ("BBB", "2026-01-01", 200, -31.0),
        ("CCC", "2026-01-01", 300, -29.5),
    ])
    flags = flag_price_jumps(panel, threshold_pct=35.0)
    assert not flags.any()


def test_missing_day_change_pct_column_flags_nothing_not_a_crash():
    panel = pd.DataFrame({
        "symbol": ["AAA"], "scan_date": pd.to_datetime(["2026-01-01"]),
        "current_price": [100], "final_score": [70],
    })
    flags = flag_price_jumps(panel)
    assert not flags.any()
    assert len(flags) == 1


def test_nan_day_change_pct_is_flagged_as_unsafe_not_skipped():
    # A symbol's first-ever appearance in the panel (new listing, or start
    # of this data's coverage) has no previous_close to diff against --
    # day_change_pct is NaN, which must be treated as "don't trust this
    # row", not silently coerced to "no jump happened".
    panel = _panel([("AAA", "2026-01-01", 100, float("nan"))])
    flags = flag_price_jumps(panel)
    assert bool(flags.iloc[0]) is True


def _stocka_plus_calm_neighbors_panel(jump_day_index, dates, stocka_prices, stocka_scores):
    """STOCKA is the symbol under test (its own price/score sequence
    supplied by the caller); 5 always-calm neighbor symbols exist purely so
    the cross-sectional median on the jump day isn't distorted by small-n
    (see _multi_symbol_panel_with_one_jump's comment for why that matters),
    and never generate their own entry signals (final_score always low)."""
    rows = []
    for i, date in enumerate(dates):
        rows.append(("STOCKA", date, stocka_prices[i], -40.0 if i == jump_day_index else 1.0))
        for j, sym in enumerate(["CALM1", "CALM2", "CALM3", "CALM4", "CALM5"]):
            rows.append((sym, date, 100 + j, 1.0))
    df = _panel(rows)
    score_by_date = dict(zip(dates, stocka_scores))
    df["final_score"] = df.apply(
        lambda r: score_by_date[r["scan_date"].strftime("%Y-%m-%d")] if r["symbol"] == "STOCKA" else 40,
        axis=1,
    )
    df["score_band"] = df["final_score"].map(lambda s: "Strong" if s >= 65 else "Neutral")
    return df


def test_backtest_excludes_a_trade_whose_window_spans_a_mid_holding_jump():
    # The jump happens on day index 2, which is neither the entry (index
    # 0) nor a 4-day-holding exit (index 4) -- this is the actual demerger
    # scenario: entry_price and exit_price are each individually "clean"
    # numbers, but the return between them is contaminated by a jump that
    # happened partway through the holding window. A naive "only check the
    # entry/exit day" implementation would miss this; this test fails
    # against that naive version and passes against the real one.
    dates = ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"]
    prices = [100, 101, 60, 61, 62]
    scores = [70, 70, 70, 70, 70]
    panel = _stocka_plus_calm_neighbors_panel(jump_day_index=2, dates=dates,
                                               stocka_prices=prices, stocka_scores=scores)
    strat = Strategy(name="t", entry_query="final_score >= 65",
                      exit_mode=ExitMode.FIXED_HOLDING, fixed_holding_days=(4,))
    trades = run_backtest(panel, strat)
    stocka_trades = [t for t in trades if t.symbol == "STOCKA"]
    assert stocka_trades == [], "trade's holding window spans the jump on day index 2 -- must be excluded"


def test_backtest_keeps_a_trade_whose_window_does_not_touch_the_jump_day():
    # Same panel, but a 1-day hold (entry index 0 -> exit index 1) never
    # touches the jump on index 2 -- this trade should NOT be excluded.
    dates = ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"]
    prices = [100, 101, 60, 61, 62]
    scores = [70, 70, 70, 70, 70]
    panel = _stocka_plus_calm_neighbors_panel(jump_day_index=2, dates=dates,
                                               stocka_prices=prices, stocka_scores=scores)
    strat = Strategy(name="t", entry_query="final_score >= 65",
                      exit_mode=ExitMode.FIXED_HOLDING, fixed_holding_days=(1,))
    trades = run_backtest(panel, strat)
    stocka_trades = [t for t in trades if t.symbol == "STOCKA"]
    assert len(stocka_trades) == 1
    assert stocka_trades[0].return_pct == pytest.approx(1.0, abs=0.01)


def test_entry_on_a_flagged_day_itself_is_excluded():
    # Entry signal fires exactly on the jump day -- excluded as an entry
    # point entirely, not just re-priced. final_score is only high on the
    # jump day, so the ONLY entry signal in this panel is on a flagged day.
    dates = ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]
    prices = [100, 60, 61, 62]
    scores = [40, 70, 40, 40]
    panel = _stocka_plus_calm_neighbors_panel(jump_day_index=1, dates=dates,
                                               stocka_prices=prices, stocka_scores=scores)
    strat = Strategy(name="t", entry_query="final_score >= 65",
                      exit_mode=ExitMode.FIXED_HOLDING, fixed_holding_days=(1,))
    trades = run_backtest(panel, strat)
    stocka_trades = [t for t in trades if t.symbol == "STOCKA"]
    assert stocka_trades == [], "the only entry signal fires on the flagged day -- must be excluded"
