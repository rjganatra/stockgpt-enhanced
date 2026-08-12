"""Scoring engine correctness -- specifically a regression test for the
original repo's risk-penalty bug (fundamental risk factors present in the
data but silently excluded from the number subtracted from the final score).
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stockgpt import scoring, schema as S


def _base_row(**overrides):
    row = dict(
        symbol="TEST", sector="Technology", current_price=1000, sma200=900,
        rsi=55, day_change_pct=0.5, distance_from_high_pct=5, volume_ratio=1.2,
        debt_to_equity=20, net_profit_margin=15, revenue_growth=10, earnings_growth=10,
        operating_cashflow_cr=100, free_cashflow_cr=50, fundamental_data_coverage=1.0,
        technical_score=70, sector_adjusted_fundamental_score=60, relative_strength_score=55,
    )
    row.update(overrides)
    return row


def test_high_debt_and_negative_margin_both_produce_a_nonzero_risk_penalty():
    """Regression test for the original repo's pandas concat/duplicated()
    bug: fundamental risk factors must actually reduce the final score, not
    just appear in the reasons text."""
    row = _base_row(debt_to_equity=275.52, net_profit_margin=-2.7,
                     current_price=811.05, sma200=355.29, rsi=69.25,
                     day_change_pct=1.62, distance_from_high_pct=4.02, volume_ratio=0.65)
    penalty, reasons = scoring.compute_risk_penalty(pd.Series(row))

    assert penalty > 0, "high debt + negative margin must produce a real penalty, not 0"
    assert penalty == pytest.approx(16.0)  # 8 (debt) + 8 (margin), no technical risk triggered
    assert "Very high debt" in reasons
    assert "Negative net margin" in reasons


def test_clean_stock_has_zero_risk_penalty():
    row = _base_row()
    penalty, reasons = scoring.compute_risk_penalty(pd.Series(row))
    assert penalty == 0.0
    assert reasons == []


def test_low_fundamental_coverage_adds_a_small_penalty_instead_of_free_pass():
    clean = _base_row(fundamental_data_coverage=1.0)
    thin = _base_row(fundamental_data_coverage=0.1)

    penalty_clean, _ = scoring.compute_risk_penalty(pd.Series(clean))
    penalty_thin, reasons_thin = scoring.compute_risk_penalty(pd.Series(thin))

    assert penalty_thin > penalty_clean
    assert "Limited fundamental data coverage" in reasons_thin


def test_final_score_reflects_the_full_risk_penalty():
    df = pd.DataFrame([
        _base_row(symbol="RISKY", debt_to_equity=275, net_profit_margin=-5,
                   technical_score=70, sector_adjusted_fundamental_score=40,
                   relative_strength_score=50),
        _base_row(symbol="CLEAN", technical_score=70,
                   sector_adjusted_fundamental_score=40, relative_strength_score=50),
    ])
    out = scoring.compute_final_scores(df)
    risky = out[out[S.SYMBOL] == "RISKY"].iloc[0]
    clean = out[out[S.SYMBOL] == "CLEAN"].iloc[0]

    assert risky[S.RISK_PENALTY] > 0
    assert risky[S.FINAL_SCORE] < clean[S.FINAL_SCORE], (
        "a stock with real fundamental risk must score lower than an "
        "otherwise-identical clean stock"
    )


def test_score_bands_match_configured_cutoffs():
    assert scoring.classify_band(80) == S.BAND_HIGH_CONVICTION
    assert scoring.classify_band(70) == S.BAND_STRONG
    assert scoring.classify_band(60) == S.BAND_WATCHLIST
    assert scoring.classify_band(50) == S.BAND_NEUTRAL
    assert scoring.classify_band(40) == S.BAND_WEAK
    assert scoring.classify_band(10) == S.BAND_AVOID


def test_final_score_never_exceeds_100_or_drops_below_0():
    extreme_high = _base_row(technical_score=100, sector_adjusted_fundamental_score=100,
                              relative_strength_score=100)
    extreme_low = _base_row(technical_score=0, sector_adjusted_fundamental_score=0,
                             relative_strength_score=0, debt_to_equity=999,
                             net_profit_margin=-99, revenue_growth=-99, earnings_growth=-99,
                             operating_cashflow_cr=-99, free_cashflow_cr=-99,
                             current_price=100, sma200=500, rsi=5, day_change_pct=-20,
                             distance_from_high_pct=90, volume_ratio=0.1)
    df = pd.DataFrame([dict(extreme_high, symbol="HI"), dict(extreme_low, symbol="LO")])
    out = scoring.compute_final_scores(df)
    assert out[S.FINAL_SCORE].between(0, 100).all()
