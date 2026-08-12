import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stockgpt import fundamentals as F, schema as S


def test_missing_ratios_stay_missing_not_zero():
    row = pd.Series({S.ROE: np.nan, S.DEBT_TO_EQUITY: np.nan, S.NET_PROFIT_MARGIN: np.nan,
                      S.REVENUE_GROWTH: np.nan, S.EARNINGS_GROWTH: np.nan, S.OPERATING_MARGIN: np.nan,
                      S.ROA: np.nan, S.CURRENT_RATIO: np.nan, S.QUICK_RATIO: np.nan,
                      S.TOTAL_CASH_CR: np.nan, S.TOTAL_DEBT_CR: np.nan, S.OPERATING_CASHFLOW_CR: np.nan,
                      S.FREE_CASHFLOW_CR: np.nan, S.TRAILING_PE: np.nan, S.FORWARD_PE: np.nan,
                      S.PRICE_TO_BOOK: np.nan, S.DIVIDEND_YIELD: np.nan})
    scored = F.score_row(row)
    assert scored[S.FUNDAMENTAL_DATA_COVERAGE] == 0.0
    assert scored[S.FUNDAMENTAL_SCORE] == 0.0
    assert scored[S.FUNDAMENTAL_RISK_REASONS] == ""  # unknown != risky; no false risk flags


def test_bank_debt_to_equity_keyword_bucket():
    assert F.sector_bucket("Financial Services", "Banks - Regional") == "Banking"
    assert F.sector_bucket("Technology", "Software - Application") == "IT / Technology"
    assert F.sector_bucket("", "") == "General"


def test_coverage_fraction_is_between_0_and_1():
    row = pd.Series({S.ROE: 15, S.DEBT_TO_EQUITY: 20, S.NET_PROFIT_MARGIN: 10,
                      S.REVENUE_GROWTH: np.nan, S.EARNINGS_GROWTH: np.nan, S.OPERATING_MARGIN: np.nan,
                      S.ROA: np.nan, S.CURRENT_RATIO: np.nan, S.QUICK_RATIO: np.nan,
                      S.TOTAL_CASH_CR: np.nan, S.TOTAL_DEBT_CR: np.nan, S.OPERATING_CASHFLOW_CR: np.nan,
                      S.FREE_CASHFLOW_CR: np.nan, S.TRAILING_PE: np.nan, S.FORWARD_PE: np.nan,
                      S.PRICE_TO_BOOK: np.nan, S.DIVIDEND_YIELD: np.nan})
    scored = F.score_row(row)
    assert 0 < scored[S.FUNDAMENTAL_DATA_COVERAGE] < 1


def test_sector_adjustment_rewards_bank_leverage_instead_of_penalizing():
    row = pd.Series({S.SECTOR_BUCKET: "Banking", S.ROE: 18, S.DEBT_TO_EQUITY: 800,
                      S.PRICE_TO_BOOK: 2.0, S.REVENUE_GROWTH: np.nan, S.OPERATING_MARGIN: np.nan})
    adj, reasons = F._sector_adjustment_row(row)
    assert adj > 0
    assert any("structural" in r for r in reasons)


def test_sector_adjustment_is_clamped_to_plus_minus_15():
    row = pd.Series({S.SECTOR_BUCKET: "Banking", S.ROE: 50, S.DEBT_TO_EQUITY: 50,
                      S.PRICE_TO_BOOK: 1.0, S.REVENUE_GROWTH: np.nan, S.OPERATING_MARGIN: np.nan})
    adj, _ = F._sector_adjustment_row(row)
    assert -15.0 <= adj <= 15.0
