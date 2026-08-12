"""Canonical column names for the whole pipeline.

Why this file exists: the original StockGPT accumulated near-duplicate
columns over time (`risk_penalty` computed twice with a pandas concat that
silently kept the wrong one; `reasons` / `technical_reasons` / `risk_reasons`
/ `fundamental_reasons` all doing overlapping jobs; `conviction_score` vs
`score` vs `final_conviction_score` all coexisting as vestigial leftovers
from three scoring-engine rewrites). Importing names from here instead of
typing raw strings means a typo becomes an ImportError instead of a silent
`KeyError` -> `.get(..., 0)` fallback three modules later, which is exactly
the class of bug that produced the risk-penalty issue in the original repo.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------
SYMBOL = "symbol"
COMPANY_NAME = "company_name"
SECTOR = "sector"
INDUSTRY = "industry"
SECTOR_BUCKET = "sector_bucket"

# ---------------------------------------------------------------------------
# Price / technical (scanner.py)
# ---------------------------------------------------------------------------
CURRENT_PRICE = "current_price"
DAY_CHANGE_PCT = "day_change_pct"
LOW_52W = "low_52w"
HIGH_52W = "high_52w"
DISTANCE_FROM_LOW_PCT = "distance_from_low_pct"
DISTANCE_FROM_HIGH_PCT = "distance_from_high_pct"
RSI = "rsi"
SMA50 = "sma50"
SMA200 = "sma200"
AVG_VOLUME_20D = "avg_volume_20d"
LATEST_VOLUME = "latest_volume"
VOLUME_RATIO = "volume_ratio"
TREND = "trend"                      # "Bullish" / "Bearish"
TECHNICAL_SCORE = "technical_score"
TECHNICAL_REASONS = "technical_reasons"

# ---------------------------------------------------------------------------
# Fundamentals (fundamentals.py) -- raw fetched ratios
# ---------------------------------------------------------------------------
MARKET_CAP_CR = "market_cap_cr"
TRAILING_PE = "trailing_pe"
FORWARD_PE = "forward_pe"
PRICE_TO_BOOK = "price_to_book"
DEBT_TO_EQUITY = "debt_to_equity"
ROE = "roe"
ROA = "roa"
OPERATING_MARGIN = "operating_margin"
NET_PROFIT_MARGIN = "net_profit_margin"
GROSS_MARGIN = "gross_margin"
REVENUE_GROWTH = "revenue_growth"
EARNINGS_GROWTH = "earnings_growth"
CURRENT_RATIO = "current_ratio"
QUICK_RATIO = "quick_ratio"
TOTAL_CASH_CR = "total_cash_cr"
TOTAL_DEBT_CR = "total_debt_cr"
FREE_CASHFLOW_CR = "free_cashflow_cr"
OPERATING_CASHFLOW_CR = "operating_cashflow_cr"
DIVIDEND_YIELD = "dividend_yield"
BETA = "beta"

# Every raw fundamental field above has a matching `<field>_known` boolean.
# This is how "we never fetched this" stays distinguishable from "this is
# genuinely zero" all the way through to the dashboard and the risk model.
FUNDAMENTAL_RATIO_FIELDS = [
    DEBT_TO_EQUITY, ROE, ROA, OPERATING_MARGIN, NET_PROFIT_MARGIN,
    GROSS_MARGIN, REVENUE_GROWTH, EARNINGS_GROWTH, CURRENT_RATIO,
    QUICK_RATIO, OPERATING_CASHFLOW_CR, FREE_CASHFLOW_CR,
    TRAILING_PE, FORWARD_PE, PRICE_TO_BOOK, DIVIDEND_YIELD, BETA,
]


def known_flag(field: str) -> str:
    """Name of the boolean 'was this actually fetched' companion column."""
    return f"{field}_known"


# ---------------------------------------------------------------------------
# Fundamental scoring (fundamentals.py) -- derived
# ---------------------------------------------------------------------------
PROFITABILITY_SCORE = "profitability_score"
GROWTH_SCORE = "growth_score"
BALANCE_SHEET_SCORE = "balance_sheet_score"
CASHFLOW_SCORE = "cashflow_score"
VALUATION_SCORE = "valuation_score"
FUNDAMENTAL_RISK_PENALTY = "fundamental_risk_penalty"
FUNDAMENTAL_SCORE = "fundamental_score"
FUNDAMENTAL_DATA_COVERAGE = "fundamental_data_coverage"   # 0-1, share of ratio fields known
SECTOR_ADJUSTMENT = "sector_adjustment"
SECTOR_ADJUSTED_FUNDAMENTAL_SCORE = "sector_adjusted_fundamental_score"
FUNDAMENTAL_REASONS = "fundamental_reasons"
FUNDAMENTAL_RISK_REASONS = "fundamental_risk_reasons"

# ---------------------------------------------------------------------------
# Relative strength (relative_strength.py)
# ---------------------------------------------------------------------------
RETURN_1M = "return_1m"
RETURN_3M = "return_3m"
RETURN_6M = "return_6m"
NIFTY_RETURN_1M = "nifty_return_1m"
NIFTY_RETURN_3M = "nifty_return_3m"
NIFTY_RETURN_6M = "nifty_return_6m"
RETURN_VS_NIFTY_1M = "return_vs_nifty_1m"
RETURN_VS_NIFTY_3M = "return_vs_nifty_3m"
RETURN_VS_NIFTY_6M = "return_vs_nifty_6m"
RELATIVE_STRENGTH_SCORE = "relative_strength_score"
SECTOR_RANK_PCT = "sector_rank_pct"

# ---------------------------------------------------------------------------
# Range-bound (range_bound.py)
# ---------------------------------------------------------------------------
RANGE_STATUS = "range_status"
RANGE_SCORE = "range_score"
RANGE_LOW = "range_low"
RANGE_HIGH = "range_high"
RANGE_WIDTH_PCT = "range_width_pct"
RANGE_POSITION_PCT = "range_position_pct"
RANGE_REASONS = "range_reasons"

# ---------------------------------------------------------------------------
# Final scoring engine (scoring.py) -- the single source of truth
# ---------------------------------------------------------------------------
SECTOR_SCORE = "sector_score"
RISK_PENALTY = "risk_penalty"                # ONE risk penalty. Not two.
RISK_REASONS = "risk_reasons"
FINAL_SCORE = "final_score"
SCORE_BAND = "score_band"
REASONS = "reasons"                          # single combined explanation

# Score bands. Kept from the original StockGPT (the "Avoid / Neutral / ..."
# ladder the user explicitly asked to preserve) but renamed to remove the
# grade-letter framing (A+/A/B/C/D/E) that implied more precision than a
# heuristic score actually has.
BAND_AVOID = "Avoid"
BAND_WEAK = "Weak"
BAND_NEUTRAL = "Neutral"
BAND_WATCHLIST = "Watchlist"
BAND_STRONG = "Strong"
BAND_HIGH_CONVICTION = "High Conviction"

BAND_ORDER = [
    BAND_AVOID, BAND_WEAK, BAND_NEUTRAL, BAND_WATCHLIST, BAND_STRONG,
    BAND_HIGH_CONVICTION,
]

# ---------------------------------------------------------------------------
# History / snapshots
# ---------------------------------------------------------------------------
SCAN_DATE = "scan_date"
SCAN_TIME = "scan_time"

ALL_NUMERIC_SCORE_COLUMNS = [
    TECHNICAL_SCORE, FUNDAMENTAL_SCORE, SECTOR_ADJUSTED_FUNDAMENTAL_SCORE,
    RELATIVE_STRENGTH_SCORE, SECTOR_SCORE, RISK_PENALTY, FINAL_SCORE,
    RANGE_SCORE,
]
