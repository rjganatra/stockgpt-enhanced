"""Fundamental data fetch + sector-aware scoring.

Two things this module fixes relative to the original repo:

1. Missing ratios stay missing. yfinance's `financialData` block (ROE, ROA,
   current/quick ratio, operating & free cash flow) is an all-or-nothing
   block that Yahoo simply doesn't populate for roughly half of NSE tickers
   -- verified against the original repo's live data: nullness across those
   six fields was 92-97% correlated, i.e. one missing module, not six
   independent failures. The original repo's score_engine.py later
   coerced several of these (debt_to_equity, net_profit_margin,
   earnings_growth, revenue_growth, cash-flow figures) to 0 with
   `fillna(0)` on the way to the final CSV. A 0 debt-to-equity reads as
   "no debt" to every risk check downstream, so an unfetched ratio was
   silently scored as *safe*. Here, every ratio keeps `NaN` when unknown,
   plus a `<field>_known` companion column, all the way to the final
   dataset.

2. A stock with thin data coverage gets a small explicit risk penalty
   instead of a free pass (see FUNDAMENTAL_DATA_COVERAGE +
   RiskPenalties.low_data_coverage_penalty in scoring.py).
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from . import schema as S
from .config import FUNDAMENTAL_CAPS

logger = logging.getLogger(__name__)

MAX_WORKERS = 3          # Yahoo throttles aggressive .info polling
RETRY_ATTEMPTS = 3


# ---------------------------------------------------------------------------
# Sector bucketing (kept from the original -- this idea was genuinely good:
# a bank's "high debt" is structural, not a risk signal, so blanket
# thresholds misprice financials without it)
# ---------------------------------------------------------------------------
SECTOR_BUCKET_KEYWORDS = {
    "Banking": ["bank"],
    "Financial Services": ["nbfc", "finance", "financial", "insurance",
                            "capital markets", "asset management", "credit"],
    "IT / Technology": ["software", "information technology", "technology",
                         "semiconductor", "electronic", "computer"],
    "Pharma / Healthcare": ["pharma", "healthcare", "hospital", "diagnostic",
                             "biotech", "medical"],
    "FMCG / Consumer": ["fmcg", "food", "beverage", "household",
                         "personal products", "consumer staples", "tobacco"],
    "Automobile": ["auto", "automobile", "tyre", "tire", "ancillaries"],
    "Metals / Mining": ["metal", "steel", "aluminium", "aluminum", "copper",
                         "mining", "coal"],
    "Energy / Utilities": ["oil", "gas", "energy", "power", "utility",
                            "utilities", "renewable", "electricity"],
    "Capital Goods / Infra": ["capital goods", "engineering", "industrial",
                               "machinery", "defence", "defense",
                               "aerospace", "rail", "infrastructure"],
    "Chemicals": ["chemical", "fertilizer", "agrochemical", "paint"],
    "Cement / Realty": ["cement", "real estate", "building material",
                         "construction material"],
    "Telecom": ["telecom", "communication"],
    "Consumer Discretionary": ["retail", "textile", "apparel", "footwear",
                                "jewellery", "jewelry", "consumer cyclical"],
}


def sector_bucket(sector_yf: str, industry_yf: str) -> str:
    text = f"{sector_yf} {industry_yf}".lower()
    for bucket, keywords in SECTOR_BUCKET_KEYWORDS.items():
        if any(k in text for k in keywords):
            return bucket
    return "General"


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------
def _safe_percent(value) -> Optional[float]:
    """yfinance usually returns e.g. 0.15 for 15%; occasionally already 15."""
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return round(value * 100, 2) if abs(value) <= 5 else round(value, 2)


def _safe_dividend_yield(value) -> Optional[float]:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return round(value * 100, 2) if value <= 1 else round(value, 2)


def _safe_number(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def _safe_crore(value) -> Optional[float]:
    n = _safe_number(value)
    return round(n / 1e7, 2) if n is not None else None


def fetch_one(symbol: str) -> Optional[dict]:
    yf_symbol = f"{symbol}.NS"
    info: dict = {}
    for attempt in range(RETRY_ATTEMPTS):
        try:
            info = yf.Ticker(yf_symbol).info or {}
            if info:
                break
        except Exception as e:  # noqa: BLE001
            logger.debug("%s fetch attempt %s failed: %s", symbol, attempt + 1, e)
        time.sleep(3 + attempt * 4)

    if not info:
        return None

    sector_yf = info.get("sector") or "Unknown"
    industry_yf = info.get("industry") or "Unknown"

    raw = {
        S.MARKET_CAP_CR: _safe_crore(info.get("marketCap")),
        S.TRAILING_PE: _safe_number(info.get("trailingPE")),
        S.FORWARD_PE: _safe_number(info.get("forwardPE")),
        S.PRICE_TO_BOOK: _safe_number(info.get("priceToBook")),
        S.DEBT_TO_EQUITY: _safe_number(info.get("debtToEquity")),
        S.ROE: _safe_percent(info.get("returnOnEquity")),
        S.ROA: _safe_percent(info.get("returnOnAssets")),
        S.OPERATING_MARGIN: _safe_percent(info.get("operatingMargins")),
        S.NET_PROFIT_MARGIN: _safe_percent(info.get("profitMargins")),
        S.GROSS_MARGIN: _safe_percent(info.get("grossMargins")),
        S.REVENUE_GROWTH: _safe_percent(info.get("revenueGrowth")),
        S.EARNINGS_GROWTH: _safe_percent(info.get("earningsGrowth")),
        S.CURRENT_RATIO: _safe_number(info.get("currentRatio")),
        S.QUICK_RATIO: _safe_number(info.get("quickRatio")),
        S.TOTAL_CASH_CR: _safe_crore(info.get("totalCash")),
        S.TOTAL_DEBT_CR: _safe_crore(info.get("totalDebt")),
        S.FREE_CASHFLOW_CR: _safe_crore(info.get("freeCashflow")),
        S.OPERATING_CASHFLOW_CR: _safe_crore(info.get("operatingCashflow")),
        S.DIVIDEND_YIELD: _safe_dividend_yield(info.get("dividendYield")),
        S.BETA: _safe_number(info.get("beta")),
    }

    row = {
        S.SYMBOL: symbol,
        S.COMPANY_NAME: info.get("longName") or info.get("shortName") or symbol,
        "sector_yf": sector_yf,
        "industry_yf": industry_yf,
        S.SECTOR_BUCKET: sector_bucket(sector_yf, industry_yf),
    }
    for field, value in raw.items():
        row[field] = value
        row[S.known_flag(field)] = value is not None

    return row


def fetch_fundamentals(symbols: list[str]) -> pd.DataFrame:
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_one, s): s for s in symbols}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                row = future.result()
                if row:
                    results.append(row)
            except Exception as e:  # noqa: BLE001
                logger.warning("%s fundamentals failed: %s", symbol, e)
            time.sleep(0.8)
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def _points(value: Optional[float], rules: list[tuple]) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    for condition, points in rules:
        if condition(value):
            return points
    return 0.0


def score_row(row: pd.Series) -> pd.Series:
    def get(field):
        v = row.get(field)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        return float(v)

    roe, roa = get(S.ROE), get(S.ROA)
    opm, npm = get(S.OPERATING_MARGIN), get(S.NET_PROFIT_MARGIN)
    rev_g, earn_g = get(S.REVENUE_GROWTH), get(S.EARNINGS_GROWTH)
    dte, cr, qr = get(S.DEBT_TO_EQUITY), get(S.CURRENT_RATIO), get(S.QUICK_RATIO)
    cash, debt = get(S.TOTAL_CASH_CR), get(S.TOTAL_DEBT_CR)
    ocf, fcf = get(S.OPERATING_CASHFLOW_CR), get(S.FREE_CASHFLOW_CR)
    pe, fpe, pb, div = get(S.TRAILING_PE), get(S.FORWARD_PE), get(S.PRICE_TO_BOOK), get(S.DIVIDEND_YIELD)

    reasons: list[str] = []
    risks: list[str] = []

    profitability = min(FUNDAMENTAL_CAPS.profitability_max,
        _points(roe, [(lambda x: x >= 25, 8), (lambda x: x >= 18, 6), (lambda x: x >= 12, 4), (lambda x: x >= 8, 2)])
        + _points(roa, [(lambda x: x >= 12, 5), (lambda x: x >= 8, 4), (lambda x: x >= 5, 2)])
        + _points(opm, [(lambda x: x >= 25, 6), (lambda x: x >= 18, 5), (lambda x: x >= 10, 3)])
        + _points(npm, [(lambda x: x >= 18, 6), (lambda x: x >= 12, 5), (lambda x: x >= 6, 3)]))
    if roe is not None and roe >= 18:
        reasons.append("Strong ROE")
    if npm is not None and npm < 0:
        risks.append("Negative net margin")

    growth = min(FUNDAMENTAL_CAPS.growth_max,
        _points(rev_g, [(lambda x: x >= 25, 10), (lambda x: x >= 15, 8), (lambda x: x >= 8, 5), (lambda x: x >= 3, 2)])
        + _points(earn_g, [(lambda x: x >= 25, 10), (lambda x: x >= 15, 8), (lambda x: x >= 8, 5), (lambda x: x >= 3, 2)]))
    if rev_g is not None and rev_g < 0:
        risks.append("Revenue contraction")
    if earn_g is not None and earn_g < 0:
        risks.append("Earnings contraction")

    balance_sheet = (
        _points(dte, [(lambda x: x <= 20, 9), (lambda x: x <= 50, 7), (lambda x: x <= 100, 4), (lambda x: x <= 150, 2)])
        + _points(cr, [(lambda x: x >= 2, 5), (lambda x: x >= 1.5, 4), (lambda x: x >= 1, 2)])
        + _points(qr, [(lambda x: x >= 1.5, 3), (lambda x: x >= 1, 2)]))
    if cash is not None and debt is not None:
        if debt <= 0 < cash:
            balance_sheet += 3
            reasons.append("Net cash position")
        elif debt > 0 and cash / debt >= 1:
            balance_sheet += 3
            reasons.append("Cash covers debt")
    balance_sheet = min(FUNDAMENTAL_CAPS.balance_sheet_max, balance_sheet)
    if dte is not None and dte > 200:
        risks.append("Very high debt")

    cashflow = 0.0
    if ocf is not None:
        cashflow += 9 if ocf > 0 else 0
        if ocf < 0:
            risks.append("Negative operating cash flow")
    if fcf is not None:
        cashflow += 8 if fcf > 0 else 0
        if fcf < 0:
            risks.append("Negative free cash flow")
    cashflow = min(FUNDAMENTAL_CAPS.cashflow_max, cashflow)

    valuation = min(FUNDAMENTAL_CAPS.valuation_max,
        _points(pe, [(lambda x: 0 < x <= 20, 6), (lambda x: x <= 35, 4), (lambda x: x <= 60, 2)])
        + (3 if (pe is not None and fpe is not None and 0 < fpe < pe) else 0)
        + _points(pb, [(lambda x: 0 < x <= 3, 4), (lambda x: x <= 6, 2)])
        + _points(div, [(lambda x: x >= 2, 2), (lambda x: x >= 1, 1)]))
    if pe is not None and pe > 80:
        risks.append("Very high PE")

    fields_known = [roe, roa, opm, npm, rev_g, earn_g, dte, cr, qr, ocf, fcf, pe, pb, div]
    coverage = sum(1 for v in fields_known if v is not None) / len(fields_known)

    raw_score = profitability + growth + balance_sheet + cashflow + valuation
    fundamental_score = max(0.0, min(100.0, raw_score))

    return pd.Series({
        S.PROFITABILITY_SCORE: round(profitability, 2),
        S.GROWTH_SCORE: round(growth, 2),
        S.BALANCE_SHEET_SCORE: round(balance_sheet, 2),
        S.CASHFLOW_SCORE: round(cashflow, 2),
        S.VALUATION_SCORE: round(valuation, 2),
        S.FUNDAMENTAL_SCORE: round(fundamental_score, 2),
        S.FUNDAMENTAL_DATA_COVERAGE: round(coverage, 2),
        S.FUNDAMENTAL_REASONS: ", ".join(dict.fromkeys(reasons)),
        S.FUNDAMENTAL_RISK_REASONS: ", ".join(dict.fromkeys(risks)),
    })


def _sector_adjustment_row(row: pd.Series) -> tuple[float, list[str]]:
    """A bank carrying 8x leverage isn't "very high debt" the way a
    consumer-goods company would be -- that's structural to how banks fund
    their balance sheet. Blanket ratio thresholds misprice financials
    without a sector-aware adjustment, which is why this exists. Kept from
    the original repo's genuinely good idea, generalised to fewer, clearer
    rules per bucket instead of one bespoke if-block per sector."""
    bucket = row.get(S.SECTOR_BUCKET, "General")

    def get(field):
        v = row.get(field)
        return None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)

    roe, dte, pb = get(S.ROE), get(S.DEBT_TO_EQUITY), get(S.PRICE_TO_BOOK)
    rev_g, opm = get(S.REVENUE_GROWTH), get(S.OPERATING_MARGIN)

    adj = 0.0
    reasons: list[str] = []

    if bucket in ("Banking", "Financial Services"):
        if dte is not None and dte > 200:
            adj += 6; reasons.append("High leverage is structural for this sector")
        if roe is not None and roe >= 15:
            adj += 6; reasons.append("Strong ROE for a leveraged-balance-sheet business")
        elif roe is not None and roe < 8:
            adj -= 5; reasons.append("Weak ROE for a leveraged-balance-sheet business")
        if pb is not None and pb > 5:
            adj -= 4; reasons.append("Expensive relative to book value for this sector")

    elif bucket == "IT / Technology":
        if roe is not None and roe >= 20:
            adj += 5; reasons.append("Strong ROE for an asset-light business")
        if opm is not None and opm >= 18:
            adj += 5; reasons.append("Strong operating margin")

    elif bucket == "FMCG / Consumer":
        if roe is not None and roe >= 20:
            adj += 5; reasons.append("Strong ROE for a consumer staples business")
        if dte is not None and dte <= 50:
            adj += 3; reasons.append("Low debt, typical of a healthy consumer business")

    elif bucket == "Capital Goods / Infra":
        if rev_g is not None and rev_g >= 15:
            adj += 5; reasons.append("Strong order-book-driven revenue growth")

    else:
        if roe is not None and roe >= 18:
            adj += 2; reasons.append("General quality ROE")
        if dte is not None and dte <= 75:
            adj += 2; reasons.append("General low-debt support")

    return round(max(-15.0, min(15.0, adj)), 2), reasons


def apply_sector_adjustment(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    adjustments = df.apply(_sector_adjustment_row, axis=1, result_type="expand")
    df[S.SECTOR_ADJUSTMENT] = adjustments[0]
    df[S.SECTOR_ADJUSTED_FUNDAMENTAL_SCORE] = (
        df[S.FUNDAMENTAL_SCORE] + df[S.SECTOR_ADJUSTMENT]
    ).clip(0, 100).round(2)
    df["sector_adjustment_reasons"] = adjustments[1].apply(lambda xs: ", ".join(dict.fromkeys(xs)))
    return df


def score_fundamentals(df: pd.DataFrame) -> pd.DataFrame:
    scored = df.apply(score_row, axis=1)
    combined = pd.concat([df, scored], axis=1)
    return apply_sector_adjustment(combined)
