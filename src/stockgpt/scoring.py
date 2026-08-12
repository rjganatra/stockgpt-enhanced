"""Final conviction score: the single authoritative blend.

This is the module that fixes the original repo's most serious bug. There,
`score_engine.py` computed a combined technical+fundamental risk penalty
correctly, then did:

    df = pd.concat([df, risk_df], axis=1)          # risk_df has 'risk_penalty'
    df = df.loc[:, ~df.columns.duplicated()]        # keeps the FIRST occurrence

...but `df` already had an OLDER, purely-technical `risk_penalty` column
from an earlier pipeline stage. `duplicated()` defaults to `keep='first'`,
so the newly-computed, fundamentals-aware risk penalty was silently
discarded on every single run, while its accompanying text column
(`risk_reasons`, which had no naming collision) survived untouched. The
result: a stock's displayed "risk reasons" could say "Very high debt,
Negative net margin" while its actual `risk_penalty` number -- the one
subtracted from the final score -- was 0, because it only reflected
technical factors. Verified against a live run: UFBL had debt/equity 275
and -2.7% net margin, both correctly flagged in the reasons text, with
risk_penalty == 0.

Here there is exactly one `compute_risk_penalty` function, it runs once,
and its output is assigned directly (`df[S.RISK_PENALTY] = ...`), never
concatenated against a same-named column that could shadow it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import schema as S
from .config import BAND_CUTOFFS, RISK, SCORE_WEIGHTS


def _num(row: pd.Series, field: str) -> float | None:
    v = row.get(field)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return float(v)


def compute_risk_penalty(row: pd.Series) -> tuple[float, list[str]]:
    penalty = 0.0
    reasons: list[str] = []

    price = _num(row, S.CURRENT_PRICE) or 0.0
    sma200 = _num(row, S.SMA200)
    rsi = _num(row, S.RSI) or 50.0
    day_change = _num(row, S.DAY_CHANGE_PCT) or 0.0
    dist_from_high = _num(row, S.DISTANCE_FROM_HIGH_PCT) or 0.0
    volume_ratio = _num(row, S.VOLUME_RATIO)

    # --- technical risk ---
    if sma200 is not None and price < sma200:
        penalty += RISK.below_sma200
        reasons.append("Below 200 DMA")
    if rsi < RISK.rsi_extreme_threshold:
        penalty += RISK.extreme_rsi_weakness
        reasons.append("Extreme RSI weakness")
    if day_change < RISK.sharp_fall_threshold_pct:
        penalty += RISK.sharp_daily_fall
        reasons.append("Sharp daily fall")
    if dist_from_high > RISK.far_from_high_threshold_pct:
        penalty += RISK.far_from_high
        reasons.append("Far from 52W high")
    if volume_ratio is not None and volume_ratio < RISK.low_volume_ratio_threshold:
        penalty += RISK.low_volume_participation
        reasons.append("Low volume participation")

    # --- fundamental risk ---
    dte = _num(row, S.DEBT_TO_EQUITY)
    npm = _num(row, S.NET_PROFIT_MARGIN)
    rev_g = _num(row, S.REVENUE_GROWTH)
    earn_g = _num(row, S.EARNINGS_GROWTH)
    ocf = _num(row, S.OPERATING_CASHFLOW_CR)
    fcf = _num(row, S.FREE_CASHFLOW_CR)

    if dte is not None and dte > RISK.debt_to_equity_threshold:
        penalty += RISK.very_high_debt
        reasons.append("Very high debt")
    if npm is not None and npm < 0:
        penalty += RISK.negative_net_margin
        reasons.append("Negative net margin")
    if rev_g is not None and rev_g < RISK.revenue_contraction_threshold_pct:
        penalty += RISK.revenue_contraction
        reasons.append("Revenue contraction")
    if earn_g is not None and earn_g < RISK.earnings_contraction_threshold_pct:
        penalty += RISK.earnings_contraction
        reasons.append("Earnings contraction")
    if ocf is not None and ocf < 0:
        penalty += RISK.negative_operating_cashflow
        reasons.append("Negative operating cash flow")
    if fcf is not None and fcf < 0:
        penalty += RISK.negative_free_cashflow
        reasons.append("Negative free cash flow")

    # --- unknown-data penalty: missing data must never score as "safe" ---
    coverage = _num(row, S.FUNDAMENTAL_DATA_COVERAGE)
    if coverage is not None and coverage < RISK.low_data_coverage_threshold:
        penalty += RISK.low_data_coverage_penalty
        reasons.append("Limited fundamental data coverage")

    return round(min(RISK.max_total, penalty), 2), reasons


def classify_band(score: float) -> str:
    if score >= BAND_CUTOFFS.high_conviction:
        return S.BAND_HIGH_CONVICTION
    if score >= BAND_CUTOFFS.strong:
        return S.BAND_STRONG
    if score >= BAND_CUTOFFS.watchlist:
        return S.BAND_WATCHLIST
    if score >= BAND_CUTOFFS.neutral:
        return S.BAND_NEUTRAL
    if score >= BAND_CUTOFFS.weak:
        return S.BAND_WEAK
    return S.BAND_AVOID


def compute_sector_score(df: pd.DataFrame) -> pd.DataFrame:
    """Sector score = sector-average of technical/relative/fundamental,
    joined back with a plain merge (not concat) so there is no risk of a
    duplicate-column shadow like the bug this module fixes."""
    sector_avg = df.groupby(S.SECTOR, dropna=False).agg(**{
        "_sector_avg_technical": (S.TECHNICAL_SCORE, "mean"),
        "_sector_avg_relative": (S.RELATIVE_STRENGTH_SCORE, "mean"),
        "_sector_avg_fundamental": (S.SECTOR_ADJUSTED_FUNDAMENTAL_SCORE, "mean"),
    }).reset_index()

    sector_avg[S.SECTOR_SCORE] = (
        sector_avg["_sector_avg_technical"] * 0.35
        + sector_avg["_sector_avg_relative"] * 0.35
        + sector_avg["_sector_avg_fundamental"] * 0.30
    ).round(2)

    out = df.merge(sector_avg[[S.SECTOR, S.SECTOR_SCORE]], on=S.SECTOR, how="left")
    out[S.SECTOR_SCORE] = out[S.SECTOR_SCORE].fillna(0.0)
    return out


def compute_final_scores(df: pd.DataFrame) -> pd.DataFrame:
    """df must already have technical_score, sector_adjusted_fundamental_score,
    relative_strength_score populated (0 where genuinely absent, per each
    upstream module's own contract)."""
    df = df.copy()

    for col in (S.TECHNICAL_SCORE, S.SECTOR_ADJUSTED_FUNDAMENTAL_SCORE, S.RELATIVE_STRENGTH_SCORE):
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df = compute_sector_score(df)

    risk_results = df.apply(compute_risk_penalty, axis=1, result_type="expand")
    df[S.RISK_PENALTY] = risk_results[0]
    df[S.RISK_REASONS] = risk_results[1].apply(lambda xs: ", ".join(dict.fromkeys(xs)))

    df[S.FINAL_SCORE] = (
        df[S.TECHNICAL_SCORE] * SCORE_WEIGHTS.technical
        + df[S.SECTOR_ADJUSTED_FUNDAMENTAL_SCORE] * SCORE_WEIGHTS.fundamental
        + df[S.RELATIVE_STRENGTH_SCORE] * SCORE_WEIGHTS.relative_strength
        + df[S.SECTOR_SCORE] * SCORE_WEIGHTS.sector
        - df[S.RISK_PENALTY]
    ).clip(0, 100).round(2)

    df[S.SCORE_BAND] = df[S.FINAL_SCORE].apply(classify_band)

    reason_cols = [S.TECHNICAL_REASONS, S.FUNDAMENTAL_REASONS, "relative_strength_reasons", S.RANGE_REASONS]
    present = [c for c in reason_cols if c in df.columns]
    if present:
        df[S.REASONS] = df[present].fillna("").agg(
            lambda parts: ", ".join([p for p in parts if p]), axis=1
        )

    return df.sort_values(S.FINAL_SCORE, ascending=False).reset_index(drop=True)
