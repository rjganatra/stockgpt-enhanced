"""Every tunable weight, threshold and window in one place.

In the original repo these numbers were scattered as inline literals across
a dozen files (a `15` here for "near 52W low", another `15` there for "risk
penalty for being below 200 DMA", no relation between them). Centralising
them here doesn't change the *values* used at launch (they're carried over
from the original model, since that scoring behaviour is what the user is
already used to reading) but it means every number has one, named,
searchable, documented home instead of an anonymous literal buried in a
lambda.

Nothing in this file is a bound on the *data* (price, market cap, volume) --
those stay fully adaptive per stockgpt/schema.py's design note. Everything
in this file is a ratio, percentage, or weight, which is the one category of
number that's legitimately fine to fix.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Final score blend
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ScoreWeights:
    technical: float = 0.25
    fundamental: float = 0.35
    relative_strength: float = 0.25
    sector: float = 0.15
    # risk_penalty is subtracted, not weighted -- see scoring.py

    def __post_init__(self) -> None:
        total = self.technical + self.fundamental + self.relative_strength + self.sector
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"ScoreWeights must sum to 1.0, got {total}")


SCORE_WEIGHTS = ScoreWeights()


# ---------------------------------------------------------------------------
# Score bands (cutoffs are on the 0-100 final score, always adaptive to the
# *inputs* being percentiles/ratios -- the cutoffs themselves are fixed
# grading lines, same category as exam grade boundaries, and that's fine)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BandCutoffs:
    high_conviction: float = 75.0
    strong: float = 65.0
    watchlist: float = 55.0
    neutral: float = 45.0
    weak: float = 35.0
    # below `weak` = Avoid


BAND_CUTOFFS = BandCutoffs()


# ---------------------------------------------------------------------------
# Technical scoring thresholds
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TechnicalThresholds:
    rsi_healthy_low: float = 50.0
    rsi_healthy_high: float = 70.0
    rsi_recovering_low: float = 45.0
    rsi_oversold: float = 30.0
    near_low_pct: float = 10.0
    near_low_extended_pct: float = 25.0
    near_high_pct: float = 15.0
    volume_strong_ratio: float = 2.0
    volume_expansion_ratio: float = 1.3


TECHNICAL = TechnicalThresholds()


# ---------------------------------------------------------------------------
# Risk model -- technical + fundamental combined into ONE penalty.
# (The original repo computed this combined penalty correctly in code, then
# lost it to a duplicate-column bug before it ever reached the CSV. Here
# there is exactly one risk_penalty column and exactly one function that
# writes it -- see scoring.py::compute_risk_penalty.)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RiskPenalties:
    below_sma200: float = 8.0
    extreme_rsi_weakness: float = 8.0
    rsi_extreme_threshold: float = 25.0
    sharp_daily_fall: float = 5.0
    sharp_fall_threshold_pct: float = -5.0
    far_from_high: float = 5.0
    far_from_high_threshold_pct: float = 65.0
    low_volume_participation: float = 3.0
    low_volume_ratio_threshold: float = 0.5

    very_high_debt: float = 8.0
    debt_to_equity_threshold: float = 200.0
    negative_net_margin: float = 8.0
    revenue_contraction: float = 5.0
    revenue_contraction_threshold_pct: float = -15.0
    earnings_contraction: float = 7.0
    earnings_contraction_threshold_pct: float = -15.0
    negative_operating_cashflow: float = 5.0
    negative_free_cashflow: float = 4.0

    # Unknown-data penalty: the original repo let a missing ratio silently
    # read as "safe" (0 = no risk). Here, a fundamental section with too
    # little coverage adds a small explicit penalty instead, because "we
    # don't know" should never score better than "we checked and it's fine".
    low_data_coverage_penalty: float = 4.0
    low_data_coverage_threshold: float = 0.5   # < 50% of ratio fields known

    max_total: float = 50.0


RISK = RiskPenalties()


# ---------------------------------------------------------------------------
# Fundamental scoring
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FundamentalCaps:
    profitability_max: float = 25.0
    growth_max: float = 20.0
    balance_sheet_max: float = 20.0
    cashflow_max: float = 20.0
    valuation_max: float = 15.0
    sector_adjustment_max: float = 15.0


FUNDAMENTAL_CAPS = FundamentalCaps()


# ---------------------------------------------------------------------------
# Windows (trading days)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Windows:
    days_1m: int = 21
    days_3m: int = 63
    days_6m: int = 126
    sma_short: int = 50
    sma_long: int = 200
    volume_avg: int = 20
    range_lookback_days: int = 120
    range_min_history_days: int = 80


WINDOWS = Windows()


# ---------------------------------------------------------------------------
# Backtest engine defaults
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BacktestDefaults:
    fixed_holding_days: tuple = (7, 15, 30, 60)
    win_return_threshold_pct: float = 0.0   # return > this counts as a win
    min_signals_for_confidence: int = 10    # below this, flag as low-sample
    # A symbol's day_change_pct diverging from that day's cross-sectional
    # median move by at least this many percentage points is treated as a
    # probable corporate action (demerger, stock split not reflected in
    # current_price, etc.) rather than a genuine price move -- see
    # backtest/corporate_actions.py's module docstring for the full
    # reasoning and why this is a market-relative test, not a raw
    # percentage cutoff. Chosen conservatively: comfortably above NSE's
    # widest routine circuit limits (20%) so it doesn't flag ordinary
    # volatile trading days, while still catching real demerger-sized drops
    # (commonly 20-60%+) even on days when the broader market also moved.
    price_jump_threshold_pct: float = 35.0


BACKTEST_DEFAULTS = BacktestDefaults()
