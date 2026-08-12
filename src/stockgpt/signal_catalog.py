"""The original repo's ~12 built-in signal types (`app/performance/signal_performance.py`
::detect_signals), reproduced here as proper Strategy objects instead of a hardcoded
detection function -- so their historical win rate runs through the SAME backtest
engine as any custom Strategy Lab query, rather than a second, weaker measurement
method.

Two real differences from the original, both improvements, both worth knowing about:

1. The original measured every signal's return against *today's* price, then
   bucketed the elapsed days into a loose horizon_bucket (1D/3D/7D/15D/30D/60D+).
   A signal from 47 days ago landed in the "30D" bucket even though its actual
   holding period was 47 days, not 30 -- the bucket label didn't match the real
   elapsed time. Here, every signal in this catalog uses FIXED_HOLDING at exact
   day offsets (see engine.py), so a "15D" row is genuinely a 15-trading-day
   holding period, always.

2. "Top Final Conviction" (the original's top-25-by-score-that-day signal) is a
   RANKING, not a threshold -- "how does this stock rank against its peers today"
   isn't expressible as a per-row pandas query, which is how every Strategy here
   (and in Strategy Lab) is defined. It's dropped from this catalog rather than
   faked with a fixed threshold that would silently stop being "top 25" the
   moment the universe size changes. "High Conviction" below (final_score >= 65)
   covers materially the same intent with a threshold that stays meaningful as
   the universe grows or shrinks.

SIGNAL_DIRECTIONS matters for reading the win rates: a "Bearish" signal is a
warning, not a buy. For those rows, a NEGATIVE return_pct is the signal being
right, so win_rate_pct as the backtest engine reports it (return > threshold)
reads backwards for bearish rows -- the dashboard's Signal Performance tab
flags this explicitly rather than silently flipping the number.
"""

from __future__ import annotations

from .backtest.strategy import ExitMode, Strategy

_HOLDING_DAYS = (7, 15, 30, 60)

SIGNAL_CATALOG: list[Strategy] = [
    Strategy(
        name="High Conviction",
        entry_query="final_score >= 65",
        exit_mode=ExitMode.FIXED_HOLDING,
        fixed_holding_days=_HOLDING_DAYS,
        description="final_score >= 65 (covers the original's 'Top Final Conviction' "
                     "and 'High Conviction' signals with one threshold-based rule).",
    ),
    Strategy(
        name="52W Low Opportunity",
        entry_query="distance_from_low_pct <= 15",
        exit_mode=ExitMode.FIXED_HOLDING,
        fixed_holding_days=_HOLDING_DAYS,
        description="Within 15% of the 52-week low.",
    ),
    Strategy(
        name="Swing Candidate",
        entry_query="distance_from_low_pct <= 25 and rsi <= 45 and volume_ratio >= 1.0",
        exit_mode=ExitMode.FIXED_HOLDING,
        fixed_holding_days=_HOLDING_DAYS,
        description="Off the low, RSI still soft, volume confirming interest.",
    ),
    Strategy(
        name="Near 52W High Momentum",
        entry_query="distance_from_high_pct <= 15 and rsi >= 50 and trend == 'Bullish'",
        exit_mode=ExitMode.FIXED_HOLDING,
        fixed_holding_days=_HOLDING_DAYS,
        description="Within 15% of the 52-week high, RSI healthy, trend bullish.",
    ),
    Strategy(
        name="Strong Fundamentals",
        entry_query="sector_adjusted_fundamental_score >= 60",
        exit_mode=ExitMode.FIXED_HOLDING,
        fixed_holding_days=_HOLDING_DAYS,
        description="Sector-adjusted fundamental score >= 60.",
    ),
    Strategy(
        name="Relative Strength Leader",
        entry_query="relative_strength_score >= 60",
        exit_mode=ExitMode.FIXED_HOLDING,
        fixed_holding_days=_HOLDING_DAYS,
        description="Outperforming on 1M/3M/6M returns.",
    ),
    Strategy(
        name="Low Risk Quality",
        entry_query="risk_penalty <= 10 and final_score >= 50 and sector_adjusted_fundamental_score >= 50",
        exit_mode=ExitMode.FIXED_HOLDING,
        fixed_holding_days=_HOLDING_DAYS,
        description="Decent score, decent fundamentals, low risk penalty.",
    ),
    Strategy(
        name="Range Accumulation Zone",
        entry_query="range_status in ['Accumulation Zone', 'Lower Range Watch']",
        exit_mode=ExitMode.FIXED_HOLDING,
        fixed_holding_days=_HOLDING_DAYS,
        description="Lower third of its own adaptive range, stable.",
    ),
    Strategy(
        name="Range Profit Booking Zone",
        entry_query="range_status == 'Profit Booking Zone'",
        exit_mode=ExitMode.FIXED_HOLDING,
        fixed_holding_days=_HOLDING_DAYS,
        description="BEARISH warning: upper third of its range -- a falling price after "
                     "this signal is the warning being correct, not a loss.",
    ),
    Strategy(
        name="Range Breakdown Risk",
        entry_query="range_status == 'Breakdown Risk'",
        exit_mode=ExitMode.FIXED_HOLDING,
        fixed_holding_days=_HOLDING_DAYS,
        description="BEARISH warning: broke below its own range on elevated volume.",
    ),
    Strategy(
        name="Avoid / Risky",
        entry_query="score_band == 'Avoid' or final_score < 35 or risk_penalty >= 25 or "
                    "(sector_adjusted_fundamental_score < 30 and relative_strength_score < 40)",
        exit_mode=ExitMode.FIXED_HOLDING,
        fixed_holding_days=_HOLDING_DAYS,
        description="BEARISH warning: the model's own 'stay away' signal.",
    ),
]

SIGNAL_DIRECTIONS: dict[str, str] = {
    "High Conviction": "Bullish",
    "52W Low Opportunity": "Bullish",
    "Swing Candidate": "Bullish",
    "Near 52W High Momentum": "Bullish",
    "Strong Fundamentals": "Bullish",
    "Relative Strength Leader": "Bullish",
    "Low Risk Quality": "Bullish",
    "Range Accumulation Zone": "Bullish",
    "Range Profit Booking Zone": "Bearish",
    "Range Breakdown Risk": "Bearish",
    "Avoid / Risky": "Bearish",
}
