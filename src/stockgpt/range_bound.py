"""Range-bound / mean-reversion scanner.

Support/resistance bands are the 10th/90th percentile of a stock's OWN
closing-price history (not a fixed rupee level), so a Rs.50 stock and a
Rs.1.5L stock each get bands sized to their own price scale automatically --
this part of the original design was already correctly adaptive and is
kept as-is.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from . import schema as S
from .config import WINDOWS
from .scanner import compute_rsi


def _clip(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def analyse(symbol: str, hist: pd.DataFrame, technical_score: float,
            risk_penalty: float) -> Optional[dict]:
    hist = hist.tail(WINDOWS.range_lookback_days).dropna(subset=["Close"])
    if len(hist) < WINDOWS.range_min_history_days:
        return None

    close, high, low, volume = hist["Close"], hist["High"], hist["Low"], hist["Volume"]
    current_price = float(close.iloc[-1])

    range_low = float(close.quantile(0.10))
    range_high = float(close.quantile(0.90))
    if range_low <= 0 or range_high <= range_low or current_price <= 0:
        return None

    width_pct = ((range_high - range_low) / range_low) * 100
    position_pct = _clip(((current_price - range_low) / (range_high - range_low)) * 100)
    inside_pct = (((close >= range_low) & (close <= range_high)).sum() / len(close)) * 100

    avg_vol_20 = float(volume.tail(20).mean() or 0)
    latest_vol = float(volume.iloc[-1])
    volume_ratio = latest_vol / avg_vol_20 if avg_vol_20 > 0 else 1.0
    rsi = float(compute_rsi(close).iloc[-1])

    stability = _clip(((inside_pct - 55) / (85 - 55)) * 100) if 85 != 55 else 0
    is_breakdown = current_price < range_low * 0.97 and volume_ratio >= 1.3
    is_breakout = current_price > range_high * 1.03 and volume_ratio >= 1.3

    quality_component = _clip(technical_score * 0.6 + (100 - _clip(risk_penalty, 0, 100)) * 0.4)
    range_score = _clip(stability * 0.35 + quality_component * 0.35 +
                         (100 - min(width_pct, 100)) * 0.15 +
                         (100 - position_pct if position_pct <= 50 else position_pct) * 0.15)

    if is_breakdown:
        status, range_score = "Breakdown Risk", min(range_score, 35)
    elif is_breakout:
        status, range_score = "Breakout Watch", max(range_score, 55)
    elif stability < 45:
        status, range_score = "Not Range Bound", min(range_score, 50)
    elif position_pct <= 30 and range_score >= 65:
        status = "Accumulation Zone"
    elif position_pct <= 45:
        status = "Lower Range Watch"
    elif position_pct >= 75:
        status = "Profit Booking Zone"
    else:
        status = "Neutral Range"

    reasons = [
        f"{inside_pct:.1f}% of closes stayed inside the range",
        f"Range width {width_pct:.1f}%",
        f"Currently {position_pct:.1f}% through the range",
    ]
    if is_breakdown:
        reasons.append("Below lower band on elevated volume")
    if is_breakout:
        reasons.append("Above upper band on elevated volume")

    return {
        S.SYMBOL: symbol,
        S.RANGE_STATUS: status,
        S.RANGE_SCORE: round(range_score, 2),
        S.RANGE_LOW: round(range_low, 2),
        S.RANGE_HIGH: round(range_high, 2),
        S.RANGE_WIDTH_PCT: round(width_pct, 2),
        S.RANGE_POSITION_PCT: round(position_pct, 2),
        S.RANGE_REASONS: " | ".join(reasons),
    }
