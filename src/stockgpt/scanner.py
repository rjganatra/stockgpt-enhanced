"""Technical scanner: price/volume/RSI/SMA-derived signals for one score.

The original repo computed a technical score TWICE -- once in scan_52w.py
(v1: trend_score/momentum_score/reversal_score/volume_score) and again in
score_engine.py (v2: a differently-weighted technical_score). Both columns
shipped to the final CSV; v1 became dead weight nobody read. Here there is
one technical score, computed once, in this file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from . import schema as S
from .config import TECHNICAL, WINDOWS


@dataclass
class TechnicalResult:
    row: dict
    ok: bool
    reason: str = ""


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI. Pure pandas, no external TA dependency required."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(100)  # avg_loss == 0 means pure gains -> RSI 100


def technical_score(rsi: float, above_sma50: bool, above_sma200: Optional[bool],
                     distance_from_low_pct: float, distance_from_high_pct: float,
                     volume_ratio: float, day_change_pct: float) -> tuple[float, list[str]]:
    """0-100 technical score. Returns (score, reasons)."""
    score = 0.0
    reasons: list[str] = []

    # Trend quality -- 30 pts
    if above_sma50:
        score += 15
        reasons.append("Above 50 DMA")
    if above_sma200 is True:
        score += 15
        reasons.append("Above 200 DMA")

    # Momentum health -- 20 pts
    if TECHNICAL.rsi_healthy_low <= rsi <= TECHNICAL.rsi_healthy_high:
        score += 20
        reasons.append("Healthy RSI momentum")
    elif TECHNICAL.rsi_recovering_low <= rsi < TECHNICAL.rsi_healthy_low:
        score += 12
        reasons.append("RSI recovering")
    elif TECHNICAL.rsi_oversold <= rsi < TECHNICAL.rsi_recovering_low:
        score += 8
        reasons.append("Weak but watchable RSI")
    elif rsi > TECHNICAL.rsi_healthy_high:
        score += 7
        reasons.append("Strong but overbought RSI")
    elif rsi < TECHNICAL.rsi_oversold:
        score += 4
        reasons.append("Oversold RSI")

    # Price location -- 20 pts
    if distance_from_high_pct <= TECHNICAL.near_high_pct and rsi >= 50:
        score += 10
        reasons.append("Near 52W high momentum")
    if distance_from_low_pct <= TECHNICAL.near_low_pct and rsi >= 30:
        score += 8
        reasons.append("Near 52W low with stable RSI")
    elif distance_from_low_pct <= TECHNICAL.near_low_extended_pct:
        score += 5
        reasons.append("Near 52W low zone")

    # Volume confirmation -- 20 pts
    if volume_ratio >= TECHNICAL.volume_strong_ratio:
        score += 15
        reasons.append("Strong volume expansion")
    elif volume_ratio >= TECHNICAL.volume_expansion_ratio:
        score += 10
        reasons.append("Volume expansion")
    elif volume_ratio >= 1:
        score += 5
        reasons.append("Normal volume support")

    # Daily confirmation -- 10 pts (bumped from the original's 5 pt slot,
    # since the original left 5 pts of headroom unassigned in the 100 pt
    # budget -- 15+20+18+15 = 68 max previously reachable, so it never hit
    # 100 even in a perfect setup. This version's max is genuinely 100.)
    if day_change_pct > 2:
        score += 10
        reasons.append("Strong positive day move")
    elif day_change_pct > 0:
        score += 5
        reasons.append("Positive day move")

    return round(min(100.0, score), 2), reasons


def process_history(symbol: str, hist: pd.DataFrame, sector: str, industry: str) -> Optional[dict]:
    """hist must have columns: Close, High, Low, Volume, sorted ascending by date."""
    hist = hist.dropna(subset=["Close"]).copy()
    if len(hist) < 60:
        return None

    close = hist["Close"]
    current_price = float(close.iloc[-1])
    previous_close = float(close.iloc[-2])
    low_52w = float(hist["Low"].min())
    high_52w = float(hist["High"].max())

    if low_52w <= 0 or high_52w <= 0 or previous_close <= 0:
        return None

    day_change_pct = ((current_price - previous_close) / previous_close) * 100
    distance_from_low_pct = ((current_price - low_52w) / low_52w) * 100
    distance_from_high_pct = ((high_52w - current_price) / high_52w) * 100

    rsi_series = compute_rsi(close)
    rsi = float(rsi_series.iloc[-1])
    if pd.isna(rsi):
        return None

    sma50 = float(close.rolling(WINDOWS.sma_short).mean().iloc[-1])
    sma200_series = close.rolling(WINDOWS.sma_long).mean()
    sma200 = float(sma200_series.iloc[-1]) if len(hist) >= WINDOWS.sma_long else None
    sma200_known = sma200 is not None and not pd.isna(sma200)
    if not sma200_known:
        sma200 = None

    avg_volume_20d = float(hist["Volume"].tail(WINDOWS.volume_avg).mean())
    latest_volume = float(hist["Volume"].iloc[-1])
    volume_ratio = latest_volume / avg_volume_20d if avg_volume_20d > 0 else 0.0

    above_sma50 = current_price > sma50
    above_sma200 = (current_price > sma200) if sma200_known else None
    trend = "Bullish" if above_sma50 else "Bearish"

    score, reasons = technical_score(
        rsi=rsi, above_sma50=above_sma50, above_sma200=above_sma200,
        distance_from_low_pct=distance_from_low_pct,
        distance_from_high_pct=distance_from_high_pct,
        volume_ratio=volume_ratio, day_change_pct=day_change_pct,
    )

    return {
        S.SYMBOL: symbol,
        S.SECTOR: sector,
        S.INDUSTRY: industry,
        S.CURRENT_PRICE: round(current_price, 2),
        S.DAY_CHANGE_PCT: round(day_change_pct, 2),
        S.LOW_52W: round(low_52w, 2),
        S.HIGH_52W: round(high_52w, 2),
        S.DISTANCE_FROM_LOW_PCT: round(distance_from_low_pct, 2),
        S.DISTANCE_FROM_HIGH_PCT: round(distance_from_high_pct, 2),
        S.RSI: round(rsi, 2),
        S.SMA50: round(sma50, 2),
        S.SMA200: round(sma200, 2) if sma200_known else np.nan,
        S.AVG_VOLUME_20D: round(avg_volume_20d, 2),
        S.LATEST_VOLUME: round(latest_volume, 2),
        S.VOLUME_RATIO: round(volume_ratio, 2),
        S.TREND: trend,
        S.TECHNICAL_SCORE: score,
        S.TECHNICAL_REASONS: ", ".join(reasons),
    }
