"""Relative strength vs Nifty 50, computed once per stock per scan.

Kept from the original: 1M/3M/6M absolute + vs-Nifty returns, sector rank.
Cleaned up: if the Nifty benchmark fetch fails, that's tracked explicitly
(NIFTY_RETURN_* stays NaN and the vs-Nifty columns stay NaN too, rather than
silently defaulting the comparison to 0 -- 0 excess return would read as
"exactly matched the index", which is a specific, wrong claim to make about
a stock we simply couldn't benchmark).
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from . import schema as S
from .config import WINDOWS


def calc_return(close: pd.Series, days: int) -> Optional[float]:
    close = close.dropna()
    if len(close) <= days:
        return None
    past = close.iloc[-days]
    if past == 0:
        return None
    return round(((close.iloc[-1] - past) / past) * 100, 2)


def relative_strength_score(return_1m, return_3m, return_6m,
                             vs_nifty_1m, vs_nifty_3m, vs_nifty_6m) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []

    if return_1m is not None:
        if return_1m > 10:
            score += 15; reasons.append("Strong 1M absolute return")
        elif return_1m > 0:
            score += 8; reasons.append("Positive 1M return")
    if return_3m is not None:
        if return_3m > 20:
            score += 20; reasons.append("Strong 3M absolute return")
        elif return_3m > 0:
            score += 10; reasons.append("Positive 3M return")
    if return_6m is not None:
        if return_6m > 30:
            score += 20; reasons.append("Strong 6M absolute return")
        elif return_6m > 0:
            score += 10; reasons.append("Positive 6M return")

    if vs_nifty_1m is not None and vs_nifty_1m > 5:
        score += 10; reasons.append("1M Nifty outperformance")
    if vs_nifty_3m is not None and vs_nifty_3m > 10:
        score += 15; reasons.append("3M Nifty outperformance")
    if vs_nifty_6m is not None and vs_nifty_6m > 15:
        score += 15; reasons.append("6M Nifty outperformance")

    if all(r is not None and r > 0 for r in (return_1m, return_3m, return_6m)):
        score += 10; reasons.append("Positive across 1M/3M/6M")

    return round(min(100.0, score), 2), reasons


def compute_for_symbol(symbol: str, close: pd.Series, sector: str,
                        nifty_1m: Optional[float], nifty_3m: Optional[float],
                        nifty_6m: Optional[float]) -> dict:
    r1m = calc_return(close, WINDOWS.days_1m)
    r3m = calc_return(close, WINDOWS.days_3m)
    r6m = calc_return(close, WINDOWS.days_6m)

    vs1 = round(r1m - nifty_1m, 2) if r1m is not None and nifty_1m is not None else None
    vs3 = round(r3m - nifty_3m, 2) if r3m is not None and nifty_3m is not None else None
    vs6 = round(r6m - nifty_6m, 2) if r6m is not None and nifty_6m is not None else None

    score, reasons = relative_strength_score(r1m, r3m, r6m, vs1, vs3, vs6)

    return {
        S.SYMBOL: symbol,
        S.SECTOR: sector,
        S.RETURN_1M: r1m, S.RETURN_3M: r3m, S.RETURN_6M: r6m,
        S.NIFTY_RETURN_1M: nifty_1m, S.NIFTY_RETURN_3M: nifty_3m, S.NIFTY_RETURN_6M: nifty_6m,
        S.RETURN_VS_NIFTY_1M: vs1, S.RETURN_VS_NIFTY_3M: vs3, S.RETURN_VS_NIFTY_6M: vs6,
        S.RELATIVE_STRENGTH_SCORE: score,
        "relative_strength_reasons": ", ".join(reasons),
    }


def add_sector_rank(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[S.SECTOR_RANK_PCT] = (
        df.groupby(S.SECTOR)[S.RELATIVE_STRENGTH_SCORE]
        .rank(ascending=False, pct=True)
        .mul(100).round(2)
    )
    return df
