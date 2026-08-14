"""Which sectors are gaining or losing conviction over time.

The Sectors tab's existing table is a single-day snapshot -- it can tell you
today's average final_score per sector, but not whether that sector has been
climbing or fading. This module answers that, using the same daily history
panel the backtest engine already reads (`data/history/*/scan.csv`): split
the available days into a "recent" window and the "prior" window right
before it, and compare each sector's average final_score between the two.
"""

from __future__ import annotations

import pandas as pd

from . import schema as S


def compute_sector_rotation(panel: pd.DataFrame, recent_days: int = 14,
                             prior_days: int = 14) -> pd.DataFrame:
    """Returns one row per sector: recent_avg_final_score, prior_avg_final_score,
    change, change_pct -- sorted by change descending (biggest gainers first).

    Adaptive to however much history actually exists: if there isn't enough
    for the full recent_days + prior_days window, it splits whatever days
    ARE available roughly in half rather than returning nothing -- a
    smaller, honest comparison beats no comparison at all. Returns an empty
    DataFrame only when there are fewer than 2 distinct scan dates (nothing
    to compare) or the required columns aren't present."""
    if panel.empty or S.SECTOR not in panel.columns or S.SCAN_DATE not in panel.columns \
            or S.FINAL_SCORE not in panel.columns:
        return pd.DataFrame()

    dates = sorted(panel[S.SCAN_DATE].unique())
    if len(dates) < 2:
        return pd.DataFrame()

    window = recent_days + prior_days
    if len(dates) >= window:
        recent_dates = dates[-recent_days:]
        prior_dates = dates[-window:-recent_days]
    else:
        # Not enough history for the requested window -- split whatever
        # exists roughly in half instead of failing outright.
        mid = max(1, len(dates) // 2)
        prior_dates = dates[:mid]
        recent_dates = dates[mid:]

    if not prior_dates or not recent_dates:
        return pd.DataFrame()

    recent = panel[panel[S.SCAN_DATE].isin(recent_dates)]
    prior = panel[panel[S.SCAN_DATE].isin(prior_dates)]

    recent_avg = recent.groupby(S.SECTOR, dropna=False)[S.FINAL_SCORE].mean().rename("recent_avg_final_score")
    prior_avg = prior.groupby(S.SECTOR, dropna=False)[S.FINAL_SCORE].mean().rename("prior_avg_final_score")

    result = pd.concat([recent_avg, prior_avg], axis=1).dropna()
    if result.empty:
        return pd.DataFrame()

    result["change"] = result["recent_avg_final_score"] - result["prior_avg_final_score"]
    # Guard divide-by-zero: a sector whose prior average was exactly 0 gets
    # change_pct left as NaN rather than an infinite/undefined percentage.
    result["change_pct"] = result["change"] / result["prior_avg_final_score"].replace(0, pd.NA) * 100

    result = result.reset_index().rename(columns={S.SECTOR: S.SECTOR})
    result = result.sort_values("change", ascending=False).reset_index(drop=True)
    return result.round(2)
