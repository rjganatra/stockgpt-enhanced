"""NSE equity universe fetch, with a small built-in fallback list.

Same idea as the original repo's fetch_nifty500.py (NSE archives can rate
limit or go down), rewritten cleanly: fetch -> validate row count -> on any
failure, fall back to a known-good static list rather than writing an empty
or tiny universe that would silently starve the rest of the pipeline.
"""

from __future__ import annotations

import logging
from io import StringIO

import pandas as pd
import requests

from . import schema as S

logger = logging.getLogger(__name__)

NSE_EQUITY_URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
NIFTY_500_URL = "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "text/csv,application/csv,text/plain,*/*",
    "Referer": "https://www.nseindia.com/",
}
MIN_ACCEPTABLE_UNIVERSE_SIZE = 800

FALLBACK_UNIVERSE = [
    ("RELIANCE", "Energy"), ("TCS", "IT"), ("INFY", "IT"),
    ("HDFCBANK", "Financial Services"), ("ICICIBANK", "Financial Services"),
    ("SBIN", "Financial Services"), ("LT", "Capital Goods"), ("ITC", "FMCG"),
    ("KOTAKBANK", "Financial Services"), ("AXISBANK", "Financial Services"),
    ("BHARTIARTL", "Telecommunication"), ("HINDUNILVR", "FMCG"),
    ("ASIANPAINT", "Consumer Durables"), ("BAJFINANCE", "Financial Services"),
    ("MARUTI", "Automobile"), ("TITAN", "Consumer Durables"),
    ("SUNPHARMA", "Healthcare"), ("ULTRACEMCO", "Cement"),
    ("NESTLEIND", "FMCG"), ("WIPRO", "IT"), ("ONGC", "Energy"),
    ("NTPC", "Power"), ("POWERGRID", "Power"), ("BEL", "Capital Goods"),
    ("HAL", "Capital Goods"), ("TRENT", "Retail"), ("DMART", "Retail"),
    ("PIDILITIND", "Chemicals"), ("ADANIPORTS", "Services"),
    ("COALINDIA", "Mining"), ("TATASTEEL", "Metals"),
]


def _fetch_csv(url: str) -> pd.DataFrame:
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return pd.read_csv(StringIO(response.text))


def fallback_universe() -> pd.DataFrame:
    symbols, sectors = zip(*FALLBACK_UNIVERSE)
    return pd.DataFrame({
        S.SYMBOL: symbols,
        S.SECTOR: sectors,
        S.COMPANY_NAME: symbols,
    })


def fetch_universe() -> tuple[pd.DataFrame, bool]:
    """Returns (universe_df, used_fallback)."""
    try:
        df = _fetch_csv(NSE_EQUITY_URL)
        df.columns = [str(c).strip().upper() for c in df.columns]
        df = df.rename(columns={
            "SYMBOL": S.SYMBOL,
            "NAME OF COMPANY": S.COMPANY_NAME,
            "SERIES": "series",
        })
        df = df[df["series"].astype(str).str.upper().str.strip() == "EQ"]
        df[S.SYMBOL] = df[S.SYMBOL].astype(str).str.strip()
        df[S.COMPANY_NAME] = df[S.COMPANY_NAME].astype(str).str.strip()

        try:
            sector_df = _fetch_csv(NIFTY_500_URL)
            sector_map = dict(zip(sector_df["Symbol"], sector_df["Industry"]))
        except Exception as e:  # noqa: BLE001 - sector enrichment is best-effort
            logger.warning("Nifty 500 sector map fetch failed: %s", e)
            sector_map = {}

        df[S.SECTOR] = df[S.SYMBOL].map(sector_map).fillna("Unknown")
        df = df.drop_duplicates(subset=[S.SYMBOL])[[S.SYMBOL, S.SECTOR, S.COMPANY_NAME]]

        if len(df) < MIN_ACCEPTABLE_UNIVERSE_SIZE:
            raise ValueError(f"Universe too small: {len(df)} rows")

        return df.reset_index(drop=True), False

    except Exception as e:  # noqa: BLE001 - any failure -> fallback
        logger.warning("Universe fetch failed (%s); using fallback list.", e)
        return fallback_universe(), True
