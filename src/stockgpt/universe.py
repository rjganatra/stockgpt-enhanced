"""NSE equity universe fetch, with a small built-in fallback list.

Same idea as the original repo's fetch_nifty500.py (NSE archives can rate
limit or go down), rewritten cleanly: fetch -> validate row count -> on any
failure, fall back to a known-good static list rather than writing an empty
or tiny universe that would silently starve the rest of the pipeline.
"""

from __future__ import annotations

import logging
import time
from io import StringIO

import pandas as pd
import requests

from . import schema as S

logger = logging.getLogger(__name__)

NSE_EQUITY_URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"

# Sector source, broadest-first. Both are niftyindices.com's own published
# index-constituent CSVs (same domain/format/reliability as the one already
# in use -- no new external dependency, no new fragility surface). Total
# Market (~750 names) covers meaningfully more of the ~2100-symbol universe
# than Nifty 500 (~500 names) did on its own, so it's tried first; Nifty 500
# stays as a fallback if that particular file is ever unavailable. This is
# still only the base layer -- run_daily_pipeline.py's sector_for() overrides
# it with Yahoo's per-symbol sector whenever that's available, which is
# where the bulk of real coverage (~96% of the universe) actually comes
# from. Neither source, nor any other free one found during a deliberate
# search (NSE's own non-archive APIs need browser session cookies and are
# more likely to break on a CI runner; BSE's public scrip-classification API
# is realistically reachable too, but was not able to be verified from this
# environment's network -- worth a follow-up if the residual "Unknown" tail
# after this change still matters), will ever reach 100%: a real slice of
# India's ~2000+ listed microcaps have no sector classification in ANY free
# source, including paid ones -- that's a data-availability floor, not a
# bug in how this pipeline sources data.
NIFTY_TOTAL_MARKET_URL = "https://www.niftyindices.com/IndexConstituent/ind_niftytotalmarket_list.csv"
NIFTY_500_URL = "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"
SECTOR_SOURCES = [
    ("Nifty Total Market", NIFTY_TOTAL_MARKET_URL),
    ("Nifty 500", NIFTY_500_URL),
]
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


def _fetch_csv(url: str, attempts: int = 3, backoff: float = 2.0) -> pd.DataFrame:
    """Same single-attempt-was-the-bug lesson as download_history() in the
    daily pipeline: a transient network hiccup shouldn't be indistinguishable
    from the source being genuinely gone. Retries a few times with a short
    backoff before the caller's fallback logic kicks in."""
    last_exc: Exception = RuntimeError(f"no attempts made for {url}")
    for attempt in range(attempts):
        try:
            response = requests.get(url, headers=HEADERS, timeout=30)
            response.raise_for_status()
            return pd.read_csv(StringIO(response.text))
        except Exception as e:  # noqa: BLE001
            last_exc = e
            if attempt < attempts - 1:
                time.sleep(backoff * (attempt + 1))
    raise last_exc


def fetch_sector_map() -> dict[str, str]:
    """Try each entry in SECTOR_SOURCES in order (broadest first), return the
    first one that succeeds. Returns {} only if every source failed -- the
    caller (fetch_universe) already treats a missing sector as "Unknown" per
    symbol, and run_daily_pipeline.py's Yahoo-based override covers most of
    the resulting gap regardless."""
    for label, url in SECTOR_SOURCES:
        try:
            sector_df = _fetch_csv(url)
            sector_map = dict(zip(sector_df["Symbol"], sector_df["Industry"]))
            logger.info("Sector map loaded from %s (%d symbols)", label, len(sector_map))
            return sector_map
        except Exception as e:  # noqa: BLE001 - try the next source
            logger.warning("%s sector map fetch failed: %s", label, e)
    logger.warning("All sector map sources failed; universe-level sector will be "
                    "'Unknown' until the daily pipeline's Yahoo-based override runs.")
    return {}


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

        sector_map = fetch_sector_map()
        df[S.SECTOR] = df[S.SYMBOL].map(sector_map).fillna("Unknown")
        df = df.drop_duplicates(subset=[S.SYMBOL])[[S.SYMBOL, S.SECTOR, S.COMPANY_NAME]]

        if len(df) < MIN_ACCEPTABLE_UNIVERSE_SIZE:
            raise ValueError(f"Universe too small: {len(df)} rows")

        return df.reset_index(drop=True), False

    except Exception as e:  # noqa: BLE001 - any failure -> fallback
        logger.warning("Universe fetch failed (%s); using fallback list.", e)
        return fallback_universe(), True
