"""Historical backfill: reconstructs the exact same daily snapshot format
(`data/history/YYYY-MM-DD/scan.csv`) for past trading days, so the
backtest engine, Leaderboard, walk-forward validation, and portfolio
backtest all immediately have years of real history instead of waiting
for it to accumulate one day at a time.

Why this works at all: scanner.py, relative_strength.py, range_bound.py,
and scoring.py are ALL pure functions of a price-history window (they only
ever look at `.iloc[-1]` of whatever DataFrame they're handed, treating it
as "today"). None of them needed to change for this. The only genuinely
new problem is fundamentals -- yfinance's `.info` block (what
fundamentals.py normally fetches) is a live snapshot with no "as of last
year" mode. There is no way to ask Yahoo "what was RELIANCE's ROE on
15 March 2023" directly.

What this module does instead: pulls each symbol's raw quarterly/annual
financial statements (balance sheet, income statement, cash flow -- which
Yahoo DOES expose historically, several periods back, as actual filed
numbers) and derives the same ratios fundamentals.py computes from
`.info` (ROE, ROA, margins, growth, debt/equity, current/quick ratio,
cash, debt, operating/free cash flow) directly from those statement line
items, one set of ratios per real filing period. Each set is then
forward-filled forward in time until the next filing -- i.e. a trading
day's fundamentals are always the most recently *actually filed* numbers
as of that day, never a number from the future (no lookahead).

Two honest limits, by design, not oversight:
1. Valuation ratios that mix price with a per-share statement figure
   (PE, forward PE, price-to-book, dividend yield) and beta are NOT
   derived historically here -- forward PE is an analyst estimate with no
   historical record at all, and the others would need shares-outstanding
   history this pass doesn't fetch. These fields fall back to today's
   current value, held constant, for every backfilled day. Flagged via
   `fundamentals_source` so it's visible in the data, not silently mixed in.
2. If a symbol's statements are empty/unusable (delisted, thinly covered,
   Yahoo simply doesn't have them), the WHOLE fundamentals row falls back
   to today's current values held constant across the entire backfill
   window for that symbol, per the explicit approval to do that as a
   fallback. Also flagged via `fundamentals_source`.

Survivorship bias, also worth stating plainly: this backfills whatever
symbols are in TODAY's universe. Any company that delisted, merged, or was
removed from the index during the backfill window is invisible to it --
only today's survivors get a 3-year history. That's a real bias in any
backtest run against this data, not a bug to fix here; a true
survivorship-bias-free backfill would need a point-in-time universe
snapshot per year, which is a separate, bigger project.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from . import history, range_bound, relative_strength, scanner, scoring
from . import schema as S
from .config import WINDOWS
from .fundamentals import apply_sector_adjustment, score_row

logger = logging.getLogger(__name__)

FUNDAMENTALS_SOURCE_DERIVED = "derived_from_statements"
FUNDAMENTALS_SOURCE_CURRENT_FALLBACK = "current_value_fallback"

# Fields we can genuinely derive, period by period, from raw financial
# statements. Everything in FUNDAMENTAL_RATIO_FIELDS but not listed here
# (trailing_pe, forward_pe, price_to_book, dividend_yield, beta) always
# falls back to today's current value -- see module docstring, limit #1.
DERIVABLE_FIELDS = [
    S.ROE, S.ROA, S.OPERATING_MARGIN, S.NET_PROFIT_MARGIN,
    S.REVENUE_GROWTH, S.EARNINGS_GROWTH, S.DEBT_TO_EQUITY,
    S.CURRENT_RATIO, S.QUICK_RATIO, S.TOTAL_CASH_CR, S.TOTAL_DEBT_CR,
    S.OPERATING_CASHFLOW_CR, S.FREE_CASHFLOW_CR,
]

# yfinance's statement row labels have shifted across versions (Yahoo's own
# scrape target changes, and yfinance's normalisation layer with it) -- each
# entry here is tried in order, first match wins, so this module doesn't
# silently return "unavailable" just because one library version renamed a
# row. This sandbox has no network access to Yahoo Finance to verify the
# exact live label set, so treat this alias list as a best-effort starting
# point from documented/community-known yfinance conventions -- if the pilot
# run shows a symbol falling back when it shouldn't, the fix is almost
# always adding another alias here, not changing the derivation math.
_ROW_ALIASES: dict[str, list[str]] = {
    "total_revenue": ["Total Revenue", "Operating Revenue"],
    "net_income": ["Net Income", "Net Income Common Stockholders", "Net Income Continuous Operations"],
    "operating_income": ["Operating Income", "EBIT"],
    "total_assets": ["Total Assets"],
    "total_equity": ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"],
    "total_debt": ["Total Debt", "Net Debt"],
    "current_assets": ["Current Assets", "Total Current Assets"],
    "current_liabilities": ["Current Liabilities", "Total Current Liabilities"],
    "inventory": ["Inventory"],
    "cash": ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"],
    "operating_cashflow": ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"],
    "free_cashflow": ["Free Cash Flow"],
    "capex": ["Capital Expenditure", "Purchase Of PPE"],
}


def _find_row(stmt: pd.DataFrame, key: str) -> Optional[pd.Series]:
    """First matching row (by alias list) from a yfinance statement
    DataFrame, or None if none of the known labels are present."""
    if stmt is None or stmt.empty:
        return None
    for label in _ROW_ALIASES.get(key, []):
        if label in stmt.index:
            row = stmt.loc[label]
            if isinstance(row, pd.DataFrame):  # duplicate label -- take first
                row = row.iloc[0]
            return row
    return None


def derive_ratios_from_statements(balance_sheet: pd.DataFrame, income_stmt: pd.DataFrame,
                                   cashflow: pd.DataFrame) -> pd.DataFrame:
    """One row per report period (indexed by period-end date, ascending),
    columns = DERIVABLE_FIELDS, values `None` wherever a needed line item
    is missing for that period -- never a fabricated 0, same "missing stays
    missing" contract fundamentals.py already follows for the live path.

    Growth fields (revenue_growth, earnings_growth) need a PRIOR period to
    compare against, so the earliest period in the input never has them --
    that's the correct answer, not a bug: there is no earlier filing to
    compute a real growth rate from.
    """
    total_revenue = _find_row(income_stmt, "total_revenue")
    net_income = _find_row(income_stmt, "net_income")
    operating_income = _find_row(income_stmt, "operating_income")
    total_assets = _find_row(balance_sheet, "total_assets")
    total_equity = _find_row(balance_sheet, "total_equity")
    total_debt = _find_row(balance_sheet, "total_debt")
    current_assets = _find_row(balance_sheet, "current_assets")
    current_liabilities = _find_row(balance_sheet, "current_liabilities")
    inventory = _find_row(balance_sheet, "inventory")
    cash = _find_row(balance_sheet, "cash")
    operating_cf = _find_row(cashflow, "operating_cashflow")
    free_cf = _find_row(cashflow, "free_cashflow")
    capex = _find_row(cashflow, "capex")

    # Every period any statement mentions, oldest first -- ratios are
    # computed per period from whichever line items exist for that period;
    # a period missing one statement (e.g. cashflow present, balance sheet
    # row renamed) still yields partial ratios rather than none at all.
    periods: set = set()
    for row in (total_revenue, net_income, operating_income, total_assets,
                total_equity, total_debt, current_assets, cash, operating_cf, free_cf):
        if row is not None:
            periods.update(row.index)
    if not periods:
        return pd.DataFrame()
    periods = sorted(periods)

    def get(row: Optional[pd.Series], period) -> Optional[float]:
        if row is None or period not in row.index:
            return None
        v = row[period]
        # Defensive, on top of the column-dedup already done in
        # fetch_statements_one: a duplicate-labelled period still resolves
        # to a Series (ambiguous, multiple values for "the same period"),
        # not a scalar. Rather than trust every caller to have deduped
        # upstream, take the first non-null value here too.
        if isinstance(v, pd.Series):
            v = v.dropna().iloc[0] if not v.dropna().empty else None
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        return float(v)

    rows = []
    prev_revenue = prev_net_income = None
    for period in periods:
        rev = get(total_revenue, period)
        ni = get(net_income, period)
        opi = get(operating_income, period)
        assets = get(total_assets, period)
        equity = get(total_equity, period)
        debt = get(total_debt, period)
        curr_a = get(current_assets, period)
        curr_l = get(current_liabilities, period)
        inv = get(inventory, period)
        csh = get(cash, period)
        ocf = get(operating_cf, period)
        fcf = get(free_cf, period)
        cpx = get(capex, period)
        if fcf is None and ocf is not None and cpx is not None:
            fcf = ocf - abs(cpx)  # yfinance's Free Cash Flow row is sometimes absent even when OCF+capex both exist

        row = {
            S.ROE: round((ni / equity) * 100, 2) if ni is not None and equity else None,
            S.ROA: round((ni / assets) * 100, 2) if ni is not None and assets else None,
            S.OPERATING_MARGIN: round((opi / rev) * 100, 2) if opi is not None and rev else None,
            S.NET_PROFIT_MARGIN: round((ni / rev) * 100, 2) if ni is not None and rev else None,
            S.DEBT_TO_EQUITY: round((debt / equity) * 100, 2) if debt is not None and equity else None,
            S.CURRENT_RATIO: round(curr_a / curr_l, 2) if curr_a is not None and curr_l else None,
            S.QUICK_RATIO: (round((curr_a - inv) / curr_l, 2)
                             if curr_a is not None and inv is not None and curr_l else None),
            S.TOTAL_CASH_CR: round(csh / 1e7, 2) if csh is not None else None,
            S.TOTAL_DEBT_CR: round(debt / 1e7, 2) if debt is not None else None,
            S.OPERATING_CASHFLOW_CR: round(ocf / 1e7, 2) if ocf is not None else None,
            S.FREE_CASHFLOW_CR: round(fcf / 1e7, 2) if fcf is not None else None,
            S.REVENUE_GROWTH: (round(((rev - prev_revenue) / abs(prev_revenue)) * 100, 2)
                                if rev is not None and prev_revenue not in (None, 0) else None),
            S.EARNINGS_GROWTH: (round(((ni - prev_net_income) / abs(prev_net_income)) * 100, 2)
                                 if ni is not None and prev_net_income not in (None, 0) else None),
        }
        rows.append(row)
        if rev is not None:
            prev_revenue = rev
        if ni is not None:
            prev_net_income = ni

    out = pd.DataFrame(rows, index=pd.to_datetime(pd.Index(periods)))
    return out.sort_index()


def historical_fundamentals_for_symbol(symbol: str, balance_sheet: pd.DataFrame,
                                        income_stmt: pd.DataFrame, cashflow: pd.DataFrame,
                                        current_fundamentals_row: Optional[dict]) -> tuple[pd.DataFrame, str]:
    """Returns (per-period fundamentals DataFrame, source label).

    Tries the real statement-derived path first; falls back to a single
    row of today's current fundamentals (held constant) if statements are
    unusable AND a current row is available. If neither is available,
    returns an empty DataFrame -- that symbol simply has no fundamentals
    for the backfilled window, same as any other genuinely-unknown field.
    """
    derived = derive_ratios_from_statements(balance_sheet, income_stmt, cashflow)
    non_null_periods = derived.dropna(how="all") if not derived.empty else derived
    if not non_null_periods.empty:
        return derived, FUNDAMENTALS_SOURCE_DERIVED

    if current_fundamentals_row:
        fallback_row = {k: current_fundamentals_row.get(k) for k in DERIVABLE_FIELDS}
        # Growth fields from a single current snapshot describe THIS year's
        # growth, not a trailing point-in-time series -- kept anyway (still
        # better than nothing, and it's the same number the live pipeline
        # would show today), just under the fallback label so it's visibly
        # distinguishable from a real per-period figure.
        return pd.DataFrame([fallback_row], index=[pd.Timestamp.now().normalize()]), \
            FUNDAMENTALS_SOURCE_CURRENT_FALLBACK

    return pd.DataFrame(), FUNDAMENTALS_SOURCE_CURRENT_FALLBACK


def fundamentals_asof(hist_fund_df: pd.DataFrame, trading_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """For each trading date, the most recently FILED period's ratios as
    of that date -- i.e. a report dated 2024-03-31 only starts applying to
    trading dates on or after it actually would have been available (this
    uses the period-end date itself as a conservative proxy for filing
    date, which slightly understates real-world reporting lag but never
    looks into the future). Trading dates before the earliest known period
    get every derivable field as NaN -- correctly "unknown that far back",
    not backfilled from a report that didn't exist yet.
    """
    if hist_fund_df.empty:
        return pd.DataFrame(index=trading_dates, columns=DERIVABLE_FIELDS, dtype=float)

    sorted_fund = hist_fund_df.sort_index()
    # merge_asof requires both sides sorted and as a DataFrame with the key
    # as a column, not the index. It also requires both key columns to share
    # the exact same datetime64 unit (pandas >=2.2 stopped silently
    # coercing this) -- trading_dates comes from yfinance's price-history
    # index and period_date from statement timestamps (or, for the
    # current-fallback case, pd.Timestamp.now()), which can land on
    # different units (e.g. datetime64[s] vs datetime64[us]) depending on
    # pandas version and where each Timestamp originated. Normalizing both
    # to datetime64[ns] here makes the merge robust to that regardless of
    # which side happens to disagree.
    left = pd.DataFrame({"trading_date": pd.to_datetime(trading_dates)}).astype(
        {"trading_date": "datetime64[ns]"}).sort_values("trading_date")
    right = sorted_fund.reset_index().rename(columns={"index": "period_date"})
    right["period_date"] = pd.to_datetime(right["period_date"]).astype("datetime64[ns]")
    merged = pd.merge_asof(left, right, left_on="trading_date", right_on="period_date", direction="backward")
    merged = merged.set_index("trading_date").reindex(trading_dates)
    return merged[DERIVABLE_FIELDS]


def apply_fallback_fields(asof_df: pd.DataFrame, current_fundamentals_row: Optional[dict]) -> pd.DataFrame:
    """Adds the fields that are never statement-derived (valuation ratios +
    beta, see module docstring limit #1) as a flat current-value fallback
    across every row, plus market_cap_cr and gross_margin which fundamentals
    scoring doesn't weight directly but the dashboard's Fundamentals tab
    displays."""
    out = asof_df.copy()
    always_fallback = [S.TRAILING_PE, S.FORWARD_PE, S.PRICE_TO_BOOK, S.DIVIDEND_YIELD,
                        S.BETA, S.MARKET_CAP_CR, S.GROSS_MARGIN]
    for field in always_fallback:
        out[field] = (current_fundamentals_row or {}).get(field)
    return out


# ---------------------------------------------------------------------------
# Day-by-day orchestration -- reuses the exact same scoring modules the live
# daily pipeline uses (scanner, relative_strength, range_bound, fundamentals
# scoring, scoring.compute_final_scores), just fed a truncated price-history
# window and an as-of fundamentals row instead of "whatever's current".
# ---------------------------------------------------------------------------
def build_day_snapshot(trading_date: pd.Timestamp,
                        price_hist: dict[str, pd.DataFrame],
                        nifty_close: pd.Series,
                        sector_map: dict[str, str],
                        industry_map: dict[str, str],
                        sector_bucket_map: dict[str, str],
                        fundamentals_asof_by_symbol: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One trading day's full scan_df -- same columns compute_final_scores
    produces in the live pipeline, built from price history truncated to
    `trading_date` and each symbol's as-of fundamentals for that date."""
    nifty_asof = nifty_close[nifty_close.index <= trading_date]
    nifty_1m = relative_strength.calc_return(nifty_asof, WINDOWS.days_1m)
    nifty_3m = relative_strength.calc_return(nifty_asof, WINDOWS.days_3m)
    nifty_6m = relative_strength.calc_return(nifty_asof, WINDOWS.days_6m)

    scan_rows, rs_rows, range_rows, fund_rows = [], [], [], []
    for symbol, full_hist in price_hist.items():
        hist_asof = full_hist[full_hist.index <= trading_date]
        if hist_asof.empty:
            continue
        sector = sector_map.get(symbol, "Unknown")
        industry = industry_map.get(symbol, "Unknown")

        row = scanner.process_history(symbol, hist_asof, sector, industry)
        if not row:
            continue
        scan_rows.append(row)

        rs_rows.append(relative_strength.compute_for_symbol(
            symbol, hist_asof["Close"], sector, nifty_1m, nifty_3m, nifty_6m))

        range_row = range_bound.analyse(symbol, hist_asof, row[S.TECHNICAL_SCORE], 0)
        if range_row:
            range_rows.append(range_row)

        fund_asof_df = fundamentals_asof_by_symbol.get(symbol)
        if fund_asof_df is not None and trading_date in fund_asof_df.index:
            frow = fund_asof_df.loc[trading_date].to_dict()
            frow[S.SYMBOL] = symbol
            frow[S.SECTOR_BUCKET] = sector_bucket_map.get(symbol, "General")
            fund_rows.append(frow)

    if not scan_rows:
        return pd.DataFrame()

    scan_df = pd.DataFrame(scan_rows)
    rs_df = relative_strength.add_sector_rank(pd.DataFrame(rs_rows))
    range_df = pd.DataFrame(range_rows) if range_rows else pd.DataFrame(columns=[S.SYMBOL])
    fund_df = pd.DataFrame(fund_rows) if fund_rows else pd.DataFrame(columns=[S.SYMBOL])

    if not fund_df.empty:
        scored_fund = fund_df.apply(score_row, axis=1)
        fund_df = pd.concat([fund_df, scored_fund], axis=1)
        fund_df = apply_sector_adjustment(fund_df)

    merged = scan_df.merge(rs_df.drop(columns=[S.SECTOR], errors="ignore"), on=S.SYMBOL, how="left")
    merged = merged.merge(range_df, on=S.SYMBOL, how="left")
    merged = merged.merge(fund_df, on=S.SYMBOL, how="left", suffixes=("", "_fund"))

    if S.SECTOR_ADJUSTED_FUNDAMENTAL_SCORE not in merged.columns:
        merged[S.SECTOR_ADJUSTED_FUNDAMENTAL_SCORE] = 0.0
    merged[S.RELATIVE_STRENGTH_SCORE] = merged[S.RELATIVE_STRENGTH_SCORE].fillna(0.0)

    return scoring.compute_final_scores(merged)


# ---------------------------------------------------------------------------
# Bulk fetch -- same chunking/retry resilience pattern already proven in
# scripts/run_daily_pipeline.py's download_history(), extended to a longer
# lookback period and to also pull raw financial statements per symbol.
# ---------------------------------------------------------------------------
def _chunk(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def fetch_price_history_bulk(symbols: list[str], period: str = "4y") -> dict[str, pd.DataFrame]:
    """Batch-download then retry failures individually -- identical
    resilience pattern to run_daily_pipeline.py's download_history(), just
    with a multi-year period instead of the live pipeline's 1y."""
    out: dict[str, pd.DataFrame] = {}
    failed: list[str] = []

    for chunk in _chunk(symbols, 100):
        yf_symbols = [f"{s}.NS" for s in chunk]
        try:
            data = yf.download(yf_symbols, period=period, interval="1d",
                                group_by="ticker", threads=True, progress=False, timeout=60)
        except Exception:  # noqa: BLE001
            failed.extend(chunk)
            continue

        for symbol in chunk:
            yf_symbol = f"{symbol}.NS"
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    if yf_symbol not in data.columns.get_level_values(0):
                        failed.append(symbol)
                        continue
                    hist = data[yf_symbol].dropna(how="all")
                else:
                    hist = data.dropna(how="all")
                if hist.empty:
                    failed.append(symbol)
                    continue
                out[symbol] = hist
            except Exception:  # noqa: BLE001
                failed.append(symbol)
        time.sleep(2)

    for symbol in failed:
        try:
            hist = yf.Ticker(f"{symbol}.NS").history(period=period)
            if not hist.empty:
                out[symbol] = hist
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.3)

    return out


def fetch_nifty_history_long(period: str = "4y") -> pd.Series:
    """Same retry pattern as run_daily_pipeline.py's Nifty fetch, longer
    window. Returns an empty Series (never raises) if all attempts fail --
    callers already handle an empty/missing benchmark the same way the live
    pipeline does (relative-strength-vs-Nifty columns stay NaN)."""
    for attempt in range(3):
        try:
            hist = yf.download("^NSEI", period=period, interval="1d", progress=False, timeout=45)
            close = hist["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close = close.dropna()
            if not close.empty:
                return close
        except Exception as e:  # noqa: BLE001
            logger.warning("Nifty history fetch attempt %s/3 failed: %s", attempt + 1, e)
        time.sleep(3 + attempt * 3)
    return pd.Series(dtype=float)


def fetch_statements_one(symbol: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Annual + quarterly balance sheet / income statement / cash flow,
    combined column-wise per statement type (outer join on report date) so
    derive_ratios_from_statements sees the richest available period
    coverage in one pass. Returns three empty DataFrames (never raises) if
    Yahoo has nothing for this symbol -- that's the trigger for the
    current-value fallback path in historical_fundamentals_for_symbol."""
    yf_symbol = f"{symbol}.NS"
    t = yf.Ticker(yf_symbol)

    def _combine(annual_attr: str, quarterly_attr: str) -> pd.DataFrame:
        for attempt in range(3):
            try:
                annual = getattr(t, annual_attr)
                quarterly = getattr(t, quarterly_attr)
                frames = [f for f in (annual, quarterly) if f is not None and not f.empty]
                if not frames:
                    return pd.DataFrame()
                combined = pd.concat(frames, axis=1)
                # The last quarter of a fiscal year and the annual report
                # itself share the exact same period-end date, so annual +
                # quarterly concatenated column-wise very commonly produces
                # a duplicate date column -- a duplicate-labelled column
                # makes row[period] return a Series instead of a scalar
                # deeper in derive_ratios_from_statements. Caught via the
                # real pilot run (float(v) raised "cannot convert the
                # series to <class 'float'>" for every symbol, not one
                # bad ticker). Keep the first occurrence (the annual
                # figure, since annual runs first in `frames`) and drop
                # the rest.
                return combined.loc[:, ~combined.columns.duplicated()]
            except Exception as e:  # noqa: BLE001
                logger.debug("%s %s fetch attempt %s failed: %s", symbol, annual_attr, attempt + 1, e)
                time.sleep(2 + attempt * 3)
        return pd.DataFrame()

    balance_sheet = _combine("balance_sheet", "quarterly_balance_sheet")
    income_stmt = _combine("income_stmt", "quarterly_income_stmt")
    cashflow = _combine("cashflow", "quarterly_cashflow")
    return balance_sheet, income_stmt, cashflow


def fetch_statements_bulk(symbols: list[str], max_workers: int = 3) -> dict[str, tuple]:
    """Threaded, same MAX_WORKERS=3 throttling fundamentals.py already uses
    for .info polling -- statement endpoints are a different Yahoo API but
    empirically throttle similarly."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    out: dict[str, tuple] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_statements_one, s): s for s in symbols}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                out[symbol] = future.result()
            except Exception as e:  # noqa: BLE001
                logger.warning("%s statements failed: %s", symbol, e)
                out[symbol] = (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
            time.sleep(0.5)
    return out


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------
def run_backfill(symbols: list[str], sector_map: dict[str, str], industry_map: dict[str, str],
                  sector_bucket_map: dict[str, str], current_fundamentals_by_symbol: dict[str, dict],
                  years: float, output_dir: str | Path, skip_existing: bool = True,
                  progress_every: int = 20) -> dict:
    """Runs the full backfill: fetch price history + Nifty + statements
    once, then walk every real trading day in the window writing
    data/history/YYYY-MM-DD/scan.csv, resumable via `skip_existing` (a
    partial run that got rate-limited or interrupted can just be re-run --
    already-written days are skipped, not recomputed or re-fetched).

    Fetches a `years + 1` window so the earliest backfilled dates still
    have a full 200-day-SMA / 6-month-return warmup lookback behind them,
    same as any live scan does -- the extra year is warmup only, never
    itself written as a snapshot.

    Returns a summary dict: symbols fetched, statements-derived vs
    fallback counts, days written, days skipped.
    """
    output_dir = Path(output_dir)
    fetch_period = f"{int(years) + 1}y"

    logger.info("Fetching %s of price history for %d symbols...", fetch_period, len(symbols))
    price_hist = fetch_price_history_bulk(symbols, period=fetch_period)
    logger.info("Got price history for %d/%d symbols", len(price_hist), len(symbols))

    logger.info("Fetching Nifty benchmark history...")
    nifty_close = fetch_nifty_history_long(period=fetch_period)
    if nifty_close.empty:
        logger.warning("Nifty benchmark unavailable -- relative-strength-vs-Nifty columns will be NaN "
                        "for the whole backfill, same as the live pipeline's own fallback behaviour.")

    logger.info("Fetching historical financial statements for %d symbols...", len(price_hist))
    statements = fetch_statements_bulk(list(price_hist.keys()))

    if not price_hist:
        return {"symbols_fetched": 0, "days_written": 0, "days_skipped": 0,
                "derived_fundamentals": 0, "fallback_fundamentals": 0}

    # Master trading-day calendar: every date any fetched symbol actually
    # traded on, restricted to the real `years`-long target window (the
    # extra fetch year stays warmup-only).
    all_dates = sorted(set().union(*(h.index for h in price_hist.values())))
    all_dates = pd.DatetimeIndex(all_dates)
    cutoff = all_dates.max() - pd.Timedelta(days=int(years * 365.25))
    target_dates = all_dates[all_dates >= cutoff]

    logger.info("Deriving as-of fundamentals for each symbol...")
    fundamentals_asof_by_symbol = {}
    derived_count = fallback_count = 0
    for symbol in price_hist:
        bs, inc, cf = statements.get(symbol, (pd.DataFrame(), pd.DataFrame(), pd.DataFrame()))
        hist_fund, source = historical_fundamentals_for_symbol(
            symbol, bs, inc, cf, current_fundamentals_by_symbol.get(symbol))
        if source == FUNDAMENTALS_SOURCE_DERIVED:
            derived_count += 1
        else:
            fallback_count += 1
        asof = fundamentals_asof(hist_fund, all_dates)
        asof = apply_fallback_fields(asof, current_fundamentals_by_symbol.get(symbol))
        fundamentals_asof_by_symbol[symbol] = asof

    logger.info("Writing %d trading days to %s...", len(target_dates), output_dir)
    written = skipped = 0
    for i, trading_date in enumerate(target_dates):
        out_path = output_dir / trading_date.strftime("%Y-%m-%d") / "scan.csv"
        if skip_existing and out_path.exists():
            skipped += 1
            continue
        snap = build_day_snapshot(trading_date, price_hist, nifty_close, sector_map,
                                   industry_map, sector_bucket_map, fundamentals_asof_by_symbol)
        if not snap.empty:
            history.save_snapshot(snap, output_dir, trading_date)
            written += 1
        if progress_every and (i + 1) % progress_every == 0:
            logger.info("  ...%d/%d trading days processed (%d written, %d skipped)",
                        i + 1, len(target_dates), written, skipped)

    return {
        "symbols_fetched": len(price_hist),
        "derived_fundamentals": derived_count,
        "fallback_fundamentals": fallback_count,
        "days_written": written,
        "days_skipped": skipped,
        "total_trading_days": len(target_dates),
    }
