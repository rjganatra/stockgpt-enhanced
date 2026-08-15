"""Historical backfill correctness -- verified with synthetic price and
financial-statement data shaped exactly like yfinance's real output, so
these tests don't depend on network access (which this sandbox doesn't
have to Yahoo Finance anyway). The point-in-time / no-lookahead guarantee
is the single most important thing to get right here -- a backtest run
against fundamentals that leaked a number from the future would look
better than reality, silently."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stockgpt import backfill as bf
from stockgpt import schema as S


def _synthetic_statements():
    periods = pd.to_datetime(["2022-03-31", "2023-03-31", "2024-03-31"])
    balance_sheet = pd.DataFrame({
        periods[0]: {"Total Assets": 1000.0, "Stockholders Equity": 400.0, "Total Debt": 200.0,
                     "Current Assets": 300.0, "Current Liabilities": 150.0, "Inventory": 50.0,
                     "Cash And Cash Equivalents": 80.0},
        periods[1]: {"Total Assets": 1100.0, "Stockholders Equity": 450.0, "Total Debt": 210.0,
                     "Current Assets": 320.0, "Current Liabilities": 140.0, "Inventory": 55.0,
                     "Cash And Cash Equivalents": 90.0},
        periods[2]: {"Total Assets": 1250.0, "Stockholders Equity": 520.0, "Total Debt": 220.0,
                     "Current Assets": 360.0, "Current Liabilities": 160.0, "Inventory": 60.0,
                     "Cash And Cash Equivalents": 110.0},
    })
    income_stmt = pd.DataFrame({
        periods[0]: {"Total Revenue": 900.0, "Net Income": 80.0, "Operating Income": 110.0},
        periods[1]: {"Total Revenue": 1000.0, "Net Income": 95.0, "Operating Income": 130.0},
        periods[2]: {"Total Revenue": 1150.0, "Net Income": 120.0, "Operating Income": 160.0},
    })
    cashflow = pd.DataFrame({
        periods[0]: {"Operating Cash Flow": 100.0, "Capital Expenditure": -40.0},
        periods[1]: {"Operating Cash Flow": 115.0, "Capital Expenditure": -45.0},
        periods[2]: {"Operating Cash Flow": 140.0, "Capital Expenditure": -50.0},
    })
    return periods, balance_sheet, income_stmt, cashflow


class TestDeriveRatiosFromStatements:
    def test_ratios_computed_correctly_from_raw_line_items(self):
        periods, bs, inc, cf = _synthetic_statements()
        derived = bf.derive_ratios_from_statements(bs, inc, cf)
        p0 = derived.iloc[0]
        assert p0[S.ROE] == pytest.approx(80 / 400 * 100, abs=0.01)
        assert p0[S.ROA] == pytest.approx(80 / 1000 * 100, abs=0.01)
        assert p0[S.OPERATING_MARGIN] == pytest.approx(110 / 900 * 100, abs=0.01)
        assert p0[S.NET_PROFIT_MARGIN] == pytest.approx(80 / 900 * 100, abs=0.01)
        assert p0[S.DEBT_TO_EQUITY] == pytest.approx(200 / 400 * 100, abs=0.01)
        assert p0[S.CURRENT_RATIO] == pytest.approx(300 / 150, abs=0.01)
        assert p0[S.QUICK_RATIO] == pytest.approx((300 - 50) / 150, abs=0.01)
        assert p0[S.TOTAL_CASH_CR] == pytest.approx(80 / 1e7, abs=0.001)
        assert p0[S.FREE_CASHFLOW_CR] == pytest.approx((100 - 40) / 1e7, abs=0.001)

    def test_earliest_period_has_no_growth_no_prior_to_compare(self):
        periods, bs, inc, cf = _synthetic_statements()
        derived = bf.derive_ratios_from_statements(bs, inc, cf)
        assert pd.isna(derived.iloc[0][S.REVENUE_GROWTH])
        assert pd.isna(derived.iloc[0][S.EARNINGS_GROWTH])

    def test_later_periods_have_real_growth_rates(self):
        periods, bs, inc, cf = _synthetic_statements()
        derived = bf.derive_ratios_from_statements(bs, inc, cf)
        assert derived.iloc[1][S.REVENUE_GROWTH] == pytest.approx((1000 - 900) / 900 * 100, abs=0.01)
        assert derived.iloc[2][S.EARNINGS_GROWTH] == pytest.approx((120 - 95) / 95 * 100, abs=0.01)

    def test_empty_statements_return_empty_dataframe(self):
        derived = bf.derive_ratios_from_statements(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        assert derived.empty

    def test_missing_line_item_yields_none_not_fabricated_zero(self):
        # Balance sheet has no "Inventory" row at all -- quick_ratio needs it.
        periods = pd.to_datetime(["2022-03-31"])
        bs = pd.DataFrame({periods[0]: {"Total Assets": 1000.0, "Stockholders Equity": 400.0,
                                         "Current Assets": 300.0, "Current Liabilities": 150.0}})
        inc = pd.DataFrame({periods[0]: {"Total Revenue": 900.0, "Net Income": 80.0}})
        cf = pd.DataFrame()
        derived = bf.derive_ratios_from_statements(bs, inc, cf)
        assert pd.isna(derived.iloc[0][S.QUICK_RATIO])
        assert derived.iloc[0][S.ROE] == pytest.approx(20.0, abs=0.01)  # unaffected fields still compute

    def test_duplicate_period_column_does_not_crash(self):
        # Regression test for a real bug caught during the pilot backfill
        # run: a fiscal year's Q4 and its annual report share the exact
        # same period-end date, so combining annual + quarterly statements
        # column-wise (as fetch_statements_one does) produces a duplicate
        # date column for every symbol. Indexing a Series by a duplicated
        # label returns another Series, not a scalar -- float(v) on that
        # raised "cannot convert the series to <class 'float'>" for the
        # whole pilot batch. This DataFrame has two identical "2022-03-31"
        # columns with DIFFERENT values, exactly reproducing what an
        # undeduplicated concat would look like.
        period = pd.Timestamp("2022-03-31")
        bs = pd.DataFrame(
            {"Total Assets": [1000.0, 1000.0], "Stockholders Equity": [400.0, 400.0],
             "Total Debt": [200.0, 200.0], "Current Assets": [300.0, 300.0],
             "Current Liabilities": [150.0, 150.0], "Inventory": [50.0, 50.0],
             "Cash And Cash Equivalents": [80.0, 80.0]},
            index=[period, period],
        ).T
        inc = pd.DataFrame(
            {"Total Revenue": [900.0, 900.0], "Net Income": [80.0, 80.0], "Operating Income": [110.0, 110.0]},
            index=[period, period],
        ).T
        cf = pd.DataFrame(
            {"Operating Cash Flow": [100.0, 100.0], "Capital Expenditure": [-40.0, -40.0]},
            index=[period, period],
        ).T
        derived = bf.derive_ratios_from_statements(bs, inc, cf)  # must not raise
        assert len(derived) == 1  # duplicate period collapses to one row, not two
        assert derived.iloc[0][S.ROE] == pytest.approx(20.0, abs=0.01)

    def test_free_cashflow_falls_back_to_ocf_minus_capex(self):
        # No explicit "Free Cash Flow" row -- must derive from OCF - |capex|.
        periods = pd.to_datetime(["2022-03-31"])
        bs = pd.DataFrame()
        inc = pd.DataFrame()
        cf = pd.DataFrame({periods[0]: {"Operating Cash Flow": 100.0, "Capital Expenditure": -30.0}})
        derived = bf.derive_ratios_from_statements(bs, inc, cf)
        assert derived.iloc[0][S.FREE_CASHFLOW_CR] == pytest.approx((100 - 30) / 1e7, abs=0.001)


class TestHistoricalFundamentalsForSymbol:
    def test_uses_derived_path_when_statements_available(self):
        periods, bs, inc, cf = _synthetic_statements()
        hist_fund, source = bf.historical_fundamentals_for_symbol("TEST", bs, inc, cf, None)
        assert source == bf.FUNDAMENTALS_SOURCE_DERIVED
        assert not hist_fund.empty

    def test_falls_back_to_current_value_when_statements_empty(self):
        current_row = {S.ROE: 22.0, S.DEBT_TO_EQUITY: 30.0}
        hist_fund, source = bf.historical_fundamentals_for_symbol(
            "TEST", pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), current_row)
        assert source == bf.FUNDAMENTALS_SOURCE_CURRENT_FALLBACK
        assert hist_fund.iloc[0][S.ROE] == 22.0

    def test_no_statements_and_no_current_row_returns_empty(self):
        hist_fund, source = bf.historical_fundamentals_for_symbol(
            "TEST", pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None)
        assert hist_fund.empty


class TestFundamentalsAsof:
    def test_before_earliest_period_is_unknown_not_backfilled(self):
        periods, bs, inc, cf = _synthetic_statements()
        derived = bf.derive_ratios_from_statements(bs, inc, cf)
        trading_dates = pd.date_range("2022-01-01", "2024-06-30", freq="D")
        asof = bf.fundamentals_asof(derived, trading_dates)
        assert pd.isna(asof.loc["2022-02-01", S.ROE])

    def test_never_looks_ahead_to_a_future_report(self):
        periods, bs, inc, cf = _synthetic_statements()
        derived = bf.derive_ratios_from_statements(bs, inc, cf)
        trading_dates = pd.date_range("2022-01-01", "2024-06-30", freq="D")
        asof = bf.fundamentals_asof(derived, trading_dates)
        # The day before period[1] (2023-03-31) must still show period[0]'s value.
        assert asof.loc["2023-03-30", S.ROE] == pytest.approx(derived.iloc[0][S.ROE], abs=0.001)

    def test_flips_to_new_period_on_its_exact_report_date(self):
        periods, bs, inc, cf = _synthetic_statements()
        derived = bf.derive_ratios_from_statements(bs, inc, cf)
        trading_dates = pd.date_range("2022-01-01", "2024-06-30", freq="D")
        asof = bf.fundamentals_asof(derived, trading_dates)
        assert asof.loc["2023-03-31", S.ROE] == pytest.approx(derived.iloc[1][S.ROE], abs=0.001)

    def test_holds_latest_period_after_the_last_report(self):
        periods, bs, inc, cf = _synthetic_statements()
        derived = bf.derive_ratios_from_statements(bs, inc, cf)
        trading_dates = pd.date_range("2022-01-01", "2024-06-30", freq="D")
        asof = bf.fundamentals_asof(derived, trading_dates)
        assert asof.loc["2024-06-30", S.ROE] == pytest.approx(derived.iloc[2][S.ROE], abs=0.001)

    def test_empty_history_returns_all_nan(self):
        trading_dates = pd.date_range("2022-01-01", "2022-01-10", freq="D")
        asof = bf.fundamentals_asof(pd.DataFrame(), trading_dates)
        assert asof.isna().all().all()


class TestApplyFallbackFields:
    def test_valuation_fields_applied_uniformly_from_current_row(self):
        trading_dates = pd.date_range("2022-01-01", "2022-01-05", freq="D")
        asof = pd.DataFrame(index=trading_dates, columns=bf.DERIVABLE_FIELDS, dtype=float)
        result = bf.apply_fallback_fields(asof, {S.TRAILING_PE: 18.5, S.BETA: 1.1})
        assert (result[S.TRAILING_PE] == 18.5).all()
        assert (result[S.BETA] == 1.1).all()

    def test_missing_current_row_leaves_fallback_fields_none(self):
        trading_dates = pd.date_range("2022-01-01", "2022-01-05", freq="D")
        asof = pd.DataFrame(index=trading_dates, columns=bf.DERIVABLE_FIELDS, dtype=float)
        result = bf.apply_fallback_fields(asof, None)
        assert result[S.TRAILING_PE].isna().all()


def _synthetic_price_hist(start_price, drift, vol, seed, periods_index):
    rng = np.random.default_rng(seed)
    n = len(periods_index)
    returns = rng.normal(drift, vol, n)
    close = start_price * np.cumprod(1 + returns)
    high = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    volume = rng.integers(100000, 5000000, n).astype(float)
    return pd.DataFrame({"Close": close, "High": high, "Low": low, "Volume": volume}, index=periods_index)


class TestBuildDaySnapshot:
    def _setup(self):
        dates = pd.date_range("2022-01-03", "2024-06-28", freq="B")
        price_hist = {
            "AAA": _synthetic_price_hist(100, 0.0006, 0.018, 1, dates),
            "BBB": _synthetic_price_hist(500, 0.0002, 0.012, 2, dates),
        }
        nifty_close = _synthetic_price_hist(18000, 0.0004, 0.009, 99, dates)["Close"]
        sector_map = {"AAA": "Technology", "BBB": "Financial Services"}
        industry_map = {"AAA": "Software", "BBB": "Banks"}
        sector_bucket_map = {"AAA": "IT / Technology", "BBB": "Banking"}

        periods = pd.to_datetime(["2022-06-30", "2023-06-30"])
        bs = pd.DataFrame({
            periods[0]: {"Total Assets": 1000.0, "Stockholders Equity": 400.0, "Total Debt": 200.0,
                         "Current Assets": 300.0, "Current Liabilities": 150.0, "Inventory": 50.0,
                         "Cash And Cash Equivalents": 80.0},
            periods[1]: {"Total Assets": 1100.0, "Stockholders Equity": 450.0, "Total Debt": 210.0,
                         "Current Assets": 320.0, "Current Liabilities": 140.0, "Inventory": 55.0,
                         "Cash And Cash Equivalents": 90.0},
        })
        inc = pd.DataFrame({
            periods[0]: {"Total Revenue": 900.0, "Net Income": 80.0, "Operating Income": 110.0},
            periods[1]: {"Total Revenue": 1000.0, "Net Income": 95.0, "Operating Income": 130.0},
        })
        cf = pd.DataFrame({
            periods[0]: {"Operating Cash Flow": 100.0, "Capital Expenditure": -40.0},
            periods[1]: {"Operating Cash Flow": 115.0, "Capital Expenditure": -45.0},
        })

        fundamentals_asof_by_symbol = {}
        for sym in price_hist:
            derived, _ = bf.historical_fundamentals_for_symbol(sym, bs, inc, cf, None)
            asof = bf.fundamentals_asof(derived, dates)
            asof = bf.apply_fallback_fields(asof, {S.TRAILING_PE: 22.0, S.BETA: 1.0})
            fundamentals_asof_by_symbol[sym] = asof

        return price_hist, nifty_close, sector_map, industry_map, sector_bucket_map, fundamentals_asof_by_symbol

    def test_produces_one_row_per_symbol_with_final_score(self):
        (price_hist, nifty_close, sector_map, industry_map,
         sector_bucket_map, fundamentals_asof_by_symbol) = self._setup()
        snap = bf.build_day_snapshot(pd.Timestamp("2023-06-15"), price_hist, nifty_close, sector_map,
                                      industry_map, sector_bucket_map, fundamentals_asof_by_symbol)
        assert set(snap[S.SYMBOL]) == {"AAA", "BBB"}
        assert S.FINAL_SCORE in snap.columns
        assert S.SCORE_BAND in snap.columns
        assert snap[S.FINAL_SCORE].between(0, 100).all()

    def test_before_enough_price_history_symbol_is_excluded_not_crashed(self):
        (price_hist, nifty_close, sector_map, industry_map,
         sector_bucket_map, fundamentals_asof_by_symbol) = self._setup()
        # scanner.process_history requires >= 60 rows of history -- a date
        # right at the very start of the series has almost none.
        snap = bf.build_day_snapshot(pd.Timestamp("2022-01-10"), price_hist, nifty_close, sector_map,
                                      industry_map, sector_bucket_map, fundamentals_asof_by_symbol)
        assert snap.empty  # correctly nothing yet, not an exception

    def test_matches_column_set_the_live_pipeline_produces(self):
        (price_hist, nifty_close, sector_map, industry_map,
         sector_bucket_map, fundamentals_asof_by_symbol) = self._setup()
        snap = bf.build_day_snapshot(pd.Timestamp("2023-06-15"), price_hist, nifty_close, sector_map,
                                      industry_map, sector_bucket_map, fundamentals_asof_by_symbol)
        expected = {S.SYMBOL, S.SECTOR, S.INDUSTRY, S.CURRENT_PRICE, S.TECHNICAL_SCORE, S.RSI,
                    S.RELATIVE_STRENGTH_SCORE, S.RANGE_STATUS, S.RANGE_SCORE,
                    S.SECTOR_ADJUSTED_FUNDAMENTAL_SCORE, S.RISK_PENALTY, S.FINAL_SCORE, S.SCORE_BAND}
        assert expected.issubset(set(snap.columns))

    def test_written_snapshot_is_readable_by_the_existing_backtest_engine(self, tmp_path):
        (price_hist, nifty_close, sector_map, industry_map,
         sector_bucket_map, fundamentals_asof_by_symbol) = self._setup()
        snap = bf.build_day_snapshot(pd.Timestamp("2023-06-15"), price_hist, nifty_close, sector_map,
                                      industry_map, sector_bucket_map, fundamentals_asof_by_symbol)

        from stockgpt import history
        from stockgpt.backtest.engine import load_history_panel

        history.save_snapshot(snap, tmp_path, pd.Timestamp("2023-06-15"))
        panel = load_history_panel(tmp_path)
        assert len(panel) == 2
        assert S.FINAL_SCORE in panel.columns
