import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stockgpt import universe as U


def _csv_df(symbols_industries: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame({
        "Symbol": list(symbols_industries.keys()),
        "Industry": list(symbols_industries.values()),
    })


class TestFetchSectorMap:
    def test_uses_broadest_source_when_it_succeeds(self):
        total_market = _csv_df({"AAA": "Tech", "BBB": "Pharma"})
        with patch("stockgpt.universe._fetch_csv", return_value=total_market) as mock_fetch:
            result = U.fetch_sector_map()
            assert result == {"AAA": "Tech", "BBB": "Pharma"}
            # Only the first (broadest) source should have been tried.
            mock_fetch.assert_called_once_with(U.NIFTY_TOTAL_MARKET_URL)

    def test_falls_back_to_next_source_when_first_fails(self):
        nifty500 = _csv_df({"CCC": "Financial Services"})
        with patch("stockgpt.universe._fetch_csv", side_effect=[Exception("boom"), nifty500]) as mock_fetch:
            result = U.fetch_sector_map()
            assert result == {"CCC": "Financial Services"}
            assert mock_fetch.call_count == 2

    def test_returns_empty_dict_when_every_source_fails(self):
        with patch("stockgpt.universe._fetch_csv", side_effect=Exception("boom")):
            result = U.fetch_sector_map()
            assert result == {}

    def test_does_not_merge_sources_only_takes_first_success(self):
        # Total Market succeeding means Nifty 500 is never consulted, even
        # though in reality Nifty 500 is a subset -- no double-fetch needed.
        total_market = _csv_df({"AAA": "Tech"})
        with patch("stockgpt.universe._fetch_csv", return_value=total_market) as mock_fetch:
            U.fetch_sector_map()
            assert mock_fetch.call_count == 1


class TestFetchNifty500Symbols:
    def test_returns_symbol_list_on_success(self):
        nifty500 = _csv_df({"AAA": "Tech", "BBB": "Pharma", "CCC": "Financial Services"})
        with patch("stockgpt.universe._fetch_csv", return_value=nifty500) as mock_fetch:
            result = U.fetch_nifty500_symbols()
            assert result == ["AAA", "BBB", "CCC"]
            mock_fetch.assert_called_once_with(U.NIFTY_500_URL)

    def test_strips_whitespace_from_symbols(self):
        nifty500 = pd.DataFrame({"Symbol": [" AAA ", "BBB"], "Industry": ["Tech", "Pharma"]})
        with patch("stockgpt.universe._fetch_csv", return_value=nifty500):
            result = U.fetch_nifty500_symbols()
            assert result == ["AAA", "BBB"]

    def test_returns_empty_list_on_failure(self):
        with patch("stockgpt.universe._fetch_csv", side_effect=Exception("boom")):
            result = U.fetch_nifty500_symbols()
            assert result == []


class TestFetchUniverseUsesSectorMap:
    def test_universe_sector_falls_back_to_unknown_when_symbol_not_in_map(self):
        equity_l = pd.DataFrame({
            "SYMBOL": ["AAA", "ZZZ"],
            "NAME OF COMPANY": ["Company A", "Company Z"],
            "SERIES": ["EQ", "EQ"],
        })
        sector_map = {"AAA": "Tech"}  # ZZZ deliberately absent

        with patch("stockgpt.universe._fetch_csv", return_value=equity_l), \
             patch("stockgpt.universe.fetch_sector_map", return_value=sector_map), \
             patch("stockgpt.universe.MIN_ACCEPTABLE_UNIVERSE_SIZE", 1):
            df, used_fallback = U.fetch_universe()

        assert used_fallback is False
        row_zzz = df[df["symbol"] == "ZZZ"].iloc[0]
        row_aaa = df[df["symbol"] == "AAA"].iloc[0]
        assert row_zzz["sector"] == "Unknown"
        assert row_aaa["sector"] == "Tech"
