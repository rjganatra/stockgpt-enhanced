"""Every catalog signal must be a valid, runnable Strategy against the real
schema -- these are shown on the live dashboard's Signal Performance tab, so
a typo'd column name here would silently produce an empty tab, not a crash."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stockgpt import schema as S
from stockgpt.signal_catalog import SIGNAL_CATALOG, SIGNAL_DIRECTIONS
from stockgpt.backtest import run_backtest


def _synthetic_panel():
    dates = pd.date_range("2026-01-01", periods=10, freq="D")
    rows = []
    for i, d in enumerate(dates):
        rows.append({
            S.SYMBOL: "TESTCO", S.SCAN_DATE: d, S.CURRENT_PRICE: 100 + i,
            S.FINAL_SCORE: 70, S.SCORE_BAND: "Strong",
            S.DISTANCE_FROM_LOW_PCT: 10, S.DISTANCE_FROM_HIGH_PCT: 5,
            S.RSI: 55, S.VOLUME_RATIO: 1.5, S.TREND: "Bullish",
            S.SECTOR_ADJUSTED_FUNDAMENTAL_SCORE: 65, S.RELATIVE_STRENGTH_SCORE: 65,
            S.RISK_PENALTY: 5, "range_status": "Accumulation Zone",
        })
    return pd.DataFrame(rows)


def test_every_catalog_signal_is_a_valid_query_against_real_schema():
    panel = _synthetic_panel()
    for strategy in SIGNAL_CATALOG:
        trades = run_backtest(panel, strategy)  # must not raise ValueError
        assert isinstance(trades, list)


def test_every_catalog_signal_has_a_direction():
    names = {s.name for s in SIGNAL_CATALOG}
    assert names == set(SIGNAL_DIRECTIONS.keys())


def test_directions_are_only_bullish_or_bearish():
    assert set(SIGNAL_DIRECTIONS.values()) <= {"Bullish", "Bearish"}


def test_catalog_names_are_unique():
    names = [s.name for s in SIGNAL_CATALOG]
    assert len(names) == len(set(names))
