import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stockgpt import scanner


def test_too_short_history_returns_none():
    hist = pd.DataFrame({"Close": [100] * 30, "High": [101] * 30,
                          "Low": [99] * 30, "Volume": [1000] * 30})
    assert scanner.process_history("X", hist, "Tech", "Software") is None


def test_uptrend_scores_higher_than_downtrend():
    n = 250
    up_close = pd.Series(np.linspace(100, 200, n))
    down_close = pd.Series(np.linspace(200, 100, n))

    up_hist = pd.DataFrame({"Close": up_close, "High": up_close + 1,
                             "Low": up_close - 1, "Volume": [1e5] * n})
    down_hist = pd.DataFrame({"Close": down_close, "High": down_close + 1,
                               "Low": down_close - 1, "Volume": [1e5] * n})

    up_row = scanner.process_history("UP", up_hist, "Tech", "Software")
    down_row = scanner.process_history("DOWN", down_hist, "Tech", "Software")

    assert up_row["technical_score"] > down_row["technical_score"]
    assert up_row["trend"] == "Bullish"
    assert down_row["trend"] == "Bearish"


def test_rsi_is_bounded_0_100():
    n = 200
    close = pd.Series(100 + np.random.RandomState(0).normal(0, 5, n).cumsum())
    rsi = scanner.compute_rsi(close)
    valid = rsi.dropna()
    assert (valid >= 0).all() and (valid <= 100).all()
