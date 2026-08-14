import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stockgpt import schema as S
from stockgpt.sector_rotation import compute_sector_rotation


def _panel(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df[S.SCAN_DATE] = pd.to_datetime(df[S.SCAN_DATE])
    return df


class TestComputeSectorRotation:
    def test_empty_panel_returns_empty(self):
        assert compute_sector_rotation(pd.DataFrame()).empty

    def test_single_date_returns_empty(self):
        panel = _panel([
            {S.SCAN_DATE: "2026-01-01", S.SECTOR: "IT", S.FINAL_SCORE: 50.0},
        ])
        assert compute_sector_rotation(panel).empty

    def test_detects_a_rising_sector(self):
        rows = []
        # Prior window: IT averages 40. Recent window: IT averages 60.
        for d in pd.date_range("2026-01-01", periods=5):
            rows.append({S.SCAN_DATE: d, S.SECTOR: "IT", S.FINAL_SCORE: 40.0})
        for d in pd.date_range("2026-01-06", periods=5):
            rows.append({S.SCAN_DATE: d, S.SECTOR: "IT", S.FINAL_SCORE: 60.0})
        panel = _panel(rows)

        result = compute_sector_rotation(panel, recent_days=5, prior_days=5)
        assert not result.empty
        it_row = result[result[S.SECTOR] == "IT"].iloc[0]
        assert it_row["prior_avg_final_score"] == 40.0
        assert it_row["recent_avg_final_score"] == 60.0
        assert it_row["change"] == 20.0
        assert it_row["change_pct"] == 50.0

    def test_sorted_biggest_gainer_first(self):
        rows = []
        for d in pd.date_range("2026-01-01", periods=3):
            rows.append({S.SCAN_DATE: d, S.SECTOR: "Falling", S.FINAL_SCORE: 60.0})
            rows.append({S.SCAN_DATE: d, S.SECTOR: "Rising", S.FINAL_SCORE: 20.0})
        for d in pd.date_range("2026-01-04", periods=3):
            rows.append({S.SCAN_DATE: d, S.SECTOR: "Falling", S.FINAL_SCORE: 30.0})
            rows.append({S.SCAN_DATE: d, S.SECTOR: "Rising", S.FINAL_SCORE: 70.0})
        panel = _panel(rows)

        result = compute_sector_rotation(panel, recent_days=3, prior_days=3)
        assert result.iloc[0][S.SECTOR] == "Rising"
        assert result.iloc[-1][S.SECTOR] == "Falling"

    def test_falls_back_to_half_split_when_history_shorter_than_requested_window(self):
        # Only 4 days total, but recent_days=14/prior_days=14 requested --
        # should still return a result via the half-split fallback, not empty.
        rows = []
        for d in pd.date_range("2026-01-01", periods=2):
            rows.append({S.SCAN_DATE: d, S.SECTOR: "IT", S.FINAL_SCORE: 40.0})
        for d in pd.date_range("2026-01-03", periods=2):
            rows.append({S.SCAN_DATE: d, S.SECTOR: "IT", S.FINAL_SCORE: 50.0})
        panel = _panel(rows)

        result = compute_sector_rotation(panel, recent_days=14, prior_days=14)
        assert not result.empty
        assert result.iloc[0][S.SECTOR] == "IT"

    def test_zero_prior_average_does_not_crash_change_pct(self):
        rows = []
        for d in pd.date_range("2026-01-01", periods=3):
            rows.append({S.SCAN_DATE: d, S.SECTOR: "Zero", S.FINAL_SCORE: 0.0})
        for d in pd.date_range("2026-01-04", periods=3):
            rows.append({S.SCAN_DATE: d, S.SECTOR: "Zero", S.FINAL_SCORE: 10.0})
        panel = _panel(rows)

        result = compute_sector_rotation(panel, recent_days=3, prior_days=3)
        assert not result.empty
        assert pd.isna(result.iloc[0]["change_pct"])
