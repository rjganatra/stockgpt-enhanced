import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stockgpt import alerts
from stockgpt import schema as S


def _changes_df():
    return pd.DataFrame({
        S.SYMBOL: ["AAA", "BBB", "CCC", "DDD"],
        "previous_score_band": ["Watchlist", "Strong", "High Conviction", "Neutral"],
        "current_score_band": ["High Conviction", "High Conviction", "High Conviction", "Weak"],
        "change_signal": [
            "Band changed: Watchlist -> High Conviction",
            "Band changed: Strong -> High Conviction",
            "No major change",
            "Score dropped sharply",
        ],
    })


def _today_df():
    return pd.DataFrame({
        S.SYMBOL: ["AAA", "BBB", "CCC", "DDD"],
        "final_score": [72, 68, 55, 30],
        "score_band": ["High Conviction", "High Conviction", "High Conviction", "Weak"],
        "rsi": [60, 45, 70, 20],
    })


class TestDetectNewHighConviction:
    def test_only_symbols_newly_entering_the_band(self):
        result = alerts.detect_new_high_conviction(_changes_df())
        assert set(result[S.SYMBOL]) == {"AAA", "BBB"}

    def test_symbol_already_in_band_is_not_flagged_again(self):
        result = alerts.detect_new_high_conviction(_changes_df())
        assert "CCC" not in set(result[S.SYMBOL])

    def test_empty_changes_returns_empty(self):
        assert alerts.detect_new_high_conviction(pd.DataFrame()).empty


class TestDetectStrategyMatches:
    def test_valid_query_returns_matching_rows(self):
        saved = [{"name": "High score", "entry_query": "final_score >= 65"}]
        matches = alerts.detect_strategy_matches(_today_df(), saved)
        assert set(matches["High score"][S.SYMBOL]) == {"AAA", "BBB"}

    def test_no_matches_omits_the_strategy_entirely(self):
        saved = [{"name": "Impossible", "entry_query": "final_score >= 999"}]
        matches = alerts.detect_strategy_matches(_today_df(), saved)
        assert "Impossible" not in matches

    def test_broken_query_is_skipped_not_raised(self):
        saved = [
            {"name": "Broken", "entry_query": "not_a_real_column >= 5"},
            {"name": "Fine", "entry_query": "final_score >= 65"},
        ]
        matches = alerts.detect_strategy_matches(_today_df(), saved)
        assert "Broken" not in matches
        assert "Fine" in matches

    def test_strategy_with_no_query_is_skipped(self):
        saved = [{"name": "Empty", "entry_query": ""}]
        assert alerts.detect_strategy_matches(_today_df(), saved) == {}

    def test_empty_today_df_returns_no_matches(self):
        saved = [{"name": "Anything", "entry_query": "final_score >= 0"}]
        assert alerts.detect_strategy_matches(pd.DataFrame(), saved) == {}


class TestDetectWatchlistAlerts:
    def test_only_flags_sharp_changes_on_watchlisted_symbols(self):
        watchlist_df = pd.DataFrame({S.SYMBOL: ["AAA", "DDD", "ZZZ"]})
        result = alerts.detect_watchlist_alerts(_changes_df(), watchlist_df)
        # AAA: band changed (sharp), DDD: score dropped sharply (sharp) -- both flagged.
        # ZZZ isn't in changes_df at all. CCC is on changes but not watchlisted.
        assert set(result[S.SYMBOL]) == {"AAA", "DDD"}

    def test_symbol_on_watchlist_with_no_major_change_not_flagged(self):
        watchlist_df = pd.DataFrame({S.SYMBOL: ["CCC"]})
        result = alerts.detect_watchlist_alerts(_changes_df(), watchlist_df)
        assert result.empty

    def test_empty_watchlist_returns_empty(self):
        assert alerts.detect_watchlist_alerts(_changes_df(), pd.DataFrame()).empty

    def test_case_insensitive_symbol_matching(self):
        watchlist_df = pd.DataFrame({S.SYMBOL: ["aaa"]})
        result = alerts.detect_watchlist_alerts(_changes_df(), watchlist_df)
        assert "AAA" in set(result[S.SYMBOL])


class TestAlertBundle:
    def test_is_empty_true_when_nothing_detected(self):
        bundle = alerts.AlertBundle(new_high_conviction=pd.DataFrame())
        assert bundle.is_empty()

    def test_is_empty_false_when_high_conviction_present(self):
        bundle = alerts.build_alert_bundle(_changes_df(), _today_df(), [], pd.DataFrame())
        assert not bundle.is_empty()
        assert set(bundle.new_high_conviction[S.SYMBOL]) == {"AAA", "BBB"}

    def test_build_alert_bundle_wires_all_three_detectors(self):
        saved = [{"name": "High score", "entry_query": "final_score >= 65"}]
        watchlist_df = pd.DataFrame({S.SYMBOL: ["DDD"]})
        bundle = alerts.build_alert_bundle(_changes_df(), _today_df(), saved, watchlist_df)
        assert not bundle.new_high_conviction.empty
        assert "High score" in bundle.strategy_matches
        assert set(bundle.watchlist_alerts[S.SYMBOL]) == {"DDD"}


class TestFormatMessage:
    def test_message_contains_symbols_from_all_three_sections(self):
        saved = [{"name": "High score", "entry_query": "final_score >= 65"}]
        watchlist_df = pd.DataFrame({S.SYMBOL: ["DDD"]})
        bundle = alerts.build_alert_bundle(_changes_df(), _today_df(), saved, watchlist_df)
        message = alerts.format_message(bundle)
        assert "AAA" in message
        assert "High score" in message
        assert "DDD" in message

    def test_empty_bundle_still_produces_a_string(self):
        bundle = alerts.AlertBundle(new_high_conviction=pd.DataFrame())
        message = alerts.format_message(bundle)
        assert isinstance(message, str) and "Alert Summary" in message

    def test_long_lists_are_truncated_with_a_count(self):
        big_changes = pd.DataFrame({
            S.SYMBOL: [f"SYM{i}" for i in range(15)],
            "previous_score_band": ["Watchlist"] * 15,
            "current_score_band": ["High Conviction"] * 15,
        })
        bundle = alerts.AlertBundle(new_high_conviction=big_changes)
        message = alerts.format_message(bundle, max_rows=5)
        assert "...and 10 more." in message


class TestSendTelegram:
    def test_missing_config_fails_without_network_call(self):
        with patch("stockgpt.alerts.requests.post") as mock_post:
            ok, err = alerts.send_telegram("", "", "hello")
            assert ok is False
            assert "not configured" in err
            mock_post.assert_not_called()

    def test_success_path(self):
        mock_resp = MagicMock(status_code=200)
        with patch("stockgpt.alerts.requests.post", return_value=mock_resp) as mock_post:
            ok, err = alerts.send_telegram("faketoken", "12345", "hello world")
            assert ok is True
            assert err == ""
            mock_post.assert_called_once()
            called_url = mock_post.call_args[0][0]
            assert "faketoken" in called_url

    def test_non_200_response_is_reported_as_failure(self):
        mock_resp = MagicMock(status_code=400, text="Bad Request: chat not found")
        with patch("stockgpt.alerts.requests.post", return_value=mock_resp):
            ok, err = alerts.send_telegram("faketoken", "12345", "hello")
            assert ok is False
            assert "400" in err

    def test_network_exception_is_caught_not_raised(self):
        with patch("stockgpt.alerts.requests.post", side_effect=ConnectionError("boom")):
            ok, err = alerts.send_telegram("faketoken", "12345", "hello")
            assert ok is False
            assert "boom" in err

    def test_long_message_truncated_to_telegram_limit(self):
        mock_resp = MagicMock(status_code=200)
        long_message = "x" * 5000
        with patch("stockgpt.alerts.requests.post", return_value=mock_resp) as mock_post:
            alerts.send_telegram("t", "c", long_message)
            sent_text = mock_post.call_args[1]["json"]["text"]
            assert len(sent_text) <= 4000


class TestSendEmail:
    def test_missing_config_fails_without_network_call(self):
        with patch("smtplib.SMTP") as mock_smtp:
            ok, err = alerts.send_email("", 587, "", "", "", "subject", "body")
            assert ok is False
            assert "not configured" in err
            mock_smtp.assert_not_called()

    def test_success_path(self):
        mock_server = MagicMock()
        mock_smtp_cm = MagicMock()
        mock_smtp_cm.__enter__.return_value = mock_server
        with patch("smtplib.SMTP", return_value=mock_smtp_cm):
            ok, err = alerts.send_email(
                "smtp.example.com", 587, "bot@example.com", "password",
                "me@example.com", "subject", "body",
            )
            assert ok is True
            mock_server.starttls.assert_called_once()
            mock_server.login.assert_called_once_with("bot@example.com", "password")
            mock_server.sendmail.assert_called_once()

    def test_smtp_exception_is_caught_not_raised(self):
        with patch("smtplib.SMTP", side_effect=OSError("connection refused")):
            ok, err = alerts.send_email(
                "smtp.example.com", 587, "bot@example.com", "password",
                "me@example.com", "subject", "body",
            )
            assert ok is False
            assert "connection refused" in err
