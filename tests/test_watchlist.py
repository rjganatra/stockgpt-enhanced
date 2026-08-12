import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stockgpt import watchlist as W


def test_no_secret_configured_denies_access_even_with_empty_key():
    assert W.has_write_access("", "") is False
    assert W.has_write_access("anything", "") is False


def test_correct_key_grants_access():
    assert W.has_write_access("supersecret", "supersecret") is True


def test_wrong_key_denies_access():
    assert W.has_write_access("wrong", "supersecret") is False
