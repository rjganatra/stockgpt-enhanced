"""A Strategy is a plain, savable, human-readable definition of a rule.

This is the feature the user specifically asked for: "if I saw conviction
score above 65 and comes under strong band, what would be my win rate."
A Strategy captures exactly that sentence -- an entry condition, phrased as
a pandas query string against the daily scan columns -- plus how you'd
decide to get back out.

Two exit styles are supported, and neither is "the right one" -- they
answer different questions:

  FIXED_HOLDING -- "if I bought and held for exactly N days, no matter
  what, what's my return?" Cheap to compute, easy to compare across
  strategies and horizons side by side. This is what the original repo's
  signal_performance.py already did (7D/15D/30D/60D buckets); kept here
  because it's a genuinely useful lens, just formalised into a reusable
  Strategy object instead of hardcoded signal types.

  CONDITION_EXIT -- "if I bought when the condition became true and sold
  the moment it stopped being true, what's my return?" Closer to how
  someone would actually trade a signal, but the holding period is now a
  random variable instead of a control, so results across different
  strategies aren't directly holding-period-comparable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from ..config import BACKTEST_DEFAULTS


class ExitMode(str, Enum):
    FIXED_HOLDING = "fixed_holding"
    CONDITION_EXIT = "condition_exit"


@dataclass
class Strategy:
    name: str
    entry_query: str
    exit_mode: ExitMode = ExitMode.FIXED_HOLDING
    fixed_holding_days: tuple = BACKTEST_DEFAULTS.fixed_holding_days
    exit_query: Optional[str] = None          # None -> exit when entry_query stops matching
    win_return_threshold_pct: float = BACKTEST_DEFAULTS.win_return_threshold_pct
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Strategy needs a name")
        if not self.entry_query.strip():
            raise ValueError("Strategy needs an entry_query")
        if self.exit_mode == ExitMode.FIXED_HOLDING and not self.fixed_holding_days:
            raise ValueError("FIXED_HOLDING strategies need at least one holding period")

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "entry_query": self.entry_query,
            "exit_mode": self.exit_mode.value,
            "fixed_holding_days": list(self.fixed_holding_days),
            "exit_query": self.exit_query,
            "win_return_threshold_pct": self.win_return_threshold_pct,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Strategy":
        return cls(
            name=d["name"],
            entry_query=d["entry_query"],
            exit_mode=ExitMode(d.get("exit_mode", "fixed_holding")),
            fixed_holding_days=tuple(d.get("fixed_holding_days") or BACKTEST_DEFAULTS.fixed_holding_days),
            exit_query=d.get("exit_query"),
            win_return_threshold_pct=d.get("win_return_threshold_pct", BACKTEST_DEFAULTS.win_return_threshold_pct),
            description=d.get("description", ""),
        )


# A handful of ready-made strategies mirroring the score-band language the
# user wants to keep, so the Strategy Lab isn't a blank page on first open.
PRESET_STRATEGIES = [
    Strategy(
        name="Strong band entry, 30D hold",
        entry_query="final_score >= 65 and score_band in ['Strong', 'High Conviction']",
        exit_mode=ExitMode.FIXED_HOLDING,
        fixed_holding_days=(7, 15, 30, 60),
        description="Enter whenever a stock crosses into Strong/High Conviction; "
                     "check return after fixed holding windows.",
    ),
    Strategy(
        name="High conviction, ride the band",
        entry_query="final_score >= 65 and score_band in ['Strong', 'High Conviction']",
        exit_mode=ExitMode.CONDITION_EXIT,
        description="Enter on Strong/High Conviction, exit the day it drops out of that band.",
    ),
    Strategy(
        name="Low risk quality",
        entry_query="risk_penalty <= 10 and sector_adjusted_fundamental_score >= 55",
        exit_mode=ExitMode.FIXED_HOLDING,
        fixed_holding_days=(15, 30, 60),
        description="Quality fundamentals with controlled risk penalty.",
    ),
    Strategy(
        name="52W low + oversold RSI",
        entry_query="distance_from_low_pct <= 15 and rsi <= 40",
        exit_mode=ExitMode.CONDITION_EXIT,
        exit_query="rsi >= 55",
        description="Enter near the 52W low while RSI is oversold; exit once RSI recovers past 55.",
    ),
]
