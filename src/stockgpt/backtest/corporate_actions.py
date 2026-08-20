"""Detects probable corporate-action price discontinuities (demergers, and
any other event that produces a large, permanent, non-trading-driven jump
in `current_price` that the data source doesn't retroactively adjust for)
so the backtest engine can stop treating them as genuine trading returns.

Why this exists: a demerger causes a real, often large, overnight drop in
the parent company's raw share price on its ex-date -- shareholders aren't
actually down that much, they received shares in the new entity as
compensation, but nothing in this pipeline's data (current_price /
previous_close, both raw/unadjusted) knows that. Any backtest trade whose
holding window spans that date computes a return contaminated by an event
that has nothing to do with the strategy being tested, silently corrupting
win rates and average returns for any symbol that happened to go through
one. A brand-new listing has the mirror problem on its first trading
day(s): day_change_pct is either undefined (no previous_close to diff
against) or dominated by IPO-pop volatility rather than a real signal.

There is no free, reliable feed of actual NSE corporate-action events
(session-cookie-gated, same problem noted in universe.py's module comment
for other non-archive NSE endpoints), so this doesn't try to identify WHAT
happened or WHY -- it's a purely price-based heuristic: flag any
(symbol, date) row whose day_change_pct diverges from that day's
cross-sectional median move (a cheap proxy for "how the market moved that
day", computed straight from the panel already in memory, no external
index feed needed) by more than BACKTEST_DEFAULTS.price_jump_threshold_pct
percentage points. Market-relative rather than a raw cutoff specifically so
a genuine broad-market crash/rally day (where most stocks move together)
doesn't get every symbol flagged -- only stocks moving unusually relative
to everything else that day.

This is a heuristic, not a certainty: it will occasionally flag a real,
large idiosyncratic move (a shock earnings surprise, a regulatory order)
that wasn't a corporate action, and can miss a demerger whose price impact
happened to be smaller than the threshold. Both failure directions are
judged an acceptable trade-off against the alternative of not filtering
anything at all and letting every demerger-corrupted trade count at full
weight in backtest statistics -- see run_backtest()'s docstring for how
the flag is actually applied (a trade is excluded if ANY day in its
[entry, exit] window is flagged, not just the entry/exit days themselves,
since a demerger's price effect persists in every later day's price, not
just the one day it occurred).

dashboard/static/backtest_engine.js's `computePriceJumpFlags` is a
line-for-line port of `flag_price_jumps` below -- see
tests/js/crossverify.node.js and tests/js/gen_fixture.py, which include a
deliberate jump case so Python/JS agreement on this specific logic is
actually exercised, not just assumed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import schema as S
from ..config import BACKTEST_DEFAULTS


def flag_price_jumps(panel: pd.DataFrame,
                      threshold_pct: float = BACKTEST_DEFAULTS.price_jump_threshold_pct) -> pd.Series:
    """Returns a boolean Series aligned with `panel`'s index/order: True for
    every row whose day_change_pct looks like a probable corporate action
    per this module's docstring.

    Rows with a missing/NaN day_change_pct (a symbol's first appearance in
    the panel -- new listing or start of this data's coverage -- has no
    previous_close to diff against) are flagged too, not just left False:
    that first row is exactly the "thin data, don't trust it" case a new
    listing produces, and treating "no data" as "safe to trade on" would be
    the wrong default here. This means a symbol's very first entry-eligible
    day in the panel is always excluded as an entry or exit point, which is
    the intended, deliberately conservative behavior, not an oversight.
    """
    if panel.empty or S.DAY_CHANGE_PCT not in panel.columns:
        return pd.Series(False, index=panel.index)

    day_change = pd.to_numeric(panel[S.DAY_CHANGE_PCT], errors="coerce")
    market_move = day_change.groupby(panel[S.SCAN_DATE]).transform("median")
    idiosyncratic_move = (day_change - market_move).abs()

    # NaN (missing day_change_pct, or a day with too few symbols for a
    # meaningful median -- groupby.transform("median") of a single value is
    # just that value, so idiosyncratic_move would be 0 there, which is
    # correctly "not a jump", not a bug) compares False against every
    # threshold, which combined with the explicit isna() check below is
    # exactly the deliberate "missing data is unsafe" rule the docstring
    # above describes.
    return (idiosyncratic_move >= threshold_pct) | day_change.isna()
