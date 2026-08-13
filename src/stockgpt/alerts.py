"""Alert detection + delivery.

Runs as a step inside the daily pipeline's GitHub Actions job (not a
separate always-on service -- no infrastructure to stand up, it reuses the
runner and repo checkout the pipeline already has). Three trigger types,
each independent and additive:

1. New High Conviction stocks -- symbols that crossed into the band since
   the last snapshot. Reuses history.compute_changes()'s own band-change
   detection rather than re-deriving "new to High Conviction" a second way,
   so there's exactly one definition of a band change in the codebase.
2. Saved Strategy Lab strategies matching TODAY's scan -- a live signal
   check ("would this strategy's entry condition fire on any stock right
   now"), not a backtest. Uses the same pandas query syntax the Strategy
   Lab tab already evaluates entry conditions with.
3. Sharp changes on stocks already in the watchlist -- reuses the same
   change_signal categories the Movers & Changes tab already surfaces,
   filtered down to symbols actually being tracked.

Delivery (Telegram, email) is best-effort and independent per channel: if
one fails, that's reported in the script output, not treated as a reason to
fail the pipeline run that already succeeded at its main job of updating
the data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd
import requests

from . import schema as S

logger = logging.getLogger(__name__)

SHARP_CHANGE_PATTERN = "Score improved sharply|Score dropped sharply|Band changed|Risk increased"


@dataclass
class AlertBundle:
    new_high_conviction: pd.DataFrame
    strategy_matches: dict[str, pd.DataFrame] = field(default_factory=dict)
    watchlist_alerts: pd.DataFrame = field(default_factory=pd.DataFrame)

    def is_empty(self) -> bool:
        return (
            self.new_high_conviction.empty
            and all(df.empty for df in self.strategy_matches.values())
            and self.watchlist_alerts.empty
        )


def detect_new_high_conviction(changes_df: pd.DataFrame) -> pd.DataFrame:
    if changes_df.empty or "current_score_band" not in changes_df.columns:
        return pd.DataFrame()
    became_hc = (
        (changes_df["current_score_band"] == S.BAND_HIGH_CONVICTION)
        & (changes_df["previous_score_band"] != S.BAND_HIGH_CONVICTION)
    )
    return changes_df[became_hc]


def detect_strategy_matches(today_df: pd.DataFrame, saved_strategies: list[dict]) -> dict[str, pd.DataFrame]:
    """A strategy with a broken/typo'd query is skipped with a warning, not
    a crash -- one bad saved strategy shouldn't silence alerts for the
    others."""
    matches: dict[str, pd.DataFrame] = {}
    if today_df.empty:
        return matches
    for strat in saved_strategies:
        name = strat.get("name", "unnamed")
        query = strat.get("entry_query", "")
        if not query:
            continue
        try:
            hit = today_df.query(query, engine="python")
        except Exception as e:  # noqa: BLE001
            logger.warning("Saved strategy %r has an invalid entry_query, skipping: %s", name, e)
            continue
        if not hit.empty:
            matches[name] = hit
    return matches


def detect_watchlist_alerts(changes_df: pd.DataFrame, watchlist_df: pd.DataFrame) -> pd.DataFrame:
    if changes_df.empty or watchlist_df.empty or "change_signal" not in changes_df.columns:
        return pd.DataFrame()
    if S.SYMBOL not in watchlist_df.columns:
        return pd.DataFrame()
    watch_symbols = set(watchlist_df[S.SYMBOL].astype(str).str.upper())
    on_watchlist = changes_df[changes_df[S.SYMBOL].astype(str).str.upper().isin(watch_symbols)]
    return on_watchlist[on_watchlist["change_signal"].str.contains(SHARP_CHANGE_PATTERN, na=False, regex=True)]


def build_alert_bundle(changes_df: pd.DataFrame, today_df: pd.DataFrame,
                        saved_strategies: list[dict], watchlist_df: pd.DataFrame) -> AlertBundle:
    return AlertBundle(
        new_high_conviction=detect_new_high_conviction(changes_df),
        strategy_matches=detect_strategy_matches(today_df, saved_strategies),
        watchlist_alerts=detect_watchlist_alerts(changes_df, watchlist_df),
    )


def format_message(bundle: AlertBundle, max_rows: int = 10) -> str:
    lines = ["StockGPT Enhanced -- Alert Summary"]

    if not bundle.new_high_conviction.empty:
        lines.append(f"\nNew High Conviction ({len(bundle.new_high_conviction)}):")
        for _, row in bundle.new_high_conviction.head(max_rows).iterrows():
            lines.append(f"  {row.get(S.SYMBOL)}: {row.get('previous_score_band')} -> {row.get('current_score_band')}")
        if len(bundle.new_high_conviction) > max_rows:
            lines.append(f"  ...and {len(bundle.new_high_conviction) - max_rows} more.")

    for name, hits in bundle.strategy_matches.items():
        lines.append(f"\nSaved strategy '{name}' matched {len(hits)} stock(s) today:")
        symbols = hits[S.SYMBOL].tolist() if S.SYMBOL in hits.columns else []
        lines.append("  " + ", ".join(symbols[:max_rows]))
        if len(symbols) > max_rows:
            lines.append(f"  ...and {len(symbols) - max_rows} more.")

    if not bundle.watchlist_alerts.empty:
        lines.append(f"\nWatchlist changes ({len(bundle.watchlist_alerts)}):")
        for _, row in bundle.watchlist_alerts.head(max_rows).iterrows():
            lines.append(f"  {row.get(S.SYMBOL)}: {row.get('change_signal')}")
        if len(bundle.watchlist_alerts) > max_rows:
            lines.append(f"  ...and {len(bundle.watchlist_alerts) - max_rows} more.")

    return "\n".join(lines)


def send_telegram(bot_token: str, chat_id: str, message: str) -> tuple[bool, str]:
    if not bot_token or not chat_id:
        return False, "Telegram not configured (missing bot token or chat id)."
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={"chat_id": chat_id, "text": message[:4000]},
            timeout=20,
        )
        if resp.status_code != 200:
            return False, f"Telegram send failed ({resp.status_code}): {resp.text}"
        return True, ""
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def send_email(smtp_host: str, smtp_port: int, smtp_user: str, smtp_password: str,
               to_addr: str, subject: str, body: str) -> tuple[bool, str]:
    if not all([smtp_host, smtp_user, smtp_password, to_addr]):
        return False, "Email not configured (missing SMTP host/user/password/recipient)."
    import smtplib
    from email.mime.text import MIMEText

    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = smtp_user
        msg["To"] = to_addr
        with smtplib.SMTP(smtp_host, smtp_port, timeout=20) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(smtp_user, [to_addr], msg.as_string())
        return True, ""
    except Exception as e:  # noqa: BLE001
        return False, str(e)
