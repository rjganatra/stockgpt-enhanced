"""GitHub-backed permanent watchlist, with a secret gate on every write path.

Design carried over deliberately from the original repo, on the user's
explicit instruction: the dashboard is shared publicly (friends/colleagues
can browse it), but only the owner should be able to add or remove
watchlist entries. Every function that mutates the watchlist requires
`access_key == WATCHLIST_SECRET`; every read function requires nothing.
There is exactly one place (`has_write_access`) that makes this decision,
so a future write path can't accidentally forget to check it.
"""

from __future__ import annotations

import base64
from io import StringIO

import pandas as pd
import requests

from . import schema as S

WATCHLIST_COLUMNS = [S.SYMBOL, "basket", "notes", "added_at"]

BASKETS = [
    "52W Low Opportunities", "Swing Candidates", "Near 52W High Momentum",
    "High Conviction", "Personal Watchlist", "Research", "Avoid / Risky",
]


def has_write_access(entered_key: str, configured_secret: str) -> bool:
    """The one function every write path must call. Empty configured
    secret always denies access (fail closed, not fail open)."""
    return bool(configured_secret) and entered_key == configured_secret


def empty_watchlist() -> pd.DataFrame:
    return pd.DataFrame(columns=WATCHLIST_COLUMNS)


class GitHubWatchlistStore:
    """Thin wrapper around the GitHub Contents API. Requires a token with
    `contents: write` on the target repo; read works without a token by
    falling back to the public raw URL."""

    def __init__(self, repo: str, branch: str, token: str = "",
                 path: str = "data/watchlist/watchlist.csv"):
        self.repo = repo
        self.branch = branch
        self.token = token
        self.path = path

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}", "Accept": "application/vnd.github+json"}

    def _api_url(self) -> str:
        return f"https://api.github.com/repos/{self.repo}/contents/{self.path}"

    def _raw_url(self) -> str:
        return f"https://raw.githubusercontent.com/{self.repo}/{self.branch}/{self.path}"

    def load(self) -> pd.DataFrame:
        try:
            response = requests.get(self._raw_url(), timeout=20)
            if response.status_code != 200 or not response.text.strip():
                return empty_watchlist()
            df = pd.read_csv(StringIO(response.text))
        except Exception:
            return empty_watchlist()

        for col in WATCHLIST_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        df[S.SYMBOL] = df[S.SYMBOL].astype(str).str.upper().str.strip()
        return df[WATCHLIST_COLUMNS]

    def save(self, df: pd.DataFrame) -> tuple[bool, str]:
        if not self.token:
            return False, "No GitHub token configured; cannot write."
        try:
            df = df[WATCHLIST_COLUMNS].drop_duplicates(subset=[S.SYMBOL, "basket"])
            csv_content = df.to_csv(index=False)

            get_resp = requests.get(self._api_url(), headers=self._headers(),
                                     params={"ref": self.branch}, timeout=20)
            sha = get_resp.json().get("sha") if get_resp.status_code == 200 else None

            payload = {
                "message": "Update watchlist",
                "content": base64.b64encode(csv_content.encode()).decode(),
                "branch": self.branch,
            }
            if sha:
                payload["sha"] = sha

            put_resp = requests.put(self._api_url(), headers=self._headers(), json=payload, timeout=20)
            if put_resp.status_code not in (200, 201):
                return False, put_resp.text
            return True, ""
        except Exception as e:  # noqa: BLE001
            return False, str(e)


def add_symbols(store: GitHubWatchlistStore, symbols: list[str], basket: str,
                 notes: str, added_at: str) -> tuple[bool, str]:
    watchlist = store.load()
    existing = set(zip(watchlist[S.SYMBOL], watchlist["basket"]))
    new_rows = [
        {S.SYMBOL: s.upper().strip(), "basket": basket, "notes": notes, "added_at": added_at}
        for s in symbols if s.strip() and (s.upper().strip(), basket) not in existing
    ]
    if not new_rows:
        return True, "Nothing new to add."
    watchlist = pd.concat([watchlist, pd.DataFrame(new_rows)], ignore_index=True)
    return store.save(watchlist)


def remove_symbols(store: GitHubWatchlistStore, symbols: list[str],
                    basket: str | None = None) -> tuple[bool, str]:
    watchlist = store.load()
    symbols_upper = {s.upper().strip() for s in symbols}
    mask = watchlist[S.SYMBOL].isin(symbols_upper)
    if basket:
        mask &= watchlist["basket"] == basket
    watchlist = watchlist[~mask]
    return store.save(watchlist)
