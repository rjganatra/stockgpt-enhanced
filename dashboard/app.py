"""StockGPT Enhanced -- Streamlit dashboard.

Filters: every numeric slider in this app is built by `adaptive_slider`,
which reads its min/max off the *live* dataset on every render. There is no
fixed price/market-cap ceiling anywhere in this file -- the original repo's
bug where a slider's hardcoded upper bound silently excluded a real stock
(a Rs.1.3L ceiling excluding a Rs.1.5L stock) cannot happen here by
construction, because there is no ceiling to go stale. See
adaptive_slider()'s docstring for the one-function guarantee.

Watchlist: browsing is open to everyone the dashboard is shared with;
adding/removing requires WATCHLIST_SECRET, checked once via
`stockgpt.watchlist.has_write_access` on every write path, per the owner's
explicit design intent (this is a public dashboard, not a public watchlist).

Tab-for-tab parity with the original's 11 tabs, plus two new ones:
Signal Performance (the original's 12-signal-type performance tracker,
recomputed through the new backtest engine instead of the original's
vs-today fuzzy-bucket method) and Strategy Lab (free-form entry/exit rules,
not restricted to a fixed signal taxonomy).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stockgpt import schema as S
from stockgpt import watchlist as W
from stockgpt.backtest import ExitMode, Strategy, run_backtest, summarize
from stockgpt.backtest.engine import load_history_panel
from stockgpt.backtest.strategy import PRESET_STRATEGIES
from stockgpt.signal_catalog import SIGNAL_CATALOG, SIGNAL_DIRECTIONS

DATA_DIR = Path("data")
STRATEGIES_PATH = DATA_DIR / "backtest" / "saved_strategies.json"

st.set_page_config(page_title="StockGPT Enhanced", layout="wide")
st.title("StockGPT Enhanced -- NSE Market Intelligence")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def adaptive_slider(label: str, series: pd.Series, step: float = 1.0,
                     key: str | None = None, sidebar: bool = True):
    """The ONE slider constructor used everywhere in this app. Bounds are
    always `series.min()`/`series.max()` off the data passed in -- never a
    literal. If you're tempted to hardcode a min/max "just this once",
    that's the exact mistake that broke the original dashboard's price
    filter for high-priced stocks. Don't; call this instead.

    When the series has no real numeric data at all (column missing, or
    entirely NaN -- e.g. the fundamentals fetch hasn't run yet), this does
    NOT fall back to a fake 0-to-step range. A slider that always shows
    "0.00 to 0.50" regardless of what's actually in the data looks adaptive
    but isn't, which is worse than no slider -- it quietly filters out real
    rows. Shows a caption and returns (None, None) instead; callers must
    skip filtering when either bound is None.

    Also has a pair of "exact" number inputs below the slider. Dragging a
    handle across a wide, skewed range (Rs 0.18 to Rs 130,985 for one
    outlier stock like MRF is a real observed case) cannot reliably land on
    an exact pixel-perfect endpoint -- a handle that visually looks like
    it's at the far edge can still sit meaningfully short of the true max,
    silently dropping the single most extreme stock with no indication
    anything was excluded. Typing an exact number sidesteps that entirely.
    The slider and the two number boxes are kept in sync both ways via
    on_change callbacks that write directly into each other's
    session_state key before the other widget is (re)constructed on the
    resulting rerun -- same key-write-before-mount mechanism as the Reset
    button's key-versioning fix. Typing an exact value moves the slider;
    dragging the slider updates the number boxes to match, so they never
    silently show a stale full range while the slider is actually
    narrower. The slider's own return value is still what every caller
    filters on."""
    target = st.sidebar if sidebar else st
    clean = pd.to_numeric(series, errors="coerce").replace([float("inf"), float("-inf")], pd.NA).dropna()
    if clean.empty:
        target.caption(f"{label}: no data available yet.")
        return None, None
    lo, hi = float(clean.min()), float(clean.max())
    if hi <= lo:
        hi = lo + float(step)

    slider_key = key
    exact_min_key = f"{key}_exact_min" if key else None
    exact_max_key = f"{key}_exact_max" if key else None

    def _sync_slider_from_inputs():
        """Typed a number -> move the slider. Reads the just-changed number
        boxes and writes the slider's own session_state entry, which the
        slider widget (constructed after this callback runs, on the
        resulting rerun) will pick up in place of its `value=` default --
        same key-write-before-mount mechanism as the Reset button fix."""
        typed_min = st.session_state.get(exact_min_key, lo)
        typed_max = st.session_state.get(exact_max_key, hi)
        if typed_min > typed_max:
            typed_min, typed_max = typed_max, typed_min
        st.session_state[slider_key] = (
            max(lo, min(typed_min, hi)),
            max(lo, min(typed_max, hi)),
        )

    def _sync_inputs_from_slider():
        """Dragged the slider -> update the number boxes to match. Without
        this, the boxes would silently keep showing the full lo/hi range
        after a drag, which is worse than no number boxes at all -- it would
        look like nothing is excluded when the slider is actually narrower.
        Only one of these two callbacks fires per interaction (whichever
        widget the user actually touched), so there's no update loop."""
        current = st.session_state.get(slider_key, (lo, hi))
        st.session_state[exact_min_key] = current[0]
        st.session_state[exact_max_key] = current[1]

    col_min, col_max = target.columns(2)
    col_min.number_input(
        f"{label} -- exact min", min_value=lo, max_value=hi, value=lo, step=float(step),
        key=exact_min_key, on_change=_sync_slider_from_inputs, label_visibility="collapsed",
    )
    col_max.number_input(
        f"{label} -- exact max", min_value=lo, max_value=hi, value=hi, step=float(step),
        key=exact_max_key, on_change=_sync_slider_from_inputs, label_visibility="collapsed",
    )

    widget = st.sidebar.slider if sidebar else st.slider
    return widget(
        label, min_value=lo, max_value=hi, value=(lo, hi), step=float(step),
        key=slider_key, on_change=_sync_inputs_from_slider,
    )


@st.cache_data(ttl=300)
def load_scan() -> pd.DataFrame:
    path = DATA_DIR / "scans" / "latest_scan.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    for col in [S.SECTOR, S.INDUSTRY, S.SECTOR_BUCKET, S.TREND, S.SCORE_BAND, S.REASONS]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)
    return df


@st.cache_data(ttl=600)
def load_history_panel_cached() -> pd.DataFrame:
    return load_history_panel(DATA_DIR / "history")


@st.cache_data(ttl=600)
def compute_signal_catalog_performance() -> pd.DataFrame:
    """Runs every catalog signal through the real backtest engine, once,
    cached -- this is the Signal Performance tab's data source."""
    panel = load_history_panel_cached()
    if panel.empty:
        return pd.DataFrame()
    rows = []
    for strategy in SIGNAL_CATALOG:
        try:
            trades = run_backtest(panel, strategy)
        except ValueError:
            continue
        if not trades:
            continue
        summary = summarize(trades, strategy.name)
        summary["direction"] = SIGNAL_DIRECTIONS.get(strategy.name, "Bullish")
        rows.append(summary)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def get_secret(name: str, default: str = "") -> str:
    try:
        return st.secrets[name]
    except Exception:
        return default


def load_saved_strategies() -> list[dict]:
    if not STRATEGIES_PATH.exists():
        return []
    try:
        return json.loads(STRATEGIES_PATH.read_text())
    except Exception:
        return []


def save_strategy(strategy: Strategy) -> None:
    STRATEGIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    saved = load_saved_strategies()
    saved = [s for s in saved if s["name"] != strategy.name]
    saved.append(strategy.to_dict())
    STRATEGIES_PATH.write_text(json.dumps(saved, indent=2))


def get_watchlist_store():
    repo = get_secret("GITHUB_REPO")
    branch = get_secret("GITHUB_BRANCH", "main")
    token = get_secret("GITHUB_TOKEN")
    return W.GitHubWatchlistStore(repo=repo, branch=branch, token=token) if repo else None


def render_add_to_watchlist(candidates: pd.DataFrame, key_prefix: str, default_basket: str) -> None:
    """One reusable add-to-watchlist control, used on every basket-style
    section (Opportunities' three sections, Stock Explorer) so a user
    browsing a specific list can add from it directly instead of retyping
    symbols on the Watchlist tab."""
    with st.expander(f"Add these to watchlist ({len(candidates)} shown)"):
        entered_key = st.text_input("Access key", type="password", key=f"{key_prefix}_key")
        if not W.has_write_access(entered_key, get_secret("WATCHLIST_SECRET")):
            st.caption("Enter the access key to add stocks from this list.")
            return
        options = sorted(candidates[S.SYMBOL].unique()) if S.SYMBOL in candidates.columns else []
        chosen = st.multiselect("Symbols", options, default=options[:10], key=f"{key_prefix}_symbols")
        basket = st.selectbox("Basket", W.BASKETS,
                               index=W.BASKETS.index(default_basket) if default_basket in W.BASKETS else 0,
                               key=f"{key_prefix}_basket")
        if st.button("Add to watchlist", key=f"{key_prefix}_add_btn"):
            store = get_watchlist_store()
            if not store:
                st.error("GITHUB_REPO secret not configured; can't persist the watchlist.")
            else:
                ok, msg = W.add_symbols(store, chosen, basket, "",
                                         pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"))
                if ok:
                    st.success("Added.")
                else:
                    st.error(msg)


def render_column_reference(reference_df: pd.DataFrame, key_prefix: str) -> None:
    """Shared "what can I actually type in this box" reference, used by both
    the Overview tab's custom filter and Strategy Lab. Built from the live
    dataframe's own columns/dtypes/sample values -- not a hand-maintained
    list -- so it never goes stale and any field that shows up in a future
    scan (new fundamentals ratio, new signal, whatever) appears here
    automatically with no code change required."""
    with st.expander("Which columns can I use? (click to see available fields + example values)"):
        numeric_cols = sorted(reference_df.select_dtypes(include="number").columns.tolist())
        text_cols = sorted(reference_df.select_dtypes(exclude="number").columns.tolist())
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Numeric fields** (use `>`, `<`, `>=`, `<=`, `==`)")
            rows = []
            for col in numeric_cols:
                clean = reference_df[col].dropna()
                sample = f"{clean.min():.2f} to {clean.max():.2f}" if not clean.empty else "no data yet"
                rows.append({"field": col, "range today": sample})
            st.dataframe(pd.DataFrame(rows), width="stretch", height=250, key=f"{key_prefix}_numeric_ref")
        with c2:
            st.markdown("**Text fields** (use `==`, `!=`, or `.str.contains('x')`)")
            rows = []
            for col in text_cols:
                clean = reference_df[col].dropna().astype(str)
                sample = ", ".join(sorted(clean.unique())[:4]) if not clean.empty else "no data yet"
                rows.append({"field": col, "example values": sample})
            st.dataframe(pd.DataFrame(rows), width="stretch", height=250, key=f"{key_prefix}_text_ref")
        st.caption(
            "Combine with `and` / `or`, e.g. `dividend_yield > 2 and rsi > 55` or "
            "`score_band == \"Strong\" and sector == \"Information Technology\"`. "
            "Text values need quotes; field names are typed exactly as shown above."
        )


df = load_scan()
if df.empty:
    st.warning("No scan data yet. Run scripts/run_daily_pipeline.py first.")
    st.stop()

if S.SCAN_TIME in df.columns and not df[S.SCAN_TIME].dropna().empty:
    st.caption(f"Last scanned: {df[S.SCAN_TIME].dropna().iloc[0]}")


# ---------------------------------------------------------------------------
# Sidebar filters (all adaptive -- see adaptive_slider docstring)
# ---------------------------------------------------------------------------
st.sidebar.header("Filters")
st.sidebar.caption("Every slider below is bounded by today's actual data, not a fixed number.")

# Deleting a widget's session_state entry and calling st.rerun() (the first
# attempt here) resets the VALUE Streamlit computes, but not necessarily
# what's painted on screen -- slider and text_input frontend components are
# stateful and don't always repaint from a cleared key, so the widget can
# keep showing its old handle position / old text even though the backend
# value (and therefore the filtered data) has genuinely reset underneath it.
# Confirmed live: after a reset, "Matching stocks" and the results table
# both updated correctly, but the price slider stayed visually parked at
# its narrowed position and the custom-query text box kept showing its old
# query. The reliable fix is to change every filter widget's *key* on
# reset, not just clear its value -- a new key makes Streamlit mount a
# genuinely new widget instance with a clean default, which always repaints
# correctly, instead of asking the old instance to reset itself.
if "filter_reset_version" not in st.session_state:
    st.session_state["filter_reset_version"] = 0
if st.sidebar.button("Reset filters", help="Clears every filter below back to \"show everything\"."):
    st.session_state["filter_reset_version"] += 1
    st.rerun()
_rv = st.session_state["filter_reset_version"]

all_symbols = sorted(df[S.SYMBOL].dropna().unique()) if S.SYMBOL in df.columns else []
search_symbols = st.sidebar.multiselect(
    "Search symbol", all_symbols, default=[], placeholder="Type to search...",
    help="Options are today's actual scanned symbols -- pulled live, never a fixed list.",
    key=f"flt_search_symbols_{_rv}",
)
sectors = sorted(df[S.SECTOR].dropna().unique()) if S.SECTOR in df.columns else []
selected_sectors = st.sidebar.multiselect("Sectors", sectors, default=sectors, key=f"flt_sectors_{_rv}")
bands = sorted(df[S.SCORE_BAND].dropna().unique()) if S.SCORE_BAND in df.columns else []
selected_bands = st.sidebar.multiselect("Score band", bands, default=bands, key=f"flt_bands_{_rv}")

price_min, price_max = adaptive_slider("Current price", df.get(S.CURRENT_PRICE, pd.Series(dtype=float)), step=1.0, key=f"price_{_rv}")
score_min, score_max = adaptive_slider("Final score", df.get(S.FINAL_SCORE, pd.Series(dtype=float)), step=1.0, key=f"score_{_rv}")
risk_min, risk_max = adaptive_slider("Risk penalty", df.get(S.RISK_PENALTY, pd.Series(dtype=float)), step=1.0, key=f"risk_{_rv}")
rsi_min, rsi_max = adaptive_slider("RSI", df.get(S.RSI, pd.Series(dtype=float)), step=0.5, key=f"rsi_{_rv}")

filtered = df.copy()
if selected_sectors:
    filtered = filtered[filtered[S.SECTOR].isin(selected_sectors)]
if selected_bands:
    filtered = filtered[filtered[S.SCORE_BAND].isin(selected_bands)]
if search_symbols:
    filtered = filtered[filtered[S.SYMBOL].isin(search_symbols)]
if S.CURRENT_PRICE in filtered.columns and price_min is not None:
    filtered = filtered[filtered[S.CURRENT_PRICE].between(price_min, price_max)]
if S.FINAL_SCORE in filtered.columns and score_min is not None:
    filtered = filtered[filtered[S.FINAL_SCORE].between(score_min, score_max)]
if S.RISK_PENALTY in filtered.columns and risk_min is not None:
    filtered = filtered[filtered[S.RISK_PENALTY].between(risk_min, risk_max)]
if S.RSI in filtered.columns and rsi_min is not None:
    filtered = filtered[filtered[S.RSI].between(rsi_min, rsi_max)]

st.sidebar.metric("Matching stocks", len(filtered))


# ---------------------------------------------------------------------------
# Tabs -- same 11 as the original, in the same order, plus Signal
# Performance and Strategy Lab.
# ---------------------------------------------------------------------------
(tab_overview, tab_heatmap, tab_opportunities, tab_sectors, tab_explorer,
 tab_history, tab_watchlist, tab_fundamentals, tab_changes, tab_range,
 tab_signal_perf, tab_strategy) = st.tabs([
    "Overview", "Heatmap", "Opportunities", "Sectors", "Stock Explorer",
    "History", "Watchlist", "Fundamentals", "Movers & Changes", "Range Bound",
    "Signal Performance", "Strategy Lab",
])

# --- Overview ----------------------------------------------------------------
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stocks scanned", len(df))
    c2.metric("Matching filters", len(filtered))
    c3.metric("Avg final score", round(df[S.FINAL_SCORE].mean(), 1) if S.FINAL_SCORE in df else "-")
    c4.metric("High Conviction", int((df[S.SCORE_BAND] == S.BAND_HIGH_CONVICTION).sum()) if S.SCORE_BAND in df else "-")

    if S.SCORE_BAND in df.columns:
        band_counts = df[S.SCORE_BAND].value_counts().reindex(S.BAND_ORDER, fill_value=0)
        fig = px.bar(band_counts, title="Stocks per score band")
        st.plotly_chart(fig, width="stretch")

    st.subheader("Custom filter")
    st.caption(
        "The sidebar sliders cover a handful of common fields. For anything else -- dividend "
        "yield, PE, any fundamental ratio, any combination -- type a filter expression here "
        "instead of us guessing which sliders you'll want next. Works with whatever columns "
        "exist in today's scan, so it stays useful as more fields land (e.g. once "
        "scripts/run_weekly_fundamentals.py has run and PE/dividend_yield/etc. are populated)."
    )
    render_column_reference(filtered, "overview_custom_filter")
    custom_query = st.text_input(
        "Filter expression", value="", key=f"overview_custom_query_{_rv}",
        placeholder="e.g. dividend_yield > 2 and rsi > 55",
    )
    custom_filtered = filtered
    if custom_query.strip():
        try:
            custom_filtered = filtered.query(custom_query, engine="python")
            st.caption(f"{len(custom_filtered)} of {len(filtered)} filtered stocks match.")
        except Exception as e:  # noqa: BLE001
            st.error(f"Couldn't evaluate that filter: {e}")
            custom_filtered = filtered

    show_cols = [c for c in [S.SYMBOL, S.SECTOR, S.CURRENT_PRICE, S.FINAL_SCORE, S.SCORE_BAND,
                              S.TECHNICAL_SCORE, S.SECTOR_ADJUSTED_FUNDAMENTAL_SCORE,
                              S.RELATIVE_STRENGTH_SCORE, S.RISK_PENALTY, S.REASONS] if c in custom_filtered.columns]
    st.dataframe(custom_filtered.sort_values(S.FINAL_SCORE, ascending=False)[show_cols].head(200),
                 width="stretch")

# --- Heatmap -------------------------------------------------------------
with tab_heatmap:
    have_cols = all(c in filtered.columns for c in (S.SECTOR, S.INDUSTRY, S.SYMBOL, S.FINAL_SCORE))
    if not have_cols or filtered.empty:
        st.info("Not enough sector/industry data in this scan to build a heatmap yet.")
    else:
        st.header("Opportunity Heatmap")
        heat_df = filtered.copy()
        heat_df["_size"] = heat_df[S.FINAL_SCORE].apply(lambda x: max(float(x), 1))
        hover = [c for c in [S.CURRENT_PRICE, S.DAY_CHANGE_PCT, S.RSI, S.VOLUME_RATIO,
                              S.TECHNICAL_SCORE, S.SECTOR_ADJUSTED_FUNDAMENTAL_SCORE,
                              S.RELATIVE_STRENGTH_SCORE, S.RISK_PENALTY, S.SCORE_BAND] if c in heat_df.columns]
        fig = px.treemap(
            heat_df, path=[S.SECTOR, S.INDUSTRY, S.SYMBOL], values="_size",
            color=S.FINAL_SCORE, color_continuous_scale=["red", "yellow", "green"],
            hover_data=hover,
            title="Green = higher final conviction, red = lower",
        )
        st.plotly_chart(fig, width="stretch")

        if S.DAY_CHANGE_PCT in heat_df.columns and S.CURRENT_PRICE in heat_df.columns:
            st.header("Daily Movement Heatmap")
            move_df = filtered.copy()
            move_df["_size"] = move_df[S.CURRENT_PRICE].apply(lambda x: max(float(x), 1))
            move_fig = px.treemap(
                move_df, path=[S.SECTOR, S.INDUSTRY, S.SYMBOL], values="_size",
                color=S.DAY_CHANGE_PCT, color_continuous_scale=["red", "white", "green"],
                hover_data=[c for c in [S.CURRENT_PRICE, S.DAY_CHANGE_PCT, S.RSI, S.VOLUME_RATIO,
                                         S.TREND, S.FINAL_SCORE, S.SCORE_BAND] if c in move_df.columns],
                title="Green = up today, red = down today",
            )
            st.plotly_chart(move_fig, width="stretch")

# --- Opportunities ---------------------------------------------------------
with tab_opportunities:
    st.caption(f"Baskets are calculated from all {len(filtered)} filtered stocks.")

    st.subheader("52W Low Opportunities")
    low_opp = pd.DataFrame()
    if S.DISTANCE_FROM_LOW_PCT in filtered.columns:
        low_opp = filtered[filtered[S.DISTANCE_FROM_LOW_PCT] <= 15].sort_values(
            [S.DISTANCE_FROM_LOW_PCT, S.FINAL_SCORE], ascending=[True, False])
        if low_opp.empty:
            st.info("No stocks currently qualify as 52W Low Opportunities under active filters.")
        else:
            st.dataframe(low_opp, width="stretch")
            render_add_to_watchlist(low_opp, "low_opp", "52W Low Opportunities")

    st.divider()
    st.subheader("Swing Candidates")
    swing = pd.DataFrame()
    if all(c in filtered.columns for c in (S.DISTANCE_FROM_LOW_PCT, S.RSI, S.VOLUME_RATIO)):
        swing = filtered[(filtered[S.DISTANCE_FROM_LOW_PCT] <= 25) & (filtered[S.RSI] <= 45)
                          & (filtered[S.VOLUME_RATIO] >= 1.0)].sort_values(S.FINAL_SCORE, ascending=False)
        if swing.empty:
            st.info("No stocks currently qualify as Swing Candidates under active filters.")
        else:
            st.dataframe(swing, width="stretch")
            render_add_to_watchlist(swing, "swing", "Swing Candidates")

    st.divider()
    st.subheader("Near 52W High Momentum")
    if all(c in filtered.columns for c in (S.DISTANCE_FROM_HIGH_PCT, S.RSI, S.TREND)):
        near_high = filtered[(filtered[S.DISTANCE_FROM_HIGH_PCT] <= 15) & (filtered[S.RSI] >= 50)
                              & (filtered[S.TREND] == "Bullish")].sort_values(
            [S.DISTANCE_FROM_HIGH_PCT, S.FINAL_SCORE], ascending=[True, False])
        if near_high.empty:
            st.info("No stocks currently qualify as Near 52W High Momentum under active filters.")
        else:
            st.dataframe(near_high, width="stretch")
            render_add_to_watchlist(near_high, "near_high", "Near 52W High Momentum")

# --- Sectors -----------------------------------------------------------------
with tab_sectors:
    if S.SECTOR not in filtered.columns or filtered.empty:
        st.info("No sector data in this scan yet.")
    else:
        st.header("Sector Overview")
        agg_map = {
            "stocks": (S.SYMBOL, "count"),
            "avg_final_score": (S.FINAL_SCORE, "mean"),
            "avg_technical": (S.TECHNICAL_SCORE, "mean"),
            "avg_sector_adj_fundamental": (S.SECTOR_ADJUSTED_FUNDAMENTAL_SCORE, "mean"),
            "avg_relative_strength": (S.RELATIVE_STRENGTH_SCORE, "mean"),
            "avg_risk_penalty": (S.RISK_PENALTY, "mean"),
            "avg_rsi": (S.RSI, "mean"),
        }
        agg_map = {k: v for k, v in agg_map.items() if v[0] in filtered.columns}
        sector_df = filtered.groupby(S.SECTOR, dropna=False).agg(**agg_map).reset_index().round(2)
        sector_df = sector_df.sort_values("avg_final_score", ascending=False)
        st.dataframe(sector_df, width="stretch")

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.bar(sector_df, x=S.SECTOR, y="avg_final_score",
                                    title="Sector average final score"), width="stretch")
        with c2:
            if "avg_relative_strength" in sector_df.columns:
                st.plotly_chart(px.bar(sector_df, x=S.SECTOR, y="avg_relative_strength",
                                        title="Sector average relative strength"), width="stretch")

        if S.INDUSTRY in filtered.columns:
            st.divider()
            st.header("Industry Overview")
            industry_df = filtered.groupby(S.INDUSTRY, dropna=False).agg(**agg_map).reset_index().round(2)
            industry_df = industry_df.drop(columns=["stocks"]).merge(
                filtered.groupby(S.INDUSTRY, dropna=False).size().rename("stocks").reset_index(),
                on=S.INDUSTRY,
            ).sort_values("avg_final_score", ascending=False)
            st.dataframe(industry_df, width="stretch")

            fig = px.treemap(industry_df, path=[S.INDUSTRY], values="stocks",
                              color="avg_final_score", color_continuous_scale=["red", "yellow", "green"],
                              title="Industry heatmap by average final score")
            st.plotly_chart(fig, width="stretch")

# --- Stock Explorer ------------------------------------------------------
with tab_explorer:
    if filtered.empty:
        st.info("No stocks match the current filters.")
    else:
        symbol = st.selectbox("Select a stock", sorted(filtered[S.SYMBOL].unique()))
        row = df[df[S.SYMBOL] == symbol].iloc[0]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Price", row.get(S.CURRENT_PRICE))
        c2.metric("Final score", row.get(S.FINAL_SCORE))
        c3.metric("Band", row.get(S.SCORE_BAND))
        c4.metric("Risk penalty", row.get(S.RISK_PENALTY))

        c5, c6, c7 = st.columns(3)
        c5.metric("Technical", row.get(S.TECHNICAL_SCORE))
        c6.metric("Fundamental (sector-adj.)", row.get(S.SECTOR_ADJUSTED_FUNDAMENTAL_SCORE))
        c7.metric("Relative strength", row.get(S.RELATIVE_STRENGTH_SCORE))

        st.subheader("Reason Engine")
        st.caption("Every score component's own reasons, kept separate rather than one blended string.")
        reason_fields = [
            ("Technical", S.TECHNICAL_REASONS), ("Fundamental", S.FUNDAMENTAL_REASONS),
            ("Relative strength", "relative_strength_reasons"), ("Range", S.RANGE_REASONS),
            ("Risk", S.RISK_REASONS),
        ]
        any_reason = False
        for label, field in reason_fields:
            value = row.get(field)
            if isinstance(value, str) and value.strip():
                any_reason = True
                st.markdown(f"**{label}:** {value}")
        if not any_reason:
            st.caption("No reason text recorded for this stock in today's scan.")

        with st.expander("Full row data"):
            st.json(row.to_dict())

        render_add_to_watchlist(pd.DataFrame([row]), f"explorer_{symbol}", "Personal Watchlist")

# --- History ---------------------------------------------------------------
with tab_history:
    history_root = DATA_DIR / "history"
    if not history_root.exists() or not any(history_root.iterdir()):
        st.warning("No historical snapshots available yet.")
    else:
        folders = sorted([p.name for p in history_root.iterdir() if p.is_dir()], reverse=True)
        if not folders:
            st.warning("No historical snapshots available yet.")
        else:
            selected_date = st.selectbox("Select snapshot date", folders)
            hist_file = history_root / selected_date / "scan.csv"
            if hist_file.exists():
                st.dataframe(pd.read_csv(hist_file, low_memory=False), width="stretch")
            else:
                st.warning("Snapshot file missing for this date.")

# --- Watchlist ---------------------------------------------------------------
with tab_watchlist:
    st.caption("Anyone can browse. Adding/removing requires the access key.")
    store = get_watchlist_store()
    watchlist_df = store.load() if store else W.empty_watchlist()
    st.dataframe(watchlist_df, width="stretch")

    if not watchlist_df.empty and "basket" in watchlist_df.columns:
        st.subheader("Watchlist by basket")
        basket_summary = watchlist_df.groupby("basket").agg(
            symbols=(S.SYMBOL, "count")).reset_index().sort_values("symbols", ascending=False)
        st.dataframe(basket_summary, width="stretch")

        st.subheader("Concentration Check")
        st.caption(
            "A basket that's accidentally all one sector isn't really diversified, even if it "
            "has 20 symbols in it. Sector/industry looked up against today's scan."
        )
        basket_options = ["All baskets combined"] + sorted(watchlist_df["basket"].unique())
        chosen_basket = st.selectbox("Basket", basket_options, key="wl_conc_basket")
        conc_source = watchlist_df if chosen_basket == "All baskets combined" \
            else watchlist_df[watchlist_df["basket"] == chosen_basket]

        if conc_source.empty:
            st.caption("No symbols in this basket yet.")
        elif S.SYMBOL not in conc_source.columns or S.SECTOR not in df.columns:
            st.caption("Can't compute concentration -- missing symbol or sector data.")
        else:
            wl_with_sector = conc_source.merge(
                df[[S.SYMBOL, S.SECTOR]].drop_duplicates(S.SYMBOL), on=S.SYMBOL, how="left")
            wl_with_sector[S.SECTOR] = wl_with_sector[S.SECTOR].fillna("Unknown")
            total = len(wl_with_sector)
            sector_counts = wl_with_sector[S.SECTOR].value_counts()
            sector_pct = (sector_counts / total * 100).round(1)

            conc_threshold = st.slider(
                "Flag a sector as concentrated above this % of the basket",
                min_value=10, max_value=100, value=30, step=5, key="wl_conc_threshold",
            )
            concentrated = sector_pct[sector_pct > conc_threshold]
            if not concentrated.empty:
                for sector, pct in concentrated.items():
                    st.warning(
                        f"{sector}: {pct}% of this basket ({sector_counts[sector]}/{total} stocks) "
                        f"-- above your {conc_threshold}% concentration line."
                    )
            else:
                st.success(f"No sector exceeds {conc_threshold}% of this basket ({total} stocks).")

            conc_df = pd.DataFrame({
                "sector": sector_pct.index, "pct_of_basket": sector_pct.values,
                "count": sector_counts.reindex(sector_pct.index).values,
            }).sort_values("pct_of_basket", ascending=False)
            st.dataframe(conc_df, width="stretch")
            fig = px.pie(conc_df, names="sector", values="count",
                         title=f"{chosen_basket} -- sector composition ({total} stocks)")
            st.plotly_chart(fig, width="stretch")

    col_add, col_remove = st.columns(2)

    with col_add:
        st.subheader("Add to watchlist")
        add_key = st.text_input("Access key", type="password", key="wl_add_key")
        if W.has_write_access(add_key, get_secret("WATCHLIST_SECRET")):
            symbols_to_add = st.multiselect("Symbols", sorted(df[S.SYMBOL].unique()), key="wl_add_symbols")
            basket = st.selectbox("Basket", W.BASKETS, key="wl_add_basket")
            if st.button("Add", key="wl_add_btn") and store:
                ok, msg = W.add_symbols(store, symbols_to_add, basket, "",
                                         pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"))
                if ok:
                    st.success("Added.")
                else:
                    st.error(msg)
        else:
            st.caption("Enter the access key to add stocks.")

    with col_remove:
        st.subheader("Remove from watchlist")
        remove_key = st.text_input("Access key", type="password", key="wl_remove_key")
        if W.has_write_access(remove_key, get_secret("WATCHLIST_SECRET")):
            existing = sorted(watchlist_df[S.SYMBOL].unique()) if not watchlist_df.empty else []
            symbols_to_remove = st.multiselect("Symbols", existing, key="wl_remove_symbols")
            if st.button("Remove", key="wl_remove_btn") and store:
                ok, msg = W.remove_symbols(store, symbols_to_remove)
                if ok:
                    st.success("Removed.")
                else:
                    st.error(msg)
        else:
            st.caption("Enter the access key to remove stocks.")

# --- Fundamentals --------------------------------------------------------
with tab_fundamentals:
    fund_path = DATA_DIR / "fundamentals" / "fundamentals_scored.csv"
    if not fund_path.exists():
        st.warning("No fundamentals file yet. Run scripts/run_weekly_fundamentals.py.")
    else:
        fund_df = pd.read_csv(fund_path, low_memory=False)
        st.caption("Fundamentals are refreshed weekly and can be up to a week stale relative "
                   "to today's price scan -- this is intentional (Yahoo throttles a full fetch).")

        st.subheader("Fundamental filters")
        if S.ROE not in fund_df.columns:
            st.caption(
                "ROE %: not present in the current fundamentals file yet. This file only carries "
                "ROE once scripts/run_weekly_fundamentals.py has run for real against live Yahoo "
                "data -- it fetches ROE reliably (see fundamentals.py::fetch_one), the column is "
                "just missing from whatever fundamentals snapshot is currently loaded."
            )
        roe_min, roe_max = adaptive_slider("ROE %", fund_df.get(S.ROE, pd.Series(dtype=float)), step=0.5, sidebar=False, key="roe")
        dte_min, dte_max = adaptive_slider("Debt/Equity", fund_df.get(S.DEBT_TO_EQUITY, pd.Series(dtype=float)), step=1.0, sidebar=False, key="dte")
        f = fund_df.copy()
        if S.ROE in f.columns and roe_min is not None:
            f = f[f[S.ROE].between(roe_min, roe_max) | f[S.ROE].isna()]
        if S.DEBT_TO_EQUITY in f.columns and dte_min is not None:
            f = f[f[S.DEBT_TO_EQUITY].between(dte_min, dte_max) | f[S.DEBT_TO_EQUITY].isna()]

        st.subheader("Fundamental Quality Table")
        sort_col = S.FUNDAMENTAL_SCORE if S.FUNDAMENTAL_SCORE in f.columns else f.columns[0]
        st.dataframe(f.sort_values(sort_col, ascending=False), width="stretch")

        if S.SECTOR_ADJUSTED_FUNDAMENTAL_SCORE in f.columns:
            st.subheader("Top Sector-Adjusted Fundamental Companies")
            st.dataframe(f.sort_values(S.SECTOR_ADJUSTED_FUNDAMENTAL_SCORE, ascending=False).head(30),
                         width="stretch")

        st.subheader("Best Combined Candidates")
        st.caption("Fundamentals (weekly) joined with today's price scan (daily) by symbol -- "
                   "these two datasets are refreshed on different schedules by design, so this "
                   "table is the one place they're brought back together.")
        if S.SYMBOL in f.columns and S.SYMBOL in df.columns:
            combined_cols = [c for c in [S.SYMBOL, S.FINAL_SCORE, S.SCORE_BAND, S.TECHNICAL_SCORE] if c in df.columns]
            combined = f.merge(df[combined_cols], on=S.SYMBOL, how="inner", suffixes=("", "_scan"))
            if S.FINAL_SCORE in combined.columns:
                st.dataframe(combined.sort_values(S.FINAL_SCORE, ascending=False).head(30), width="stretch")
            else:
                st.dataframe(combined.head(30), width="stretch")

# --- Movers & Changes ----------------------------------------------------
with tab_changes:
    changes_path = DATA_DIR / "history" / "latest_changes.csv"
    if not changes_path.exists():
        st.info("No change data yet -- needs at least two daily snapshots.")
    else:
        changes_df = pd.read_csv(changes_path, low_memory=False)
        if changes_df.empty:
            st.info("Change file exists but has no rows yet.")
        else:
            st.subheader("Change Summary")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Stocks tracked", len(changes_df))
            c2.metric("New stocks", int((changes_df["change_signal"] == "New stock").sum())
                      if "change_signal" in changes_df else "-")
            c3.metric("Band changes", int(changes_df["change_signal"].str.contains("Band changed", na=False).sum())
                      if "change_signal" in changes_df else "-")
            c4.metric("Risk increases", int(changes_df["change_signal"].str.contains("Risk increased", na=False).sum())
                      if "change_signal" in changes_df else "-")

            st.subheader("Change Filters")
            changes_symbols = sorted(changes_df[S.SYMBOL].dropna().unique()) if S.SYMBOL in changes_df.columns else []
            search_selected = st.multiselect(
                "Search symbol", changes_symbols, default=[], key="changes_search",
                placeholder="Type to search...",
            )
            cf = changes_df.copy()
            if search_selected and S.SYMBOL in cf.columns:
                cf = cf[cf[S.SYMBOL].isin(search_selected)]

            if "score_change" in cf.columns:
                st.subheader("Biggest Score Improvers")
                st.dataframe(cf.sort_values("score_change", ascending=False).head(20), width="stretch")

                st.subheader("Biggest Score Droppers")
                st.dataframe(cf.sort_values("score_change", ascending=True).head(20), width="stretch")

            if "change_signal" in cf.columns:
                st.subheader("New / Improved Conviction")
                improved = cf[cf["change_signal"].str.contains(
                    "New stock|Score improved sharply|Band changed", na=False, regex=True)]
                st.dataframe(improved, width="stretch")

                st.subheader("Risk / Weakness Alerts")
                risky = cf[cf["change_signal"].str.contains(
                    "Risk increased|Score dropped sharply", na=False, regex=True)]
                st.dataframe(risky, width="stretch")

            st.subheader("Full Change Table")
            st.dataframe(cf, width="stretch")

# --- Range Bound -----------------------------------------------------------
with tab_range:
    if S.RANGE_STATUS not in filtered.columns:
        st.info("Range-bound columns not present in this scan yet.")
    else:
        st.subheader("Range Filters")
        status_options = sorted(filtered[S.RANGE_STATUS].dropna().unique())
        chosen = st.multiselect("Range status", status_options, default=status_options)
        range_filtered = filtered[filtered[S.RANGE_STATUS].isin(chosen)]

        st.subheader("Accumulation Zone")
        accum = range_filtered[range_filtered[S.RANGE_STATUS].isin(["Accumulation Zone", "Lower Range Watch"])]
        if not accum.empty:
            st.dataframe(accum.sort_values(S.RANGE_SCORE, ascending=False), width="stretch")
        else:
            st.caption("None under current filters.")

        st.subheader("Profit Booking Zone")
        booking = range_filtered[range_filtered[S.RANGE_STATUS] == "Profit Booking Zone"]
        if not booking.empty:
            st.dataframe(booking.sort_values(S.RANGE_SCORE, ascending=False), width="stretch")
        else:
            st.caption("None under current filters.")

        st.subheader("Breakdown / Volatility Risk")
        breakdown = range_filtered[range_filtered[S.RANGE_STATUS] == "Breakdown Risk"]
        if not breakdown.empty:
            st.dataframe(breakdown.sort_values(S.RANGE_SCORE, ascending=True), width="stretch")
        else:
            st.caption("None under current filters.")

        st.subheader("Full Range Bound Table")
        st.dataframe(range_filtered.sort_values(S.RANGE_SCORE, ascending=False), width="stretch")

# --- Signal Performance ----------------------------------------------------
with tab_signal_perf:
    st.caption(
        "Historical win rate for the same ~12 built-in signal types the original dashboard "
        "tracked, computed through the new backtest engine (exact fixed-day holding periods, "
        "not the original's vs-today fuzzy bucketing). Bearish rows are warnings, not buys -- "
        "for those, a negative return means the warning was right."
    )
    st.caption(
        "This runs every catalog signal through the backtest engine across all of history, which "
        "gets slower as more daily snapshots accumulate -- gated behind a button rather than run "
        "on every page load, both so other tabs aren't stuck waiting on it (Streamlit executes "
        "every tab's code on every rerun, not just the one you're looking at) and so a full page "
        "load stays fast."
    )
    if st.button("Compute signal performance", key="signal_perf_run"):
        st.session_state["signal_perf_computed"] = True

    if not st.session_state.get("signal_perf_computed"):
        st.info("Click 'Compute signal performance' to run the backtest across all catalog signals.")
    else:
        with st.spinner("Running every catalog signal through the backtest engine..."):
            perf_df = compute_signal_catalog_performance()

        if perf_df.empty:
            st.info("No history snapshots to backtest against yet.")
        else:
            p1, p2 = st.columns(2)
            with p1:
                signal_options = sorted(perf_df["strategy_name"].unique())
                chosen_signals = st.multiselect("Signal type", signal_options, default=[], placeholder="All")
            with p2:
                horizon_options = sorted(perf_df["horizon_label"].unique())
                chosen_horizons = st.multiselect("Horizon", horizon_options, default=[], placeholder="All")

            pf = perf_df.copy()
            if chosen_signals:
                pf = pf[pf["strategy_name"].isin(chosen_signals)]
            if chosen_horizons:
                pf = pf[pf["horizon_label"].isin(chosen_horizons)]

            st.dataframe(
                pf.sort_values(["avg_return_pct", "win_rate_pct"], ascending=[False, False]),
                width="stretch",
            )
            for _, row in pf.iterrows():
                if row.get("low_sample_warning"):
                    st.caption(f"'{row['strategy_name']}' / {row['horizon_label']}: only "
                               f"{row['closed_trades']} closed trades -- treat as a rough signal.")

# --- Strategy Lab ------------------------------------------------------------
with tab_strategy:
    st.caption(
        "Phrase a rule like \"final_score >= 65 and score_band == 'Strong'\" and see what "
        "your actual historical win rate would have been -- computed from real daily snapshots, "
        "not eyeballed."
    )
    st.caption(
        "Preset strategies below come from backtest/strategy.py's PRESET_STRATEGIES list -- "
        "adding a new Strategy() there (or saving one from this tab) makes it show up in the "
        "dropdown automatically, no other code changes needed. Same for the Signal Performance "
        "tab's catalog (src/stockgpt/signal_catalog.py)."
    )
    _history_panel_for_ref = load_history_panel_cached()
    render_column_reference(_history_panel_for_ref if not _history_panel_for_ref.empty else df,
                             "strategy_lab_ref")

    saved = load_saved_strategies()
    all_strategies = {s.name: s for s in PRESET_STRATEGIES}
    all_strategies.update({s["name"]: Strategy.from_dict(s) for s in saved})

    chosen_name = st.selectbox("Start from a strategy", ["-- New custom strategy --"] + list(all_strategies.keys()))
    base = all_strategies.get(chosen_name)
    if base and base.description:
        st.info(base.description)

    name = st.text_input("Strategy name", value=base.name if base else "My strategy")
    entry_query = st.text_area(
        "Entry condition (pandas query syntax, evaluated against daily scan columns -- "
        "see the column reference above for exact field names and today's actual value ranges)",
        value=base.entry_query if base else "final_score >= 65 and score_band == 'Strong'",
    )
    exit_mode = st.radio("Exit style", [ExitMode.FIXED_HOLDING.value, ExitMode.CONDITION_EXIT.value],
                          index=0 if not base or base.exit_mode == ExitMode.FIXED_HOLDING else 1)

    fixed_days = base.fixed_holding_days if base else (7, 15, 30, 60)
    exit_query = base.exit_query if base else ""
    if exit_mode == ExitMode.FIXED_HOLDING.value:
        days_text = st.text_input("Holding periods (days, comma separated)", value=",".join(str(d) for d in fixed_days))
        try:
            fixed_days = tuple(int(x.strip()) for x in days_text.split(",") if x.strip())
        except ValueError:
            st.error("Holding periods must be comma-separated integers.")
            fixed_days = (7, 15, 30, 60)
    else:
        exit_query = st.text_input(
            "Exit condition (leave blank to exit as soon as the entry condition stops matching)",
            value=exit_query or "",
        )

    win_threshold = st.number_input("A trade counts as a win if return % is above", value=0.0, step=0.5)

    col_run, col_save = st.columns(2)
    run_clicked = col_run.button("Run backtest", type="primary")
    save_clicked = col_save.button("Save this strategy")

    strategy = Strategy(
        name=name, entry_query=entry_query,
        exit_mode=ExitMode(exit_mode), fixed_holding_days=fixed_days,
        exit_query=exit_query or None, win_return_threshold_pct=win_threshold,
    )

    if save_clicked:
        try:
            save_strategy(strategy)
            st.success(f"Saved '{name}'.")
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not save: {e}")

    if run_clicked:
        panel = load_history_panel_cached()
        if panel.empty:
            st.warning("No history snapshots found yet -- the backtester needs at least a few "
                       "days of data/history/YYYY-MM-DD/scan.csv to test against.")
        else:
            try:
                trades = run_backtest(panel, strategy)
            except ValueError as e:
                st.error(str(e))
                trades = None

            if trades is not None:
                if not trades:
                    st.info("No historical signals matched this entry condition.")
                else:
                    summary = summarize(trades, strategy.name, win_threshold)
                    st.dataframe(summary, width="stretch")
                    for _, row in summary.iterrows():
                        if row["low_sample_warning"]:
                            st.caption(
                                f"'{row['horizon_label']}': only {row['closed_trades']} closed trades -- "
                                "treat this win rate as a rough signal, not a reliable statistic."
                            )

    st.divider()
    st.subheader("Parameter sweep")
    st.caption(
        "Testing one threshold at a time above means guessing, running, tweaking, running again. "
        "This instead runs the SAME entry condition across a whole range of threshold values in "
        "one pass and ranks them by historical performance -- write your condition with `{t}` "
        "wherever the number you want to sweep goes, e.g. "
        "`final_score >= {t} and score_band == \"Strong\"`. Fixed-holding exit only for now "
        "(condition-based exit sweeps are a fair bit more complex and not built yet)."
    )
    sweep_template = st.text_input(
        "Entry condition template (use {t} as the swept value)",
        value="final_score >= {t}", key="sweep_template",
    )
    sc1, sc2, sc3 = st.columns(3)
    sweep_min = sc1.number_input("Sweep from", value=50.0, step=5.0, key="sweep_min")
    sweep_max = sc2.number_input("Sweep to", value=80.0, step=5.0, key="sweep_max")
    sweep_step = sc3.number_input("Step", value=5.0, step=1.0, min_value=0.5, key="sweep_step")
    sweep_days_text = st.text_input("Holding periods (days, comma separated)", value="15,30", key="sweep_days")
    sweep_win_threshold = st.number_input(
        "A trade counts as a win if return % is above", value=0.0, step=0.5, key="sweep_win_threshold",
    )

    if st.button("Run parameter sweep", key="sweep_run"):
        st.session_state["sweep_computed"] = True

    if st.session_state.get("sweep_computed"):
        try:
            sweep_days = tuple(int(x.strip()) for x in sweep_days_text.split(",") if x.strip())
            thresholds = []
            v = sweep_min
            while v <= sweep_max + 1e-9:
                thresholds.append(round(v, 4))
                v += sweep_step
            if not thresholds:
                raise ValueError("Sweep range produced no values -- check from/to/step.")
        except ValueError as e:
            st.error(f"Invalid sweep settings: {e}")
            thresholds = []
            sweep_days = ()

        if thresholds:
            panel = load_history_panel_cached()
            if panel.empty:
                st.warning("No history snapshots found yet -- the backtester needs at least a few "
                           "days of data/history/YYYY-MM-DD/scan.csv to test against.")
            else:
                sweep_rows = []
                errors = []
                no_signal_thresholds = []
                with st.spinner(f"Running {len(thresholds)} backtests..."):
                    for t in thresholds:
                        try:
                            query = sweep_template.format(t=t)
                        except (KeyError, IndexError) as e:
                            errors.append(f"t={t}: couldn't fill template ({e})")
                            continue
                        sweep_strategy = Strategy(
                            name=f"sweep(t={t})", entry_query=query,
                            exit_mode=ExitMode.FIXED_HOLDING, fixed_holding_days=sweep_days,
                        )
                        try:
                            sweep_trades = run_backtest(panel, sweep_strategy)
                        except ValueError as e:
                            errors.append(f"t={t}: {e}")
                            continue
                        if not sweep_trades:
                            no_signal_thresholds.append(t)
                            continue
                        sweep_summary = summarize(sweep_trades, f"t={t}", sweep_win_threshold)
                        sweep_summary["threshold"] = t
                        sweep_rows.append(sweep_summary)

                if errors:
                    with st.expander(f"{len(errors)} threshold(s) failed to run"):
                        for e in errors:
                            st.caption(e)
                if no_signal_thresholds:
                    st.caption(
                        f"No historical signals matched at threshold(s): "
                        f"{', '.join(str(t) for t in no_signal_thresholds)} -- not a bug, just "
                        f"too strict for this history window, so they're left out of the table below."
                    )

                if not sweep_rows:
                    st.info("No threshold in this range produced any historical signals.")
                else:
                    sweep_result = pd.concat(sweep_rows, ignore_index=True)
                    sweep_result = sweep_result.sort_values(
                        ["horizon_label", "avg_return_pct"], ascending=[True, False])
                    st.dataframe(
                        sweep_result[["threshold", "horizon_label", "closed_trades", "win_rate_pct",
                                       "avg_return_pct", "median_return_pct", "low_sample_warning"]],
                        width="stretch",
                    )
                    for horizon in sorted(sweep_result["horizon_label"].unique()):
                        hz = sweep_result[sweep_result["horizon_label"] == horizon]
                        fig = px.line(hz.sort_values("threshold"), x="threshold", y="avg_return_pct",
                                      markers=True, title=f"Avg return % vs threshold ({horizon})")
                        st.plotly_chart(fig, width="stretch")
