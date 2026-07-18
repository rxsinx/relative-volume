"""
RVOL + RSI Divergence Scanner
=============================
Streamlit app -- RVOL/Chg%/Strong-Start from the original strategy, plus RSI
divergence on 15-minute and 1-day timeframes, over a NSE equity watchlist.
Data source: yfinance (Yahoo Finance) -- no broker API/login required.

    RVOL = today's volume / SMA(prior N days' volume)   [today excluded]
    Chg% = (CMP - prev_close) / prev_close * 100
    SS   = today_open > prev_close AND today_low >= prev_close * 0.995

    RSI divergence (per timeframe, over the last 50/100 candles):
      Bullish = price makes a LOWER swing low while RSI makes a HIGHER low
      Bearish = price makes a HIGHER swing high while RSI makes a LOWER high

NOTE ON DATA: Yahoo Finance data is free but delayed (typically 15-20+ min
for NSE) and rate-limited -- this is not a substitute for a broker feed for
time-sensitive execution. 15-minute history is capped at the last 60 days by
Yahoo itself; that's a Yahoo limit, not something this app can work around.
"""
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh

# ============================================================================
# indicators -- RSI, pivot-based divergence, RVOL/Chg%/SS
# ============================================================================

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(avg_loss != 0, 100.0)
    rsi = rsi.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)
    return rsi


def _dedupe_adjacent(idx_list):
    """Collapses runs of adjacent pivot indices (flat plateaus) to their midpoint."""
    if not idx_list:
        return []
    groups = [[idx_list[0]]]
    for i in idx_list[1:]:
        if i - groups[-1][-1] <= 1:
            groups[-1].append(i)
        else:
            groups.append([i])
    return [g[len(g) // 2] for g in groups]


def local_pivots(values: np.ndarray, order: int = 3):
    """Swing lows/highs: a bar is a pivot if it's the min/max within `order`
    bars on each side."""
    n = len(values)
    minima, maxima = [], []
    for i in range(order, n - order):
        window = values[i - order: i + order + 1]
        v = values[i]
        if v == window.min():
            minima.append(i)
        if v == window.max():
            maxima.append(i)
    return _dedupe_adjacent(minima), _dedupe_adjacent(maxima)


def detect_divergence(close: pd.Series, rsi: pd.Series, lookback: int = 50, order: int = 3) -> dict:
    """Compares the two most recent price swing lows (bullish) / swing highs
    (bearish) against RSI at those same bars -- the standard divergence
    definition.

    Returns a dict, not just booleans, so the caller can show/chart exactly
    where the divergence was found:
        {
          "bullish": bool, "bearish": bool,
          "bullish_detail": {"prev": {"time", "price", "rsi"}, "curr": {...}} or None,
          "bearish_detail": {...} or None,
        }
    "prev"/"curr" are the two pivot bars compared, oldest first.
    """
    empty = {"bullish": False, "bearish": False, "bullish_detail": None, "bearish_detail": None}
    n = len(close)
    lb = min(lookback, n)
    if lb < order * 2 + 2:
        return empty

    # NOTE: deliberately NOT using reset_index() here (unlike an earlier
    # version) -- .iloc still gives positional access on a Series with its
    # original index intact, and keeping the DatetimeIndex is exactly what
    # lets us report *when* each pivot happened, not just its bar position.
    c = close.iloc[-lb:]
    r = rsi.iloc[-lb:]
    first_valid = r.first_valid_index()
    if first_valid is None:
        return empty
    c = c.loc[first_valid:]
    r = r.loc[first_valid:]
    if len(c) < order * 2 + 2:
        return empty

    minima, maxima = local_pivots(c.values, order=order)

    def _point(pos: int) -> dict:
        return {"time": c.index[pos], "price": float(c.iloc[pos]), "rsi": float(r.iloc[pos])}

    bullish, bullish_detail = False, None
    if len(minima) >= 2:
        i_prev, i_curr = minima[-2], minima[-1]
        bullish = bool((c.iloc[i_curr] < c.iloc[i_prev]) and (r.iloc[i_curr] > r.iloc[i_prev]))
        if bullish:
            bullish_detail = {"prev": _point(i_prev), "curr": _point(i_curr)}

    bearish, bearish_detail = False, None
    if len(maxima) >= 2:
        j_prev, j_curr = maxima[-2], maxima[-1]
        bearish = bool((c.iloc[j_curr] > c.iloc[j_prev]) and (r.iloc[j_curr] < r.iloc[j_prev]))
        if bearish:
            bearish_detail = {"prev": _point(j_prev), "curr": _point(j_curr)}

    return {"bullish": bullish, "bearish": bearish,
            "bullish_detail": bullish_detail, "bearish_detail": bearish_detail}


def compute_rvol_chg_ss(daily: pd.DataFrame, lookback: int = 20, intraday: pd.DataFrame = None) -> Optional[dict]:
    """`daily`: OHLCV sorted ascending; the last row is treated as "today" --
    if the market's live this is Yahoo's developing bar for today, if closed
    it's simply the last completed session. No special-casing needed either
    way, unlike a live broker feed.

    Yahoo's "today" daily bar can occasionally have a NaN Close mid-session
    (especially on a fast/volatile move) while Open/Low/Volume are already
    populated -- if so, fall back to the most recent valid 15m close, which
    Yahoo's intraday feed tends to populate more reliably than the daily
    aggregate bar."""
    if daily is None or len(daily) < lookback + 2:
        return None
    today = daily.iloc[-1]
    prev = daily.iloc[-2]
    avg_vol = daily["Volume"].iloc[-(lookback + 1):-1].mean()
    volume = today["Volume"]

    cmp_ = today["Close"]
    if pd.isna(cmp_) and intraday is not None and len(intraday) > 0:
        valid_intraday_close = intraday["Close"].dropna()
        if len(valid_intraday_close) > 0:
            cmp_ = valid_intraday_close.iloc[-1]

    prev_close = prev["Close"]
    volume_ok = not pd.isna(volume) and volume > 0
    rvol = (volume / avg_vol) if (avg_vol and volume_ok) else None

    cmp_ok = not pd.isna(cmp_)
    prev_close_ok = not pd.isna(prev_close) and prev_close != 0
    chg_pct = ((cmp_ - prev_close) / prev_close * 100) if (cmp_ok and prev_close_ok) else None

    today_open, today_low = today["Open"], today["Low"]
    strong_start = bool(
        not pd.isna(today_open) and not pd.isna(today_low) and prev_close_ok
        and today_open > prev_close and today_low >= prev_close * 0.995
    )

    return {
        "cmp": float(cmp_) if cmp_ok else None,
        "rvol": rvol,
        "chg_pct": chg_pct,
        "strong_start": strong_start,
        "volume": float(volume) if not pd.isna(volume) else None,
        "avg_volume": avg_vol,
        "cmp_from_fallback": cmp_ok and pd.isna(today["Close"]),
    }


# ============================================================================
# watchlist parsing
# ============================================================================

def parse_symbols(raw: str, suffix: str = ".NS") -> list:
    norm = raw.replace("\r", ",").replace("\n", ",")
    tokens = [t.strip().upper() for t in norm.split(",")]
    out = []
    for tok in tokens:
        if not tok or tok.startswith("###"):
            continue
        out.append(tok if "." in tok else f"{tok}{suffix}")
    return out


# ============================================================================
# data fetch (batched, cached)
# ============================================================================

@st.cache_data(ttl=300, show_spinner=False)
def fetch_batch(tickers: tuple, period: str, interval: str) -> dict:
    """One batched yfinance call for the whole watchlist. Returns
    {ticker: DataFrame} -- a ticker maps to None if its fetch failed/was empty."""
    if not tickers:
        return {}
    raw = yf.download(list(tickers), period=period, interval=interval,
                       group_by="ticker", threads=True, progress=False, auto_adjust=False)
    result = {}
    for t in tickers:
        try:
            df = raw[t].dropna(how="all")
            result[t] = df if len(df) > 0 else None
        except (KeyError, TypeError):
            result[t] = None
    return result


@dataclass
class ScanRow:
    symbol: str
    cmp: Optional[float]
    rvol: Optional[float]
    chg_pct: Optional[float]
    strong_start: bool
    rsi_15m: Optional[float]
    div_15m: Optional[str]   # "Bullish" | "Bearish" | None
    div_15m_detail: Optional[dict]
    rsi_1d: Optional[float]
    div_1d: Optional[str]
    div_1d_detail: Optional[dict]
    note: Optional[str] = None


def scan(tickers: list, rvol_lookback: int, div_lookback: int, pivot_order: int) -> list:
    daily_data = fetch_batch(tuple(tickers), period="1y", interval="1d")
    intraday_data = fetch_batch(tuple(tickers), period="60d", interval="15m")

    rows = []
    for t in tickers:
        daily = daily_data.get(t)
        intraday = intraday_data.get(t)

        if daily is None or len(daily) < rvol_lookback + 2:
            rows.append(ScanRow(t, None, None, None, False, None, None, None, None, None, None,
                                 note="No/insufficient daily data from Yahoo Finance for this ticker."))
            continue

        base = compute_rvol_chg_ss(daily, lookback=rvol_lookback, intraday=intraday)
        if base is None:
            rows.append(ScanRow(t, None, None, None, False, None, None, None, None, None, None,
                                 note="Could not compute RVOL (insufficient history)."))
            continue

        row_note = None
        if base.get("cmp_from_fallback"):
            row_note = "Today's daily Close was missing from Yahoo's feed -- CMP/Chg% use the latest 15m close instead."

        # pandas' ewm().mean() carries the PREVIOUS value forward when the
        # trailing input is NaN rather than emitting NaN -- so if today's
        # Close is missing, a naive rsi.iloc[-1] silently returns yesterday's
        # RSI looking like a fresh number. Guard explicitly instead of trusting it.
        daily_close_valid = not pd.isna(daily["Close"].iloc[-1])
        rsi_1d_series = compute_rsi(daily["Close"])
        rsi_1d = float(rsi_1d_series.iloc[-1]) if (daily_close_valid and not pd.isna(rsi_1d_series.iloc[-1])) else None
        div_result_1d = detect_divergence(daily["Close"], rsi_1d_series, lookback=div_lookback, order=pivot_order)
        div_1d = "Bullish" if div_result_1d["bullish"] else ("Bearish" if div_result_1d["bearish"] else None)
        div_1d_detail = div_result_1d["bullish_detail"] if div_result_1d["bullish"] else div_result_1d["bearish_detail"]

        rsi_15m, div_15m, div_15m_detail = None, None, None
        if intraday is not None and len(intraday) >= pivot_order * 2 + 2:
            intraday_close_valid = not pd.isna(intraday["Close"].iloc[-1])
            rsi_15m_series = compute_rsi(intraday["Close"])
            rsi_15m = float(rsi_15m_series.iloc[-1]) if (intraday_close_valid and not pd.isna(rsi_15m_series.iloc[-1])) else None
            div_result_15m = detect_divergence(intraday["Close"], rsi_15m_series, lookback=div_lookback, order=pivot_order)
            div_15m = "Bullish" if div_result_15m["bullish"] else ("Bearish" if div_result_15m["bearish"] else None)
            div_15m_detail = div_result_15m["bullish_detail"] if div_result_15m["bullish"] else div_result_15m["bearish_detail"]

        rows.append(ScanRow(
            symbol=t, cmp=base["cmp"], rvol=base["rvol"], chg_pct=base["chg_pct"],
            strong_start=base["strong_start"], rsi_15m=rsi_15m, div_15m=div_15m, div_15m_detail=div_15m_detail,
            rsi_1d=rsi_1d, div_1d=div_1d, div_1d_detail=div_1d_detail, note=row_note,
        ))
    return rows


# ============================================================================
# RVOL trend (up/down arrow) -- persisted in session_state across reruns
# ============================================================================

def apply_rvol_trend(rows: list) -> dict:
    if "rvol_prev" not in st.session_state:
        st.session_state.rvol_prev = {}
    prev_map = st.session_state.rvol_prev
    trend = {}
    for r in rows:
        if r.rvol is None:
            continue
        prev = prev_map.get(r.symbol)
        if prev is None:
            trend[r.symbol] = None
        elif r.rvol > prev:
            trend[r.symbol] = "up"
        elif r.rvol < prev:
            trend[r.symbol] = "down"
        else:
            trend[r.symbol] = None
        prev_map[r.symbol] = r.rvol
    return trend


# ============================================================================
# display
# ============================================================================

def rows_to_dataframe(rows: list, trend: dict, rvol_as: str) -> pd.DataFrame:
    def fmt_rvol(r):
        if r.rvol is None:
            return "\u2013"
        pct = r.rvol * 100
        label = f"{pct:.1f}%" if (rvol_as == "percent" and pct < 10) else (f"{pct:.0f}%" if rvol_as == "percent" else f"{r.rvol:.2f}")
        arrow = {"up": " \u2191", "down": " \u2193"}.get(trend.get(r.symbol), "")
        return label + arrow

    def fmt_div(d):
        return {"Bullish": "\u25b2 Bullish", "Bearish": "\u25bc Bearish"}.get(d, "")

    data = {
        "Symbol": [r.symbol.replace(".NS", "") for r in rows],
        "CMP": [f"{r.cmp:.2f}" if r.cmp is not None else "\u2013" for r in rows],
        "RVOL": [fmt_rvol(r) for r in rows],
        "Chg%": [f"{r.chg_pct:+.1f}%" if r.chg_pct is not None else "\u2013" for r in rows],
        "SS": ["\u2605" if r.strong_start else "" for r in rows],
        "RSI 15m": [f"{r.rsi_15m:.0f}" if r.rsi_15m is not None else "\u2013" for r in rows],
        "Div 15m": [fmt_div(r.div_15m) for r in rows],
        "RSI 1D": [f"{r.rsi_1d:.0f}" if r.rsi_1d is not None else "\u2013" for r in rows],
        "Div 1D": [fmt_div(r.div_1d) for r in rows],
    }
    return pd.DataFrame(data)


def style_table(df: pd.DataFrame):
    def color_chg(val):
        if val == "\u2013":
            return ""
        return "color: green" if val.startswith("+") else "color: red"

    def color_div(val):
        if "Bullish" in val:
            return "color: green; font-weight: bold"
        if "Bearish" in val:
            return "color: red; font-weight: bold"
        return ""

    styler = df.style.map(color_chg, subset=["Chg%"])
    styler = styler.map(color_div, subset=["Div 15m", "Div 1D"])
    return styler


def get_price_rsi_series(ticker: str, timeframe: str, all_tickers: tuple):
    """Re-derives the raw Close/RSI series for one ticker so the detail view
    can chart them. Deliberately calls fetch_batch with the SAME tickers
    tuple used by the main scan (not just [ticker]) so this hits the
    existing st.cache_data entry instead of triggering a fresh yfinance call."""
    if timeframe == "1d":
        data = fetch_batch(all_tickers, period="1y", interval="1d")
    else:
        data = fetch_batch(all_tickers, period="60d", interval="15m")
    df = data.get(ticker)
    if df is None:
        return None, None
    close = df["Close"]
    rsi = compute_rsi(close)
    return close, rsi


def build_divergence_chart(close: pd.Series, rsi: pd.Series, detail: dict, kind: str, context_bars: int = 60):
    """Two-panel price+RSI chart with the divergence's two pivot points
    marked and connected on both panes -- the visual crosscheck. Shows some
    context before/after the pivots, not just the two points in isolation."""
    end_time = detail["curr"]["time"]
    end_pos = close.index.get_indexer([end_time], method="nearest")[0]
    start_pos = max(0, end_pos - context_bars)
    stop_pos = min(len(close), end_pos + max(5, context_bars // 6))

    price_window = close.iloc[start_pos:stop_pos]
    rsi_window = rsi.iloc[start_pos:stop_pos]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4],
                         vertical_spacing=0.06, subplot_titles=("Price", "RSI"))

    fig.add_trace(go.Scatter(x=price_window.index, y=price_window.values, mode="lines",
                              name="Price", line=dict(color="#1f77b4", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=rsi_window.index, y=rsi_window.values, mode="lines",
                              name="RSI", line=dict(color="#9467bd", width=1.5)), row=2, col=1)

    color = "#0b8043" if kind == "Bullish" else "#cc2222"
    prev, curr = detail["prev"], detail["curr"]
    fig.add_trace(go.Scatter(x=[prev["time"], curr["time"]], y=[prev["price"], curr["price"]],
                              mode="markers+lines", marker=dict(size=11, color=color, symbol="diamond"),
                              line=dict(color=color, dash="dash", width=2), name=f"{kind} (price)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[prev["time"], curr["time"]], y=[prev["rsi"], curr["rsi"]],
                              mode="markers+lines", marker=dict(size=11, color=color, symbol="diamond"),
                              line=dict(color=color, dash="dash", width=2), name=f"{kind} (RSI)"), row=2, col=1)

    fig.add_hline(y=70, line_dash="dot", line_color="gray", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="gray", row=2, col=1)

    fig.update_layout(height=480, showlegend=False, margin=dict(t=40, b=10, l=10, r=10))
    return fig


def render_divergence_detail(row: "ScanRow", all_tickers: tuple):
    """The click-through detail panel: exact From/To bars (time, price, RSI)
    plus a chart, for each timeframe that has a flagged divergence."""
    st.subheader(f"Divergence detail \u2014 {row.symbol.replace('.NS', '')}")

    panels = [("15m", row.div_15m, row.div_15m_detail), ("1D", row.div_1d, row.div_1d_detail)]
    panels = [(tf, kind, detail) for tf, kind, detail in panels if kind and detail]

    if not panels:
        st.info("No divergence flagged for this symbol on either timeframe.")
        return

    for tf, kind, detail in panels:
        arrow = "\u25b2" if kind == "Bullish" else "\u25bc"
        st.markdown(f"**{tf} \u2014 {arrow} {kind} divergence**")
        prev, curr = detail["prev"], detail["curr"]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("*From (earlier swing)*")
            st.write(f"Time: {prev['time']}")
            st.write(f"Price: {prev['price']:.2f}")
            st.write(f"RSI: {prev['rsi']:.1f}")
        with c2:
            st.markdown("*To (latest swing)*")
            st.write(f"Time: {curr['time']}")
            st.write(f"Price: {curr['price']:.2f}")
            st.write(f"RSI: {curr['rsi']:.1f}")

        close, rsi = get_price_rsi_series(row.symbol, "1d" if tf == "1D" else "15m", all_tickers)
        if close is not None:
            st.plotly_chart(build_divergence_chart(close, rsi, detail, kind), use_container_width=True)
        st.divider()


# ============================================================================
# main
# ============================================================================

def main():
    st.set_page_config(page_title="RVOL + RSI Divergence Scanner", layout="wide")
    st.title("RVOL + RSI Divergence Scanner")
    st.caption("RVOL/Chg%/Strong-Start (original strategy) + RSI divergence on 15m & 1D. Data: Yahoo Finance (delayed).")

    with st.sidebar:
        st.header("Watchlist")
        default_watchlist = "RELIANCE\nTCS\nINFY\nHDFCBANK\nICICIBANK\nSBIN"
        watchlist_text = st.text_area("Symbols (one per line, or comma-separated)", value=default_watchlist, height=150)
        uploaded = st.file_uploader("...or upload watchlist.txt", type=["txt"])
        if uploaded is not None:
            watchlist_text = uploaded.read().decode("utf-8")

        st.header("Settings")
        rvol_lookback = st.number_input("RVOL average-volume lookback (days)", min_value=5, max_value=100, value=20)
        div_lookback = st.selectbox("RSI divergence lookback (candles)", options=[50, 100], index=0)
        pivot_order = st.slider("Swing-pivot sensitivity (bars each side)", min_value=2, max_value=6, value=3,
                                 help="Lower = more sensitive (more, smaller swings flagged). Higher = only larger swings count.")
        sort_by = st.selectbox("Sort by", options=["RVOL", "Chg%", "SS"], index=0)
        rvol_as = st.radio("RVOL as", options=["percent", "ratio"], horizontal=True)

        st.header("Refresh")
        auto_refresh = st.checkbox("Auto-refresh", value=False)
        interval_s = st.number_input("Interval (seconds)", min_value=30, max_value=1800, value=300, step=30)
        run_clicked = st.button("Run scan", type="primary")

    if auto_refresh:
        st_autorefresh(interval=interval_s * 1000, key="autorefresh")

    tickers = parse_symbols(watchlist_text)
    if not tickers:
        st.warning("No symbols parsed from the watchlist.")
        return

    if not (run_clicked or auto_refresh or "last_rows" in st.session_state):
        st.info("Set up your watchlist and click **Run scan**.")
        return

    with st.spinner(f"Fetching {len(tickers)} symbols from Yahoo Finance..."):
        rows = scan(tickers, rvol_lookback=rvol_lookback, div_lookback=div_lookback, pivot_order=pivot_order)
    st.session_state.last_rows = rows

    trend = apply_rvol_trend(rows)

    key_fn = {
        "RVOL": lambda r: r.rvol if r.rvol is not None else float("-inf"),
        "Chg%": lambda r: r.chg_pct if r.chg_pct is not None else float("-inf"),
        "SS": lambda r: (1 if r.strong_start else 0,
                          r.chg_pct if r.chg_pct is not None else float("-inf"),
                          r.rvol if r.rvol is not None else float("-inf")),
    }[sort_by]
    valid = [r for r in rows if r.rvol is not None]
    invalid = [r for r in rows if r.rvol is None]
    ordered = sorted(valid, key=key_fn, reverse=True) + invalid

    df = rows_to_dataframe(ordered, trend, rvol_as)
    event = st.dataframe(
        style_table(df), use_container_width=True, hide_index=True,
        on_select="rerun", selection_mode="single-row",
    )

    selected_positions = event.selection.rows if event and event.selection else []
    if selected_positions:
        selected_row = ordered[selected_positions[0]]
        st.divider()
        render_divergence_detail(selected_row, tuple(tickers))
    else:
        st.caption("Click a row above to see exactly where a flagged divergence was found (From/To bars + chart).")

    notes = [r for r in rows if r.note]
    if notes:
        with st.expander(f"\u26a0 {len(notes)} symbol(s) with issues"):
            for r in notes:
                st.write(f"**{r.symbol}**: {r.note}")

    st.caption(f"Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')} | Data via Yahoo Finance (delayed, not for execution timing)")


if __name__ == "__main__":
    main()
