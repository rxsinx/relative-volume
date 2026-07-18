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
import streamlit as st
import yfinance as yf
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


def detect_divergence(close: pd.Series, rsi: pd.Series, lookback: int = 50, order: int = 3):
    """Compares the two most recent price swing lows (bullish) / swing highs
    (bearish) against RSI at those same bars -- the standard divergence
    definition. Returns (bullish: bool, bearish: bool)."""
    n = len(close)
    lb = min(lookback, n)
    if lb < order * 2 + 2:
        return False, False

    c = close.iloc[-lb:].reset_index(drop=True)
    r = rsi.iloc[-lb:].reset_index(drop=True)
    first_valid = r.first_valid_index()
    if first_valid is None:
        return False, False
    c = c.iloc[first_valid:].reset_index(drop=True)
    r = r.iloc[first_valid:].reset_index(drop=True)

    minima, maxima = local_pivots(c.values, order=order)

    bullish = False
    if len(minima) >= 2:
        i_prev, i_curr = minima[-2], minima[-1]
        bullish = bool((c.iloc[i_curr] < c.iloc[i_prev]) and (r.iloc[i_curr] > r.iloc[i_prev]))

    bearish = False
    if len(maxima) >= 2:
        j_prev, j_curr = maxima[-2], maxima[-1]
        bearish = bool((c.iloc[j_curr] > c.iloc[j_prev]) and (r.iloc[j_curr] < r.iloc[j_prev]))

    return bullish, bearish


def compute_rvol_chg_ss(daily: pd.DataFrame, lookback: int = 20) -> Optional[dict]:
    """`daily`: OHLCV sorted ascending; the last row is treated as "today" --
    if the market's live this is Yahoo's developing bar for today, if closed
    it's simply the last completed session. No special-casing needed either
    way, unlike a live broker feed."""
    if daily is None or len(daily) < lookback + 2:
        return None
    today = daily.iloc[-1]
    prev = daily.iloc[-2]
    avg_vol = daily["Volume"].iloc[-(lookback + 1):-1].mean()
    volume = today["Volume"]
    rvol = (volume / avg_vol) if avg_vol else None
    cmp_ = today["Close"]
    prev_close = prev["Close"]
    chg_pct = ((cmp_ - prev_close) / prev_close * 100) if prev_close else None
    strong_start = bool(today["Open"] > prev_close and today["Low"] >= prev_close * 0.995)
    return {"cmp": float(cmp_), "rvol": rvol, "chg_pct": chg_pct,
            "strong_start": strong_start, "volume": float(volume), "avg_volume": avg_vol}


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
    rsi_1d: Optional[float]
    div_1d: Optional[str]
    note: Optional[str] = None


def scan(tickers: list, rvol_lookback: int, div_lookback: int, pivot_order: int) -> list:
    daily_data = fetch_batch(tuple(tickers), period="1y", interval="1d")
    intraday_data = fetch_batch(tuple(tickers), period="60d", interval="15m")

    rows = []
    for t in tickers:
        daily = daily_data.get(t)
        intraday = intraday_data.get(t)

        if daily is None or len(daily) < rvol_lookback + 2:
            rows.append(ScanRow(t, None, None, None, False, None, None, None, None,
                                 note="No/insufficient daily data from Yahoo Finance for this ticker."))
            continue

        base = compute_rvol_chg_ss(daily, lookback=rvol_lookback)
        if base is None:
            rows.append(ScanRow(t, None, None, None, False, None, None, None, None,
                                 note="Could not compute RVOL (insufficient history)."))
            continue

        rsi_1d_series = compute_rsi(daily["Close"])
        rsi_1d = float(rsi_1d_series.iloc[-1]) if not pd.isna(rsi_1d_series.iloc[-1]) else None
        bull_1d, bear_1d = detect_divergence(daily["Close"], rsi_1d_series, lookback=div_lookback, order=pivot_order)
        div_1d = "Bullish" if bull_1d else ("Bearish" if bear_1d else None)

        rsi_15m, div_15m = None, None
        if intraday is not None and len(intraday) >= pivot_order * 2 + 2:
            rsi_15m_series = compute_rsi(intraday["Close"])
            rsi_15m = float(rsi_15m_series.iloc[-1]) if not pd.isna(rsi_15m_series.iloc[-1]) else None
            bull_15m, bear_15m = detect_divergence(intraday["Close"], rsi_15m_series, lookback=div_lookback, order=pivot_order)
            div_15m = "Bullish" if bull_15m else ("Bearish" if bear_15m else None)

        rows.append(ScanRow(
            symbol=t, cmp=base["cmp"], rvol=base["rvol"], chg_pct=base["chg_pct"],
            strong_start=base["strong_start"], rsi_15m=rsi_15m, div_15m=div_15m,
            rsi_1d=rsi_1d, div_1d=div_1d,
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

    styler = df.style.applymap(color_chg, subset=["Chg%"])
    styler = styler.applymap(color_div, subset=["Div 15m", "Div 1D"])
    return styler


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
    st.dataframe(style_table(df), use_container_width=True, hide_index=True)

    notes = [r for r in rows if r.note]
    if notes:
        with st.expander(f"\u26a0 {len(notes)} symbol(s) with issues"):
            for r in notes:
                st.write(f"**{r.symbol}**: {r.note}")

    st.caption(f"Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')} | Data via Yahoo Finance (delayed, not for execution timing)")


if __name__ == "__main__":
    main()
