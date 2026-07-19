"""
RVOL + Divergence Scanner
=========================
Streamlit app -- RVOL/Chg%/Strong-Start from the original strategy, plus
multi-indicator divergence (RSI, OBV, MFI) on 15-minute and 1-day
timeframes, over a NSE equity watchlist. Data source: yfinance (Yahoo
Finance) -- no broker API/login required.

    RVOL = today's volume / SMA(prior N days' volume)   [today excluded]
    Chg% = (CMP - prev_close) / prev_close * 100
    SS   = today_open > prev_close AND today_low >= prev_close * 0.995

DIVERGENCE, generalized across RSI / OBV / MFI:
    Regular bullish = price makes a flat-or-lower swing low, indicator rises  (reversal-up signal)
    Hidden  bullish = price makes a clear HIGHER swing low, indicator falls  (uptrend-continuation signal)
    Regular bearish = price makes a flat-or-higher swing high, indicator falls (reversal-down signal)
    Hidden  bearish = price makes a clear LOWER swing high, indicator rises  (downtrend-continuation signal)

CLASS GRADING (A strongest, C weakest) -- both price and indicator moves
are measured as a fraction of their own recent range within the lookback
window, so the same A/B/C rule works for RSI/MFI (0-100) and OBV
(unbounded, stock-specific scale) alike:
    Class B = price pivot-to-pivot move is ~flat (a double top/bottom) -- the classic "double divergence"
    Class C = price makes a clear new extreme, but the indicator's counter-move is weak
    Class A = both price's new extreme AND the indicator's counter-move are clear/strong

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

INDICATOR_LABEL = {"rsi": "RSI", "obv": "OBV", "mfi": "MFI"}
CLASS_RANK = {"A": 3, "B": 2, "C": 1}
SIGNAL_TYPES = ("regular_bullish", "hidden_bullish", "regular_bearish", "hidden_bearish")


def _empty_signals() -> dict:
    return {sig: None for sig in SIGNAL_TYPES}


def _empty_div() -> dict:
    return {"rsi": _empty_signals(), "obv": _empty_signals(), "mfi": _empty_signals()}


# ============================================================================
# indicators -- RSI, OBV, MFI
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


def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume.fillna(0)).cumsum()


def compute_mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
    typical_price = (high + low + close) / 3
    raw_money_flow = typical_price * volume
    tp_diff = typical_price.diff()
    positive_flow = raw_money_flow.where(tp_diff > 0, 0.0)
    negative_flow = raw_money_flow.where(tp_diff < 0, 0.0)
    positive_sum = positive_flow.rolling(period, min_periods=period).sum()
    negative_sum = negative_flow.rolling(period, min_periods=period).sum()
    mfr = positive_sum / negative_sum
    mfi = 100 - (100 / (1 + mfr))
    mfi = mfi.where(negative_sum != 0, 100.0)
    mfi = mfi.where(~((positive_sum == 0) & (negative_sum == 0)), 50.0)
    return mfi


# ============================================================================
# pivots + generalized divergence detection (regular/hidden, class A/B/C)
# ============================================================================

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


def _classify_pivot_pair(price_delta, ind_delta, price_range, ind_range,
                          flat_thresh: float, weak_thresh: float, family: str):
    """`family`: 'bullish' (minima pivots) or 'bearish' (maxima pivots).
    Returns (signal_type, class_grade) -- signal_type in {'regular','hidden',None}.

    Both deltas are expressed as a fraction of their own recent range before
    any thresholding, which is what lets the same A/B/C rule apply to RSI,
    MFI (0-100 bounded) and OBV (unbounded) uniformly."""
    price_frac = (price_delta / price_range) if price_range > 0 else 0.0
    ind_frac = (ind_delta / ind_range) if ind_range > 0 else 0.0

    price_state = "higher" if price_frac > flat_thresh else ("lower" if price_frac < -flat_thresh else "flat")
    ind_dir = "up" if ind_frac > 0 else ("down" if ind_frac < 0 else "flat")
    ind_strong = abs(ind_frac) > weak_thresh

    if ind_dir == "flat":
        return None, None

    if family == "bullish":
        if price_state in ("flat", "lower") and ind_dir == "up":
            return "regular", ("B" if price_state == "flat" else ("A" if ind_strong else "C"))
        if price_state == "higher" and ind_dir == "down":
            return "hidden", ("A" if ind_strong else "C")
    else:
        if price_state in ("flat", "higher") and ind_dir == "down":
            return "regular", ("B" if price_state == "flat" else ("A" if ind_strong else "C"))
        if price_state == "lower" and ind_dir == "up":
            return "hidden", ("A" if ind_strong else "C")

    return None, None


def analyze_pivots(close: pd.Series, indicator: pd.Series, lookback: int = 50, order: int = 3,
                    flat_thresh: float = 0.15, weak_thresh: float = 0.15) -> dict:
    """Generalized divergence detector -- works for RSI, OBV, or MFI as the
    `indicator` series (any series aligned to the same index as `close`).

    Compares the two most recent price swing lows (bullish family) and
    swing highs (bearish family) against the indicator at those same bars,
    and classifies each pair as regular or hidden divergence (or neither),
    graded A/B/C.

    Returns a dict with keys "regular_bullish", "hidden_bullish",
    "regular_bearish", "hidden_bearish" -- each either None or
    {"class": "A"/"B"/"C", "prev": {...}, "curr": {...}}, where prev/curr
    hold {"time", "price", "indicator"}.
    """
    result = _empty_signals()
    n = len(close)
    lb = min(lookback, n)
    if lb < order * 2 + 2:
        return result

    c = close.iloc[-lb:]
    ind = indicator.iloc[-lb:]
    first_valid = ind.first_valid_index()
    if first_valid is None:
        return result
    c = c.loc[first_valid:]
    ind = ind.loc[first_valid:]
    if len(c) < order * 2 + 2:
        return result

    price_range = c.max() - c.min()
    ind_range = ind.max() - ind.min()
    minima, maxima = local_pivots(c.values, order=order)

    def _point(pos: int) -> dict:
        return {"time": c.index[pos], "price": float(c.iloc[pos]), "indicator": float(ind.iloc[pos])}

    if len(minima) >= 2:
        i_prev, i_curr = minima[-2], minima[-1]
        price_delta = c.iloc[i_curr] - c.iloc[i_prev]
        ind_delta = ind.iloc[i_curr] - ind.iloc[i_prev]
        sig, grade = _classify_pivot_pair(price_delta, ind_delta, price_range, ind_range, flat_thresh, weak_thresh, "bullish")
        if sig:
            result[f"{sig}_bullish"] = {"class": grade, "prev": _point(i_prev), "curr": _point(i_curr)}

    if len(maxima) >= 2:
        j_prev, j_curr = maxima[-2], maxima[-1]
        price_delta = c.iloc[j_curr] - c.iloc[j_prev]
        ind_delta = ind.iloc[j_curr] - ind.iloc[j_prev]
        sig, grade = _classify_pivot_pair(price_delta, ind_delta, price_range, ind_range, flat_thresh, weak_thresh, "bearish")
        if sig:
            result[f"{sig}_bearish"] = {"class": grade, "prev": _point(j_prev), "curr": _point(j_curr)}

    return result


def summarize_divergence(div_by_indicator: dict) -> Optional[dict]:
    """Picks the single strongest signal across RSI/OBV/MFI x regular/hidden
    x bullish/bearish for compact main-table display, and lists which
    indicators agree on that exact same signal type (confluence)."""
    best = None
    for ind_name, signals in div_by_indicator.items():
        for sig_type in SIGNAL_TYPES:
            info = signals.get(sig_type)
            if info is None:
                continue
            is_regular = sig_type.startswith("regular")
            rank = (CLASS_RANK[info["class"]], 1 if is_regular else 0)
            if best is None or rank > best["rank"]:
                best = {"rank": rank, "indicator": ind_name, "sig_type": sig_type,
                        "is_regular": is_regular, "class": info["class"], "detail": info}

    if best is None:
        return None

    agreeing = [INDICATOR_LABEL[name] for name, signals in div_by_indicator.items()
                if signals.get(best["sig_type"]) is not None]
    direction = "Bullish" if best["sig_type"].endswith("bullish") else "Bearish"

    return {
        "direction": direction, "is_regular": best["is_regular"], "class": best["class"],
        "primary_indicator": INDICATOR_LABEL[best["indicator"]], "agreeing_indicators": agreeing,
        "detail": best["detail"], "sig_type": best["sig_type"],
    }


# ============================================================================
# RVOL / Chg% / Strong-Start (original strategy)
# ============================================================================

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
    rsi_1d: Optional[float]
    div_15m: dict   # {"rsi": {...4 signal types...}, "obv": {...}, "mfi": {...}}
    div_1d: dict
    note: Optional[str] = None


def _patch_trailing_nan_close(df: pd.DataFrame, fallback_close: float) -> pd.DataFrame:
    """Returns a copy of `df` with the last row's Close (and High, if it
    would otherwise sit below the patched Close) replaced by
    `fallback_close`. Used so RSI/OBV/MFI don't choke on -- or silently
    mask via ewm's carry-forward behavior -- a NaN trailing Close."""
    patched = df.copy()
    close_col = patched.columns.get_loc("Close")
    patched.iloc[-1, close_col] = fallback_close
    high_col = patched.columns.get_loc("High")
    current_high = patched.iloc[-1, high_col]
    if pd.isna(current_high) or current_high < fallback_close:
        patched.iloc[-1, high_col] = fallback_close
    return patched


def scan(tickers: list, rvol_lookback: int, div_lookback: int, pivot_order: int,
         flat_thresh: float = 0.15, weak_thresh: float = 0.15) -> list:
    daily_data = fetch_batch(tuple(tickers), period="1y", interval="1d")
    intraday_data = fetch_batch(tuple(tickers), period="60d", interval="15m")

    rows = []
    for t in tickers:
        daily = daily_data.get(t)
        intraday = intraday_data.get(t)

        if daily is None or len(daily) < rvol_lookback + 2:
            rows.append(ScanRow(t, None, None, None, False, None, None, _empty_div(), _empty_div(),
                                 note="No/insufficient daily data from Yahoo Finance for this ticker."))
            continue

        base = compute_rvol_chg_ss(daily, lookback=rvol_lookback, intraday=intraday)
        if base is None:
            rows.append(ScanRow(t, None, None, None, False, None, None, _empty_div(), _empty_div(),
                                 note="Could not compute RVOL (insufficient history)."))
            continue

        row_note = None
        daily_for_indicators = daily
        if base.get("cmp_from_fallback"):
            row_note = "Today's daily Close was missing from Yahoo's feed -- CMP/Chg%/indicators use the latest 15m close instead."
            daily_for_indicators = _patch_trailing_nan_close(daily, base["cmp"])

        rsi_1d_series = compute_rsi(daily_for_indicators["Close"])
        obv_1d_series = compute_obv(daily_for_indicators["Close"], daily_for_indicators["Volume"])
        mfi_1d_series = compute_mfi(daily_for_indicators["High"], daily_for_indicators["Low"],
                                     daily_for_indicators["Close"], daily_for_indicators["Volume"])
        rsi_1d = float(rsi_1d_series.iloc[-1]) if not pd.isna(rsi_1d_series.iloc[-1]) else None

        div_1d = {
            "rsi": analyze_pivots(daily_for_indicators["Close"], rsi_1d_series, div_lookback, pivot_order, flat_thresh, weak_thresh),
            "obv": analyze_pivots(daily_for_indicators["Close"], obv_1d_series, div_lookback, pivot_order, flat_thresh, weak_thresh),
            "mfi": analyze_pivots(daily_for_indicators["Close"], mfi_1d_series, div_lookback, pivot_order, flat_thresh, weak_thresh),
        }

        rsi_15m, div_15m = None, _empty_div()
        if intraday is not None and len(intraday) >= pivot_order * 2 + 2:
            intraday_for_indicators = intraday
            if pd.isna(intraday["Close"].iloc[-1]):
                valid = intraday["Close"].dropna()
                if len(valid) > 0:
                    intraday_for_indicators = _patch_trailing_nan_close(intraday, valid.iloc[-1])

            rsi_15m_series = compute_rsi(intraday_for_indicators["Close"])
            obv_15m_series = compute_obv(intraday_for_indicators["Close"], intraday_for_indicators["Volume"])
            mfi_15m_series = compute_mfi(intraday_for_indicators["High"], intraday_for_indicators["Low"],
                                          intraday_for_indicators["Close"], intraday_for_indicators["Volume"])
            rsi_15m = float(rsi_15m_series.iloc[-1]) if not pd.isna(rsi_15m_series.iloc[-1]) else None

            div_15m = {
                "rsi": analyze_pivots(intraday_for_indicators["Close"], rsi_15m_series, div_lookback, pivot_order, flat_thresh, weak_thresh),
                "obv": analyze_pivots(intraday_for_indicators["Close"], obv_15m_series, div_lookback, pivot_order, flat_thresh, weak_thresh),
                "mfi": analyze_pivots(intraday_for_indicators["Close"], mfi_15m_series, div_lookback, pivot_order, flat_thresh, weak_thresh),
            }

        rows.append(ScanRow(
            symbol=t, cmp=base["cmp"], rvol=base["rvol"], chg_pct=base["chg_pct"],
            strong_start=base["strong_start"], rsi_15m=rsi_15m, rsi_1d=rsi_1d,
            div_15m=div_15m, div_1d=div_1d, note=row_note,
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
# display -- main table
# ============================================================================

def rows_to_dataframe(rows: list, trend: dict, rvol_as: str) -> pd.DataFrame:
    def fmt_rvol(r):
        if r.rvol is None:
            return "\u2013"
        pct = r.rvol * 100
        label = f"{pct:.1f}%" if (rvol_as == "percent" and pct < 10) else (f"{pct:.0f}%" if rvol_as == "percent" else f"{r.rvol:.2f}")
        arrow = {"up": " \u2191", "down": " \u2193"}.get(trend.get(r.symbol), "")
        return label + arrow

    def fmt_div(div_dict):
        summary = summarize_divergence(div_dict)
        if summary is None:
            return ""
        arrow = "\u25b2" if summary["direction"] == "Bullish" else "\u25bc"
        kind = "" if summary["is_regular"] else "H"
        indicators = "+".join(summary["agreeing_indicators"])
        return f"{arrow}{kind}{summary['class']} {indicators}"

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
        if val.startswith("\u25b2"):
            return "color: green; font-weight: bold"
        if val.startswith("\u25bc"):
            return "color: red; font-weight: bold"
        return ""

    styler = df.style.map(color_chg, subset=["Chg%"])
    styler = styler.map(color_div, subset=["Div 15m", "Div 1D"])
    return styler


# ============================================================================
# display -- divergence detail (click-through)
# ============================================================================

def get_price_indicator_series(ticker: str, timeframe: str, indicator_name: str, all_tickers: tuple):
    """Re-derives the raw Close + indicator series for one ticker so the
    detail view can chart them. Deliberately calls fetch_batch with the SAME
    tickers tuple used by the main scan (not just [ticker]) so this hits the
    existing st.cache_data entry instead of triggering a fresh yfinance call."""
    if timeframe == "1d":
        data = fetch_batch(all_tickers, period="1y", interval="1d")
    else:
        data = fetch_batch(all_tickers, period="60d", interval="15m")
    df = data.get(ticker)
    if df is None:
        return None, None
    close = df["Close"]
    if indicator_name == "rsi":
        ind = compute_rsi(close)
    elif indicator_name == "obv":
        ind = compute_obv(close, df["Volume"])
    elif indicator_name == "mfi":
        ind = compute_mfi(df["High"], df["Low"], close, df["Volume"])
    else:
        return None, None
    return close, ind


def build_divergence_chart(close: pd.Series, indicator_series: pd.Series, detail: dict,
                            direction: str, indicator_name: str, context_bars: int = 60):
    """Two-panel price+indicator chart with the divergence's two pivot
    points marked and connected on both panes -- the visual crosscheck."""
    end_time = detail["curr"]["time"]
    end_pos = close.index.get_indexer([end_time], method="nearest")[0]
    start_pos = max(0, end_pos - context_bars)
    stop_pos = min(len(close), end_pos + max(5, context_bars // 6))

    price_window = close.iloc[start_pos:stop_pos]
    ind_window = indicator_series.iloc[start_pos:stop_pos]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4],
                         vertical_spacing=0.06, subplot_titles=("Price", indicator_name))

    fig.add_trace(go.Scatter(x=price_window.index, y=price_window.values, mode="lines",
                              name="Price", line=dict(color="#1f77b4", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=ind_window.index, y=ind_window.values, mode="lines",
                              name=indicator_name, line=dict(color="#9467bd", width=1.5)), row=2, col=1)

    color = "#0b8043" if direction == "Bullish" else "#cc2222"
    prev, curr = detail["prev"], detail["curr"]
    fig.add_trace(go.Scatter(x=[prev["time"], curr["time"]], y=[prev["price"], curr["price"]],
                              mode="markers+lines", marker=dict(size=11, color=color, symbol="diamond"),
                              line=dict(color=color, dash="dash", width=2), name=f"{direction} (price)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[prev["time"], curr["time"]], y=[prev["indicator"], curr["indicator"]],
                              mode="markers+lines", marker=dict(size=11, color=color, symbol="diamond"),
                              line=dict(color=color, dash="dash", width=2), name=f"{direction} ({indicator_name})"), row=2, col=1)

    if indicator_name in ("RSI", "MFI"):
        fig.add_hline(y=70, line_dash="dot", line_color="gray", row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="gray", row=2, col=1)

    fig.update_layout(height=480, showlegend=False, margin=dict(t=40, b=10, l=10, r=10))
    return fig


def all_signals_for_row(row: ScanRow) -> list:
    """Flattens row.div_15m / row.div_1d into a list of
    (label, timeframe, indicator, sig_type, info) for every signal found,
    sorted strongest-class first."""
    out = []
    for tf_label, div_dict in [("15m", row.div_15m), ("1D", row.div_1d)]:
        for ind_name, signals in div_dict.items():
            for sig_type in SIGNAL_TYPES:
                info = signals.get(sig_type)
                if info is None:
                    continue
                direction = "Bullish" if sig_type.endswith("bullish") else "Bearish"
                kind = "Regular" if sig_type.startswith("regular") else "Hidden"
                label = f"{tf_label} | {INDICATOR_LABEL[ind_name]} | {kind} {direction} | Class {info['class']}"
                out.append((label, tf_label, ind_name, sig_type, info))
    out.sort(key=lambda x: CLASS_RANK[x[4]["class"]], reverse=True)
    return out


def render_divergence_detail(row: ScanRow, all_tickers: tuple):
    st.subheader(f"Divergence detail \u2014 {row.symbol.replace('.NS', '')}")

    signals = all_signals_for_row(row)
    if not signals:
        st.info("No divergence flagged for this symbol on either timeframe/indicator.")
        return

    labels = [s[0] for s in signals]
    choice = st.selectbox("Signal to inspect", options=labels, key=f"div_choice_{row.symbol}")
    _, tf_label, ind_name, sig_type, info = next(s for s in signals if s[0] == choice)

    direction = "Bullish" if sig_type.endswith("bullish") else "Bearish"
    kind = "Regular" if sig_type.startswith("regular") else "Hidden"
    arrow = "\u25b2" if direction == "Bullish" else "\u25bc"
    indicator_label = INDICATOR_LABEL[ind_name]
    st.markdown(f"**{arrow} {kind} {direction} divergence \u2014 {indicator_label}, Class {info['class']}**")

    prev, curr = info["prev"], info["curr"]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("*From (earlier swing)*")
        st.write(f"Time: {prev['time']}")
        st.write(f"Price: {prev['price']:.2f}")
        st.write(f"{indicator_label}: {prev['indicator']:.2f}")
    with c2:
        st.markdown("*To (latest swing)*")
        st.write(f"Time: {curr['time']}")
        st.write(f"Price: {curr['price']:.2f}")
        st.write(f"{indicator_label}: {curr['indicator']:.2f}")

    close, ind_series = get_price_indicator_series(row.symbol, "1d" if tf_label == "1D" else "15m", ind_name, all_tickers)
    if close is not None:
        fig = build_divergence_chart(close, ind_series, info, direction, indicator_label)
        st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# main
# ============================================================================

def main():
    st.set_page_config(page_title="RVOL + Divergence Scanner", layout="wide")
    st.title("RVOL + Divergence Scanner")
    st.caption("RVOL/Chg%/Strong-Start (original strategy) + RSI/OBV/MFI divergence (regular & hidden, Class A/B/C) on 15m & 1D. Data: Yahoo Finance (delayed).")

    with st.sidebar:
        st.header("Watchlist")
        default_watchlist = "RELIANCE\nTCS\nINFY\nHDFCBANK\nICICIBANK\nSBIN"
        watchlist_text = st.text_area("Symbols (one per line, or comma-separated)", value=default_watchlist, height=150)
        uploaded = st.file_uploader("...or upload watchlist.txt", type=["txt"])
        if uploaded is not None:
            watchlist_text = uploaded.read().decode("utf-8")

        st.header("Settings")
        rvol_lookback = st.number_input("RVOL average-volume lookback (days)", min_value=5, max_value=100, value=20)
        div_lookback = st.selectbox("Divergence lookback (candles)", options=[50, 100], index=0)
        pivot_order = st.slider("Swing-pivot sensitivity (bars each side)", min_value=2, max_value=6, value=3,
                                 help="Lower = more sensitive (more, smaller swings flagged). Higher = only larger swings count.")
        sort_by = st.selectbox("Sort by", options=["RVOL", "Chg%", "SS"], index=0)
        rvol_as = st.radio("RVOL as", options=["percent", "ratio"], horizontal=True)

        with st.expander("Advanced: divergence class thresholds"):
            flat_thresh = st.slider("Class B threshold (price 'flat' if move < this % of its range)",
                                     0.05, 0.30, 0.15, 0.01)
            weak_thresh = st.slider("Class C threshold (indicator 'weak' if move < this % of its range)",
                                     0.05, 0.30, 0.15, 0.01)

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
        rows = scan(tickers, rvol_lookback=rvol_lookback, div_lookback=div_lookback,
                    pivot_order=pivot_order, flat_thresh=flat_thresh, weak_thresh=weak_thresh)
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

    st.caption(
        "Div column legend: \u25b2/\u25bc = bullish/bearish, "
        "H = hidden (blank = regular), letter = Class A/B/C (A strongest), "
        "trailing names = which indicators agree (confluence)."
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
