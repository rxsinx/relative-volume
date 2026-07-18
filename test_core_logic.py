import numpy as np
import pandas as pd


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


def _dedupe(idx_list):
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
    n = len(values)
    minima, maxima = [], []
    for i in range(order, n - order):
        window = values[i - order: i + order + 1]
        v = values[i]
        if v == window.min():
            minima.append(i)
        if v == window.max():
            maxima.append(i)
    return _dedupe(minima), _dedupe(maxima)


def detect_divergence(close: pd.Series, rsi: pd.Series, lookback: int = 50, order: int = 3):
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


def compute_rvol_chg_ss(daily: pd.DataFrame, lookback: int = 20):
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
    return {"cmp": cmp_, "rvol": rvol, "chg_pct": chg_pct, "strong_start": strong_start,
            "volume": volume, "avg_volume": avg_vol}


# ============================================================ tests

def test_rsi_sanity():
    # a strictly rising series should push RSI to 100 (no losses at all)
    close = pd.Series(np.linspace(100, 200, 60))
    rsi = compute_rsi(close)
    assert rsi.iloc[-1] == 100.0, rsi.iloc[-1]

    # a strictly falling series should push RSI to 0
    close2 = pd.Series(np.linspace(200, 100, 60))
    rsi2 = compute_rsi(close2)
    assert rsi2.iloc[-1] < 5, rsi2.iloc[-1]
    print("RSI sanity checks OK")


def test_bullish_divergence():
    # Dip 1: sharp V down to 90 at bar 20 (steep approach -> low RSI at the low).
    # Rally to a mid-high around bar 40.
    # Dip 2: gentle, decelerating approach down to 82 at bar 70 -- a LOWER low
    # than dip 1, but flattening out near the bottom so recent losses are
    # smaller there -> HIGHER RSI at the deeper low. Classic bullish divergence.
    n = 90
    x = np.arange(n).astype(float)
    price = np.zeros(n)
    for i, xi in enumerate(x):
        if xi < 20:
            price[i] = 150 - xi * 3.0
        elif xi < 40:
            price[i] = 90 + (xi - 20) * 2.0
        elif xi < 70:
            t = (xi - 40) / 30.0
            price[i] = 130 - (130 - 82) * np.sqrt(t)
        else:
            price[i] = 82 + (xi - 70) * 1.5

    close = pd.Series(price)
    rsi = compute_rsi(close, period=7)
    bullish, bearish = detect_divergence(close, rsi, lookback=90, order=3)
    print("bullish divergence detected:", bullish, "| bearish:", bearish)
    assert bullish is True, "expected a bullish divergence in this constructed series"
    assert bearish is False


def test_no_divergence_on_pure_trend():
    close = pd.Series(np.linspace(100, 200, 80) + np.sin(np.linspace(0, 6, 80)) * 2)
    rsi = compute_rsi(close, period=7)
    bullish, bearish = detect_divergence(close, rsi, lookback=80, order=3)
    print("pure uptrend -> bullish:", bullish, "bearish:", bearish)
    # a clean uptrend shouldn't flag bullish divergence
    assert bullish is False


def test_bearish_divergence():
    # Mirror of the bullish case: rally 1 sharp up to a high, pullback, rally 2
    # gentle/decelerating up to a HIGHER high but with weakening momentum near
    # the top -> LOWER RSI at the higher high. Classic bearish divergence.
    n = 90
    x = np.arange(n).astype(float)
    price = np.zeros(n)
    for i, xi in enumerate(x):
        if xi < 20:
            price[i] = 90 + xi * 3.0             # steep rally to 150 at bar 20
        elif xi < 40:
            price[i] = 150 - (xi - 20) * 2.0      # pullback to 110 at bar 40
        elif xi < 70:
            t = (xi - 40) / 30.0
            price[i] = 110 + (158 - 110) * np.sqrt(t)   # decelerating rally to 158
        else:
            price[i] = 158 - (xi - 70) * 1.5

    close = pd.Series(price)
    rsi = compute_rsi(close, period=7)
    bullish, bearish = detect_divergence(close, rsi, lookback=90, order=3)
    print("bearish divergence detected:", bearish, "| bullish:", bullish)
    assert bearish is True, "expected a bearish divergence in this constructed series"
    assert bullish is False


def test_rvol_chg_ss():
    dates = pd.date_range("2026-01-01", periods=25, freq="D")
    vols = [1_000_000] * 24 + [9_000_000]
    opens = [100] * 24 + [101]
    lows = [98] * 24 + [100.6]
    closes = [100] * 24 + [103]
    df = pd.DataFrame({"Open": opens, "High": closes, "Low": lows, "Close": closes, "Volume": vols}, index=dates)
    result = compute_rvol_chg_ss(df, lookback=20)
    print(result)
    assert abs(result["rvol"] - 9.0) < 1e-6
    assert result["strong_start"] is True
    assert result["chg_pct"] > 2.9
    print("RVOL/Chg%/SS math OK")


if __name__ == "__main__":
    test_rsi_sanity()
    test_bullish_divergence()
    test_bearish_divergence()
    test_no_divergence_on_pure_trend()
    test_rvol_chg_ss()
    print("\nALL CORE LOGIC TESTS PASSED")
