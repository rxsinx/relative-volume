# RVOL + RSI Divergence Scanner

Streamlit app: RVOL / Chg% / Strong-Start (the original strategy) plus RSI
divergence on **15-minute** and **1-day** timeframes, over an NSE equity
watchlist. Data comes from **yfinance (Yahoo Finance)** -- no broker API,
no login, no API keys.

## What it computes

| Signal | Definition |
|---|---|
| RVOL | today's volume / SMA(prior N days' volume) -- today excluded, same as the original strategy |
| Chg% | (CMP - prev close) / prev close * 100 |
| SS (Strong Start) | today's open > prev close AND today's low held above prev close * 0.995 |
| RSI | standard Wilder 14-period RSI, computed separately per timeframe |
| Divergence | Bullish = price makes a **lower swing low** while RSI makes a **higher low**. Bearish = price makes a **higher swing high** while RSI makes a **lower high**. Compared over the last 50 or 100 candles (your choice), per timeframe. |

## Data source notes (read this before trusting it for execution timing)

- Yahoo Finance is free but **delayed** -- typically 15-20+ minutes for NSE, sometimes more. This is fine for scanning/screening, not for time-sensitive entries.
- Yahoo caps **15-minute history at the last 60 days**. That's a Yahoo platform limit, not something this app can extend.
- Yahoo also rate-limits; the app batches all symbols into a single request per timeframe and caches results for 5 minutes (`st.cache_data(ttl=300)`) to stay well under that.

## Run locally first

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`. Test it here before deploying -- this sandbox couldn't reach Yahoo Finance to test live data (network-restricted), so **this is the first real live-data test** the app will get.

## Deploy: GitHub + Streamlit Community Cloud

1. **Push this folder to a new GitHub repo:**
   ```bash
   cd rvol_rsi_app
   git init
   git add .
   git commit -m "RVOL + RSI divergence scanner"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<repo-name>.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud:**
   - Go to https://share.streamlit.io
   - Sign in with GitHub, click "New app"
   - Pick the repo, branch `main`, main file path `app.py`
   - Click Deploy

No secrets to configure -- yfinance needs no API key. Give it a couple of
minutes on first deploy while it installs dependencies.

3. **Add to your phone's home screen** (this is the "mobile app" part): open
   the deployed `*.streamlit.app` URL in your phone's browser, then use
   "Add to Home Screen" (iOS Safari: Share -> Add to Home Screen; Android
   Chrome: menu -> Add to Home Screen). Launches full-screen with its own
   icon, no browser chrome -- closest thing to a native app without an app
   store.

## Using the app

- Paste or upload a watchlist in the sidebar (same format as before: one
  ticker per line or comma-separated, bare tickers get `.NS` appended,
  `###` lines are comments).
- Set RVOL lookback (days), divergence lookback (50 or 100 candles), and
  swing-pivot sensitivity (lower = more, smaller swings flagged as
  divergence; higher = only larger swings count).
- Sort by RVOL, Chg%, or SS.
- Optional auto-refresh (interval configurable) using `streamlit-autorefresh`.
- Any symbol that fails to resolve or has insufficient data shows up in an
  expandable "issues" panel below the table instead of just going blank.

## Files

```
app.py             the Streamlit app (indicators, data fetch, UI)
requirements.txt   pip dependencies
watchlist.txt       sample watchlist for the file-upload option
test_core_logic.py  offline tests for RSI/divergence/RVOL math (no network needed)
```

Run the tests anytime with:
```bash
python3 test_core_logic.py
```
