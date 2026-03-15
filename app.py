from flask import Flask, render_template, request, jsonify
import yfinance as yf
import numpy as np
import pandas as pd
import requests as req_lib
from bs4 import BeautifulSoup
import time

app = Flask(__name__)

# ── In-memory cache with TTL ────────────────────────────────────────────────
_cache = {}  # key -> {"data": ..., "ts": ...}
CACHE_TTL = 3600  # 1 hour in seconds


def cache_get(key):
    """Return cached value if it exists and hasn't expired, else None."""
    entry = _cache.get(key)
    if entry and (time.time() - entry["ts"]) < CACHE_TTL:
        return entry["data"]
    return None


def cache_set(key, data):
    """Store a value in cache with current timestamp."""
    _cache[key] = {"data": data, "ts": time.time()}


# S&P 500 constituents (cached after first request)
_sp500_cache = None


def get_sp500_tickers():
    global _sp500_cache
    if _sp500_cache is not None:
        return _sp500_cache
    try:
        r = req_lib.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )
        soup = BeautifulSoup(r.text, "lxml")
        table = soup.find("table", {"id": "constituents"})
        rows = table.find_all("tr")[1:]
        _sp500_cache = []
        for row in rows:
            cols = row.find_all("td")
            if len(cols) >= 2:
                _sp500_cache.append(
                    {"ticker": cols[0].text.strip(), "name": cols[1].text.strip()}
                )
    except Exception:
        _sp500_cache = []
    return _sp500_cache

PERIOD_OPTIONS = {
    "2y": "2 Years",
    "5y": "5 Years",
    "10y": "10 Years",
}

INTERVAL_OPTIONS = {
    "1d": "Daily",
    "1wk": "Weekly",
    "1mo": "Monthly",
}

# Annualisation factors: number of periods per year
ANNUALISE_FACTOR = {
    "1d": 252,
    "1wk": 52,
    "1mo": 12,
}


def _download_cached(symbol: str, period: str, interval: str) -> pd.DataFrame:
    """Download price data with in-memory caching."""
    key = f"dl:{symbol}:{period}:{interval}"
    cached = cache_get(key)
    if cached is not None:
        return cached
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=True)
    cache_set(key, df)
    return df


def _get_rf_cached() -> float:
    """Get risk-free rate with caching."""
    key = "rf:tnx"
    cached = cache_get(key)
    if cached is not None:
        return cached
    try:
        tnx = yf.download("^TNX", period="5d", interval="1d", auto_adjust=True)
        if not tnx.empty:
            rf = float(tnx["Close"].squeeze().dropna().iloc[-1])
        else:
            rf = 4.0
    except Exception:
        rf = 4.0
    cache_set(key, rf)
    return rf


def calculate_beta(ticker: str, period: str = "2y", interval: str = "1d") -> dict:
    """Calculate the beta of a stock against the S&P 500."""
    stock = _download_cached(ticker, period, interval)
    market = _download_cached("^GSPC", period, interval)

    if stock.empty:
        raise ValueError(f"No data found for ticker '{ticker}'.")

    # Align dates
    stock_close = stock["Close"].squeeze()
    market_close = market["Close"].squeeze()
    combined = pd.DataFrame({"stock": stock_close, "market": market_close}).dropna()

    if len(combined) < 20:
        raise ValueError("Not enough overlapping data to calculate beta.")

    returns = combined.pct_change().dropna()

    cov_matrix = np.cov(returns["stock"], returns["market"])
    beta = cov_matrix[0, 1] / cov_matrix[1, 1]

    # Additional stats
    ann_factor = ANNUALISE_FACTOR.get(interval, 252)
    correlation = returns["stock"].corr(returns["market"])
    stock_vol = returns["stock"].std() * np.sqrt(ann_factor) * 100
    market_vol = returns["market"].std() * np.sqrt(ann_factor) * 100

    # Per-period (non-annualised) stats
    avg_stock_return = returns["stock"].mean() * 100
    avg_market_return = returns["market"].mean() * 100
    stock_vol_period = returns["stock"].std() * 100
    market_vol_period = returns["market"].std() * 100

    # CAPM: E(R) = Rf + Beta * (Rm - Rf)
    rf_annual = _get_rf_cached()

    # Annualised market return over the selected period
    market_ann_return = returns["market"].mean() * ann_factor * 100  # in %
    equity_risk_premium = market_ann_return - rf_annual
    capm_expected_return = rf_annual + beta * equity_risk_premium

    # Build time-series table (date, stock price, stock % chg, market price, market % chg)
    stock_pct = returns["stock"] * 100
    market_pct = returns["market"] * 100
    table_rows = []
    for dt in returns.index:
        date_str = dt.strftime("%Y-%m-%d")
        table_rows.append({
            "date": date_str,
            "stock_price": round(float(combined.loc[dt, "stock"]), 2),
            "stock_chg": round(float(stock_pct.loc[dt]), 2),
            "market_price": round(float(combined.loc[dt, "market"]), 2),
            "market_chg": round(float(market_pct.loc[dt]), 2),
        })

    return {
        "ticker": ticker.upper(),
        "beta": round(float(beta), 4),
        "correlation": round(float(correlation), 4),
        "stock_volatility": round(float(stock_vol), 2),
        "market_volatility": round(float(market_vol), 2),
        "period": PERIOD_OPTIONS.get(period, period),
        "interval": INTERVAL_OPTIONS.get(interval, interval),
        "data_points": len(returns),
        "avg_stock_return": round(float(avg_stock_return), 4),
        "avg_market_return": round(float(avg_market_return), 4),
        "stock_vol_period": round(float(stock_vol_period), 4),
        "market_vol_period": round(float(market_vol_period), 4),
        "interval_label": INTERVAL_OPTIONS.get(interval, interval),
        "rf_annual": round(float(rf_annual), 2),
        "market_ann_return": round(float(market_ann_return), 2),
        "equity_risk_premium": round(float(equity_risk_premium), 2),
        "capm_expected_return": round(float(capm_expected_return), 2),
        "table": table_rows,
    }


@app.route("/")
def index():
    return render_template(
        "index.html", periods=PERIOD_OPTIONS, intervals=INTERVAL_OPTIONS
    )


@app.route("/tickers")
def tickers():
    return jsonify(get_sp500_tickers())


@app.route("/calculate", methods=["POST"])
def calculate():
    data = request.get_json()
    ticker = data.get("ticker", "").strip()
    period = data.get("period", "2y")
    interval = data.get("interval", "1d")

    if not ticker:
        return jsonify({"error": "Please enter a ticker symbol."}), 400
    if period not in PERIOD_OPTIONS:
        return jsonify({"error": "Invalid period."}), 400
    if interval not in INTERVAL_OPTIONS:
        return jsonify({"error": "Invalid interval."}), 400

    try:
        result = calculate_beta(ticker, period, interval)
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Failed to fetch data: {e}"}), 500


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5050))
    debug = os.environ.get("FLASK_ENV") != "production"
    app.run(debug=debug, host="0.0.0.0", port=port)
