"""
FYP Dashboard — Shared Utilities
=================================
Constants, data fetching, parameter loading for the Streamlit dashboard.
"""

import csv
import glob
import json
import pandas as pd
import yfinance as yf
from pathlib import Path

# ---------------------------------------------------------------------------
# Asset configuration
# ---------------------------------------------------------------------------
ASSET_OPTIONS = {
    "Forex": ["EURUSD=X", "USDJPY=X"],
    "Stocks": ["SPY", "QQQ", "AAPL", "TSLA"],
}

DISPLAY_NAMES = {
    "EURUSD=X": "EUR/USD",
    "USDJPY=X": "USD/JPY",
    "SPY": "SPY (S&P 500 ETF)",
    "QQQ": "QQQ (Nasdaq 100 ETF)",
    "AAPL": "AAPL (Apple)",
    "TSLA": "TSLA (Tesla)",
}

STYLE_OPTIONS = {
    "Forex": ["Day Trade"],
    "Stocks": ["Day Trade", "Swing"],
}

STYLE_MAP = {
    "Day Trade": "day_trade",
    "Swing": "swing",
}

FAMILY_NAMES = {
    1: "RSI Mean Reversion",
    2: "Bollinger Band Mean Reversion",
    3: "Connors RSI (2-Period)",
    4: "EMA Trend + ADX",
    5: "Williams %R + RSI",
}

FOREX_FAMILY_NAMES = {
    1: "RSI Mean Reversion",
    2: "Bollinger Band Bounce",
    3: "EMA Crossover",
    4: "Stochastic Mean Reversion",
    5: "EMA Trend + ADX",
}

# Timeframe for data fetching based on style
STYLE_CONFIG = {
    "day_trade": {"period": "60d", "interval": "5m"},
    "swing": {"period": "1y", "interval": "1h"},
}

# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------
def fetch_data(ticker: str, style: str = "day_trade") -> pd.DataFrame:
    """Fetch OHLCV data from yfinance based on trading style."""
    cfg = STYLE_CONFIG.get(style, STYLE_CONFIG["day_trade"])
    try:
        df = yf.download(ticker, period=cfg["period"], interval=cfg["interval"],
                         progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        # Flatten multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Parameter loading
# ---------------------------------------------------------------------------
PARAMS_PATH = Path(__file__).resolve().parent / "strategies" / "optimized_params.json"


def load_optimized_params(ticker: str, style: str = "day_trade") -> dict | None:
    """Load GA-optimized params for a given ticker+style. Returns None if not found."""
    if not PARAMS_PATH.exists():
        return None
    try:
        data = json.loads(PARAMS_PATH.read_text())
        asset_data = data.get(ticker, {})
        return asset_data.get(style)
    except Exception:
        return None


def load_all_params() -> dict:
    """Load the entire optimized_params.json."""
    if not PARAMS_PATH.exists():
        return {}
    try:
        return json.loads(PARAMS_PATH.read_text())
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Load GA results from CSVs (for per-family metrics)
# ---------------------------------------------------------------------------
RESULTS_DIR = Path(__file__).resolve().parent.parent / "genetic_algorithm" / "results"

# Map dashboard tickers → GA pair names
TICKER_TO_GA_PAIR = {
    "EURUSD=X": "EURUSD",
    "USDJPY=X": "USDJPY",
    "SPY": "SPY",
    "QQQ": "QQQ",
    "AAPL": "AAPL",
    "TSLA": "TSLA",
    "BTC-USD": "BTCUSDT",
    "SOL-USD": "SOLUSDT",
}


def load_all_ga_results(ticker: str, style: str = "day_trade") -> dict:
    """Load best GA test results per family from results CSVs.

    Returns: {family_num: {"metrics": {...}, "params": {...}}} for each
    family that has results.
    """
    if not RESULTS_DIR.exists():
        return {}

    ga_pair = TICKER_TO_GA_PAIR.get(ticker, ticker)
    results_by_family = {}

    patterns = [
        f"results_{ga_pair}_family*_{style}_*.csv",  
        f"results_{ga_pair}_family*_*.csv",          
    ]

    seen_files = set()
    for pattern in patterns:
        for csv_path in RESULTS_DIR.glob(pattern):
            if csv_path.name in seen_files:
                continue
            seen_files.add(csv_path.name)

            # Extract family number from filename
            name = csv_path.stem
            try:
                parts = name.split("_")
                fam_idx = next(i for i, p in enumerate(parts) if p.startswith("family"))
                family_num = int(parts[fam_idx].replace("family", ""))
            except (StopIteration, ValueError):
                continue

            if style in name or f"_{style}_" not in name:
                if style != "day_trade" and f"_{style}_" not in name:
                    continue

            try:
                with open(csv_path, "r") as f:
                    reader = csv.DictReader(f)
                    best_ret = None
                    best_row = None
                    for row in reader:
                        if row.get("mode") != "test":
                            continue
                        try:
                            ret = float(row.get("return_pct", ""))
                        except (ValueError, TypeError):
                            continue
                        if best_ret is None or ret > best_ret:
                            best_ret = ret
                            best_row = row

                if best_row is not None:
                    if family_num in results_by_family:
                        existing_ret = results_by_family[family_num]["metrics"].get("return_pct")
                        if existing_ret is not None and existing_ret >= best_ret:
                            continue

                    metrics = {}
                    for key in ["return_pct", "sharpe", "drawdown_pct", "win_rate_pct", "entries"]:
                        val = best_row.get(key, "")
                        try:
                            metrics[key] = float(val)
                        except (ValueError, TypeError):
                            metrics[key] = None

                    params = {}
                    for k, v in best_row.items():
                        if k.startswith("param_"):
                            param_name = k[6:]  
                            try:
                                params[param_name] = float(v) if "." in str(v) else int(v)
                            except (ValueError, TypeError):
                                params[param_name] = v

                    results_by_family[family_num] = {
                        "metrics": metrics,
                        "params": params,
                        "family_name": best_row.get("family_name", FAMILY_NAMES.get(family_num, "")),
                    }
            except Exception:
                continue

    return results_by_family
