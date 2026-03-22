"""
FYP Dashboard — Signal Engine
===============================
Implements strategy families for both stock and forex engines.

Stock families (mean-reversion):
    Family 1: RSI Mean Reversion
    Family 2: Bollinger Band Mean Reversion
    Family 3: Connors RSI (2-Period)
    Family 4: EMA Trend + ADX
    Family 5: Williams %R + RSI

Forex families (mean reversion + trend):
    Family 1: RSI Mean Reversion
    Family 2: Bollinger Band Bounce
    Family 3: EMA Crossover
    Family 4: Stochastic Mean Reversion
    Family 5: EMA Trend + ADX

Each family returns a dict:
    {
        "signal": "BUY" | "SELL" | "HOLD",
        "entry_price": float | None,
        "stop_loss": float | None,
        "take_profit": float | None,
        "confidence": str,           # "high", "medium", "low"
        "reason": str,
    }
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Default parameters (overridden by GA-optimized params when available)
# ---------------------------------------------------------------------------
DEFAULT_PARAMS = {
    # Shared
    "atr_period": 14,
    "stop_atr_mult": 1.5,
    "reward_ratio": 1.5,
    "trend_ema_period": 200,
    # Stock Family 1 — RSI Mean Reversion
    "rsi_period": 14,
    "rsi_oversold": 30,
    "rsi_overbought": 70,
    # Stock Family 2 — Bollinger Band MR
    "bb_period": 20,
    "bb_std_dev": 2.0,
    "bb_rsi_long": 35,
    "bb_rsi_short": 65,
    # Stock Family 3 — Connors RSI
    "crsi_period": 2,
    "crsi_entry": 10,
    "crsi_entry_short": 90,
    "exit_sma_period": 5,
    # Shared EMA + ADX (Stock F4, Forex F5)
    "ema_fast_period": 20,
    "ema_slow_period": 50,
    "adx_period": 14,
    "adx_min": 25,
    # Stock Family 5 — Williams %R + RSI
    "wr_period": 14,
    "wr_oversold": -80,
    "wr_overbought": -20,
    "wr_rsi_long": 35,
    "wr_rsi_short": 65,
    # Forex Family 1 — RSI Mean Reversion (same params as Stock F1)
    # rsi_period, rsi_oversold, rsi_overbought, trend_ema_period — already defined above
    # Forex Family 2 — Bollinger Band Bounce (same params as Stock F2)
    # bb_period, bb_std_dev, bb_rsi_long, bb_rsi_short — already defined above
    # Forex Family 3 — EMA Crossover
    # ema_fast_period, ema_slow_period, trend_ema_period — already defined above
    # Forex Family 4 — Stochastic Mean Reversion
    "stoch_k": 14,
    "stoch_d": 3,
    "stoch_oversold": 20,
    "stoch_overbought": 80,
}


# ---------------------------------------------------------------------------
# Indicator helpers
# ---------------------------------------------------------------------------
def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def _bollinger(series: pd.Series, period: int = 20, std_dev: float = 2.0):
    mid = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    return upper, mid, lower


def _williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    highest_high = df["High"].rolling(window=period).max()
    lowest_low = df["Low"].rolling(window=period).min()
    wr = -100 * (highest_high - df["Close"]) / (highest_high - lowest_low).replace(0, np.nan)
    return wr


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm < plus_dm] = 0
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr_s = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(span=period, adjust=False).mean() / atr_s.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr_s.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_val = dx.ewm(span=period, adjust=False).mean()
    return adx_val


def _stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3):
    """Stochastic oscillator returning %K and %D."""
    lowest_low = df["Low"].rolling(window=k_period).min()
    highest_high = df["High"].rolling(window=k_period).max()
    k = 100 * (df["Close"] - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    d = k.rolling(window=d_period).mean()
    return k, d


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD indicator returning macd_line, signal_line, histogram."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _donchian(df: pd.DataFrame, period: int = 20):
    """Donchian channel returning upper and lower."""
    upper = df["High"].rolling(window=period).max()
    lower = df["Low"].rolling(window=period).min()
    return upper, lower


def _make_result(signal: str, price, atr_val, params, reason: str, confidence: str = "medium"):
    if signal == "HOLD" or price is None or atr_val is None or np.isnan(atr_val) or atr_val <= 0:
        return {"signal": "HOLD", "entry_price": None, "stop_loss": None,
                "take_profit": None, "confidence": "low", "reason": reason}
    stop_dist = params.get("stop_atr_mult", 1.5) * atr_val
    reward_ratio = params.get("reward_ratio", 1.5)
    tp_dist = reward_ratio * stop_dist
    if signal == "BUY":
        sl = price - stop_dist
        tp = price + tp_dist
    else:
        sl = price + stop_dist
        tp = price - tp_dist
    return {
        "signal": signal,
        "entry_price": round(float(price), 5),
        "stop_loss": round(float(sl), 5),
        "take_profit": round(float(tp), 5),
        "confidence": confidence,
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# STOCK strategy families (mean-reversion)
# ---------------------------------------------------------------------------
def signal_family_1(df: pd.DataFrame, params: dict) -> dict:
    """Stock Family 1: RSI Mean Reversion"""
    p = {**DEFAULT_PARAMS, **params}
    if len(df) < max(int(p["trend_ema_period"]), int(p["rsi_period"])) + 10:
        return _make_result("HOLD", None, None, p, "Insufficient data")

    close = df["Close"]
    rsi = _rsi(close, int(p["rsi_period"]))
    trend_ema = _ema(close, int(p["trend_ema_period"]))
    atr_val = _atr(df, int(p["atr_period"]))

    last_rsi = rsi.iloc[-1]
    last_close = close.iloc[-1]
    last_ema = trend_ema.iloc[-1]
    last_atr = atr_val.iloc[-1]

    if last_rsi < p["rsi_oversold"] and last_close > last_ema:
        return _make_result("BUY", last_close, last_atr, p,
                            f"RSI({last_rsi:.0f}) oversold, price above trend EMA", "high")
    elif last_rsi > p["rsi_overbought"] and last_close < last_ema:
        return _make_result("SELL", last_close, last_atr, p,
                            f"RSI({last_rsi:.0f}) overbought, price below trend EMA", "high")
    return _make_result("HOLD", last_close, last_atr, p,
                        f"RSI({last_rsi:.0f}) neutral, no signal")


def signal_family_2(df: pd.DataFrame, params: dict) -> dict:
    """Stock Family 2: Bollinger Band Mean Reversion"""
    p = {**DEFAULT_PARAMS, **params}
    if len(df) < max(int(p["bb_period"]), int(p["rsi_period"])) + 10:
        return _make_result("HOLD", None, None, p, "Insufficient data")

    close = df["Close"]
    upper, mid, lower = _bollinger(close, int(p["bb_period"]), p["bb_std_dev"])
    rsi = _rsi(close, int(p["rsi_period"]))
    atr_val = _atr(df, int(p["atr_period"]))

    last_close = close.iloc[-1]
    last_rsi = rsi.iloc[-1]
    last_atr = atr_val.iloc[-1]

    if last_close < lower.iloc[-1] and last_rsi < p["bb_rsi_long"]:
        return _make_result("BUY", last_close, last_atr, p,
                            f"Price below lower BB, RSI({last_rsi:.0f}) confirms oversold", "high")
    elif last_close > upper.iloc[-1] and last_rsi > p["bb_rsi_short"]:
        return _make_result("SELL", last_close, last_atr, p,
                            f"Price above upper BB, RSI({last_rsi:.0f}) confirms overbought", "high")
    return _make_result("HOLD", last_close, last_atr, p,
                        f"Price within Bollinger Bands, RSI({last_rsi:.0f})")


def signal_family_3(df: pd.DataFrame, params: dict) -> dict:
    """Stock Family 3: Connors RSI (2-Period) Mean Reversion"""
    p = {**DEFAULT_PARAMS, **params}
    crsi_period = int(p.get("crsi_period", 2))
    exit_sma_period = int(p.get("exit_sma_period", 5))
    trend_ema_period = int(p["trend_ema_period"])

    if len(df) < max(trend_ema_period, crsi_period, exit_sma_period) + 10:
        return _make_result("HOLD", None, None, p, "Insufficient data")

    close = df["Close"]
    crsi = _rsi(close, crsi_period)
    trend_ema = _ema(close, trend_ema_period)
    sma = _sma(close, exit_sma_period)
    atr_val = _atr(df, int(p["atr_period"]))

    last_crsi = crsi.iloc[-1]
    last_close = close.iloc[-1]
    last_ema = trend_ema.iloc[-1]
    last_sma = sma.iloc[-1]
    last_atr = atr_val.iloc[-1]

    if last_crsi < p.get("crsi_entry", 10) and last_close > last_ema and last_close < last_sma:
        return _make_result("BUY", last_close, last_atr, p,
                            f"Connors RSI({last_crsi:.0f}) extreme oversold, pullback to SMA", "high")
    elif last_crsi > p.get("crsi_entry_short", 90) and last_close < last_ema and last_close > last_sma:
        return _make_result("SELL", last_close, last_atr, p,
                            f"Connors RSI({last_crsi:.0f}) extreme overbought, bounce from SMA", "high")
    return _make_result("HOLD", last_close, last_atr, p,
                        f"Connors RSI({last_crsi:.0f}) no extreme")


def signal_family_4(df: pd.DataFrame, params: dict) -> dict:
    """Stock Family 4: EMA Trend Following + ADX"""
    p = {**DEFAULT_PARAMS, **params}
    fast_period = int(p.get("ema_fast_period", 20))
    slow_period = int(p.get("ema_slow_period", 50))
    adx_period = int(p.get("adx_period", 14))

    if len(df) < max(slow_period, adx_period) + 10:
        return _make_result("HOLD", None, None, p, "Insufficient data")

    close = df["Close"]
    ema_fast = _ema(close, fast_period)
    ema_slow = _ema(close, slow_period)
    adx_val = _adx(df, adx_period)
    atr_val = _atr(df, int(p["atr_period"]))

    last_fast = ema_fast.iloc[-1]
    prev_fast = ema_fast.iloc[-2]
    last_slow = ema_slow.iloc[-1]
    prev_slow = ema_slow.iloc[-2]
    last_adx = adx_val.iloc[-1]
    prev_adx = adx_val.iloc[-2]
    last_close = close.iloc[-1]
    last_atr = atr_val.iloc[-1]

    cross_up = prev_fast <= prev_slow and last_fast > last_slow
    cross_down = prev_fast >= prev_slow and last_fast < last_slow
    adx_strong = last_adx > p.get("adx_min", 25) and last_adx > prev_adx

    if cross_up and adx_strong:
        return _make_result("BUY", last_close, last_atr, p,
                            f"EMA cross up, ADX({last_adx:.0f}) strong & rising", "medium")
    elif cross_down and adx_strong:
        return _make_result("SELL", last_close, last_atr, p,
                            f"EMA cross down, ADX({last_adx:.0f}) strong & rising", "medium")
    return _make_result("HOLD", last_close, last_atr, p,
                        f"No EMA crossover or ADX({last_adx:.0f}) weak")


def signal_family_5(df: pd.DataFrame, params: dict) -> dict:
    """Stock Family 5: Williams %R + RSI Confluence"""
    p = {**DEFAULT_PARAMS, **params}
    wr_period = int(p.get("wr_period", 14))
    rsi_period = int(p.get("rsi_period", 14))

    if len(df) < max(wr_period, rsi_period) + 10:
        return _make_result("HOLD", None, None, p, "Insufficient data")

    close = df["Close"]
    wr = _williams_r(df, wr_period)
    rsi = _rsi(close, rsi_period)
    atr_val = _atr(df, int(p["atr_period"]))

    last_wr = wr.iloc[-1]
    last_rsi = rsi.iloc[-1]
    last_close = close.iloc[-1]
    last_atr = atr_val.iloc[-1]

    if last_wr < p.get("wr_oversold", -80) and last_rsi < p.get("wr_rsi_long", 35):
        return _make_result("BUY", last_close, last_atr, p,
                            f"Williams %R({last_wr:.0f}) + RSI({last_rsi:.0f}) both oversold", "high")
    elif last_wr > p.get("wr_overbought", -20) and last_rsi > p.get("wr_rsi_short", 65):
        return _make_result("SELL", last_close, last_atr, p,
                            f"Williams %R({last_wr:.0f}) + RSI({last_rsi:.0f}) both overbought", "high")
    return _make_result("HOLD", last_close, last_atr, p,
                        f"Williams %R({last_wr:.0f}), RSI({last_rsi:.0f}) no confluence")


# ---------------------------------------------------------------------------
# FOREX strategy families (momentum/trend)
# ---------------------------------------------------------------------------
def forex_signal_family_1(df: pd.DataFrame, params: dict) -> dict:
    """Forex Family 1: RSI Mean Reversion — fade oversold/overbought extremes"""
    p = {**DEFAULT_PARAMS, **params}
    rsi_period = int(p.get("rsi_period", 10))
    trend_ema_period = int(p.get("trend_ema_period", 50))

    if len(df) < max(rsi_period, trend_ema_period) + 10:
        return _make_result("HOLD", None, None, p, "Insufficient data")

    close = df["Close"]
    rsi = _rsi(close, rsi_period)
    trend_ema = _ema(close, trend_ema_period)
    atr_val = _atr(df, int(p["atr_period"]))

    last_rsi = rsi.iloc[-1]
    last_close = close.iloc[-1]
    last_ema = trend_ema.iloc[-1]
    last_atr = atr_val.iloc[-1]
    oversold = p.get("rsi_oversold", 30)
    overbought = p.get("rsi_overbought", 70)

    # Long: RSI oversold, buy dip in uptrend
    if last_rsi < oversold and last_close > last_ema:
        return _make_result("BUY", last_close, last_atr, p,
                            f"RSI({last_rsi:.0f}) oversold, price above trend EMA", "high")
    # Short: RSI overbought, sell rally in downtrend
    elif last_rsi > overbought and last_close < last_ema:
        return _make_result("SELL", last_close, last_atr, p,
                            f"RSI({last_rsi:.0f}) overbought, price below trend EMA", "high")
    return _make_result("HOLD", last_close, last_atr, p,
                        f"RSI({last_rsi:.0f}) neutral, no signal")


def forex_signal_family_2(df: pd.DataFrame, params: dict) -> dict:
    """Forex Family 2: Bollinger Band Bounce — mean reversion from band extremes"""
    p = {**DEFAULT_PARAMS, **params}
    bb_period = int(p.get("bb_period", 20))
    rsi_period = int(p.get("rsi_period", 14))

    if len(df) < max(bb_period, rsi_period) + 10:
        return _make_result("HOLD", None, None, p, "Insufficient data")

    close = df["Close"]
    upper, mid, lower = _bollinger(close, bb_period, p.get("bb_std_dev", 2.0))
    rsi = _rsi(close, rsi_period)
    atr_val = _atr(df, int(p["atr_period"]))

    last_close = close.iloc[-1]
    last_rsi = rsi.iloc[-1]
    last_atr = atr_val.iloc[-1]
    bb_rsi_long = p.get("bb_rsi_long", 35)
    bb_rsi_short = p.get("bb_rsi_short", 65)

    # Long: price below lower band + RSI confirms oversold
    if last_close < lower.iloc[-1] and last_rsi < bb_rsi_long:
        return _make_result("BUY", last_close, last_atr, p,
                            f"Price below lower BB, RSI({last_rsi:.0f}) confirms oversold", "high")
    # Short: price above upper band + RSI confirms overbought
    elif last_close > upper.iloc[-1] and last_rsi > bb_rsi_short:
        return _make_result("SELL", last_close, last_atr, p,
                            f"Price above upper BB, RSI({last_rsi:.0f}) confirms overbought", "high")
    return _make_result("HOLD", last_close, last_atr, p,
                        f"Price within Bollinger Bands, RSI({last_rsi:.0f})")


def forex_signal_family_3(df: pd.DataFrame, params: dict) -> dict:
    """Forex Family 3: EMA Crossover + Trend Filter"""
    p = {**DEFAULT_PARAMS, **params}
    fast_period = int(p.get("ema_fast_period", 15))
    slow_period = int(p.get("ema_slow_period", 50))
    trend_period = int(p.get("trend_ema_period", 50))

    min_bars = max(fast_period, slow_period, trend_period) + 10
    if len(df) < min_bars:
        return _make_result("HOLD", None, None, p, "Insufficient data")

    close = df["Close"]
    ema_fast = close.ewm(span=fast_period, adjust=False).mean()
    ema_slow = close.ewm(span=slow_period, adjust=False).mean()
    trend_ema = close.ewm(span=trend_period, adjust=False).mean()
    atr_val = _atr(df, int(p["atr_period"]))

    last_close = close.iloc[-1]
    last_atr = atr_val.iloc[-1]
    fast_now = ema_fast.iloc[-1]
    slow_now = ema_slow.iloc[-1]
    fast_prev = ema_fast.iloc[-2]
    slow_prev = ema_slow.iloc[-2]
    trend_now = trend_ema.iloc[-1]

    # Bullish crossover: fast crosses above slow, uptrend confirmed
    if fast_prev <= slow_prev and fast_now > slow_now and last_close > trend_now:
        return _make_result("BUY", last_close, last_atr, p,
                            f"EMA crossover bullish (fast={fast_now:.5f} > slow={slow_now:.5f})", "medium")
    # Bearish crossover: fast crosses below slow, downtrend confirmed
    elif fast_prev >= slow_prev and fast_now < slow_now and last_close < trend_now:
        return _make_result("SELL", last_close, last_atr, p,
                            f"EMA crossover bearish (fast={fast_now:.5f} < slow={slow_now:.5f})", "medium")
    return _make_result("HOLD", last_close, last_atr, p,
                        f"EMA fast={fast_now:.5f} slow={slow_now:.5f} no crossover")


def forex_signal_family_4(df: pd.DataFrame, params: dict) -> dict:
    """Forex Family 4: Stochastic Mean Reversion — fade oversold/overbought zones"""
    p = {**DEFAULT_PARAMS, **params}
    k_period = int(p.get("stoch_k", 14))
    d_period = int(p.get("stoch_d", 3))
    trend_ema_period = int(p.get("trend_ema_period", 50))

    if len(df) < max(k_period, trend_ema_period) + 10:
        return _make_result("HOLD", None, None, p, "Insufficient data")

    close = df["Close"]
    k, d = _stochastic(df, k_period, d_period)
    trend_ema = _ema(close, trend_ema_period)
    atr_val = _atr(df, int(p["atr_period"]))

    last_k = k.iloc[-1]
    last_close = close.iloc[-1]
    last_ema = trend_ema.iloc[-1]
    last_atr = atr_val.iloc[-1]
    oversold = p.get("stoch_oversold", 20)
    overbought = p.get("stoch_overbought", 80)

    # Long: %K in oversold zone + price above trend EMA (buy dips in uptrend)
    if last_k < oversold and last_close > last_ema:
        return _make_result("BUY", last_close, last_atr, p,
                            f"Stochastic %K({last_k:.0f}) in oversold zone, uptrend", "high")
    # Short: %K in overbought zone + price below trend EMA (sell rallies in downtrend)
    elif last_k > overbought and last_close < last_ema:
        return _make_result("SELL", last_close, last_atr, p,
                            f"Stochastic %K({last_k:.0f}) in overbought zone, downtrend", "high")
    return _make_result("HOLD", last_close, last_atr, p,
                        f"Stochastic %K({last_k:.0f}) no mean reversion signal")


def forex_signal_family_5(df: pd.DataFrame, params: dict) -> dict:
    """Forex Family 5: EMA Trend + ADX (same as Stock Family 4)"""
    return signal_family_4(df, params)


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------
FAMILY_FUNCS = {
    1: signal_family_1,
    2: signal_family_2,
    3: signal_family_3,
    4: signal_family_4,
    5: signal_family_5,
}

FOREX_FAMILY_FUNCS = {
    1: forex_signal_family_1,
    2: forex_signal_family_2,
    3: forex_signal_family_3,
    4: forex_signal_family_4,
    5: forex_signal_family_5,
}


class SignalEngine:
    """Run strategy signals on live data."""

    def __init__(self):
        pass

    def analyze(self, df: pd.DataFrame, family: int = 1, params: dict | None = None,
                is_forex: bool = False) -> dict:
        """Run a strategy family on OHLCV data and return signal dict."""
        if df is None or df.empty or len(df) < 30:
            return {"signal": "HOLD", "entry_price": None, "stop_loss": None,
                    "take_profit": None, "confidence": "low",
                    "reason": "Insufficient market data"}

        funcs = FOREX_FAMILY_FUNCS if is_forex else FAMILY_FUNCS
        func = funcs.get(family)
        if func is None:
            return {"signal": "HOLD", "entry_price": None, "stop_loss": None,
                    "take_profit": None, "confidence": "low",
                    "reason": f"Unknown family {family}"}

        p = params or {}
        return func(df, p)

    def analyze_all_families(self, df: pd.DataFrame, params_by_family: dict | None = None,
                             is_forex: bool = False) -> dict:
        """Run all 5 families and return results keyed by family number."""
        results = {}
        for fam in range(1, 6):
            p = (params_by_family or {}).get(fam, {})
            results[fam] = self.analyze(df, fam, p, is_forex=is_forex)
        return results
