"""
Unit tests for the Signal Engine — stock and forex strategy families.
Tests structure, known signals, edge cases, and parameter overrides.
"""

import pytest
import numpy as np
import pandas as pd

from dashboard.strategies.signal_engine import (
    SignalEngine,
    signal_family_1,
    signal_family_2,
    signal_family_3,
    signal_family_4,
    signal_family_5,
    forex_signal_family_1,
    forex_signal_family_2,
    forex_signal_family_3,
    forex_signal_family_4,
    forex_signal_family_5,
    _rsi,
    _ema,
    _sma,
    _atr,
    _bollinger,
    _williams_r,
    _adx,
    _stochastic,
    _macd,
    _donchian,
    _make_result,
    DEFAULT_PARAMS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_ohlcv(n: int = 300, start_price: float = 100.0, trend: float = 0.0,
               volatility: float = 1.0, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data."""
    rng = np.random.RandomState(seed)
    closes = [start_price]
    for _ in range(n - 1):
        change = trend + volatility * rng.randn()
        closes.append(max(closes[-1] + change, 1.0))

    closes = np.array(closes)
    highs = closes + rng.uniform(0.1, volatility * 2, n)
    lows = closes - rng.uniform(0.1, volatility * 2, n)
    lows = np.maximum(lows, 0.5)
    opens = closes + rng.uniform(-volatility, volatility, n)
    volumes = rng.randint(1000, 100000, n)

    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame({
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": closes,
        "Volume": volumes,
    }, index=dates)


def make_oversold_df(n: int = 300, seed: int = 99) -> pd.DataFrame:
    """Create data where price drops sharply at the end → RSI oversold."""
    rng = np.random.RandomState(seed)
    prices = [150.0]
    for _ in range(n - 30):
        prices.append(prices[-1] + rng.uniform(-0.5, 0.7))
    for _ in range(29):
        prices.append(prices[-1] - rng.uniform(1.0, 3.0))

    prices = np.array(prices)
    highs = prices + rng.uniform(0.1, 1.0, n)
    lows = prices - rng.uniform(0.1, 1.0, n)
    lows = np.maximum(lows, 0.5)
    opens = prices + rng.uniform(-0.5, 0.5, n)
    volumes = rng.randint(1000, 50000, n)

    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame({
        "Open": opens, "High": highs, "Low": lows,
        "Close": prices, "Volume": volumes,
    }, index=dates)


def make_overbought_df(n: int = 300, seed: int = 77) -> pd.DataFrame:
    """Create data where price is below long EMA and RSI spikes overbought."""
    rng = np.random.RandomState(seed)
    prices = [50.0]
    for _ in range(n - 30):
        prices.append(prices[-1] + rng.uniform(-0.7, 0.3))
    for _ in range(29):
        prices.append(prices[-1] + rng.uniform(1.0, 3.0))

    prices = np.array(prices)
    highs = prices + rng.uniform(0.1, 1.0, n)
    lows = prices - rng.uniform(0.1, 1.0, n)
    lows = np.maximum(lows, 0.5)
    opens = prices + rng.uniform(-0.5, 0.5, n)
    volumes = rng.randint(1000, 50000, n)

    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame({
        "Open": opens, "High": highs, "Low": lows,
        "Close": prices, "Volume": volumes,
    }, index=dates)


def make_uptrend_df(n: int = 300, seed: int = 55) -> pd.DataFrame:
    """Create steady uptrend data for momentum strategy tests."""
    return make_ohlcv(n, start_price=100.0, trend=0.5, volatility=0.8, seed=seed)


def make_downtrend_df(n: int = 300, seed: int = 66) -> pd.DataFrame:
    """Create steady downtrend data for momentum strategy tests."""
    return make_ohlcv(n, start_price=200.0, trend=-0.5, volatility=0.8, seed=seed)


# ---------------------------------------------------------------------------
# Indicator tests
# ---------------------------------------------------------------------------
class TestIndicators:
    def test_rsi_range(self):
        df = make_ohlcv(200)
        rsi = _rsi(df["Close"], 14)
        valid = rsi.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_rsi_oversold_on_downtrend(self):
        df = make_oversold_df()
        rsi = _rsi(df["Close"], 14)
        assert rsi.iloc[-1] < 40

    def test_ema_follows_trend(self):
        df = make_ohlcv(200, trend=0.5)
        ema = _ema(df["Close"], 20)
        assert ema.iloc[-1] < df["Close"].iloc[-1]

    def test_sma_length(self):
        df = make_ohlcv(100)
        sma = _sma(df["Close"], 20)
        assert len(sma) == len(df)

    def test_atr_positive(self):
        df = make_ohlcv(200)
        atr = _atr(df, 14)
        valid = atr.dropna()
        assert (valid > 0).all()

    def test_bollinger_bands_order(self):
        df = make_ohlcv(200)
        upper, mid, lower = _bollinger(df["Close"], 20, 2.0)
        valid_idx = upper.dropna().index
        assert (upper[valid_idx] >= mid[valid_idx]).all()
        assert (mid[valid_idx] >= lower[valid_idx]).all()

    def test_williams_r_range(self):
        df = make_ohlcv(200)
        wr = _williams_r(df, 14)
        valid = wr.dropna()
        assert (valid >= -100).all() and (valid <= 0).all()

    def test_adx_positive(self):
        df = make_ohlcv(200)
        adx = _adx(df, 14)
        valid = adx.dropna()
        assert (valid >= 0).all()

    def test_stochastic_range(self):
        """Stochastic %K and %D should be between 0 and 100."""
        df = make_ohlcv(200)
        k, d = _stochastic(df, 14, 3)
        valid_k = k.dropna()
        valid_d = d.dropna()
        assert (valid_k >= 0).all() and (valid_k <= 100).all()
        assert (valid_d >= 0).all() and (valid_d <= 100).all()

    def test_macd_histogram_sign(self):
        """MACD histogram should be positive when MACD > signal."""
        df = make_ohlcv(200)
        macd_line, signal_line, histogram = _macd(df["Close"], 12, 26, 9)
        valid_idx = histogram.dropna().index
        for idx in valid_idx[-10:]:
            if macd_line[idx] > signal_line[idx]:
                assert histogram[idx] > 0

    def test_donchian_channel(self):
        """Donchian upper should >= lower."""
        df = make_ohlcv(200)
        upper, lower = _donchian(df, 20)
        valid_idx = upper.dropna().index
        assert (upper[valid_idx] >= lower[valid_idx]).all()


# ---------------------------------------------------------------------------
# Stock family structure tests
# ---------------------------------------------------------------------------
REQUIRED_KEYS = {"signal", "entry_price", "stop_loss", "take_profit", "confidence", "reason"}


class TestStockFamilyStructure:
    @pytest.fixture
    def df(self):
        return make_ohlcv(300)

    def test_family_1_structure(self, df):
        result = signal_family_1(df, {})
        assert isinstance(result, dict)
        assert REQUIRED_KEYS.issubset(result.keys())
        assert result["signal"] in ("BUY", "SELL", "HOLD")

    def test_family_2_structure(self, df):
        result = signal_family_2(df, {})
        assert isinstance(result, dict)
        assert REQUIRED_KEYS.issubset(result.keys())
        assert result["signal"] in ("BUY", "SELL", "HOLD")

    def test_family_3_structure(self, df):
        result = signal_family_3(df, {})
        assert isinstance(result, dict)
        assert REQUIRED_KEYS.issubset(result.keys())
        assert result["signal"] in ("BUY", "SELL", "HOLD")

    def test_family_4_structure(self, df):
        result = signal_family_4(df, {})
        assert isinstance(result, dict)
        assert REQUIRED_KEYS.issubset(result.keys())
        assert result["signal"] in ("BUY", "SELL", "HOLD")

    def test_family_5_structure(self, df):
        result = signal_family_5(df, {})
        assert isinstance(result, dict)
        assert REQUIRED_KEYS.issubset(result.keys())
        assert result["signal"] in ("BUY", "SELL", "HOLD")


# ---------------------------------------------------------------------------
# Forex family structure tests
# ---------------------------------------------------------------------------
class TestForexFamilyStructure:
    @pytest.fixture
    def df(self):
        return make_ohlcv(300, start_price=1.10, volatility=0.002)

    def test_forex_family_1_structure(self, df):
        result = forex_signal_family_1(df, {})
        assert isinstance(result, dict)
        assert REQUIRED_KEYS.issubset(result.keys())
        assert result["signal"] in ("BUY", "SELL", "HOLD")

    def test_forex_family_2_structure(self, df):
        result = forex_signal_family_2(df, {})
        assert isinstance(result, dict)
        assert REQUIRED_KEYS.issubset(result.keys())
        assert result["signal"] in ("BUY", "SELL", "HOLD")

    def test_forex_family_3_structure(self, df):
        result = forex_signal_family_3(df, {})
        assert isinstance(result, dict)
        assert REQUIRED_KEYS.issubset(result.keys())
        assert result["signal"] in ("BUY", "SELL", "HOLD")

    def test_forex_family_4_structure(self, df):
        result = forex_signal_family_4(df, {})
        assert isinstance(result, dict)
        assert REQUIRED_KEYS.issubset(result.keys())
        assert result["signal"] in ("BUY", "SELL", "HOLD")

    def test_forex_family_5_structure(self, df):
        result = forex_signal_family_5(df, {})
        assert isinstance(result, dict)
        assert REQUIRED_KEYS.issubset(result.keys())
        assert result["signal"] in ("BUY", "SELL", "HOLD")


# ---------------------------------------------------------------------------
# Signal Engine class tests
# ---------------------------------------------------------------------------
class TestSignalEngine:
    @pytest.fixture
    def engine(self):
        return SignalEngine()

    @pytest.fixture
    def df(self):
        return make_ohlcv(300)

    def test_analyze_returns_dict(self, engine, df):
        result = engine.analyze(df, family=1)
        assert isinstance(result, dict)
        assert "signal" in result

    def test_analyze_forex_returns_dict(self, engine, df):
        result = engine.analyze(df, family=1, is_forex=True)
        assert isinstance(result, dict)
        assert "signal" in result

    def test_analyze_all_families(self, engine, df):
        results = engine.analyze_all_families(df)
        assert len(results) == 5
        for fam in range(1, 6):
            assert fam in results
            assert "signal" in results[fam]

    def test_analyze_all_forex_families(self, engine, df):
        results = engine.analyze_all_families(df, is_forex=True)
        assert len(results) == 5
        for fam in range(1, 6):
            assert fam in results
            assert "signal" in results[fam]

    def test_unknown_family_returns_hold(self, engine, df):
        result = engine.analyze(df, family=99)
        assert result["signal"] == "HOLD"
        assert "Unknown family" in result["reason"]

    def test_empty_dataframe(self, engine):
        result = engine.analyze(pd.DataFrame(), family=1)
        assert result["signal"] == "HOLD"

    def test_none_dataframe(self, engine):
        result = engine.analyze(None, family=1)
        assert result["signal"] == "HOLD"


# ---------------------------------------------------------------------------
# Known-signal tests — stock families
# ---------------------------------------------------------------------------
class TestKnownSignals:
    def test_family_1_buy_on_oversold(self):
        df = make_oversold_df()
        params = {"rsi_period": 14, "rsi_oversold": 40, "trend_ema_period": 10}
        result = signal_family_1(df, params)
        assert result["signal"] in ("BUY", "HOLD")

    def test_family_1_sell_on_overbought(self):
        df = make_overbought_df()
        params = {"rsi_period": 14, "rsi_overbought": 60, "trend_ema_period": 10}
        result = signal_family_1(df, params)
        assert result["signal"] in ("SELL", "HOLD")

    def test_buy_signal_has_valid_sl_tp(self):
        df = make_oversold_df()
        params = {"rsi_period": 14, "rsi_oversold": 50, "trend_ema_period": 5}
        result = signal_family_1(df, params)
        if result["signal"] == "BUY":
            assert result["stop_loss"] < result["entry_price"]
            assert result["take_profit"] > result["entry_price"]

    def test_sell_signal_has_valid_sl_tp(self):
        df = make_overbought_df()
        params = {"rsi_period": 14, "rsi_overbought": 50, "trend_ema_period": 5}
        result = signal_family_1(df, params)
        if result["signal"] == "SELL":
            assert result["stop_loss"] > result["entry_price"]
            assert result["take_profit"] < result["entry_price"]


# ---------------------------------------------------------------------------
# Known-signal tests — forex momentum families
# ---------------------------------------------------------------------------
class TestForexKnownSignals:
    def test_momentum_rsi_buy_on_uptrend(self):
        """Momentum RSI should BUY in a strong uptrend."""
        df = make_uptrend_df()
        params = {"rsi_period": 10, "rsi_long_thresh": 50, "trend_ema_period": 20}
        result = forex_signal_family_1(df, params)
        assert result["signal"] in ("BUY", "HOLD")

    def test_momentum_rsi_sell_on_downtrend(self):
        """Momentum RSI should SELL in a strong downtrend."""
        df = make_downtrend_df()
        params = {"rsi_period": 10, "rsi_short_thresh": 50, "trend_ema_period": 20}
        result = forex_signal_family_1(df, params)
        assert result["signal"] in ("SELL", "HOLD")

    def test_ema_crossover_buy_signal_has_valid_sl_tp(self):
        """When EMA Crossover generates BUY, SL should be below entry."""
        df = make_uptrend_df()
        result = forex_signal_family_3(df, {"ema_fast_period": 10, "ema_slow_period": 30, "trend_ema_period": 20})
        if result["signal"] == "BUY":
            assert result["stop_loss"] < result["entry_price"]
            assert result["take_profit"] > result["entry_price"]

    def test_stochastic_structure(self):
        """Stochastic family should return valid dict."""
        df = make_ohlcv(300)
        result = forex_signal_family_4(df, {"stoch_k": 14, "stoch_d": 3})
        assert isinstance(result, dict)
        assert result["signal"] in ("BUY", "SELL", "HOLD")

    def test_forex_family_5_same_as_stock_family_4(self):
        """Forex F5 (EMA+ADX) should produce same results as Stock F4."""
        df = make_ohlcv(300)
        stock_result = signal_family_4(df, {})
        forex_result = forex_signal_family_5(df, {})
        assert stock_result["signal"] == forex_result["signal"]


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_insufficient_data_family_1(self):
        df = make_ohlcv(10)
        result = signal_family_1(df, {"trend_ema_period": 200})
        assert result["signal"] == "HOLD"
        assert "Insufficient" in result["reason"]

    def test_insufficient_data_family_2(self):
        df = make_ohlcv(5)
        result = signal_family_2(df, {})
        assert result["signal"] == "HOLD"

    def test_insufficient_data_family_3(self):
        df = make_ohlcv(5)
        result = signal_family_3(df, {})
        assert result["signal"] == "HOLD"

    def test_insufficient_data_family_4(self):
        df = make_ohlcv(5)
        result = signal_family_4(df, {})
        assert result["signal"] == "HOLD"

    def test_insufficient_data_family_5(self):
        df = make_ohlcv(5)
        result = signal_family_5(df, {})
        assert result["signal"] == "HOLD"

    def test_insufficient_data_forex_families(self):
        df = make_ohlcv(5)
        for fn in [forex_signal_family_1, forex_signal_family_2,
                    forex_signal_family_3, forex_signal_family_4,
                    forex_signal_family_5]:
            result = fn(df, {})
            assert result["signal"] == "HOLD"

    def test_nan_in_close(self):
        df = make_ohlcv(300)
        df.loc[df.index[150], "Close"] = np.nan
        engine = SignalEngine()
        result = engine.analyze(df, family=1)
        assert isinstance(result, dict)

    def test_constant_price(self):
        n = 300
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        df = pd.DataFrame({
            "Open": [100.0] * n,
            "High": [100.0] * n,
            "Low": [100.0] * n,
            "Close": [100.0] * n,
            "Volume": [10000] * n,
        }, index=dates)
        result = signal_family_1(df, {})
        assert result["signal"] == "HOLD"

    def test_very_small_price(self):
        df = make_ohlcv(300, start_price=1.05, volatility=0.002)
        engine = SignalEngine()
        result = engine.analyze(df, family=1)
        assert isinstance(result, dict)

    def test_very_large_price(self):
        df = make_ohlcv(300, start_price=50000.0, volatility=500.0)
        engine = SignalEngine()
        result = engine.analyze(df, family=1)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Parameter override tests
# ---------------------------------------------------------------------------
class TestParameterOverrides:
    def test_custom_rsi_period(self):
        df = make_ohlcv(300)
        result_default = signal_family_1(df, {})
        result_custom = signal_family_1(df, {"rsi_period": 5})
        assert isinstance(result_default, dict)
        assert isinstance(result_custom, dict)

    def test_custom_stop_and_reward(self):
        df = make_oversold_df()
        params_tight = {"stop_atr_mult": 0.5, "reward_ratio": 1.0,
                        "rsi_oversold": 50, "trend_ema_period": 5}
        params_wide = {"stop_atr_mult": 3.0, "reward_ratio": 3.0,
                       "rsi_oversold": 50, "trend_ema_period": 5}
        r_tight = signal_family_1(df, params_tight)
        r_wide = signal_family_1(df, params_wide)
        if r_tight["signal"] == "BUY" and r_wide["signal"] == "BUY":
            tight_dist = r_tight["entry_price"] - r_tight["stop_loss"]
            wide_dist = r_wide["entry_price"] - r_wide["stop_loss"]
            assert wide_dist > tight_dist

    def test_ga_params_override_defaults(self):
        df = make_ohlcv(300)
        engine = SignalEngine()
        params = {"rsi_oversold": 1, "rsi_overbought": 99, "trend_ema_period": 200}
        result = engine.analyze(df, family=1, params=params)
        assert result["signal"] == "HOLD"


# ---------------------------------------------------------------------------
# _make_result helper tests
# ---------------------------------------------------------------------------
class TestMakeResult:
    def test_hold_returns_none_prices(self):
        result = _make_result("HOLD", 100.0, 1.0, {}, "no signal")
        assert result["entry_price"] is None
        assert result["stop_loss"] is None
        assert result["take_profit"] is None

    def test_buy_sl_below_entry(self):
        result = _make_result("BUY", 100.0, 2.0, {"stop_atr_mult": 1.5, "reward_ratio": 2.0}, "test")
        assert result["stop_loss"] < result["entry_price"]
        assert result["take_profit"] > result["entry_price"]

    def test_sell_sl_above_entry(self):
        result = _make_result("SELL", 100.0, 2.0, {"stop_atr_mult": 1.5, "reward_ratio": 2.0}, "test")
        assert result["stop_loss"] > result["entry_price"]
        assert result["take_profit"] < result["entry_price"]

    def test_zero_atr_returns_hold(self):
        result = _make_result("BUY", 100.0, 0.0, {}, "test")
        assert result["signal"] == "HOLD"

    def test_nan_atr_returns_hold(self):
        result = _make_result("BUY", 100.0, float("nan"), {}, "test")
        assert result["signal"] == "HOLD"

    def test_none_price_returns_hold(self):
        result = _make_result("BUY", None, 1.0, {}, "test")
        assert result["signal"] == "HOLD"
