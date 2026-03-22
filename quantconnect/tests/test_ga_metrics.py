"""
Unit Tests for GA Metrics and Parameter Loading
=================================================
Tests cover:
  - Loading optimized params from JSON
  - Ticker-to-GA pair mapping
  - Loading GA results from CSVs
  - Handling missing files gracefully
  - Best result selection logic
"""

import pytest
import json
import csv
import tempfile
from pathlib import Path
from unittest.mock import patch

from dashboard.utils import (
    load_optimized_params,
    load_all_params,
    load_all_ga_results,
    TICKER_TO_GA_PAIR,
    FAMILY_NAMES,
    FOREX_FAMILY_NAMES,
    ASSET_OPTIONS,
    STYLE_MAP,
)


class TestTickerMapping:
    """Test ticker to GA pair name mapping."""

    def test_forex_mapping(self):
        assert TICKER_TO_GA_PAIR["EURUSD=X"] == "EURUSD"
        assert TICKER_TO_GA_PAIR["USDJPY=X"] == "USDJPY"

    def test_stock_mapping(self):
        assert TICKER_TO_GA_PAIR["SPY"] == "SPY"
        assert TICKER_TO_GA_PAIR["QQQ"] == "QQQ"
        assert TICKER_TO_GA_PAIR["AAPL"] == "AAPL"
        assert TICKER_TO_GA_PAIR["TSLA"] == "TSLA"

    def test_crypto_mapping(self):
        assert TICKER_TO_GA_PAIR["BTC-USD"] == "BTCUSDT"
        assert TICKER_TO_GA_PAIR["SOL-USD"] == "SOLUSDT"

    def test_all_dashboard_assets_have_mapping(self):
        """Every asset available in the dashboard should have a GA pair mapping."""
        for market, tickers in ASSET_OPTIONS.items():
            for ticker in tickers:
                assert ticker in TICKER_TO_GA_PAIR, (
                    f"{ticker} in {market} has no TICKER_TO_GA_PAIR entry"
                )


class TestFamilyNames:
    """Test strategy family name constants."""

    def test_five_stock_families(self):
        assert len(FAMILY_NAMES) == 5
        for i in range(1, 6):
            assert i in FAMILY_NAMES

    def test_five_forex_families(self):
        assert len(FOREX_FAMILY_NAMES) == 5
        for i in range(1, 6):
            assert i in FOREX_FAMILY_NAMES

    def test_family_names_are_strings(self):
        for name in FAMILY_NAMES.values():
            assert isinstance(name, str) and len(name) > 0
        for name in FOREX_FAMILY_NAMES.values():
            assert isinstance(name, str) and len(name) > 0


class TestStyleMap:
    """Test trading style mapping."""

    def test_day_trade_mapping(self):
        assert STYLE_MAP["Day Trade"] == "day_trade"

    def test_swing_mapping(self):
        assert STYLE_MAP["Swing"] == "swing"


class TestLoadOptimizedParams:
    """Test loading GA-optimized parameters from JSON."""

    def test_returns_none_for_missing_file(self):
        with patch("dashboard.utils.PARAMS_PATH", Path("/nonexistent/params.json")):
            result = load_optimized_params("EURUSD=X", "day_trade")
            assert result is None

    def test_returns_none_for_missing_ticker(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"EURUSD=X": {"day_trade": {"family": 3}}}, f)
            f.flush()
            with patch("dashboard.utils.PARAMS_PATH", Path(f.name)):
                result = load_optimized_params("UNKNOWN_TICKER", "day_trade")
                assert result is None

    def test_returns_none_for_missing_style(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"EURUSD=X": {"day_trade": {"family": 3}}}, f)
            f.flush()
            with patch("dashboard.utils.PARAMS_PATH", Path(f.name)):
                result = load_optimized_params("EURUSD=X", "swing")
                assert result is None

    def test_loads_valid_params(self):
        data = {
            "EURUSD=X": {
                "day_trade": {
                    "family": 3,
                    "family_name": "EMA Crossover",
                    "params": {"ema_fast_period": 20, "ema_slow_period": 30},
                    "metrics": {"return_pct": 8.55, "sharpe": -0.405},
                }
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            with patch("dashboard.utils.PARAMS_PATH", Path(f.name)):
                result = load_optimized_params("EURUSD=X", "day_trade")
                assert result is not None
                assert result["family"] == 3
                assert result["metrics"]["return_pct"] == 8.55


class TestLoadAllParams:
    """Test loading the entire optimized_params.json."""

    def test_returns_empty_dict_for_missing_file(self):
        with patch("dashboard.utils.PARAMS_PATH", Path("/nonexistent/params.json")):
            result = load_all_params()
            assert result == {}

    def test_loads_full_json(self):
        data = {"EURUSD=X": {"day_trade": {"family": 3}}, "SPY": {"swing": {"family": 1}}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            with patch("dashboard.utils.PARAMS_PATH", Path(f.name)):
                result = load_all_params()
                assert "EURUSD=X" in result
                assert "SPY" in result


class TestLoadAllGAResults:
    """Test loading GA results from CSV files."""

    def _create_csv(self, tmpdir, pair, family, style, rows):
        """Helper to create a fake GA results CSV."""
        filename = f"results_{pair}_family{family}_{style}_20260315_120000.csv"
        csv_path = tmpdir / filename
        fieldnames = [
            "mode", "return_pct", "sharpe", "drawdown_pct",
            "win_rate_pct", "entries", "total_orders", "score",
            "param_rsi_period", "param_stop_atr_mult",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        return csv_path

    def test_returns_empty_for_missing_dir(self):
        with patch("dashboard.utils.RESULTS_DIR", Path("/nonexistent/path")):
            result = load_all_ga_results("EURUSD=X", "day_trade")
            assert result == {}

    def test_returns_empty_for_no_matching_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("dashboard.utils.RESULTS_DIR", Path(tmpdir)):
                result = load_all_ga_results("EURUSD=X", "day_trade")
                assert result == {}

    def test_loads_test_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            self._create_csv(tmppath, "EURUSD", 3, "day_trade", [
                {"mode": "train", "return_pct": "7.32", "sharpe": "1.5",
                 "drawdown_pct": "3.0", "win_rate_pct": "55", "entries": "100",
                 "total_orders": "300", "score": "10.5",
                 "param_rsi_period": "14", "param_stop_atr_mult": "1.5"},
                {"mode": "test", "return_pct": "8.55", "sharpe": "-0.41",
                 "drawdown_pct": "7.0", "win_rate_pct": "36", "entries": "78",
                 "total_orders": "234", "score": "5.2",
                 "param_rsi_period": "14", "param_stop_atr_mult": "1.5"},
            ])
            with patch("dashboard.utils.RESULTS_DIR", tmppath):
                result = load_all_ga_results("EURUSD=X", "day_trade")
                assert 3 in result
                assert result[3]["metrics"]["return_pct"] == 8.55
                assert result[3]["metrics"]["win_rate_pct"] == 36.0
                assert result[3]["params"]["rsi_period"] == 14

    def test_picks_best_test_return(self):
        """When multiple test rows exist, should pick highest return."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            self._create_csv(tmppath, "QQQ", 3, "day_trade", [
                {"mode": "test", "return_pct": "3.0", "sharpe": "0.5",
                 "drawdown_pct": "2.0", "win_rate_pct": "45", "entries": "50",
                 "total_orders": "150", "score": "5.0",
                 "param_rsi_period": "10", "param_stop_atr_mult": "2.0"},
                {"mode": "test", "return_pct": "6.56", "sharpe": "0.8",
                 "drawdown_pct": "4.5", "win_rate_pct": "32", "entries": "117",
                 "total_orders": "351", "score": "7.0",
                 "param_rsi_period": "14", "param_stop_atr_mult": "1.5"},
            ])
            with patch("dashboard.utils.RESULTS_DIR", tmppath):
                result = load_all_ga_results("QQQ", "day_trade")
                assert 3 in result
                assert result[3]["metrics"]["return_pct"] == 6.56

    def test_ignores_train_rows(self):
        """Should only return metrics from test-mode rows."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            self._create_csv(tmppath, "SPY", 1, "day_trade", [
                {"mode": "train", "return_pct": "50.0", "sharpe": "3.0",
                 "drawdown_pct": "2.0", "win_rate_pct": "80", "entries": "200",
                 "total_orders": "600", "score": "20.0",
                 "param_rsi_period": "14", "param_stop_atr_mult": "1.0"},
                {"mode": "test", "return_pct": "1.8", "sharpe": "0.3",
                 "drawdown_pct": "1.5", "win_rate_pct": "38", "entries": "8",
                 "total_orders": "24", "score": "3.0",
                 "param_rsi_period": "14", "param_stop_atr_mult": "1.0"},
            ])
            with patch("dashboard.utils.RESULTS_DIR", tmppath):
                result = load_all_ga_results("SPY", "day_trade")
                assert 1 in result
                assert result[1]["metrics"]["return_pct"] == 1.8

    def test_multiple_families(self):
        """Should return results for all families present."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            for fam, ret in [(1, "2.5"), (3, "8.55"), (5, "-1.2")]:
                self._create_csv(tmppath, "EURUSD", fam, "day_trade", [
                    {"mode": "test", "return_pct": ret, "sharpe": "0.5",
                     "drawdown_pct": "3.0", "win_rate_pct": "40", "entries": "50",
                     "total_orders": "150", "score": "5.0",
                     "param_rsi_period": "14", "param_stop_atr_mult": "1.5"},
                ])
            with patch("dashboard.utils.RESULTS_DIR", tmppath):
                result = load_all_ga_results("EURUSD=X", "day_trade")
                assert len(result) == 3
                assert 1 in result and 3 in result and 5 in result
