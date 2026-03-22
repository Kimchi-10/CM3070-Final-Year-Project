"""
Unit tests for the FinBERT Sentiment Component.
Tests structure, veto logic, and graceful fallbacks.
"""

import pytest
from unittest.mock import patch, MagicMock

from dashboard.components.sentiment import (
    analyze_sentiment,
    should_veto,
    _fetch_headlines,
)


# ---------------------------------------------------------------------------
# should_veto tests
# ---------------------------------------------------------------------------
class TestShouldVeto:
    def test_buy_negative_strong_vetoes(self):
        """BUY + strong negative sentiment → VETO."""
        sentiment = {"label": "negative", "score": 0.8}
        assert should_veto("BUY", sentiment) is True

    def test_sell_positive_strong_vetoes(self):
        """SELL + strong positive sentiment → VETO."""
        sentiment = {"label": "positive", "score": 0.75}
        assert should_veto("SELL", sentiment) is True

    def test_buy_positive_no_veto(self):
        """BUY + positive sentiment → NO VETO (aligned)."""
        sentiment = {"label": "positive", "score": 0.9}
        assert should_veto("BUY", sentiment) is False

    def test_sell_negative_no_veto(self):
        """SELL + negative sentiment → NO VETO (aligned)."""
        sentiment = {"label": "negative", "score": 0.85}
        assert should_veto("SELL", sentiment) is False

    def test_hold_never_vetoed(self):
        """HOLD signal should never be vetoed."""
        sentiment = {"label": "negative", "score": 0.95}
        assert should_veto("HOLD", sentiment) is False

    def test_none_sentiment_no_veto(self):
        """None sentiment (FinBERT unavailable) → NO VETO."""
        assert should_veto("BUY", None) is False

    def test_weak_sentiment_no_veto(self):
        """Weak sentiment (score < 0.6) should not trigger veto."""
        sentiment = {"label": "negative", "score": 0.4}
        assert should_veto("BUY", sentiment) is False

    def test_neutral_no_veto(self):
        """Neutral sentiment should never veto."""
        sentiment = {"label": "neutral", "score": 0.8}
        assert should_veto("BUY", sentiment) is False
        assert should_veto("SELL", sentiment) is False

    def test_borderline_score_no_veto(self):
        """Score exactly at 0.6 threshold should not veto (< 0.6 check)."""
        sentiment = {"label": "negative", "score": 0.5999}
        assert should_veto("BUY", sentiment) is False

    def test_just_above_threshold_vetoes(self):
        """Score just above 0.6 should veto."""
        sentiment = {"label": "negative", "score": 0.61}
        assert should_veto("BUY", sentiment) is True


# ---------------------------------------------------------------------------
# analyze_sentiment tests (with mocked pipeline)
# ---------------------------------------------------------------------------
class TestAnalyzeSentiment:
    @patch("dashboard.components.sentiment._load_pipeline")
    @patch("dashboard.components.sentiment._fetch_headlines")
    def test_returns_correct_structure(self, mock_headlines, mock_pipeline):
        """Mocked sentiment analysis should return correct dict structure."""
        mock_headlines.return_value = ["Stock surges on earnings beat", "Market rallies"]
        mock_pipe = MagicMock()
        mock_pipe.return_value = [
            {"label": "positive", "score": 0.92},
            {"label": "positive", "score": 0.88},
        ]
        mock_pipeline.return_value = mock_pipe

        result = analyze_sentiment("AAPL")
        assert result is not None
        assert "label" in result
        assert "score" in result
        assert "headlines" in result
        assert "details" in result
        assert result["label"] in ("positive", "negative", "neutral")
        assert 0 <= result["score"] <= 1

    @patch("dashboard.components.sentiment._load_pipeline")
    def test_returns_none_when_pipeline_unavailable(self, mock_pipeline):
        """Should return None when FinBERT is not installed."""
        mock_pipeline.return_value = None
        result = analyze_sentiment("AAPL")
        assert result is None

    @patch("dashboard.components.sentiment._load_pipeline")
    @patch("dashboard.components.sentiment._fetch_headlines")
    def test_no_headlines_returns_neutral(self, mock_headlines, mock_pipeline):
        """No headlines → neutral sentiment."""
        mock_headlines.return_value = []
        mock_pipeline.return_value = MagicMock()

        result = analyze_sentiment("AAPL")
        assert result is not None
        assert result["label"] == "neutral"
        assert result["score"] == 0.5

    @patch("dashboard.components.sentiment._load_pipeline")
    @patch("dashboard.components.sentiment._fetch_headlines")
    def test_negative_sentiment(self, mock_headlines, mock_pipeline):
        """Negative headlines should produce negative sentiment."""
        mock_headlines.return_value = ["Stock plunges", "Market crashes"]
        mock_pipe = MagicMock()
        mock_pipe.return_value = [
            {"label": "negative", "score": 0.95},
            {"label": "negative", "score": 0.90},
        ]
        mock_pipeline.return_value = mock_pipe

        result = analyze_sentiment("AAPL")
        assert result is not None
        assert result["label"] == "negative"

    @patch("dashboard.components.sentiment._load_pipeline")
    @patch("dashboard.components.sentiment._fetch_headlines")
    def test_details_match_headlines(self, mock_headlines, mock_pipeline):
        """Each headline should have a corresponding detail entry."""
        headlines = ["Good news", "Bad news", "Neutral news"]
        mock_headlines.return_value = headlines
        mock_pipe = MagicMock()
        mock_pipe.return_value = [
            {"label": "positive", "score": 0.8},
            {"label": "negative", "score": 0.7},
            {"label": "neutral", "score": 0.6},
        ]
        mock_pipeline.return_value = mock_pipe

        result = analyze_sentiment("AAPL")
        assert len(result["details"]) == 3
        assert result["details"][0]["headline"] == "Good news"
        assert result["details"][1]["label"] == "negative"


# ---------------------------------------------------------------------------
# _fetch_headlines tests
# ---------------------------------------------------------------------------
class TestFetchHeadlines:
    @patch("yfinance.Ticker")
    def test_returns_list(self, mock_ticker_cls):
        """Should return a list of strings."""
        mock_ticker = MagicMock()
        mock_ticker.news = [
            {"title": "Headline 1"},
            {"title": "Headline 2"},
        ]
        mock_ticker_cls.return_value = mock_ticker

        result = _fetch_headlines("AAPL")
        assert isinstance(result, list)
        assert len(result) == 2

    @patch("yfinance.Ticker")
    def test_max_headlines(self, mock_ticker_cls):
        """Should respect max_headlines limit."""
        mock_ticker = MagicMock()
        mock_ticker.news = [{"title": f"Headline {i}"} for i in range(20)]
        mock_ticker_cls.return_value = mock_ticker

        result = _fetch_headlines("AAPL", max_headlines=3)
        assert len(result) == 3

    @patch("yfinance.Ticker")
    def test_handles_exception(self, mock_ticker_cls):
        """Should return empty list on exception."""
        mock_ticker_cls.side_effect = Exception("API error")
        result = _fetch_headlines("INVALID")
        assert result == []
