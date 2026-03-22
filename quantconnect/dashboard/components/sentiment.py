"""
FYP Dashboard — FinBERT Sentiment Component
=============================================
Fetches recent news headlines and scores them using ProsusAI/finbert.
Provides a veto layer: if sentiment contradicts the trading signal,
the signal is vetoed.

Supports two veto modes:
  1. Rule-based (fallback): hardcoded 0.6 threshold
  2. RL-adaptive: Q-Learning agent decides based on market context
"""

import streamlit as st

# ---------------------------------------------------------------------------
# Sentiment analysis with FinBERT
# ---------------------------------------------------------------------------
_pipeline = None


def _load_pipeline():
    """Lazy-load the FinBERT pipeline (cached)."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    try:
        import os
        os.environ["TRANSFORMERS_NO_TF"] = "1"
        os.environ["USE_TF"] = "0"
        from transformers import pipeline
        _pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert",
                             truncation=True, max_length=512)
        return _pipeline
    except ImportError:
        return None
    except Exception:
        return None


def _fetch_headlines(ticker: str, max_headlines: int = 5) -> list[str]:
    """Fetch recent news headlines using yfinance."""
    try:
        import yfinance as yf
        # Map forex tickers
        lookup = ticker.replace("=X", "")
        tk = yf.Ticker(lookup)
        news = tk.news or []
        headlines = []
        for item in news[:max_headlines]:
            title = item.get("title", "")
            if not title:
                content = item.get("content", {})
                title = content.get("title", "") if isinstance(content, dict) else ""
            if title:
                headlines.append(title)
        return headlines
    except Exception:
        return []


def analyze_sentiment(ticker: str) -> dict | None:
    """
    Analyze sentiment for a ticker.

    Returns:
        {
            "label": "positive" | "negative" | "neutral",
            "score": float (0-1),
            "headlines": list[str],
            "details": list[dict],  # per-headline results
        }
    or None if sentiment analysis is unavailable.
    """
    pipe = _load_pipeline()
    if pipe is None:
        return None

    headlines = _fetch_headlines(ticker)
    if not headlines:
        return {
            "label": "neutral",
            "score": 0.5,
            "headlines": [],
            "details": [],
        }

    try:
        results = pipe(headlines)
    except Exception:
        return None

    # Aggregate sentiment
    scores = {"positive": 0, "negative": 0, "neutral": 0}
    details = []
    for headline, result in zip(headlines, results):
        label = result["label"].lower()
        conf = result["score"]
        scores[label] += conf
        details.append({"headline": headline, "label": label, "score": round(conf, 3)})

    total = sum(scores.values()) or 1
    # Determine overall sentiment
    dominant = max(scores, key=scores.get)
    overall_score = scores[dominant] / total

    return {
        "label": dominant,
        "score": round(overall_score, 3),
        "headlines": headlines,
        "details": details,
    }


def should_veto(signal: str, sentiment: dict | None) -> bool:
    """
    Rule-based veto check (fallback when RL agent is not available).
    BUY + negative sentiment (score > 0.6) → VETO
    SELL + positive sentiment (score > 0.6) → VETO
    """
    if sentiment is None or signal == "HOLD":
        return False

    label = sentiment.get("label", "neutral")
    score = sentiment.get("score", 0.5)

    # Only veto if sentiment is strong enough (>0.6)
    if score < 0.6:
        return False

    if signal == "BUY" and label == "negative":
        return True
    if signal == "SELL" and label == "positive":
        return True

    return False


def should_veto_rl(signal: str, sentiment: dict | None,
                   rsi: float = 50.0, atr: float = 1.0, atr_median: float = 1.0) -> tuple:
    """
    RL-adaptive veto check using trained Q-Learning agent.

    Returns:
        (vetoed: bool, method: str)
        method is "rl" if RL agent made the decision, "rule" if fallback was used
    """
    if sentiment is None or signal == "HOLD":
        return False, "rule"

    label = sentiment.get("label", "neutral")
    score = sentiment.get("score", 0.5)

    # Try RL agent first
    try:
        from dashboard.components.rl_agent import load_trained_agent
        agent = load_trained_agent()
        if agent is not None:
            vetoed = agent.should_veto(signal, score, label, rsi, atr, atr_median)
            return vetoed, "rl"
    except Exception:
        pass

    # Fallback to rule-based
    return should_veto(signal, sentiment), "rule"


def render_sentiment(ticker: str, signal: str = "HOLD",
                     rsi: float = 50.0, atr: float = 1.0, atr_median: float = 1.0):
    """Render sentiment analysis panel in Streamlit."""
    st.subheader("FinBERT Sentiment Analysis")

    sentiment = analyze_sentiment(ticker)

    if sentiment is None:
        st.warning("FinBERT not available. Install transformers + torch to enable sentiment analysis.")
        st.code("pip install transformers torch", language="bash")
        return sentiment, False

    # Display overall sentiment
    label = sentiment["label"]
    score = sentiment["score"]

    if label == "positive":
        st.success(f"Overall Sentiment: **{label.upper()}** (confidence: {score:.1%})")
    elif label == "negative":
        st.error(f"Overall Sentiment: **{label.upper()}** (confidence: {score:.1%})")
    else:
        st.info(f"Overall Sentiment: **{label.upper()}** (confidence: {score:.1%})")

    # Check veto using RL agent (with rule-based fallback)
    vetoed, veto_method = should_veto_rl(signal, sentiment, rsi, atr, atr_median)

    if vetoed:
        method_label = "RL Agent" if veto_method == "rl" else "Rule-Based"
        st.error(f"SIGNAL VETOED ({method_label}) — {signal} signal contradicted by {label} sentiment")
    else:
        # Show which method is being used
        if veto_method == "rl":
            st.caption("🤖 Veto decision by RL adaptive agent")
        else:
            st.caption("📏 Veto decision by rule-based threshold (train RL agent for adaptive decisions)")

    # Show individual headlines
    if sentiment["details"]:
        with st.expander("News Headlines & Scores"):
            for d in sentiment["details"]:
                icon = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}.get(d["label"], "⚪")
                st.text(f"{icon} [{d['label']:>8s} {d['score']:.0%}] {d['headline']}")

    return sentiment, vetoed
