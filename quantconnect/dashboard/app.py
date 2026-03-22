"""
FYP Dashboard — AI-Driven Dynamic Portfolio Advisor
=====================================================
Streamlit dashboard that displays live trading signals from 5 GA-optimized
mean-reversion strategy families, with FinBERT sentiment veto layer.

Usage:
    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

# Ensure dashboard package is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from dashboard.utils import (
    ASSET_OPTIONS, DISPLAY_NAMES, STYLE_OPTIONS, STYLE_MAP,
    FAMILY_NAMES, FOREX_FAMILY_NAMES, fetch_data, load_optimized_params, load_all_params,
)
from dashboard.strategies.signal_engine import (
    SignalEngine, _rsi, _ema, _sma, _bollinger, _williams_r, _adx, _atr,
)
from dashboard.components.ga_metrics import render_ga_metrics
from dashboard.components.sentiment import render_sentiment

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Portfolio Advisor — FYP",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for polish
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Tighter metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 10px;
        padding: 12px 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    [data-testid="stMetricLabel"] { font-size: 0.85rem !important; }
    /* Signal badge pulse animation */
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.7} }
    .signal-live { animation: pulse 2s infinite; }
    /* Sidebar footer */
    .sidebar-footer { font-size: 0.75rem; color: #6c757d; line-height: 1.5; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar — Navigation + Market/Asset/Style only 
# ---------------------------------------------------------------------------
st.sidebar.title("AI-Driven Dynamic\nPortfolio Advisor")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigation", ["Signal View", "Portfolio Overview", "GA Results Explorer"], index=0)

# Market/Asset/Style filters — only shown on Signal View
if page == "Signal View":
    st.sidebar.markdown("---")

    # Market & asset selection
    market = st.sidebar.selectbox("Market", list(ASSET_OPTIONS.keys()))
    assets = ASSET_OPTIONS[market]
    ticker = st.sidebar.selectbox("Asset", assets, format_func=lambda x: DISPLAY_NAMES.get(x, x))

    # Style selection
    styles = STYLE_OPTIONS.get(market, ["Day Trade"])
    style_display = st.sidebar.selectbox("Trading Style", styles)
    style = STYLE_MAP.get(style_display, "day_trade")
    is_forex = market == "Forex"
    active_family_names = FOREX_FAMILY_NAMES if is_forex else FAMILY_NAMES

    # Auto-refresh option (useful during market hours)
    st.sidebar.markdown("---")
    auto_refresh = st.sidebar.checkbox("Auto-refresh (60s)", value=False,
                                        help="Automatically refresh data every 60 seconds during market hours")
    st.sidebar.caption(f"Last updated: {pd.Timestamp.now().strftime('%H:%M:%S')}")
else:
    auto_refresh = False

st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div class="sidebar-footer">CM3070 Final Year Project<br>'
    'Ng Chang Yan (230367197)<br>'
    'University of London</div>',
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helper: format price
# ---------------------------------------------------------------------------
def fmt_price(val):
    if val is None:
        return "—"
    return f"${val:,.5f}" if val < 10 else f"${val:,.2f}"


# ---------------------------------------------------------------------------
# Signal View Page
# ---------------------------------------------------------------------------
def signal_view():
    st.title(f"📈 {DISPLAY_NAMES.get(ticker, ticker)} — {style_display}")

    # --- Strategy Family selector IN the main page ---
    winner = load_optimized_params(ticker, style)
    default_family = winner["family"] if winner else 1

    # Determine which families are available for this asset/style
    all_params = load_all_params()
    tested_families_data = {}
    if ticker in all_params and style in all_params[ticker]:
        tested_families_data = all_params[ticker][style].get("all_families", {})

    if tested_families_data:
        # Only show families that were actually GA-tested
        available_families = sorted(int(k) for k in tested_families_data.keys())
    else:
        # Fallback to all 5
        available_families = list(active_family_names.keys())

    # Make sure default_family is in the list
    if default_family in available_families:
        default_idx = available_families.index(default_family)
    else:
        default_idx = 0

    fam_cols = st.columns([3, 2])
    with fam_cols[0]:
        family = st.selectbox(
            "Strategy Family",
            available_families,
            index=default_idx,
            format_func=lambda x: f"F{x}: {active_family_names[x]}",
        )
    with fam_cols[1]:
        st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
        if winner and winner["family"] == family:
            st.success(f"✅ GA-optimized params loaded (best: F{winner['family']})")
        elif winner:
            st.info(f"💡 GA recommends F{winner['family']} for best results")
        else:
            st.warning("No GA results for this asset/style — using defaults")

    # Fetch data
    with st.spinner("Fetching market data..."):
        df = fetch_data(ticker, style)

    if df.empty:
        st.error("Failed to fetch market data. Please check your internet connection.")
        return

    # Check if market might be closed (weekend/holiday)
    if len(df) > 0:
        last_bar = df.index[-1]
        now = pd.Timestamp.now(tz=last_bar.tzinfo if last_bar.tzinfo else None)
        hours_since = (now - last_bar).total_seconds() / 3600
        market_closed = hours_since > 4  
    else:
        market_closed = True

    # Initialize signal engine
    engine = SignalEngine()

    # Get params
    params = {}
    if winner and winner["family"] == family:
        params = winner.get("params", {})

    # Run signal
    result = engine.analyze(df, family, params, is_forex=is_forex)
    signal = result["signal"]

    # Override to HOLD when market is closed (stale data)
    if market_closed and signal != "HOLD":
        signal = "HOLD"
        result["signal"] = "HOLD"
        result["entry_price"] = None
        result["stop_loss"] = None
        result["take_profit"] = None
        result["reason"] = "Market closed — signal based on stale data, check back during market hours"
        result["confidence"] = "low"

    # Compute RSI and ATR for RL veto agent
    close = df["Close"]
    rsi_series = _rsi(close, 14)
    atr_series = _atr(df, 14)
    last_rsi = float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else 50.0
    last_atr = float(atr_series.iloc[-1]) if not pd.isna(atr_series.iloc[-1]) else 1.0
    atr_med = float(atr_series.median()) if not pd.isna(atr_series.median()) else 1.0

    # --- Sentiment analysis (with RL adaptive veto) ---
    sentiment_data, vetoed = render_sentiment(ticker, signal, last_rsi, last_atr, atr_med)

    # --- Signal Panel ---
    st.markdown("---")

    if vetoed:
        signal_display = f"{signal} — VETOED"
        signal_color = "orange"
    else:
        signal_display = signal
        signal_color = {"BUY": "green", "SELL": "red", "HOLD": "gray"}.get(signal, "gray")

    # Current market price
    current_price = float(df["Close"].iloc[-1])

    col_sig, col_price, col_entry, col_sl, col_tp = st.columns(5)

    with col_sig:
        st.markdown("### Signal")
        st.markdown(
            f'<div style="background-color:{signal_color};color:white;'
            f'padding:20px;border-radius:10px;text-align:center;font-size:28px;'
            f'font-weight:bold;">{signal_display}</div>',
            unsafe_allow_html=True,
        )
        if vetoed:
            st.caption("Sentiment contradicts signal direction")

    with col_price:
        st.markdown("### Current Price")
        st.markdown(f"<div style='text-align:center;font-size:24px;padding:20px;'>"
                    f"{fmt_price(current_price)}</div>",
                    unsafe_allow_html=True)

    with col_entry:
        ep = result.get("entry_price")
        st.markdown("### Entry Price")
        st.markdown(f"<div style='text-align:center;font-size:24px;padding:20px;'>"
                    f"{fmt_price(ep)}</div>",
                    unsafe_allow_html=True)

    with col_sl:
        sl = result.get("stop_loss")
        st.markdown("### Stop Loss")
        st.markdown(f"<div style='text-align:center;font-size:24px;padding:20px;color:red;'>"
                    f"{fmt_price(sl)}</div>",
                    unsafe_allow_html=True)

    with col_tp:
        tp = result.get("take_profit")
        st.markdown("### Take Profit")
        st.markdown(f"<div style='text-align:center;font-size:24px;padding:20px;color:green;'>"
                    f"{fmt_price(tp)}</div>",
                    unsafe_allow_html=True)

    # Signal reason
    st.caption(f"**Reason:** {result.get('reason', 'N/A')} | **Confidence:** {result.get('confidence', 'N/A')}")

    # If HOLD or market closed, show helpful message
    if signal == "HOLD":
        if market_closed:
            st.info("📅 **Market is currently closed.** Entry/SL/TP prices are only generated "
                    "when the strategy detects an active BUY or SELL signal during market hours. "
                    "Check back when the market reopens.")
        else:
            st.info("No active signal. The strategy is waiting for entry conditions to be met. "
                    "Try switching to a different family or asset to see active signals.")

    # --- Price Chart (clean line chart + indicators) ---
    st.markdown("---")

    # Chart period/timeframe selector
    chart_col1, chart_col2 = st.columns([3, 1])
    with chart_col1:
        st.subheader("Price Chart")
    with chart_col2:
        # Different timeframe options based on trading style
        if style == "day_trade":
            tf_options = {
                "5 Min": {"period": "5d", "interval": "5m"},
                "15 Min": {"period": "60d", "interval": "15m"},
                "1 Hour": {"period": "6mo", "interval": "1h"},
                "Daily": {"period": "2y", "interval": "1d"},
            }
            default_tf = "5 Min"
        else:  # swing
            tf_options = {
                "1 Hour": {"period": "6mo", "interval": "1h"},
                "Daily": {"period": "2y", "interval": "1d"},
                "Weekly": {"period": "5y", "interval": "1wk"},
            }
            default_tf = "1 Hour"

        selected_tf = st.selectbox(
            "Timeframe",
            list(tf_options.keys()),
            index=list(tf_options.keys()).index(default_tf),
            label_visibility="collapsed",
        )
        tf_cfg = tf_options[selected_tf]

    # Fetch chart data based on selected timeframe
    import yfinance as yf
    try:
        chart_df = yf.download(ticker, period=tf_cfg["period"], interval=tf_cfg["interval"],
                               progress=False, auto_adjust=True)
        if isinstance(chart_df.columns, pd.MultiIndex):
            chart_df.columns = chart_df.columns.get_level_values(0)
        if chart_df.empty:
            chart_df = df.copy()
    except Exception:
        chart_df = df.copy()
    close = chart_df["Close"]
    p = {**params}

    # Determine which indicator subplot to show based on family
    has_oscillator = family in [1, 3, 5]
    rows = 3 if has_oscillator else 2
    row_heights = [0.6, 0.15, 0.25] if has_oscillator else [0.7, 0.3]

    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
    )

    # --- Row 1: Candlestick + overlays ---
    fig.add_trace(go.Candlestick(
        x=chart_df.index,
        open=chart_df["Open"], high=chart_df["High"],
        low=chart_df["Low"], close=chart_df["Close"],
        name="Price",
        increasing_line_color="#26a69a", increasing_fillcolor="#26a69a",
        decreasing_line_color="#ef5350", decreasing_fillcolor="#ef5350",
    ), row=1, col=1)

    # Family-specific overlays on price chart
    if family == 1:
        ema_period = int(p.get("trend_ema_period", 200))
        trend_ema = _ema(close, ema_period)
        fig.add_trace(go.Scatter(x=chart_df.index, y=trend_ema, name=f"EMA({ema_period})",
                                 line=dict(color="#1976d2", width=1.5, dash="dot")), row=1, col=1)
    elif family == 2:
        bb_period = int(p.get("bb_period", 20))
        bb_std = float(p.get("bb_std_dev", 2.0))
        upper, mid, lower = _bollinger(close, bb_period, bb_std)
        fig.add_trace(go.Scatter(x=chart_df.index, y=upper, name="Upper BB",
                                 line=dict(color="#42a5f5", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=chart_df.index, y=mid, name=f"SMA({bb_period})",
                                 line=dict(color="#ff9800", width=1, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=chart_df.index, y=lower, name="Lower BB",
                                 line=dict(color="#42a5f5", width=1),
                                 fill="tonexty", fillcolor="rgba(66,165,245,0.1)"), row=1, col=1)
    elif family == 3:
        ema_period = int(p.get("trend_ema_period", 200))
        sma_period = int(p.get("exit_sma_period", 5))
        trend_ema = _ema(close, ema_period)
        exit_sma = _sma(close, sma_period)
        fig.add_trace(go.Scatter(x=chart_df.index, y=trend_ema, name=f"Trend EMA({ema_period})",
                                 line=dict(color="#1976d2", width=1.5, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=chart_df.index, y=exit_sma, name=f"Exit SMA({sma_period})",
                                 line=dict(color="#9c27b0", width=1)), row=1, col=1)
    elif family == 4:
        fast = int(p.get("ema_fast_period", 20))
        slow = int(p.get("ema_slow_period", 50))
        ema_f = _ema(close, fast)
        ema_s = _ema(close, slow)
        fig.add_trace(go.Scatter(x=chart_df.index, y=ema_f, name=f"Fast EMA({fast})",
                                 line=dict(color="#00bcd4", width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=chart_df.index, y=ema_s, name=f"Slow EMA({slow})",
                                 line=dict(color="#ff9800", width=1.5)), row=1, col=1)
    elif family == 5:
        ema_period = int(p.get("trend_ema_period", 200))
        trend_ema = _ema(close, ema_period)
        fig.add_trace(go.Scatter(x=chart_df.index, y=trend_ema, name=f"EMA({ema_period})",
                                 line=dict(color="#1976d2", width=1.5, dash="dot")), row=1, col=1)

    # --- Buy/Sell signal marker on chart ---
    if signal != "HOLD":
        last_time = chart_df.index[-1]
        last_price = float(chart_df["Close"].iloc[-1])
        marker_color = "#26a69a" if signal == "BUY" else "#ef5350"
        marker_symbol = "triangle-up" if signal == "BUY" else "triangle-down"
        marker_label = f"{'VETOED ' if vetoed else ''}{signal}"
        if vetoed:
            marker_color = "#ff9800" 

        fig.add_trace(go.Scatter(
            x=[last_time], y=[last_price],
            mode="markers+text",
            marker=dict(symbol=marker_symbol, size=18, color=marker_color,
                        line=dict(width=2, color="white")),
            text=[marker_label],
            textposition="top center" if signal == "BUY" else "bottom center",
            textfont=dict(size=13, color=marker_color, family="Arial Black"),
            name=f"{signal} Signal",
            showlegend=False,
        ), row=1, col=1)

    # Entry/SL/TP lines
    if signal != "HOLD" and not vetoed:
        ep = result.get("entry_price")
        sl = result.get("stop_loss")
        tp = result.get("take_profit")
        if ep:
            fig.add_hline(y=ep, line_dash="dash", line_color="#2196f3", line_width=1.5,
                          annotation_text=f"Entry {ep:.4f}", row=1, col=1)
        if sl:
            fig.add_hline(y=sl, line_dash="dash", line_color="#ef5350", line_width=1.5,
                          annotation_text=f"SL {sl:.4f}", row=1, col=1)
        if tp:
            fig.add_hline(y=tp, line_dash="dash", line_color="#26a69a", line_width=1.5,
                          annotation_text=f"TP {tp:.4f}", row=1, col=1)

    # --- Row 2: Volume ---
    colors = ["#26a69a" if c >= o else "#ef5350"
              for c, o in zip(chart_df["Close"], chart_df["Open"])]
    fig.add_trace(go.Bar(x=chart_df.index, y=chart_df["Volume"], name="Volume",
                         marker_color=colors, opacity=0.5, showlegend=False), row=2, col=1)

    # --- Row 3: Oscillator (only for families 1, 3, 5) ---
    if has_oscillator:
        osc_row = 3
        if family == 1:
            rsi_period = int(p.get("rsi_period", 14))
            rsi_vals = _rsi(close, rsi_period)
            fig.add_trace(go.Scatter(x=chart_df.index, y=rsi_vals, name=f"RSI({rsi_period})",
                                     line=dict(color="#7b1fa2", width=1.5)), row=osc_row, col=1)
            oversold = float(p.get("rsi_oversold", 30))
            overbought = float(p.get("rsi_overbought", 70))
            fig.add_hline(y=oversold, line_dash="dot", line_color="#26a69a", line_width=0.8, row=osc_row, col=1)
            fig.add_hline(y=overbought, line_dash="dot", line_color="#ef5350", line_width=0.8, row=osc_row, col=1)
            fig.add_hline(y=50, line_dash="dot", line_color="gray", line_width=0.5, row=osc_row, col=1)
            fig.update_yaxes(title_text="RSI", row=osc_row, col=1, range=[0, 100])

        elif family == 3:
            crsi_period = int(p.get("crsi_period", 2))
            crsi_vals = _rsi(close, crsi_period)
            fig.add_trace(go.Scatter(x=chart_df.index, y=crsi_vals, name=f"CRSI({crsi_period})",
                                     line=dict(color="#7b1fa2", width=1.5)), row=osc_row, col=1)
            entry_lvl = float(p.get("crsi_entry", 10))
            entry_short = float(p.get("crsi_entry_short", 90))
            fig.add_hline(y=entry_lvl, line_dash="dot", line_color="#26a69a", line_width=0.8, row=osc_row, col=1)
            fig.add_hline(y=entry_short, line_dash="dot", line_color="#ef5350", line_width=0.8, row=osc_row, col=1)
            fig.update_yaxes(title_text="Connors RSI", row=osc_row, col=1, range=[0, 100])

        elif family == 5:
            wr_period = int(p.get("wr_period", 14))
            wr_vals = _williams_r(chart_df, wr_period)
            fig.add_trace(go.Scatter(x=chart_df.index, y=wr_vals, name=f"W%R({wr_period})",
                                     line=dict(color="#7b1fa2", width=1.5)), row=osc_row, col=1)
            oversold = float(p.get("wr_oversold", -80))
            overbought = float(p.get("wr_overbought", -20))
            fig.add_hline(y=oversold, line_dash="dot", line_color="#26a69a", line_width=0.8, row=osc_row, col=1)
            fig.add_hline(y=overbought, line_dash="dot", line_color="#ef5350", line_width=0.8, row=osc_row, col=1)
            fig.update_yaxes(title_text="Williams %R", row=osc_row, col=1, range=[-100, 0])

    # For family 4, show ADX in volume row
    if family == 4 and not has_oscillator:
        adx_period = int(p.get("adx_period", 14))
        adx_vals = _adx(chart_df, adx_period)
        fig.add_trace(go.Scatter(x=chart_df.index, y=adx_vals, name=f"ADX({adx_period})",
                                 line=dict(color="#7b1fa2", width=1.5)), row=2, col=1)
        adx_min = float(p.get("adx_min", 25))
        fig.add_hline(y=adx_min, line_dash="dot", line_color="#ff9800", line_width=0.8, row=2, col=1)

    # --- Clean light layout with standard chart zoom ---
    fig.update_layout(
        height=650 if has_oscillator else 500,
        template="plotly_white",
        xaxis_rangeslider_visible=False,
        margin=dict(l=50, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(size=10)),
        showlegend=True,
        dragmode="pan", 
    )

    for i in range(1, rows + 1):
        fig.update_xaxes(gridcolor="#e0e0e0", zeroline=False, row=i, col=1)
        fig.update_yaxes(gridcolor="#e0e0e0", zeroline=False, side="right", row=i, col=1)

    fig.update_yaxes(showticklabels=False, row=2, col=1)

    # Remove weekend/holiday gaps for stocks
    if market == "Stocks":
        fig.update_xaxes(
            rangebreaks=[
                dict(bounds=["sat", "mon"]),          
                dict(bounds=[20, 4], pattern="hour"),  
            ]
        )

    config = {
        "scrollZoom": True,        
        "displayModeBar": False,   
        "displaylogo": False,
    }
    st.plotly_chart(fig, width="stretch", config=config)

    # --- GA Metrics (with explanations)
    st.markdown("---")
    render_ga_metrics(ticker, style, selected_family=family)

    # --- Multi-Family Comparison ---
    st.markdown("---")
    st.subheader("All Families Comparison")

    if market_closed:
        st.info("📅 **Market is closed.** All families may show HOLD because there are no live "
                "price movements to trigger entry signals. Entry/SL/TP prices only appear when "
                "the strategy detects an active signal. This is normal behavior — check back during "
                "market hours (Mon-Fri).")

    # Load per-family GA test data if available
    all_params = load_all_params()
    all_families_data = {}
    if ticker in all_params and style in all_params[ticker]:
        all_families_data = all_params[ticker][style].get("all_families", {})

    # Only show families that were GA-tested
    if all_families_data:
        families_to_show = sorted(int(k) for k in all_families_data.keys())
    else:
        families_to_show = list(range(1, 6))

    all_results = engine.analyze_all_families(df, is_forex=is_forex)
    comparison_data = []
    w = load_optimized_params(ticker, style)
    for fam_num in families_to_show:
        r = all_results[fam_num]
        # Override to HOLD when market is closed
        fam_signal = r["signal"]
        if market_closed and fam_signal != "HOLD":
            fam_signal = "HOLD"
            r["entry_price"] = None
            r["stop_loss"] = None
            r["take_profit"] = None
        # GA winner tag
        ga_tag = ""
        if w and w["family"] == fam_num:
            ga_tag = " ⭐"
        # GA test return for this family
        fam_ga = all_families_data.get(str(fam_num), {}).get("metrics", {})
        ga_ret = fam_ga.get("return_pct")
        ga_ret_str = f"{ga_ret:+.2f}%" if ga_ret is not None else "—"

        comparison_data.append({
            "Family": f"F{fam_num}: {active_family_names[fam_num]}{ga_tag}",
            "Signal": fam_signal,
            "GA Test Return": ga_ret_str,
            "Entry": fmt_price(r["entry_price"]) if r["entry_price"] else "—",
            "Stop Loss": fmt_price(r["stop_loss"]) if r["stop_loss"] else "—",
            "Take Profit": fmt_price(r["take_profit"]) if r["take_profit"] else "—",
            "Confidence": r.get("confidence", "—"),
            "Reason": r.get("reason", "—"),
        })

    st.dataframe(comparison_data, width="stretch", hide_index=True)
    st.caption("⭐ = GA-optimized best family for this asset/style")


# ---------------------------------------------------------------------------
# GA Results Explorer Page
# ---------------------------------------------------------------------------
def ga_results_explorer():
    st.title("📊 GA Backtesting Results Explorer")
    st.markdown("Browse and compare GA optimization results across all assets, "
                "styles, and strategy families.")

    all_params = load_all_params()
    if not all_params:
        st.warning("No GA results found. Run extract_winners.py to generate optimized_params.json.")
        return

    # --- Metric explanations ---
    with st.expander("📖 What do these metrics mean?", expanded=False):
        st.markdown("""
| Metric | Description |
|--------|-------------|
| **Test Return** | Percentage profit/loss on unseen 2024-2025 data. Positive = strategy made money on data it was never trained on. |
| **Sharpe Ratio** | Risk-adjusted return. Measures return per unit of risk. Higher is better. Above 1.0 is good, above 2.0 is excellent. Negative means the strategy lost money relative to a risk-free investment. |
| **Max Drawdown** | Largest peak-to-trough decline during the test period. Lower is better — it shows the worst-case loss you would have experienced. |
| **Win Rate** | Percentage of trades that were profitable. Mean reversion strategies typically target 50-70%. Even 30-40% can be profitable if winners are much larger than losers. |
| **Entries** | Total number of trades opened during the test period. Too few (<10) = may be unreliable. Too many (>500) = possible overtrading. |
""")

    # --- Filter controls ---
    col1, col2 = st.columns(2)
    with col1:
        filter_market = st.multiselect(
            "Filter by Market",
            ["Forex", "Stocks"],
            default=["Forex", "Stocks"],
        )
    with col2:
        filter_style = st.multiselect(
            "Filter by Style",
            ["Day Trade", "Swing"],
            default=["Day Trade", "Swing"],
        )

    style_filter_map = {"Day Trade": "day_trade", "Swing": "swing"}
    market_tickers = set()
    for m in filter_market:
        market_tickers.update(ASSET_OPTIONS.get(m, []))

    # Build results table
    results = []
    for ticker_key, styles_data in all_params.items():
        if ticker_key not in market_tickers:
            continue
        for style_key, data in styles_data.items():
            style_label = style_key.replace("_", " ").title()
            if style_label not in filter_style:
                continue
            m = data.get("metrics", {})
            ret = m.get("return_pct")
            if ret is not None and ret == 0.0:
                continue
            results.append({
                "Asset": DISPLAY_NAMES.get(ticker_key, ticker_key),
                "Style": style_label,
                "Best Family": f"F{data['family']}: {data.get('family_name', '')}",
                "Test Return": ret,
                "Sharpe": m.get("sharpe"),
                "Max Drawdown": m.get("drawdown_pct"),
                "Win Rate": m.get("win_rate_pct"),
                "Entries": m.get("entries"),
            })

    if not results:
        st.info("No results match the current filters.")
        return

    results.sort(key=lambda x: x["Test Return"] if x["Test Return"] is not None else -999, reverse=True)

    # Display with formatting
    display_results = []
    for r in results:
        display_results.append({
            "Asset": r["Asset"],
            "Style": r["Style"],
            "Best Family": r["Best Family"],
            "Test Return": f"{r['Test Return']:+.2f}%" if r["Test Return"] is not None else "N/A",
            "Sharpe": f"{r['Sharpe']:.3f}" if r["Sharpe"] is not None else "N/A",
            "Max Drawdown": f"{r['Max Drawdown']:.1f}%" if r["Max Drawdown"] is not None else "N/A",
            "Win Rate": f"{r['Win Rate']:.0f}%" if r["Win Rate"] is not None else "N/A",
            "Entries": str(int(r["Entries"])) if r["Entries"] is not None else "N/A",
        })

    st.dataframe(display_results, width="stretch", hide_index=True)

    # --- Summary stats ---
    st.markdown("---")
    st.subheader("Summary Statistics")

    positive = [r for r in results if r["Test Return"] is not None and r["Test Return"] > 0]
    negative = [r for r in results if r["Test Return"] is not None and r["Test Return"] <= 0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Combos", len(results))
    col2.metric("Profitable", len(positive), help="Asset/style combos with positive test return")
    col3.metric("Unprofitable", len(negative))
    if positive:
        best = max(positive, key=lambda x: x["Test Return"])
        col4.metric("Best Return", f"{best['Test Return']:+.2f}%",
                     help=f"{best['Asset']} {best['Style']}")

    # --- Detailed view per asset ---
    st.markdown("---")
    st.subheader("Detailed View")

    selected_asset = st.selectbox(
        "Select asset for detailed view",
        [r["Asset"] for r in results],
        format_func=lambda x: x,
    )

    # Find matching ticker key
    selected_ticker = None
    for tk, name in DISPLAY_NAMES.items():
        if name == selected_asset:
            selected_ticker = tk
            break
    if selected_ticker is None:
        selected_ticker = selected_asset

    if selected_ticker in all_params:
        asset_data = all_params[selected_ticker]
        for style_key, data in asset_data.items():
            m = data.get("metrics", {})
            ret = m.get("return_pct")
            if ret is not None and ret == 0.0:
                continue

            style_label = style_key.replace("_", " ").title()
            st.markdown(f"#### {selected_asset} — {style_label}")

            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            ret = m.get("return_pct")
            mc1.metric("Test Return", f"{ret:+.2f}%" if ret is not None else "N/A",
                       help="Percentage profit/loss on unseen 2024-2025 test data")
            sharpe = m.get("sharpe")
            mc2.metric("Sharpe Ratio", f"{sharpe:.3f}" if sharpe is not None else "N/A",
                       help="Risk-adjusted return — higher is better, >1.0 is good")
            dd = m.get("drawdown_pct")
            mc3.metric("Max Drawdown", f"{dd:.1f}%" if dd is not None else "N/A",
                       help="Largest peak-to-trough loss during the test period")
            wr = m.get("win_rate_pct")
            mc4.metric("Win Rate", f"{wr:.0f}%" if wr is not None else "N/A",
                       help="Percentage of trades that were profitable")
            entries = m.get("entries")
            mc5.metric("Entries", f"{int(entries)}" if entries is not None else "N/A",
                       help="Total trades opened during the 2024-2025 test period")

            st.caption(f"**Best Family:** F{data['family']} — {data.get('family_name', 'Unknown')}")

            # Show optimized parameters
            params_data = data.get("params", {})
            if params_data:
                with st.expander("View GA-Optimized Parameters"):
                    param_cols = st.columns(3)
                    for i, (k, v) in enumerate(sorted(params_data.items())):
                        param_cols[i % 3].text(f"{k}: {v}")


# ---------------------------------------------------------------------------
# Portfolio Overview Page
# ---------------------------------------------------------------------------
def portfolio_overview():
    st.title("Portfolio Overview — All Assets Scan")

    all_params = load_all_params()
    engine = SignalEngine()

    if st.button("Scan All Assets", type="primary"):
        results = []

        all_assets = []
        for market_name, tickers in ASSET_OPTIONS.items():
            for t in tickers:
                available_styles = STYLE_OPTIONS.get(market_name, ["Day Trade"])
                for s_display in available_styles:
                    s = STYLE_MAP[s_display]
                    all_assets.append((t, s, s_display, market_name))

        progress = st.progress(0)
        for i, (t, s, s_display, mkt) in enumerate(all_assets):
            progress.progress((i + 1) / len(all_assets))
            df = fetch_data(t, s)
            if df.empty:
                continue

            w = load_optimized_params(t, s)
            fam = w["family"] if w else 1
            params = w.get("params", {}) if w else {}

            t_is_forex = mkt == "Forex"
            result = engine.analyze(df, fam, params, is_forex=t_is_forex)
            metrics = w.get("metrics", {}) if w else {}

            # Override signal to HOLD if market is closed
            sig = result["signal"]
            if len(df) > 0:
                last_bar = df.index[-1]
                now = pd.Timestamp.now(tz=last_bar.tzinfo if last_bar.tzinfo else None)
                hours_gap = (now - last_bar).total_seconds() / 3600
                if hours_gap > 4:
                    sig = "HOLD"

            fnames = FOREX_FAMILY_NAMES if t_is_forex else FAMILY_NAMES
            results.append({
                "Asset": DISPLAY_NAMES.get(t, t),
                "Market": mkt,
                "Style": s_display,
                "Best Family": f"F{fam}",
                "Signal": sig,
                "GA Return": f"{metrics.get('return_pct', 0):+.2f}%" if metrics.get("return_pct") is not None else "N/A",
                "GA Win Rate": f"{metrics.get('win_rate_pct', 0):.0f}%" if metrics.get("win_rate_pct") is not None else "N/A",
                "Confidence": result.get("confidence", "—") if sig != "HOLD" else "—",
            })

        progress.empty()

        if results:
            # Check if today is a weekend
            today = pd.Timestamp.now()
            if today.weekday() >= 5:
                st.info("📅 **Market is closed** (weekend). All signals default to HOLD. "
                        "Check back during market hours (Mon–Fri) for live signals.")

            st.dataframe(results, width="stretch", hide_index=True)
            buy_count = sum(1 for r in results if r["Signal"] == "BUY")
            sell_count = sum(1 for r in results if r["Signal"] == "SELL")
            hold_count = sum(1 for r in results if r["Signal"] == "HOLD")
            st.markdown(f"**Summary:** 🟢 {buy_count} BUY | 🔴 {sell_count} SELL | ⚪ {hold_count} HOLD")
        else:
            st.warning("No data could be fetched for any assets.")
    else:
        st.info("Click 'Scan All Assets' to analyze all assets at once. "
                "Note: during market closed hours (weekends/holidays), "
                "most signals will show HOLD.")

        # Show a quick summary of GA results
        st.subheader("GA Optimization Results Summary")

        # Add metric explanations
        with st.expander("📖 What do these metrics mean?"):
            st.markdown("""
| Metric | Description |
|--------|-------------|
| **Test Return** | Profit/loss on unseen 2024-2025 data. Positive = strategy made money on data it never saw during training. |
| **Win Rate** | Percentage of trades that were profitable. |
| **Drawdown** | Largest peak-to-trough decline — the worst loss you'd have experienced. |
| **Entries** | Total number of trades during the test period. |
""")

        if all_params:
            summary = []
            for ticker_key, styles_data in all_params.items():
                for style_key, data in styles_data.items():
                    m = data.get("metrics", {})
                    ret = m.get("return_pct")
                    if ret is not None and ret == 0.0:
                        continue
                    summary.append({
                        "Asset": DISPLAY_NAMES.get(ticker_key, ticker_key),
                        "Style": style_key.replace("_", " ").title(),
                        "Best Family": f"F{data['family']}: {data.get('family_name', '')}",
                        "Test Return": f"{ret:+.2f}%" if ret is not None else "N/A",
                        "Win Rate": f"{m.get('win_rate_pct', 0):.0f}%" if m.get("win_rate_pct") else "N/A",
                        "Drawdown": f"{m.get('drawdown_pct', 0):.1f}%" if m.get("drawdown_pct") else "N/A",
                        "Entries": str(int(m.get("entries", 0))) if m.get("entries") else "N/A",
                    })
            if summary:
                summary.sort(key=lambda x: float(x["Test Return"].replace("+", "").replace("%", "").replace("N/A", "-999")), reverse=True)
                st.dataframe(summary, width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# Main routing
# ---------------------------------------------------------------------------
if page == "Signal View":
    signal_view()
elif page == "GA Results Explorer":
    ga_results_explorer()
else:
    portfolio_overview()

# ---------------------------------------------------------------------------
# Auto-refresh trigger
# ---------------------------------------------------------------------------
if auto_refresh:
    import time as _time
    _time.sleep(60)
    st.rerun()
