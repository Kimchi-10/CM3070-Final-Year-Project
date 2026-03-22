"""
GA Metrics Component
--------------------
Shows GA backtest results for the selected strategy family.
Loads per-family test metrics from optimized_params.json.
"""

import streamlit as st
from dashboard.utils import load_optimized_params, load_all_params, FAMILY_NAMES, FOREX_FAMILY_NAMES

FOREX_TICKERS = {"EURUSD=X", "USDJPY=X"}


def _get_family_metrics(ticker, style, family_num):
    """Try to load metrics for a specific family from all_families data."""
    all_params = load_all_params()
    if ticker in all_params and style in all_params[ticker]:
        all_fam = all_params[ticker][style].get("all_families", {})
        fam_data = all_fam.get(str(family_num))
        if fam_data:
            return fam_data.get("metrics", {})
    return None


def _get_tested_families(ticker, style):
    """Return the set of family numbers that have test data."""
    all_params = load_all_params()
    if ticker in all_params and style in all_params[ticker]:
        af = all_params[ticker][style].get("all_families", {})
        return {int(k) for k in af.keys()}
    return set()


def _render_metrics_row(metrics):
    """Display the 5 metric columns."""
    col1, col2, col3, col4, col5 = st.columns(5)

    ret = metrics.get("return_pct")
    col1.metric("Test Return", f"{ret:+.2f}%" if ret is not None else "N/A",
                help="Profit/loss on unseen 2024-2025 test data")

    sharpe = metrics.get("sharpe")
    col2.metric("Sharpe Ratio", f"{sharpe:.3f}" if sharpe is not None else "N/A",
                help="Risk-adjusted return — higher is better, >1.0 is good")

    dd = metrics.get("drawdown_pct")
    col3.metric("Max Drawdown", f"{dd:.1f}%" if dd is not None else "N/A",
                help="Largest peak-to-trough loss — lower is better")

    wr = metrics.get("win_rate_pct")
    col4.metric("Win Rate", f"{wr:.0f}%" if wr is not None else "N/A",
                help="Percentage of profitable trades")

    entries = metrics.get("entries")
    col5.metric("Entries", f"{entries}" if entries is not None else "N/A",
                help="Total trades during the 2024-2025 test period")


def render_ga_metrics(ticker: str, style: str = "day_trade", selected_family: int = None):
    """Show GA backtest metrics for the selected family.

    Per-family test data coverage:
    - Forex day trade: all 5 families
    - Stocks day trade: all 5 families
    - Stocks swing: F1 and F3 only (compute budget constraint)
    """
    fnames = FOREX_FAMILY_NAMES if ticker in FOREX_TICKERS else FAMILY_NAMES
    winner = load_optimized_params(ticker, style)

    if winner is None:
        st.subheader("GA Backtest Results")
        st.info("No GA backtest results available for this asset/style combination.")
        return

    winner_fam = winner.get("family", "?")
    winner_name = winner.get("family_name", fnames.get(winner_fam, "Unknown"))
    params = winner.get("params", {})

    # Decide which family to show
    show_fam = selected_family if selected_family else winner_fam
    show_name = fnames.get(show_fam, "Unknown")
    is_winner = (show_fam == winner_fam)

    # Try to load per-family test metrics
    metrics = _get_family_metrics(ticker, style, show_fam)
    has_family_data = metrics is not None
    tested_families = _get_tested_families(ticker, style)

    # Header
    label = f"F{show_fam}: {show_name}"
    if is_winner:
        st.subheader(f"GA Backtest Results — {label} ⭐ Best")
    else:
        st.subheader(f"GA Backtest Results — {label}")

    # Metric explanations
    with st.expander("📖 What do these metrics mean?"):
        st.markdown("""
| Metric | Description |
|--------|-------------|
| **Test Return** | Percentage profit/loss on **unseen** 2024-2025 data. The GA trained on 2020-2022 data and validated on 2023 — this metric shows how the optimized strategy performed on data it never saw during training. Positive values mean the strategy made money. |
| **Sharpe Ratio** | Risk-adjusted return — measures how much return you get per unit of volatility (risk). A Sharpe > 1.0 is considered good, > 2.0 is excellent. Negative values mean the strategy underperformed a risk-free investment like treasury bills. |
| **Max Drawdown** | The largest peak-to-trough percentage decline during the test period. For example, 4.5% means the portfolio dropped at most 4.5% from its highest point before recovering. Lower is better — it shows your worst-case scenario. |
| **Win Rate** | Percentage of trades that ended in profit. Mean reversion strategies typically achieve 40-65%. Even a 30% win rate can be profitable if the average win is much larger than the average loss (high reward-to-risk ratio). |
| **Entries** | Total number of trades opened during the 2-year test period. Very few entries (<10) may mean the result is not statistically reliable. Too many (>500) could indicate overtrading. |
""")

    if has_family_data:
        # Show this family's actual test metrics
        if not is_winner:
            st.caption(f"⭐ GA winner is F{winner_fam}: {winner_name} "
                       f"({winner['metrics'].get('return_pct', 0):+.2f}% return)")
        _render_metrics_row(metrics)
    elif is_winner:
        # Winner always has data from the main entry
        _render_metrics_row(winner.get("metrics", {}))
    else:
        # No test data for this family
        winner_ret = winner["metrics"].get("return_pct", 0)
        tested_list = ", ".join(f"F{f}" for f in sorted(tested_families))

        if style == "swing" and tested_families:
            st.info(
                f"No test results for F{show_fam}: {show_name} in swing mode. "
                f"Swing trade GA runs were only done for **{tested_list}** due to "
                f"QuantConnect cloud compute limits. "
                f"The GA winner is **F{winner_fam}: {winner_name}** "
                f"(Return: **{winner_ret:+.2f}%**)."
            )
        else:
            st.info(
                f"No test results for F{show_fam}: {show_name}. "
                f"Only the GA winner **F{winner_fam}: {winner_name}** "
                f"(Return: **{winner_ret:+.2f}%**) was tested out-of-sample. "
                f"Select F{winner_fam} above to see its results."
            )
        return  

    # Backtesting configuration
    with st.expander("⚙️ Backtesting Configuration"):
        cfg_col1, cfg_col2 = st.columns(2)
        with cfg_col1:
            st.markdown("**Training & Testing Periods**")
            st.markdown("""
- **Train:** 2020 – 2022 (GA optimization)
- **Validation:** 2023 (parameter selection)
- **Test:** 2024 – 2025 (out-of-sample evaluation)
- **Starting Capital:** $10,000
""")
        with cfg_col2:
            if style == "day_trade":
                st.markdown("**Timeframe Configuration — Day Trade**")
                st.markdown("""
- **Entry Bars:** 5-minute candles
- **HTF Trend Filter:** 1-hour EMA (50-period) — determines trend direction
- **Volatility Filter:** Bollinger Band Width on 1-hour bars — skips flat/extreme markets
- **Session Filter:** ON (trades only during active market hours)
- **Force Flat:** ON (closes positions at end of day)
""")
            else:
                st.markdown("**Timeframe Configuration — Swing**")
                st.markdown("""
- **Entry Bars:** 1-hour candles
- **No session filter** (holds positions overnight)
- **No force flat** (positions held for days/weeks)
- **Wider stops** (2-3× ATR) for swing trades
""")

    # Show optimized parameters (only for the winner family)
    if is_winner and params:
        with st.expander("View GA-Optimized Parameters"):
            param_cols = st.columns(3)
            for i, (k, v) in enumerate(sorted(params.items())):
                param_cols[i % 3].text(f"{k}: {v}")
