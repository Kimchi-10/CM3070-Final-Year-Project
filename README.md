# AI-Driven Dynamic Portfolio Advisor

**Student:** Ng Chang Yan
**Student Number:** 230367197
**Module:** CM3070 — Final Year Project
**University of London (Goldsmiths)**
**Supervisor:** Koon Heng Ronnie Peh

---

## Overview

An AI-driven trading signal system that uses **Genetic Algorithm (GA) optimisation** on QuantConnect's LEAN engine to discover profitable intraday trading strategies, then delivers live signals through a **Streamlit dashboard** with **FinBERT sentiment analysis** as a veto layer.

The system implements 5 strategy families across two categories (mean reversion and trend-following), optimised on Forex and US Stocks via walk-forward validation (Train: 2020–2022, Validation: 2023, Test: 2024–2025).

## Architecture

```
                    ┌─────────────────────────────┐
                    │   QuantConnect LEAN Cloud    │
                    │   (Backtesting Engine)       │
                    └──────────┬──────────────────┘
                               │
                    ┌──────────▼──────────────────┐
                    │   Genetic Algorithm Runner   │
                    │   ga_runner.py               │
                    │   - Population: 15           │
                    │   - Generations: 6           │
                    │   - Walk-forward validation  │
                    └──────────┬──────────────────┘
                               │
                    ┌──────────▼──────────────────┐
                    │   extract_winners.py         │
                    │   → optimized_params.json    │
                    └──────────┬──────────────────┘
                               │
    ┌──────────────────────────▼───────────────────────────┐
    │              Streamlit Dashboard (app.py)             │
    │  ┌─────────┐  ┌──────────┐  ┌─────────────────────┐ │
    │  │ Signal   │  │ FinBERT  │  │ GA Results Explorer │ │
    │  │ Engine   │  │ Sentiment│  │                     │ │
    │  │ (5 fam.) │  │ Veto     │  │ Interactive browse  │ │
    │  └─────────┘  └──────────┘  └─────────────────────┘ │
    └──────────────────────────────────────────────────────┘
```

## Key Features

- **GA-optimised trading signals** across 5 strategy families (mean reversion and trend-following)
- **FinBERT sentiment analysis** as a veto layer to filter out signals during negative news events
- **Q-Learning RL adaptive veto agent** that learns when to trust or override the sentiment veto
- **Interactive Plotly candlestick charts** with indicator overlays (Bollinger Bands, EMA, RSI, etc.)
- **Walk-forward validation** (Train 2020–2022 / Validate 2023 / Test 2024–2025) to guard against overfitting
- **Multi-asset support:** Forex (EURUSD, USDJPY) and Stocks (SPY, QQQ, AAPL, TSLA)

## Dashboard Pages

The Streamlit dashboard is organised into three pages:

1. **Signal View** — The primary trading interface. Select an asset and strategy family to view a live candlestick chart with indicator overlays. When a signal is detected, it displays the entry price, stop-loss, take-profit levels, and the current FinBERT sentiment veto status. An all-families comparison table shows signals from every strategy family side by side.

2. **Portfolio Overview** — A scan across all supported assets at once. Quickly identify which tickers have active trading signals and from which strategy families, providing a high-level summary of current market opportunities.

3. **GA Results Explorer** — Browse and compare all GA optimisation results interactively. View fitness progression across generations, compare parameter distributions, and inspect individual backtest metrics for any ticker-family combination.

## Strategy Families

### Forex (EURUSD, USDJPY)
| ID | Family | Type | Best OOS Result |
|----|--------|------|:---------------:|
| F1 | RSI Mean Reversion | Mean Reversion | — |
| F2 | Bollinger Band Bounce | Mean Reversion | USDJPY +12.73% |
| F3 | EMA Crossover | Trend Following | EURUSD +8.55% |
| F4 | Stochastic Mean Reversion | Mean Reversion | — |
| F5 | EMA Trend + ADX | Trend Following | — |

### Stocks (SPY, QQQ, AAPL, TSLA)
| ID | Family | Type | Best OOS Result |
|----|--------|------|:---------------:|
| F1 | RSI Mean Reversion | Mean Reversion | AAPL +11.87% |
| F2 | Bollinger Band Mean Reversion | Mean Reversion | — |
| F3 | Connors RSI (2-Period) | Mean Reversion | QQQ +6.56% |
| F4 | EMA Trend + ADX | Trend Following | — |
| F5 | Williams %R + RSI | Mean Reversion | — |

## GA Run Coverage

| Asset Class | Assets | Day Trade | Swing Trade | Total CSVs |
|-------------|--------|-----------|-------------|------------|
| **Forex** | EURUSD, USDJPY | All 5 families | — | 10 |
| **Stocks** | SPY, QQQ, AAPL, TSLA | All 5 families | F1 & F3 only | 28 |
| **Crypto** | BTC-USD, SOL-USD | All 5 families | — | 7 |

**Why only 2 families for stock swing trades?** Running all 5 families for both day-trade and swing across 4 stocks would mean 40 GA runs on QuantConnect cloud — too many for our compute budget. We picked F1 (RSI Mean Reversion) and F3 (Connors RSI) for swing because they had the best day-trade results and their signals work well on longer timeframes. Crypto was tested but dropped after every family came back at 0% — more on that in the backtesting notebook.

## Out-of-Sample Results (2024–2025)

The following table shows the top-performing asset-family combinations on the held-out test period (Jan 2024 – Dec 2025), using GA-optimised parameters selected during walk-forward validation:

| Asset  | Best Family | Test Return | Max Drawdown | Win Rate | Sharpe | Entries |
|--------|-------------|:-----------:|:------------:|:--------:|:------:|:-------:|
| USDJPY | F2 — Bollinger Band Bounce | **+12.73%** | 51.6% | 40% | +0.199 | 15 |
| AAPL   | F1 — RSI Mean Reversion | **+11.87%** | 110.0% | 47% | -0.422 | 137 |
| EURUSD | F3 — EMA Crossover | **+8.55%** | 7.0% | 36% | -0.405 | 78 |
| QQQ    | F3 — Connors RSI (2-Period) | **+6.56%** | 4.5% | 32% | -0.671 | 117 |

## Project Structure

```
quantconnect/
├── forex_engine/
│   ├── main.py              # QC LEAN algorithm — Forex
│   └── config.json          # Default parameters
├── stock_engine/
│   ├── main.py              # QC LEAN algorithm — Stocks
│   └── config.json
├── crypto_engine/
│   ├── main.py              # QC LEAN algorithm — Crypto (excluded from results)
│   └── config.json
├── genetic_algorithm/
│   ├── ga_runner.py          # GA optimisation runner
│   ├── extract_winners.py    # Extract best params → JSON
│   ├── backtesting_report.ipynb  # Analysis notebook
│   ├── run_all_ga.sh         # Batch runner script
│   └── results/              # GA result CSVs + checkpoint JSONs
├── dashboard/
│   ├── app.py                # Streamlit main app
│   ├── utils.py              # Shared constants, data loading
│   ├── strategies/
│   │   ├── signal_engine.py  # 5 strategy families signal logic
│   │   └── optimized_params.json  # GA-optimised parameters
│   └── components/
│       ├── ga_metrics.py     # GA results display component
│       ├── rl_agent.py       # Q-Learning sentiment agent
│       └── sentiment.py      # FinBERT sentiment analysis
├── tests/
│   ├── conftest.py           # Test configuration
│   ├── test_signal_engine.py # Signal engine unit tests (56 tests)
│   ├── test_sentiment.py     # Sentiment component tests (16 tests)
│   ├── test_rl_agent.py      # RL agent tests (21 tests)
│   └── test_ga_metrics.py    # GA metrics & utils tests (24 tests)
├── lean.json                 # LEAN CLI configuration
└── README.md
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backtesting | QuantConnect LEAN (C# engine, Python API) |
| Optimisation | Custom GA runner (Python) |
| Dashboard | Streamlit + Plotly |
| Market Data | yfinance |
| Sentiment | ProsusAI/finbert (HuggingFace Transformers) |
| RL Agent | Q-Learning (custom implementation) |
| Testing | pytest (117 tests) |

## Setup

### Prerequisites

- Python 3.10+
- QuantConnect LEAN CLI (for running backtests only — not needed for the dashboard)

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd "Final Year"

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# Install core dependencies
pip install -r requirements.txt

# Optional: Install FinBERT for sentiment analysis (~2GB download)
pip install transformers torch
```

### Running the Dashboard

```bash
cd quantconnect
streamlit run dashboard/app.py
```

Opens at http://localhost:8501

### Training the RL Veto Agent (Optional)

```bash
cd quantconnect
python -m dashboard.components.rl_agent --tickers SPY QQQ AAPL TSLA --episodes 50
```

A pre-trained Q-table is included at `dashboard/strategies/rl_qtable.json`.

### Running Tests

```bash
cd quantconnect
python -m pytest tests/ -v
```

All 117 tests should pass.

## GA Optimisation Configuration

| Parameter | Value |
|-----------|-------|
| Population Size | 15 |
| Generations | 6 |
| Elitism | 5 (top 33%) |
| Crossover | Uniform |
| Mutation Rate | 15% per gene |
| Fitness | 3.0 x Sharpe + 1.0 x Return - Drawdown Penalty |
| Training Period | Jan 2020 – Dec 2022 |
| Validation Period | Jan 2023 – Dec 2023 |
| Test Period | Jan 2024 – Dec 2025 |
| Starting Capital | $10,000 |

## Testing

The project includes 117 unit tests across four test modules, covering the core dashboard logic:

- **`test_signal_engine.py`** — 56 tests covering indicator calculations, signal generation logic, edge cases (empty data, missing columns), and parameter override behaviour across all 5 strategy families.
- **`test_sentiment.py`** — 16 tests covering FinBERT veto logic, sentiment score thresholds, headline fetching, and graceful fallback when the model is unavailable.
- **`test_rl_agent.py`** — 21 tests covering Q-learning update rules, state discretisation, epsilon-greedy exploration, Q-table save/load persistence, and end-to-end training loops.
- **`test_ga_metrics.py`** — 24 tests covering GA parameter loading from JSON, ticker-to-family mapping, CSV result parsing, and fitness metric calculations.

Run the full suite with:

```bash
python -m pytest tests/ -v
```

