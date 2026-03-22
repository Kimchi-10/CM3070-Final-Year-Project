#!/usr/bin/env python3
"""
FYP Genetic Algorithm Runner
=============================
Optimizes trading strategy parameters for each family on QuantConnect Cloud.

Engines:
    forex  = Forex (OANDA) — default
    stock  = US Equities (Interactive Brokers)
    crypto = Crypto (Binance)

Families:
    1 = RSI Mean Reversion
    2 = Bollinger Band Mean Reversion
    3 = Connors RSI (2-Period)
    4 = EMA Trend + ADX
    5 = Williams %R + RSI
"""

import argparse
import csv
import json
import os
import random
import re
import subprocess
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict, fields
from datetime import datetime
from pathlib import Path
from statistics import mean

# Ensure CWD is the quantconnect/ root so `lean` CLI can find all engine projects
_QC_ROOT = Path(__file__).resolve().parent.parent
os.chdir(_QC_ROOT)

# Resolve full path to `lean` CLI — may be in a different venv
import shutil as _shutil
LEAN_BIN = _shutil.which("lean") or str(Path(__file__).resolve().parents[2] / "Project" / ".venv" / "bin" / "lean")
if not Path(LEAN_BIN).exists():
    # Fallback: search common locations
    for _candidate in [
        Path.home() / "Desktop" / "Final Year Project" / ".venv" / "bin" / "lean",
        Path.home() / "Desktop" / "Final Year" / ".venv" / "bin" / "lean",
    ]:
        if _candidate.exists():
            LEAN_BIN = str(_candidate)
            break


# ══════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════

TRAIN_START = "2020-01-01"
TRAIN_END   = "2022-12-31"
VALID_START = "2023-01-01"
VALID_END   = "2023-12-31"
TEST_START  = "2024-01-01"
TEST_END    = "2025-12-31"

# Engine-specific defaults
ENGINE_CONFIG = {
    "forex": {
        "project_name": "forex_engine",
        "starting_cash": 10000,
        "risk_per_trade_pct": 0.25,
        "styles": {
            "day_trade":  {"bar_minutes": 5,    "use_session_filter": 1, "use_force_flat": 1,
                           "session_start_hour": 7, "session_end_hour": 16, "last_entry_hour": 14,
                           "last_entry_minute": 0, "force_flat_hour": 15, "force_flat_minute": 55,
                           "max_trades_per_day": 5,
                           "htf_minutes": 60, "htf_ema_period": 50,
                           "vol_filter_enabled": 0, "vol_bb_period": 20},
            "swing":      {"bar_minutes": 60,   "use_session_filter": 0, "use_force_flat": 0,
                           "max_trades_per_day": 4,
                           "htf_minutes": 240, "htf_ema_period": 50,
                           "vol_filter_enabled": 1, "vol_bb_period": 20},
            "long_term":  {"bar_minutes": 1440, "use_session_filter": 0, "use_force_flat": 0,
                           "max_trades_per_day": 4,
                           "htf_minutes": 1440, "htf_ema_period": 50,
                           "vol_filter_enabled": 0},
        },
        "default_style": "day_trade",
    },
    "stock": {
        "project_name": "stock_engine",
        "starting_cash": 10000,
        "risk_per_trade_pct": 0.5,
        "styles": {
            "day_trade":  {"bar_minutes": 5,    "regime_minutes": 60,   "use_session_filter": 1, "use_force_flat": 1,
                           "session_start_hour": 9, "session_start_minute": 30, "session_end_hour": 16,
                           "last_entry_hour": 15, "last_entry_minute": 30, "force_flat_hour": 15,
                           "force_flat_minute": 55, "max_trades_per_day": 8},
            "swing":      {"bar_minutes": 60,   "regime_minutes": 240,  "use_session_filter": 0, "use_force_flat": 0,
                           "max_trades_per_day": 6},
            "long_term":  {"bar_minutes": 1440, "regime_minutes": 1440, "use_session_filter": 0, "use_force_flat": 0,
                           "max_trades_per_day": 4},
        },
        "default_style": "day_trade",
    },
    # CRYPTO REMOVED — Mean reversion has no edge on crypto markets.
    # Crypto is trend-driven (momentum), not mean-reverting.
    # GA results (BTCUSDT, all families): 0% returns across 6 generations.
    # Results CSVs kept in results/ for FYP report evidence.
    # "crypto": {
    #     "project_name": "crypto_engine",
    #     "starting_cash": 10000,
    #     "risk_per_trade_pct": 0.5,
    #     "styles": {
    #         "day_trade":  {"bar_minutes": 15,   "regime_minutes": 60,   "use_session_filter": 0, "use_force_flat": 0,
    #                        "max_trades_per_day": 12},
    #         "swing":      {"bar_minutes": 240,  "regime_minutes": 960,  "use_session_filter": 0, "use_force_flat": 0,
    #                        "max_trades_per_day": 8},
    #         "long_term":  {"bar_minutes": 1440, "regime_minutes": 1440, "use_session_filter": 0, "use_force_flat": 0,
    #                        "max_trades_per_day": 4},
    #     },
    #     "default_style": "day_trade",
    # },
}

PROJECT_NAME         = "forex_engine"
STARTING_CASH        = 10000
RISK_PER_TRADE_PCT   = 0.5
STYLE_PARAMS         = {} 

# GA Settings
POP_SIZE             = 15
GENERATIONS          = 6
ELITE_KEEP           = 5
TRAIN_SHORTLIST      = 6
VALIDATION_KEEP      = 3
FINAL_KEEP           = 2
MUTATION_RATE        = 0.35
RANDOM_INJECTION     = 2
SLEEP_BETWEEN_RUNS   = 1
PARALLEL_WORKERS     = 5   

# Fitness thresholds
MIN_TRADES_REQUIRED  = 5
MAX_DRAWDOWN_FAIL    = 60.0

STOCK_FAMILY_NAMES = {
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

def get_family_names():
    """Get active family names based on current engine."""
    return FOREX_FAMILY_NAMES if PROJECT_NAME == "forex_engine" else STOCK_FAMILY_NAMES

# Default for backward compat 
FAMILY_NAMES = FOREX_FAMILY_NAMES

# ══════════════════════════════════════════════════
#  PARAMETER CHOICES (Clean Multiples)
# ══════════════════════════════════════════════════

# Shared parameters across all families
SHARED_CHOICES = {
    "atr_period":          [10, 14, 20],
    "stop_atr_mult":       [1.0, 1.5, 2.0, 2.5, 3.0],
    "reward_ratio":        [1.5, 2.0, 2.5, 3.0],          
    "max_hold_bars":       [10, 15, 20, 30, 48],
    "cooldown_bars":       [0, 1, 2, 4, 6],
    "risk_per_trade_pct":  [0.15, 0.25, 0.35, 0.5],
}


FOREX_EXTRA_CHOICES = {
    "bar_minutes":        [5, 15, 30],             
    "htf_minutes":        [60, 240],
    "htf_ema_period":     [20, 30, 50],            
    "vol_filter_enabled": [0, 0, 0, 1],           
    "use_force_flat":     [0, 0, 1],               
}


FOREX_SHARED_OVERRIDES = {
    "risk_per_trade_pct":  [0.15, 0.25, 0.35, 0.5],     
    "stop_atr_mult":       [1.5, 2.0, 2.5, 3.0],       
    "reward_ratio":        [1.0, 1.25, 1.5, 2.0],           
    "max_hold_bars":       [20, 30, 48, 60, 96],         
    "cooldown_bars":       [3, 5, 8, 12],                
}

# Family-specific parameters — STOCK ENGINE (mean-reversion)
STOCK_FAMILY_CHOICES = {
    1: {  # RSI Mean Reversion
        "rsi_period":        [5, 7, 10, 14],
        "rsi_oversold":      [25, 30, 35, 40],
        "rsi_overbought":    [60, 65, 70, 75],
        "trend_ema_period":  [100, 150, 200],
    },
    2: {  # Bollinger Band Mean Reversion
        "bb_period":     [15, 20, 25],
        "bb_std_dev":    [1.5, 2.0, 2.5],
        "bb_rsi_long":   [30, 35, 40],
        "bb_rsi_short":  [60, 65, 70],
        "rsi_period":    [10, 14, 20],
    },
    3: {  # Connors RSI (2-Period)
        "crsi_period":       [2, 3],
        "crsi_entry":        [5, 10, 15, 20],
        "crsi_entry_short":  [80, 85, 90, 95],
        "trend_ema_period":  [100, 150, 200],
        "exit_sma_period":   [3, 5, 8],
    },
    4: {  # EMA Trend + ADX
        "ema_fast_period":  [10, 15, 20, 25],
        "ema_slow_period":  [40, 50, 60, 80],
        "adx_period":       [10, 14, 20],
        "adx_min":          [20, 25, 30, 35],
    },
    5: {  # Williams %R + RSI
        "wr_period":      [10, 14, 20],
        "wr_oversold":    [-90, -85, -80],
        "wr_overbought":  [-20, -15, -10],
        "wr_rsi_long":    [30, 35, 40],
        "wr_rsi_short":   [60, 65, 70],
        "rsi_period":     [10, 14, 20],
    },
}

# Family-specific parameters — FOREX ENGINE (momentum/trend)
FOREX_FAMILY_CHOICES = {
    1: {  # RSI Mean Reversion (fade oversold/overbought)
        "rsi_period":        [5, 7, 10, 14],
        "rsi_oversold":      [20, 25, 30, 35],              # Buy below this level
        "rsi_overbought":    [65, 70, 75, 80],              # Sell above this level
        "trend_ema_period":  [20, 50, 100, 200],
    },
    2: {  # Bollinger Band Bounce (mean reversion from bands)
        "bb_period":         [15, 20, 25, 30],
        "bb_std_dev":        [1.5, 2.0, 2.5],
        "bb_rsi_long":       [25, 30, 35, 40],              # RSI must be below this to buy
        "bb_rsi_short":      [60, 65, 70, 75],              # RSI must be above this to sell
        "rsi_period":        [7, 10, 14],
    },
    3: {  # EMA Crossover (trend following)
        "ema_fast_period":   [8, 10, 15, 20],
        "ema_slow_period":   [30, 40, 50, 60],
        "trend_ema_period":  [20, 30, 50, 100],
    },
    4: {  # Stochastic Mean Reversion (fade from oversold/overbought zones)
        "stoch_k":           [5, 8, 10, 14],
        "stoch_d":           [3, 5],
        "stoch_oversold":    [20, 25, 30, 35],               # Buy when %K is below this level
        "stoch_overbought":  [65, 70, 75, 80],               # Sell when %K is above this level
        "trend_ema_period":  [20, 50, 100, 200],
    },
    5: {  # EMA Trend + ADX (same as stock F4)
        "ema_fast_period":  [8, 10, 15, 20, 25],
        "ema_slow_period":  [30, 40, 50, 60, 80],
        "adx_period":       [10, 14, 20],
        "adx_min":          [15, 18, 20, 25],              # Lower threshold = more entries
    },
}

# Backward compat alias
FAMILY_CHOICES = STOCK_FAMILY_CHOICES


# ══════════════════════════════════════════════════
#  GENOME
# ══════════════════════════════════════════════════

def get_all_choices(family):
    """Get combined parameter choices for a specific family."""
    choices = dict(SHARED_CHOICES)
    # Use engine-specific family choices
    if PROJECT_NAME == "forex_engine":
        choices.update(FOREX_FAMILY_CHOICES.get(family, {}))
        choices.update(FOREX_EXTRA_CHOICES)
        choices.update(FOREX_SHARED_OVERRIDES)
    else:
        choices.update(STOCK_FAMILY_CHOICES.get(family, {}))
    return choices


def random_genome(family):
    """Generate a random genome for the given family."""
    choices = get_all_choices(family)
    genome = {}
    for param, values in choices.items():
        genome[param] = random.choice(values)

    # Enforce constraints
    if "ema_fast_period" in genome and "ema_slow_period" in genome:
        if genome["ema_fast_period"] >= genome["ema_slow_period"]:
            valid_slow = [s for s in choices["ema_slow_period"] if s > genome["ema_fast_period"]]
            genome["ema_slow_period"] = random.choice(valid_slow) if valid_slow else genome["ema_fast_period"] + 10

    return genome


def crossover(parent_a, parent_b, family):
    """Crossover two parent genomes."""
    child = {}
    for key in parent_a:
        child[key] = random.choice([parent_a[key], parent_b[key]])

    # Re-enforce constraints
    choices = get_all_choices(family)

    if "ema_fast_period" in child and "ema_slow_period" in child:
        if child["ema_fast_period"] >= child["ema_slow_period"]:
            valid = [s for s in choices["ema_slow_period"] if s > child["ema_fast_period"]]
            child["ema_slow_period"] = random.choice(valid) if valid else child["ema_fast_period"] + 10

    return child


def mutate(genome, family):
    """Randomly mutate one parameter."""
    if random.random() > MUTATION_RATE:
        return genome

    choices = get_all_choices(family)
    param = random.choice(list(genome.keys()))
    if param in choices:
        genome[param] = random.choice(choices[param])

    # Re-enforce constraints after mutation
    if "ema_fast_period" in genome and "ema_slow_period" in genome:
        if genome["ema_fast_period"] >= genome["ema_slow_period"]:
            valid = [s for s in choices.get("ema_slow_period", []) if s > genome["ema_fast_period"]]
            if valid:
                genome["ema_slow_period"] = random.choice(valid)

    return genome


def genome_key(genome):
    """Unique key for deduplication."""
    return tuple(sorted(genome.items()))


# ══════════════════════════════════════════════════
#  QUANTCONNECT CLOUD BACKTEST
# ══════════════════════════════════════════════════

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def clean_ansi(text):
    return ANSI_RE.sub("", text).replace("\r", "")


def build_params(pair, mode, family, genome, trading_style="day_trade"):
    """Build the full parameter dict for QC backtest."""
    params = {
        "mode": mode,
        "pairs": pair,
        "strategy_family": family,
        "starting_cash": STARTING_CASH,
    }
    # Apply style-specific defaults
    params.update(STYLE_PARAMS)
    # Genome values override everything
    params.update(genome)
    return params


def parse_stats(output):
    """Parse QC backtest output into a stats dict."""
    text = clean_ansi(output)
    stats = {}

    # Parse table rows
    table_values = {}
    for line in text.splitlines():
        line = line.strip()
        if "│" in line:
            parts = [p.strip() for p in line.split("│")]
            if len(parts) >= 5:
                for i in range(1, len(parts) - 1, 2):
                    if i + 1 < len(parts):
                        label, value = parts[i], parts[i + 1]
                        if label and value and label.lower() != "statistic":
                            table_values[label] = value

    def get_number(raw):
        if raw is None:
            return None
        s = raw.replace("$", "").replace(",", "").replace("%", "").strip()
        try:
            return float(s)
        except ValueError:
            return None

    def pick(*labels):
        for label in labels:
            if label in table_values:
                val = get_number(table_values[label])
                if val is not None:
                    return val
        return None

    stats["return_pct"] = pick("Net Profit", "Return", "Compounding Annual Return")
    stats["drawdown_pct"] = pick("Drawdown")
    stats["total_orders"] = pick("Total Orders")
    stats["sharpe"] = pick("Sharpe Ratio")
    stats["win_rate_pct"] = pick("Win Rate")
    stats["start_equity"] = pick("Start Equity")
    stats["end_equity"] = pick("End Equity")
    stats["entries"] = pick("Entries")
    stats["exit_stop_loss"] = pick("Exit_StopLoss")
    stats["exit_take_profit"] = pick("Exit_TakeProfit")
    stats["exit_end_of_session"] = pick("Exit_EndOfSession")
    stats["exit_max_hold"] = pick("Exit_MaxHoldReached")

    # Fix return_pct from equity if missing
    if stats["return_pct"] is None:
        start = stats.get("start_equity")
        end = stats.get("end_equity")
        if start and end and start > 0:
            stats["return_pct"] = ((end / start) - 1.0) * 100.0

    # If return has % in table, it's already percentage
    if "Net Profit" in table_values and "%" in table_values["Net Profit"]:
        stats["return_pct"] = get_number(table_values["Net Profit"])

    return stats


def extract_url(output):
    """Extract the backtest URL from QC output."""
    for token in output.replace("\n", " ").split():
        if token.startswith("https://www.quantconnect.com/project/"):
            return token.strip()
    return ""


_project_pushed = False


def ensure_project_pushed():
    """Push the project to QC cloud once before the first backtest."""
    global _project_pushed
    if _project_pushed:
        return True

    print("  [Setup] Pushing project to QuantConnect cloud...")
    result = subprocess.run(
        [LEAN_BIN, "cloud", "push", "--project", PROJECT_NAME],
        capture_output=True, text=True,
    )
    output = (result.stdout or "") + "\n" + (result.stderr or "")

    if result.returncode != 0:
        print(f"  [ERROR] Failed to push project:")
        for line in output.strip().splitlines()[-5:]:
            print(f"    {line}")
        return False

    _project_pushed = True
    print("  [Setup] Project pushed successfully.")
    return True


def run_backtest(name, params):
    """Run a single cloud backtest and return (url, stats)."""
    if not ensure_project_pushed():
        return "", {}

    max_attempts = 3

    for attempt in range(1, max_attempts + 1):
        cmd = [LEAN_BIN, "cloud", "backtest", "--name", name, PROJECT_NAME]
        for key, value in params.items():
            cmd += ["--parameter", str(key), str(value)]

        result = subprocess.run(cmd, capture_output=True, text=True)
        output = (result.stdout or "") + "\n" + (result.stderr or "")
        output_lower = output.lower()

        # Retry on transient errors
        transient = any(x in output_lower for x in [
            "no spare nodes", "sslerror", "timed out", "rate limit",
            "too many backtest", "429", "connection aborted",
            "internal server error",
        ])

        if result.returncode != 0 and transient and attempt < max_attempts:
            wait = 30 * attempt
            print(f"    [Retry] Network/rate limit issue. Waiting {wait}s...")
            time.sleep(wait)
            continue

        if result.returncode != 0:
            # Print last few lines of error for debugging
            err_lines = [l for l in output.strip().splitlines() if l.strip()]
            for line in err_lines[-3:]:
                print(f"    {clean_ansi(line)}")
            if any(x in output_lower for x in ["runtime error", "unhandled exception", "traceback"]):
                print(f"    [ERROR] Algorithm crashed. Check QC logs.")
                return "", {}
            print(f"    [ERROR] Backtest failed (exit code {result.returncode})")
            return "", {}

        url = extract_url(output)
        stats = parse_stats(output)
        return url, stats

    return "", {}


# ══════════════════════════════════════════════════
#  FITNESS FUNCTION
# ══════════════════════════════════════════════════

def calculate_fitness(stats):
    """
    Fitness = 3.0 * Sharpe + 1.0 * Return - 1.0 * max(0, Drawdown - 15)

    Hard failures return -1,000,000,000
    """
    ret = stats.get("return_pct")
    dd = stats.get("drawdown_pct")
    orders = stats.get("total_orders")
    sharpe = stats.get("sharpe") or 0.0

    if ret is None or dd is None or orders is None:
        return -1_000_000_000

    if orders < MIN_TRADES_REQUIRED:
        return -1_000_000_000

    if dd >= MAX_DRAWDOWN_FAIL:
        return -1_000_000_000

    drawdown_penalty = max(0.0, dd - 15.0) * 1.5

    fitness = (3.0 * sharpe) + (1.0 * ret) - drawdown_penalty

    # Overtrading penalty: penalize > 250 entries per year
    entries = stats.get("entries") or orders
    if entries and entries > 250:
        overtrading_penalty = (entries - 250) * 0.05
        fitness -= overtrading_penalty

    # Undertrading penalty: day trading should have enough trades
    # Penalize < 100 entries over 2-year test period (< 1 trade/week)
    if entries and entries < 100:
        undertrading_penalty = (100 - entries) * 0.15
        fitness -= undertrading_penalty

    return fitness


# ══════════════════════════════════════════════════
#  DISPLAY HELPERS
# ══════════════════════════════════════════════════

def format_pct(value, width=8):
    if value is None:
        return "N/A".rjust(width)
    return f"{value:+.2f}%".rjust(width)


def format_num(value, width=6):
    if value is None:
        return "N/A".rjust(width)
    return f"{value:.2f}".rjust(width)


def print_header(pair, family, gen, total_gens):
    family_name = get_family_names().get(family, "Unknown")
    print()
    print("=" * 60)
    print(f"  GENERATION {gen} of {total_gens}  |  {pair}  |  {family_name}")
    print("=" * 60)
    print()


def print_candidate(index, total, stats, score):
    ret = format_pct(stats.get("return_pct"))
    dd = format_pct(stats.get("drawdown_pct"))
    sharpe = format_num(stats.get("sharpe"))
    win = format_pct(stats.get("win_rate_pct"))
    trades = stats.get("total_orders") or 0
    entries = stats.get("entries") or 0
    stops = stats.get("exit_stop_loss") or 0
    tps = stats.get("exit_take_profit") or 0

    closed = stops + tps
    print(f"  Candidate {index} of {total}:")
    print(f"    Return: {ret}  |  Drawdown: {dd}  |  Sharpe: {sharpe}")
    print(f"    Win Rate: {win}  |  Entries: {entries}  |  Closed Trades: {closed}")
    print(f"    Stop Loss Exits: {stops}  |  Take Profit Exits: {tps}  |  Total Orders: {trades}")
    print(f"    Fitness Score: {score:.2f}")
    print()


def print_top_winners(rows, title, count=3):
    print()
    print(f"  {'=' * 50}")
    print(f"  {title}")
    print(f"  {'=' * 50}")

    for i, row in enumerate(rows[:count], 1):
        ret = format_pct(row.get("return_pct"))
        sharpe = format_num(row.get("sharpe"))
        dd = format_pct(row.get("drawdown_pct"))
        win = format_pct(row.get("win_rate_pct"))
        trades = row.get("total_orders") or 0
        score = row.get("score", 0)

        print(f"  #{i}  Return: {ret}  |  Sharpe: {sharpe}  |  Drawdown: {dd}")
        print(f"      Win Rate: {win}  |  Trades: {trades}  |  Score: {score:.2f}")
        if row.get("url"):
            print(f"      URL: {row['url']}")
        print()


# ══════════════════════════════════════════════════
#  CHECKPOINT (Save/Resume)
# ══════════════════════════════════════════════════

def get_checkpoint_path(pair, family, style="day_trade"):
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir / f"checkpoint_{pair}_family{family}_{style}.json"


def save_checkpoint(pair, family, gen, population, all_rows, style="day_trade"):
    path = get_checkpoint_path(pair, family, style)
    data = {
        "pair": pair,
        "family": family,
        "style": style,
        "generation": gen,
        "population": population,
        "all_rows": all_rows,
        "timestamp": datetime.now().isoformat(),
    }
    path.write_text(json.dumps(data, indent=2, default=str))


def load_checkpoint(pair, family, style="day_trade"):
    path = get_checkpoint_path(pair, family, style)
    if not path.exists():
        old_path = path.parent / f"checkpoint_{pair}_family{family}.json"
        if old_path.exists():
            old_path.rename(path)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return data
    except Exception:
        return None


# ══════════════════════════════════════════════════
#  MAIN GA LOOP
# ══════════════════════════════════════════════════

_eval_cache_lock = threading.Lock()

def evaluate_genome(pair, family, genome, mode, eval_cache, trading_style="day_trade"):
    """Run a single backtest and return result dict."""
    cache_key = (pair, mode, genome_key(genome))
    with _eval_cache_lock:
        if cache_key in eval_cache:
            print(f"    [Cache Hit] Skipping duplicate genome")
            return eval_cache[cache_key]

    run_name = f"{pair}_{mode}_F{family}_{datetime.now().strftime('%H%M%S%f')[:10]}"
    params = build_params(pair, mode, family, genome, trading_style)

    url, stats = run_backtest(run_name, params)
    score = calculate_fitness(stats)

    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "pair": pair,
        "mode": mode,
        "family": family,
        "family_name": get_family_names().get(family, "Unknown"),
        "score": score,
        "url": url,
        **stats,
        **{f"param_{k}": v for k, v in genome.items()},
    }

    with _eval_cache_lock:
        eval_cache[cache_key] = result
    time.sleep(SLEEP_BETWEEN_RUNS)
    return result


def evaluate_batch_parallel(pair, family, genomes, mode, eval_cache, trading_style="day_trade"):
    """Evaluate a batch of genomes in parallel using ThreadPoolExecutor."""
    results = [None] * len(genomes)

    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
        future_to_idx = {}
        for i, genome in enumerate(genomes):
            future = executor.submit(
                evaluate_genome, pair, family, genome, mode, eval_cache, trading_style
            )
            future_to_idx[future] = i

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"    [ERROR] Genome {idx+1} failed: {e}")
                results[idx] = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "pair": pair, "mode": mode, "family": family,
                    "family_name": get_family_names().get(family, "Unknown"),
                    "score": -1_000_000_000,
                    **{f"param_{k}": v for k, v in genomes[idx].items()},
                }

    # Print results in order
    for i, result in enumerate(results):
        print_candidate(i + 1, len(genomes), result, result["score"])

    return results


def run_ga(pair, family, generations, resume=False, trading_style="day_trade"):
    """Run the full GA optimization for one pair + family."""
    family_name = get_family_names().get(family, "Unknown")
    eval_cache = {}
    all_rows = []
    start_gen = 1

    # Resume from checkpoint
    if resume:
        checkpoint = load_checkpoint(pair, family, trading_style)
        if checkpoint:
            start_gen = checkpoint["generation"] + 1
            all_rows = checkpoint.get("all_rows", [])
            pop = checkpoint["population"]
            print(f"\n  Resuming from Generation {start_gen} ({len(all_rows)} previous results)")
        else:
            print("\n  No checkpoint found. Starting fresh.")
            resume = False

    if not resume:
        # Generate initial population
        pop = []
        seen = set()
        while len(pop) < POP_SIZE:
            g = random_genome(family)
            key = genome_key(g)
            if key in seen:
                continue
            seen.add(key)
            pop.append(g)

    print()
    print("*" * 60)
    print(f"  GENETIC ALGORITHM OPTIMIZATION")
    print(f"  Pair: {pair}  |  Strategy: {family_name}")
    print(f"  Population: {POP_SIZE}  |  Generations: {generations}")
    print(f"  Parameters per genome: {len(get_all_choices(family))}")
    print("*" * 60)

    for gen in range(start_gen, generations + 1):
        print_header(pair, family, gen, generations)
        print(f"  ⚡ Running {len(pop)} backtests in parallel ({PARALLEL_WORKERS} workers)...")

        gen_rows = evaluate_batch_parallel(pair, family, pop, "train", eval_cache, trading_style)
        all_rows.extend(gen_rows)

        # Sort by score
        gen_rows.sort(key=lambda r: r["score"], reverse=True)
        print_top_winners(gen_rows, f"TOP WINNERS (Generation {gen})")

        # Save checkpoint
        save_checkpoint(pair, family, gen, pop, all_rows, trading_style)

        # Select elites
        elites = gen_rows[:min(ELITE_KEEP, len(gen_rows))]
        elite_genomes = []
        for row in elites:
            g = {k.replace("param_", ""): v for k, v in row.items() if k.startswith("param_")}
            elite_genomes.append(g)

        # Build next generation
        next_pop = list(elite_genomes)
        seen = {genome_key(g) for g in next_pop}

        # Crossover children
        safety = 0
        while len(next_pop) < (POP_SIZE - RANDOM_INJECTION) and safety < POP_SIZE * 100:
            safety += 1
            if len(elite_genomes) >= 2:
                p1, p2 = random.sample(elite_genomes, 2)
            else:
                p1 = p2 = elite_genomes[0]
            child = mutate(crossover(p1, p2, family), family)
            key = genome_key(child)
            if key in seen:
                continue
            seen.add(key)
            next_pop.append(child)

        # Random injection for diversity
        while len(next_pop) < POP_SIZE:
            g = random_genome(family)
            key = genome_key(g)
            if key in seen:
                continue
            seen.add(key)
            next_pop.append(g)

        pop = next_pop

    # ── VALIDATION PHASE ──
    train_rows = [r for r in all_rows if r["mode"] == "train"]
    train_rows.sort(key=lambda r: r["score"], reverse=True)

    # Deduplicate
    seen_keys = set()
    unique_train = []
    for r in train_rows:
        key = tuple(v for k, v in sorted(r.items()) if k.startswith("param_"))
        if key not in seen_keys:
            seen_keys.add(key)
            unique_train.append(r)

    shortlist = unique_train[:TRAIN_SHORTLIST]

    print()
    print("=" * 60)
    print(f"  VALIDATION PHASE  |  Testing Top {len(shortlist)} on Unseen Data (2023)")
    print("=" * 60)

    validation_rows = []
    for i, row in enumerate(shortlist, 1):
        genome = {k.replace("param_", ""): v for k, v in row.items() if k.startswith("param_")}
        result = evaluate_genome(pair, family, genome, "validation", eval_cache, trading_style)
        result["train_score"] = row["score"]
        validation_rows.append(result)
        all_rows.append(result)
        print_candidate(i, len(shortlist), result, result["score"])

    validation_rows.sort(key=lambda r: r["score"], reverse=True)
    validated = validation_rows[:VALIDATION_KEEP]
    print_top_winners(validated, "TOP VALIDATION CANDIDATES")

    # ── TEST PHASE (Out-of-Sample) ──
    print()
    print("=" * 60)
    print(f"  OUT-OF-SAMPLE TEST  |  Final Test on 2024-2025 Data")
    print("=" * 60)

    test_rows = []
    for i, row in enumerate(validated, 1):
        genome = {k.replace("param_", ""): v for k, v in row.items() if k.startswith("param_")}
        result = evaluate_genome(pair, family, genome, "test", eval_cache, trading_style)
        result["train_score"] = row.get("train_score")
        result["validation_score"] = row["score"]
        test_rows.append(result)
        all_rows.append(result)
        print_candidate(i, len(validated), result, result["score"])

    test_rows.sort(key=lambda r: r["score"], reverse=True)
    finalists = test_rows[:FINAL_KEEP]

    # ── FINAL SUMMARY ──
    print()
    print("*" * 60)
    print(f"  FINAL RESULTS  |  {pair}  |  {family_name}")
    print("*" * 60)

    for i, row in enumerate(finalists, 1):
        print(f"\n  Champion #{i}:")
        print(f"    Return:     {format_pct(row.get('return_pct'))}")
        print(f"    Drawdown:   {format_pct(row.get('drawdown_pct'))}")
        print(f"    Sharpe:     {format_num(row.get('sharpe'))}")
        print(f"    Win Rate:   {format_pct(row.get('win_rate_pct'))}")
        print(f"    Trades:     {row.get('total_orders') or 'N/A'}")
        print(f"    Entries:    {row.get('entries') or 'N/A'}")
        print(f"    Score:      {row['score']:.2f}")
        if row.get("url"):
            print(f"    Backtest:   {row['url']}")

        print(f"\n    Optimized Parameters:")
        for k, v in sorted(row.items()):
            if k.startswith("param_"):
                param_name = k.replace("param_", "")
                print(f"      {param_name}: {v}")

    # ── SAVE RESULTS ──
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"results_{pair}_family{family}_{trading_style}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    if all_rows:
        all_keys = []
        seen_cols = set()
        for row in all_rows:
            for k in row:
                if k not in seen_cols:
                    all_keys.append(k)
                    seen_cols.add(k)

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(all_rows)

        print(f"\n  Results saved to: {csv_path}")

    return finalists


# ══════════════════════════════════════════════════
#  CLI ENTRY POINT
# ══════════════════════════════════════════════════

def main():
    global PROJECT_NAME, STARTING_CASH, RISK_PER_TRADE_PCT, STYLE_PARAMS, PARALLEL_WORKERS

    parser = argparse.ArgumentParser(
        description="FYP Genetic Algorithm Runner — Optimize trading strategies on QuantConnect"
    )
    parser.add_argument("--engine", type=str, default="forex", choices=["forex", "stock", "crypto"],
                        help="Engine type: forex, stock, or crypto (default: forex)")
    parser.add_argument("--pair", type=str, default=None,
                        help="Asset to optimize (default: EURUSD for forex, SPY for stock, BTCUSD for crypto)")
    parser.add_argument("--family", type=int, default=1, choices=[1, 2, 3, 4, 5],
                        help="Strategy family: 1=RSI_MR, 2=Bollinger_MR, 3=ConnorsRSI, 4=EMA+ADX, 5=Williams+RSI")
    parser.add_argument("--style", type=str, default=None, choices=["day_trade", "swing", "long_term"],
                        help="Trading style (default depends on engine)")
    parser.add_argument("--generations", type=int, default=GENERATIONS,
                        help=f"Number of GA generations (default: {GENERATIONS})")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    parser.add_argument("--workers", type=int, default=None,
                        help=f"Parallel backtest workers (default: {PARALLEL_WORKERS}). Use 3 when running multiple scripts.")

    args = parser.parse_args()

    # Apply engine config
    engine_name = args.engine.lower()
    eng_cfg = ENGINE_CONFIG[engine_name]

    PROJECT_NAME = eng_cfg["project_name"]
    STARTING_CASH = eng_cfg["starting_cash"]
    RISK_PER_TRADE_PCT = eng_cfg["risk_per_trade_pct"]

    trading_style = args.style or eng_cfg["default_style"]
    STYLE_PARAMS = eng_cfg["styles"].get(trading_style, eng_cfg["styles"][eng_cfg["default_style"]])

    # Apply worker count override
    if args.workers is not None:
        PARALLEL_WORKERS = max(1, min(args.workers, 10))

    # Default pair per engine
    default_pairs = {"forex": "EURUSD", "stock": "SPY", "crypto": "BTCUSD"}
    pair = (args.pair or default_pairs[engine_name]).upper()

    print(f"\n  FYP Genetic Algorithm Runner")
    print(f"  {'─' * 40}")
    print(f"  Engine:      {engine_name}")
    print(f"  Pair:        {pair}")
    print(f"  Style:       {trading_style}")
    print(f"  Family:      {args.family} — {get_family_names().get(args.family, 'Unknown')}")
    print(f"  Generations: {args.generations}")
    print(f"  Population:  {POP_SIZE}")
    print(f"  Capital:     ${STARTING_CASH:,}")
    print(f"  Workers:     {PARALLEL_WORKERS}")
    print(f"  Resume:      {'Yes' if args.resume else 'No'}")
    print()

    start_time = time.time()

    finalists = run_ga(
        pair=pair,
        family=args.family,
        generations=args.generations,
        resume=args.resume,
        trading_style=trading_style,
    )

    elapsed = time.time() - start_time
    print(f"\n  Total runtime: {elapsed / 60:.1f} minutes")
    print(f"  Done!\n")


if __name__ == "__main__":
    main()
