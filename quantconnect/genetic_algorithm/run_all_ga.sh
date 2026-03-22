#!/bin/bash
# ============================================================
#  FYP Master GA Runner — All Engines, All Assets, All Families
# ============================================================
#
#  Usage:
#    ./run_all_ga.sh              # Run everything (forex + stock day trade + stock swing)
#    ./run_all_ga.sh forex        # Re-run forex day trade (with HTF + vol filter)
#    ./run_all_ga.sh stock        # Run stock day trade only
#    ./run_all_ga.sh stock_swing  # Run stock swing (best families only)
#
#  NOTE: Crypto removed — mean reversion has no edge on crypto markets.
#        See crypto results CSVs in results/ for report evidence.
#
#  Options (env vars):
#    NO_PUSH=1 ./run_all_ga.sh crypto   # Skip pushing to QC Cloud (already pushed)
#    WORKERS=3 ./run_all_ga.sh crypto   # Override parallel worker count
#
#  This script pushes each engine to QC Cloud once, then runs
#  all families sequentially for each asset.
# ============================================================

set -e

# Always work from the genetic_algorithm directory where this script lives
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
QC_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

FILTER="${1:-all}"
PYTHON="python3"
WORKERS_ARG=""

if [ -n "$WORKERS" ]; then
    WORKERS_ARG="--workers $WORKERS"
fi

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║     FYP GENETIC ALGORITHM — MASTER RUNNER           ║"
echo "║     Filter: $FILTER                                      ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

run_ga() {
    local engine=$1
    local pair=$2
    local family=$3
    local style=$4

    echo ""
    echo "────────────────────────────────────────────────────"
    echo "  ENGINE: $engine | PAIR: $pair | FAMILY: $family | STYLE: $style"
    echo "────────────────────────────────────────────────────"
    echo ""

    $PYTHON "$SCRIPT_DIR/ga_runner.py" --engine "$engine" --pair "$pair" --family "$family" --style "$style" $WORKERS_ARG
}

push_engine() {
    if [ "${NO_PUSH:-0}" = "1" ]; then
        echo ">>> Skipping push for ${1}_engine (NO_PUSH=1)"
        return
    fi

    local engine=$1
    echo ""
    echo ">>> Pushing ${engine}_engine to QuantConnect Cloud..."
    cd "$QC_ROOT"
    lean cloud push --project "${engine}_engine"
    echo ">>> ${engine}_engine pushed successfully."
    echo ""
}

# ============================================================
#  FOREX — Day Trade (5-min bars + 1hr HTF trend filter)
# ============================================================
if [ "$FILTER" = "all" ] || [ "$FILTER" = "forex" ]; then
    push_engine "forex"

    echo "╔══════════════════════════════════════════════════╗"
    echo "║  FOREX — Day Trade (5min + HTF trend filter)    ║"
    echo "╚══════════════════════════════════════════════════╝"

    for PAIR in EURUSD USDJPY; do
        for FAM in 1 2 3 4 5; do
            run_ga forex "$PAIR" "$FAM" day_trade
        done
    done
fi

# ============================================================
#  STOCKS — Day Trade (5-min bars, session filter 9:30-16:00)
# ============================================================
if [ "$FILTER" = "all" ] || [ "$FILTER" = "stock" ]; then
    push_engine "stock"

    echo "╔══════════════════════════════════════════════════╗"
    echo "║  STOCKS — Day Trade                             ║"
    echo "╚══════════════════════════════════════════════════╝"

    for PAIR in SPY QQQ AAPL TSLA; do
        for FAM in 1 2 3 4 5; do
            run_ga stock "$PAIR" "$FAM" day_trade
        done
    done
fi

# ============================================================
#  STOCKS — Swing (1hr bars, no session filter)
#  Only run winning families (1, 3) to save cloud credits
# ============================================================
if [ "$FILTER" = "stock_swing" ]; then
    push_engine "stock"

    echo "╔══════════════════════════════════════════════════╗"
    echo "║  STOCKS — Swing Trade (Families 1, 3)           ║"
    echo "╚══════════════════════════════════════════════════╝"

    for PAIR in SPY QQQ AAPL TSLA; do
        for FAM in 1 3; do
            run_ga stock "$PAIR" "$FAM" swing
        done
    done
fi

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  ALL GA RUNS COMPLETE!                              ║"
echo "║  Results saved in: results/                         ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
