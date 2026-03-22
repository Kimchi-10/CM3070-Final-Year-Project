#!/bin/bash
# ============================================================
#  FYP Master PARALLEL GA Runner — Forex + Stock Swing
# ============================================================
#
#  Runs 2 GA batches in parallel (1 worker each = 2 QC nodes):
#    Batch 1: Forex Day Trade  (EURUSD + USDJPY, F1-F5)
#    Batch 2: Stock Swing      (SPY + QQQ + AAPL + TSLA, F1,F3)
#
#  NOTE: Crypto removed — mean reversion has no edge on crypto.
#        See crypto results CSVs in results/ for report evidence.
#
#  Usage:
#    ./run_all_parallel.sh          # Run both (1 worker each)
#    ./run_all_parallel.sh 2        # Override workers (risky!)
#
#  Monitor:
#    tail -f logs/*.log
#
#  Time estimate: ~10-12 hours
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
QC_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"

# Worker count per script (default 1, safe for 2 parallel scripts on 2 QC nodes)
WORKERS="${1:-1}"

mkdir -p "$LOG_DIR"

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║     FYP GENETIC ALGORITHM — PARALLEL RUNNER              ║"
echo "║     Workers per batch: $WORKERS  |  Max concurrent: $((WORKERS * 2))    ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

echo ">>> Step 1: Pushing engines to QuantConnect Cloud..."
cd "$QC_ROOT"

echo "  Pushing forex_engine..."
lean cloud push --project forex_engine 2>&1 | tail -1

echo "  Pushing stock_engine..."
lean cloud push --project stock_engine 2>&1 | tail -1

echo ">>> Engines pushed successfully."
echo ""

export NO_PUSH=1
export WORKERS="$WORKERS"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Forex Day Trade + Stock Swing (2 parallel batches)      ║"
echo "║  (2 batches × $WORKERS workers = $((WORKERS * 2)) concurrent backtests)       ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

echo "  [1/2] Starting: Forex Day Trade (EURUSD + USDJPY, F1-F5)..."
"$SCRIPT_DIR/run_all_ga.sh" forex > "$LOG_DIR/forex_day.log" 2>&1 &
PID_FOREX=$!
echo "        PID: $PID_FOREX | Log: logs/forex_day.log"

echo "  [2/2] Starting: Stock Swing (SPY + QQQ + AAPL + TSLA, F1,F3)..."
"$SCRIPT_DIR/run_all_ga.sh" stock_swing > "$LOG_DIR/stock_swing.log" 2>&1 &
PID_STOCK=$!
echo "        PID: $PID_STOCK | Log: logs/stock_swing.log"

echo ""
echo "  Monitor:  tail -f logs/forex_day.log logs/stock_swing.log"
echo ""
echo ">>> Waiting for both batches to complete..."
echo ""

TOTAL_FAILED=0
wait $PID_FOREX && echo "  [1/2] Forex Day Trade    — DONE" || { echo "  [1/2] Forex Day Trade    — FAILED (check logs/forex_day.log)"; TOTAL_FAILED=$((TOTAL_FAILED+1)); }
wait $PID_STOCK && echo "  [2/2] Stock Swing        — DONE" || { echo "  [2/2] Stock Swing        — FAILED (check logs/stock_swing.log)"; TOTAL_FAILED=$((TOTAL_FAILED+1)); }

echo ""

# ── FINAL SUMMARY ──
if [ "$TOTAL_FAILED" -eq 0 ]; then
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║  ALL BATCHES COMPLETE!                                   ║"
    echo "║  Results saved in: results/                              ║"
    echo "║  Logs saved in: logs/                                    ║"
    echo "║                                                          ║"
    echo "║  Next step: python extract_winners.py                    ║"
    echo "╚══════════════════════════════════════════════════════════╝"
else
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║  $TOTAL_FAILED of 2 batch(es) FAILED — check logs/ for details   ║"
    echo "╚══════════════════════════════════════════════════════════╝"
fi
echo ""
