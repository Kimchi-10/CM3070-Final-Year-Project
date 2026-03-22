#!/usr/bin/env python3
"""
Extract GA Winners
------------------
After running the GA optimiser across multiple assets and strategy families,
this script scans every results CSV and picks the best performer (by test
return) for each asset + trading-style combo. Output goes to the dashboard's
optimized_params.json so the live signal engine can load the winning params.

Run:
    python extract_winners.py
"""

import csv, json, re
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).resolve().parent
RESULTS_DIR = BASE / "results"
OUT_JSON = BASE.parent / "dashboard" / "strategies" / "optimized_params.json"

# ── family name lookups ──
STOCK_FAMILIES = {
    1: "RSI Mean Reversion",
    2: "Bollinger Band Mean Reversion",
    3: "Connors RSI (2-Period)",
    4: "EMA Trend + ADX",
    5: "Williams %R + RSI",
}
FOREX_FAMILIES = {
    1: "RSI Mean Reversion",
    2: "Bollinger Band Bounce",
    3: "EMA Crossover",
    4: "Stochastic Mean Reversion",
    5: "EMA Trend + ADX",
}
FOREX_PAIRS = {"EURUSD", "USDJPY"}

TICKER_MAP = {
    "EURUSD": "EURUSD=X", "USDJPY": "USDJPY=X",
    "SPY": "SPY", "QQQ": "QQQ", "AAPL": "AAPL", "TSLA": "TSLA",
    "BTCUSDT": "BTC-USD", "SOLUSDT": "SOL-USD",
}


def family_name(pair, fam):
    lookup = FOREX_FAMILIES if pair in FOREX_PAIRS else STOCK_FAMILIES
    return lookup.get(fam, f"Family {fam}")

_RE_V2 = re.compile(r"results_(.+?)_family(\d+)_(.+?)_\d{8}_\d{6}\.csv")
_RE_V1 = re.compile(r"results_(.+?)_family(\d+)_\d{8}_\d{6}\.csv")

def parse_filename(name):
    """Return {'pair', 'family', 'style'} or None if the name doesn't match."""
    m = _RE_V2.match(name)
    if m:
        return {"pair": m[1], "family": int(m[2]), "style": m[3]}
    m = _RE_V1.match(name)
    if m:
        return {"pair": m[1], "family": int(m[2]), "style": "day_trade"}
    return None


def to_float(val, fallback=None):
    """Safely cast a CSV cell to float (handles blanks, 'None', etc.)."""
    if not val or val == "None":
        return fallback
    try:
        return float(val)
    except (ValueError, TypeError):
        return fallback


# ── main logic ──

def collect_candidates():
    """Read every results CSV and return the best test-period row per (pair, style)."""
    if not RESULTS_DIR.exists():
        print(f"  results/ not found at {RESULTS_DIR}")
        return {}

    csvs = sorted(RESULTS_DIR.glob("results_*.csv"))
    if not csvs:
        print("  No results_*.csv files found.")
        return {}
    print(f"  Scanning {len(csvs)} result files ...\n")

    bucket = defaultdict(list) 

    for path in csvs:
        info = parse_filename(path.name)
        if info is None:
            continue

        pair, fam, style = info["pair"], info["family"], info["style"]

        try:
            rows = list(csv.DictReader(path.open(encoding="utf-8")))
        except Exception as exc:
            print(f"  Could not read {path.name}: {exc}")
            continue

        # only care about test-period rows
        for row in rows:
            if row.get("mode") != "test":
                continue
            ret = to_float(row.get("return_pct"))
            if ret is None:
                continue

            # pull out GA-tuned params
            params = {}
            for col, val in row.items():
                if col.startswith("param_"):
                    v = to_float(val)
                    if v is not None:
                        params[col[6:]] = int(v) if v == int(v) else v

            bucket[(pair, style)].append({
                "pair": pair, "family": fam,
                "family_name": family_name(pair, fam),
                "style": style, "return_pct": ret,
                "sharpe": to_float(row.get("sharpe"), 0),
                "drawdown_pct": to_float(row.get("drawdown_pct"), 0),
                "win_rate_pct": to_float(row.get("win_rate_pct"), 0),
                "entries": to_float(row.get("entries"), 0),
                "total_orders": to_float(row.get("total_orders"), 0),
                "params": params,
            })

    # pick highest test return for each (pair, style)
    winners = {}
    for (pair, style), cands in bucket.items():
        best = max(cands, key=lambda c: c["return_pct"])
        tk = TICKER_MAP.get(pair, pair)
        winners.setdefault(tk, {})[style] = {
            "family": best["family"],
            "family_name": best["family_name"],
            "params": best["params"],
            "metrics": {
                "return_pct":   round(best["return_pct"], 2),
                "sharpe":       round(best["sharpe"], 3)       or None,
                "drawdown_pct": round(best["drawdown_pct"], 2) or None,
                "win_rate_pct": round(best["win_rate_pct"], 1) or None,
                "entries":      int(best["entries"])            or None,
            },
        }
        sign = "+" if best["return_pct"] >= 0 else ""
        print(f"  {pair:10s} {style:12s}  ->  F{best['family']} "
              f"({best['family_name']})  {sign}{best['return_pct']:.2f}%")

    return winners


def main():
    print("\n  Extract GA Winners\n" + "  " + "-" * 40)
    winners = collect_candidates()
    if not winners:
        print("\n  Nothing found — run the GA backtests first.")
        return

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(winners, indent=2))
    print(f"\n  Wrote {len(winners)} assets -> {OUT_JSON}\n")


if __name__ == "__main__":
    main()
