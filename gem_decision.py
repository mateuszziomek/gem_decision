#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
GEM Decision Tool — Global Equity Momentum for Polish IKE/IKZE investor.

Strategy: modified GEM (Gary Antonacci) with extended asset universe.
Base currency: PLN.  Rebalancing: monthly check, trade only on signal change.

Usage:
    python gem_decision.py              # current month
    python gem_decision.py --chart      # include normalized chart
    python gem_decision.py --date 2025-06-30  # custom end date
"""

import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

LOOKBACK_MONTHS = 12
SIGNAL_LAG_MONTHS = 1  # use data ending N months before current month

# Assets compared in momentum ranking
# bossa_ticker: what to search/buy on Bossa platform
ASSETS = [
    {"name": "S&P 500",     "ticker": "SPXS.L",  "source": "yahoo", "currency": "USD",
     "bossa_ticker": "SPXS", "bossa_exchange": "AMS (Euronext)", "bossa_currency": "EUR"},
    {"name": "NASDAQ 100",  "ticker": "CNDX.L",  "source": "yahoo", "currency": "USD",
     "bossa_ticker": "CNDX", "bossa_exchange": "AMS (Euronext)", "bossa_currency": "EUR"},
    {"name": "MSCI EM IMI", "ticker": "EIMI.L",  "source": "yahoo", "currency": "USD",
     "bossa_ticker": "EMIM", "bossa_exchange": "AMS (Euronext)", "bossa_currency": "EUR"},
    {"name": "Gold",        "ticker": "IGLN.L",  "source": "yahoo", "currency": "USD",
     "bossa_ticker": "EGLN", "bossa_exchange": "LSE (London)",   "bossa_currency": "EUR"},
    {"name": "mWIG40 TR",   "ticker": "mwig40tr","source": "stooq", "currency": "PLN",
     "bossa_ticker": "ETFBM40TR", "bossa_exchange": "GPW (Warszawa)", "bossa_currency": "PLN"},
]

# Risk-off asset (used when absolute momentum is negative)
RISK_OFF = {
    "name": "US Treasury 0-1yr", "ticker": "IB01.L", "source": "yahoo", "currency": "USD",
    "bossa_ticker": "IB01", "bossa_exchange": "LSE (London)", "bossa_currency": "USD",
}

# FX rates needed (target: PLN)
FX_TICKERS = {
    "USD": "USDPLN=X",   # Yahoo Finance
}

# Stooq CSV base URL
STOOQ_CSV_URL = "https://stooq.pl/q/d/l/?s={symbol}&i=d"

# Decision log path
DECISION_LOG = Path(__file__).parent / "decision_log.csv"

# Data coverage: max gap (days) between requested and actual date
DATA_GAP_WARN_START = 15   # warn if start-date data is >15 days off
DATA_GAP_WARN_END = 7      # warn if end-date data is >7 days off
DATA_GAP_FATAL = 60        # exit if any data point is >60 days off

# FX sanity bounds — exit if any rate falls outside these ranges
FX_SANITY_BOUNDS = {
    "USDPLN=X": (2.0, 7.0),
}


# ═══════════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════════

def validate_fx_rates(fx_data: dict[str, pd.Series]) -> None:
    """Check all FX rate values fall within sane bounds. Exit on violation."""
    for currency, series in fx_data.items():
        fx_ticker = FX_TICKERS.get(currency)
        if fx_ticker is None or fx_ticker not in FX_SANITY_BOUNDS:
            continue
        lo, hi = FX_SANITY_BOUNDS[fx_ticker]
        clean = series.dropna()
        if clean.empty:
            print(f"❌ FX sanity check: {fx_ticker} has no data", file=sys.stderr)
            sys.exit(1)
        out_of_bounds = clean[(clean < lo) | (clean > hi)]
        if not out_of_bounds.empty:
            worst = out_of_bounds.iloc[0]
            date = out_of_bounds.index[0]
            print(f"❌ FX sanity check: {fx_ticker} = {worst:.4f} on {date.date()} "
                  f"outside bounds [{lo}, {hi}]", file=sys.stderr)
            sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════
# DATE HELPERS
# ═══════════════════════════════════════════════════════════════════════

def compute_rolling_dates(reference_date: datetime | None = None) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Compute rolling 12M period with SIGNAL_LAG_MONTHS offset.

    Example with SIGNAL_LAG_MONTHS=1:
      Today = Feb 13, 2026
      End   = Dec 31, 2025   (go back 1+1=2 months from current month)
      Start = Dec 31, 2024   (12 months before end)

    Returns (start_date, end_date) as pd.Timestamp.
    """
    if reference_date is None:
        reference_date = datetime.now()

    ref = pd.Timestamp(reference_date)

    # Go to 1st of current month, then subtract (1 + SIGNAL_LAG_MONTHS) months
    # to get to the 1st of the target month, then -1 day = last day of month before
    first_of_current = ref.replace(day=1)
    end_date = (first_of_current - pd.DateOffset(months=SIGNAL_LAG_MONTHS) - pd.Timedelta(days=1)).normalize()

    # Start = 12 months before end
    start_date = (end_date - pd.DateOffset(months=LOOKBACK_MONTHS)).normalize()

    return start_date, end_date


# ═══════════════════════════════════════════════════════════════════════
# DATA FETCHING
# ═══════════════════════════════════════════════════════════════════════

def fetch_yahoo_close(tickers: list[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Download adjusted close prices from Yahoo Finance."""
    margin = pd.Timedelta(days=10)
    data = yf.download(
        tickers,
        start=(start - margin).strftime("%Y-%m-%d"),
        end=(end + margin).strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )
    if data.empty:
        print("❌ Yahoo Finance returned no data.", file=sys.stderr)
        sys.exit(1)

    # yf.download returns MultiIndex columns when multiple tickers
    if isinstance(data.columns, pd.MultiIndex):
        closes = data["Close"]
    else:
        # Single ticker case
        closes = data[["Close"]].rename(columns={"Close": tickers[0]})

    return closes.dropna(axis=1, how="all")


def fetch_stooq_series(symbol: str) -> pd.Series:
    """Download daily close from Stooq CSV."""
    url = STOOQ_CSV_URL.format(symbol=symbol)
    try:
        df = pd.read_csv(url)
    except Exception as e:
        print(f"❌ Stooq download failed for {symbol}: {e}", file=sys.stderr)
        sys.exit(1)

    if df.empty:
        print(f"❌ Stooq returned empty data for {symbol}.", file=sys.stderr)
        sys.exit(1)

    # Detect date column
    date_col = next((c for c in df.columns if c.lower() in ("data", "date")), df.columns[0])
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()

    # Detect close column
    close_col = _pick_close_col(df)
    series = pd.to_numeric(df[close_col], errors="coerce")
    series.name = symbol.upper()
    return series


def _pick_close_col(df: pd.DataFrame) -> str:
    """Pick the most likely close/price column from Stooq CSV."""
    lower_map = {c.lower(): c for c in df.columns}
    for candidate in ("close", "zamkniecie", "zamknięcie"):
        if candidate in lower_map:
            return lower_map[candidate]
    # Fallback: first numeric non-volume column
    for c in df.columns:
        if c.lower() not in ("volume", "wolumen") and pd.api.types.is_numeric_dtype(df[c]):
            return c
    return df.columns[0]


# ═══════════════════════════════════════════════════════════════════════
# PRICE HELPERS
# ═══════════════════════════════════════════════════════════════════════

def last_price_on_or_before(series: pd.Series, target: pd.Timestamp) -> tuple[float, pd.Timestamp | None]:
    """Get the last available value on or before target date.

    Returns (price, actual_date) tuple. actual_date is None when no data found.
    """
    subset = series.loc[:target].dropna()
    if subset.empty:
        return float("nan"), None
    return float(subset.iloc[-1]), subset.index[-1]


def _check_data_gap(label: str, requested: pd.Timestamp, actual: pd.Timestamp | None, warn_days: int) -> None:
    """Check gap between requested and actual data date; warn or exit."""
    if actual is None:
        print(f"❌ {label}: no data found on or before {requested.date()}", file=sys.stderr)
        sys.exit(1)
    gap = abs((requested - actual).days)
    if gap > DATA_GAP_FATAL:
        print(f"❌ {label}: data gap {gap} days (requested {requested.date()}, "
              f"got {actual.date()}) exceeds {DATA_GAP_FATAL}-day limit", file=sys.stderr)
        sys.exit(1)
    if gap > warn_days:
        print(f"⚠️  {label}: data gap {gap} days (requested {requested.date()}, "
              f"got {actual.date()})", file=sys.stderr)


def compute_return_pln(
    price_series: pd.Series,
    fx_series: pd.Series | None,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict:
    """
    Compute 12M return in PLN for a single asset.

    If fx_series is None, asset is already in PLN.
    Returns dict with prices and return percentage.
    """
    if fx_series is not None:
        # Convert price to PLN
        fx_aligned = fx_series.reindex(price_series.index, method="ffill")
        pln_series = price_series * fx_aligned
    else:
        pln_series = price_series

    pln_series = pln_series.dropna()

    start_price, start_actual = last_price_on_or_before(pln_series, start)
    end_price, end_actual = last_price_on_or_before(pln_series, end)

    series_name = price_series.name or "unknown"
    _check_data_gap(f"{series_name} start", start, start_actual, DATA_GAP_WARN_START)
    _check_data_gap(f"{series_name} end", end, end_actual, DATA_GAP_WARN_END)

    if pd.isna(start_price) or pd.isna(end_price) or start_price == 0:
        return {"start_pln": start_price, "end_pln": end_price, "return_pct": float("nan")}

    ret = (end_price / start_price - 1) * 100.0
    return {"start_pln": start_price, "end_pln": end_price, "return_pct": ret}


# ═══════════════════════════════════════════════════════════════════════
# DECISION ENGINE
# ═══════════════════════════════════════════════════════════════════════

def make_decision(ranking: list[dict]) -> dict:
    """
    Apply GEM decision rules.

    Returns dict with 'action' ('BUY' or 'RISK_OFF'), 'asset', 'return_pct'.
    """
    # Filter out assets with NaN returns
    valid = [r for r in ranking if not pd.isna(r["return_pct"])]
    if not valid:
        return {"action": "NO_DATA", "asset": None, "return_pct": None}

    best = valid[0]  # ranking is sorted descending

    if best["return_pct"] > 0:
        return {"action": "BUY", "asset": best["name"], "return_pct": best["return_pct"]}
    else:
        return {
            "action": "RISK_OFF",
            "asset": RISK_OFF["name"],
            "return_pct": best["return_pct"],
        }


def log_decision(decision: dict, start_date: pd.Timestamp, end_date: pd.Timestamp, ranking: list[dict]):
    """Append decision to CSV log."""
    file_exists = DECISION_LOG.exists()

    with open(DECISION_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "run_date", "period_start", "period_end",
                "action", "asset", "best_return_pct",
                "full_ranking",
            ])
        ranking_str = " | ".join(
            f"{r['name']}: {r['return_pct']:+.2f}%" if not pd.isna(r["return_pct"]) else f"{r['name']}: N/A"
            for r in ranking
        )
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M"),
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
            decision["action"],
            decision["asset"],
            f"{decision['return_pct']:.2f}" if decision["return_pct"] is not None else "N/A",
            ranking_str,
        ])


# ═══════════════════════════════════════════════════════════════════════
# CHART
# ═══════════════════════════════════════════════════════════════════════

def plot_momentum_chart(
    all_series_pln: dict[str, pd.Series],
    start: pd.Timestamp,
    end: pd.Timestamp,
):
    """Plot normalized performance chart in PLN."""
    import matplotlib.pyplot as plt

    # Reindex all series to a common daily date range to handle
    # different trading calendars (Yahoo vs Stooq)
    daily_index = pd.date_range(start=start, end=end, freq="D")
    aligned = {}
    for name, series in all_series_pln.items():
        aligned[name] = series.reindex(daily_index, method="ffill")
    combined = pd.DataFrame(aligned).dropna(how="all")

    if combined.empty or combined.iloc[0].isna().all():
        print("\n⚠️  Brak wspólnych dat do narysowania wykresu.", file=sys.stderr)
        return

    # Normalize to 100 at start
    norm = combined / combined.iloc[0] * 100.0

    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(13, 7))
    for col in norm.columns:
        ax.plot(norm.index, norm[col], label=col, linewidth=2)

    ax.set_title(
        f"GEM Momentum — Performance w PLN (znormalizowane, {start.date()} = 100)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylabel("Wartość znormalizowana (PLN)")
    ax.set_xlabel("Data")
    ax.legend(title="Instrumenty", loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="GEM Decision Tool — momentum ranking in PLN")
    parser.add_argument("--chart", action="store_true", help="Show normalized performance chart")
    parser.add_argument("--date", type=str, default=None,
                        help="Reference date (YYYY-MM-DD). Default: today.")
    parser.add_argument("--no-log", action="store_true", help="Skip writing to decision_log.csv")
    args = parser.parse_args()

    ref_date = datetime.strptime(args.date, "%Y-%m-%d") if args.date else None
    start_date, end_date = compute_rolling_dates(ref_date)

    print(f"📅 Okres analizy: {start_date.date()} → {end_date.date()} ({LOOKBACK_MONTHS}M rolling)")
    print()

    # ── Collect Yahoo tickers ───────────────────────────────────────────
    yahoo_tickers = [a["ticker"] for a in ASSETS if a["source"] == "yahoo"]
    yahoo_tickers.append(RISK_OFF["ticker"])  # IB01 for info

    # Add FX tickers
    fx_needed = set()
    for a in ASSETS + [RISK_OFF]:
        if a["currency"] != "PLN" and a["currency"] in FX_TICKERS:
            fx_needed.add(FX_TICKERS[a["currency"]])

    all_yahoo = yahoo_tickers + list(fx_needed)

    print(f"📡 Pobieranie danych z Yahoo Finance: {', '.join(all_yahoo)}...")
    yahoo_closes = fetch_yahoo_close(all_yahoo, start_date, end_date)

    # ── Fetch Stooq data ────────────────────────────────────────────────
    stooq_series = {}
    stooq_assets = [a for a in ASSETS if a["source"] == "stooq"]
    for a in stooq_assets:
        print(f"📡 Pobieranie danych z Stooq: {a['ticker']}...")
        stooq_series[a["ticker"]] = fetch_stooq_series(a["ticker"])

    # ── Extract FX series ───────────────────────────────────────────────
    fx_data = {}
    for currency, fx_ticker in FX_TICKERS.items():
        if fx_ticker in yahoo_closes.columns:
            fx_data[currency] = yahoo_closes[fx_ticker].dropna()
        else:
            print(f"⚠️  FX data for {currency} ({fx_ticker}) not found.", file=sys.stderr)

    validate_fx_rates(fx_data)

    # ── Compute returns for each asset ──────────────────────────────────
    print()
    ranking = []
    all_pln_series = {}  # for chart

    for asset in ASSETS:
        ticker = asset["ticker"]
        currency = asset["currency"]

        # Get price series
        if asset["source"] == "yahoo":
            if ticker not in yahoo_closes.columns:
                print(f"⚠️  No data for {ticker}, skipping.", file=sys.stderr)
                ranking.append({"name": asset["name"], "ticker": ticker, "return_pct": float("nan"),
                                "start_pln": float("nan"), "end_pln": float("nan")})
                continue
            price_series = yahoo_closes[ticker].dropna()
        else:
            price_series = stooq_series.get(ticker)
            if price_series is None or price_series.empty:
                ranking.append({"name": asset["name"], "ticker": ticker, "return_pct": float("nan"),
                                "start_pln": float("nan"), "end_pln": float("nan")})
                continue

        # Get FX series (None if PLN)
        fx_series = fx_data.get(currency) if currency != "PLN" else None

        # Compute return
        result = compute_return_pln(price_series, fx_series, start_date, end_date)
        ranking.append({
            "name": asset["name"],
            "ticker": ticker,
            "return_pct": result["return_pct"],
            "start_pln": result["start_pln"],
            "end_pln": result["end_pln"],
        })

        # Build PLN series for chart
        if fx_series is not None:
            fx_aligned = fx_series.reindex(price_series.index, method="ffill")
            all_pln_series[asset["name"]] = price_series * fx_aligned
        else:
            all_pln_series[asset["name"]] = price_series

    # Sort by return descending
    ranking.sort(key=lambda r: r["return_pct"] if not pd.isna(r["return_pct"]) else -999, reverse=True)

    # ── Also compute IB01 return for info ───────────────────────────────
    ib01_return = None
    if RISK_OFF["ticker"] in yahoo_closes.columns:
        ib01_price = yahoo_closes[RISK_OFF["ticker"]].dropna()
        fx_series = fx_data.get(RISK_OFF["currency"])
        ib01_result = compute_return_pln(ib01_price, fx_series, start_date, end_date)
        ib01_return = ib01_result["return_pct"]

    # ── Decision ────────────────────────────────────────────────────────
    decision = make_decision(ranking)

    # ── FX context ──────────────────────────────────────────────────────
    usdpln_start, usdpln_end, usdpln_change = None, None, None
    if "USD" in fx_data:
        usdpln_start, _ = last_price_on_or_before(fx_data["USD"], start_date)
        usdpln_end, _ = last_price_on_or_before(fx_data["USD"], end_date)
        if usdpln_start and usdpln_end:
            usdpln_change = (usdpln_end / usdpln_start - 1) * 100.0

    # ══════════════════════════════════════════════════════════════════
    # OUTPUT
    # ══════════════════════════════════════════════════════════════════

    print("═" * 60)
    print(f"  GEM DECISION — {datetime.now().strftime('%Y-%m-%d')}")
    print(f"  Okres: {start_date.date()} → {end_date.date()} ({LOOKBACK_MONTHS}M rolling)")
    print("═" * 60)

    print("\n📊 Ranking Momentum (w PLN):")
    for i, r in enumerate(ranking, 1):
        ret_str = f"{r['return_pct']:+.2f}%" if not pd.isna(r["return_pct"]) else "N/A"
        print(f"  #{i}  {r['name']:<18} {ret_str:>10}")

    if ib01_return is not None:
        print(f"\n🛡️  Risk-off ({RISK_OFF['name']}): {ib01_return:+.2f}%")

    print()
    if decision["action"] == "BUY":
        # Find the asset definition for Bossa info
        buy_asset = next((a for a in ASSETS if a["name"] == decision["asset"]), None)
        print(f"🎯 DECYZJA: KUP {decision['asset']} — momentum {decision['return_pct']:+.2f}%")
        print(f"   (absolute momentum: POZYTYWNY ✅)")
        if buy_asset:
            print(f"\n🏦 Jak kupić na Bossa:")
            print(f"   Ticker:  {buy_asset['bossa_ticker']}")
            print(f"   Giełda:  {buy_asset['bossa_exchange']}")
            print(f"   Waluta:  {buy_asset['bossa_currency']}")
    elif decision["action"] == "RISK_OFF":
        print(f"🛑 DECYZJA: RISK-OFF → KUP {decision['asset']}")
        print(f"   Najlepsze aktywo: {decision['return_pct']:+.2f}% (ujemne momentum ❌)")
        print(f"\n🏦 Jak kupić na Bossa:")
        print(f"   Ticker:  {RISK_OFF['bossa_ticker']}")
        print(f"   Giełda:  {RISK_OFF['bossa_exchange']}")
        print(f"   Waluta:  {RISK_OFF['bossa_currency']}")
    else:
        print("❌ DECYZJA: BRAK DANYCH — nie można podjąć decyzji")

    if usdpln_start and usdpln_end:
        print(f"\n💱 Kontekst walutowy:")
        print(f"   USD/PLN: {usdpln_start:.4f} → {usdpln_end:.4f} ({usdpln_change:+.1f}%)")

    # ── Log ──────────────────────────────────────────────────────────
    if not args.no_log:
        log_decision(decision, start_date, end_date, ranking)
        print(f"\n📁 Zapisano do {DECISION_LOG}")

    # ── Chart ────────────────────────────────────────────────────────
    if args.chart:
        print("\n📈 Generowanie wykresu...")
        plot_momentum_chart(all_pln_series, start_date, end_date)


if __name__ == "__main__":
    main()
