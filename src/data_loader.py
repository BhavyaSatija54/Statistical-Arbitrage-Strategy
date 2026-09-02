"""
data_loader.py (v2 -- 20-year, 128-ticker universe)
----------------------------------------------------
Loads real daily OHLCV data spanning **20 years** (1997-11-10 to 2017-11-10)
for a 128-ticker large + mid-cap universe, sourced by consolidating
per-ticker files from the "Huge Stock Market Dataset" (full historical daily
price/volume data for US-listed stocks, dividend/split-adjusted), covering
tech, financials, energy, healthcare, staples, discretionary, industrials,
materials and utilities.

This replaces the earlier 5-year / 63-ticker prototype: 4x the history,
2x the cross-sectional breadth, and mid-caps included (not just S&P-100-style
megacaps), which materially changes the opportunity set available to the
cointegration screen.
"""

import os
import numpy as np
import pandas as pd

RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "all_stocks_20yr_raw.csv")

WINDOW_YEARS = 20

# GICS-style sector/sub-industry buckets for the 128 surviving tickers
# (>=98% trading-day coverage over the full 20-year window).
SECTOR_UNIVERSE = {
    "Tech - Semiconductors": ["ADI", "AMAT", "AMD", "INTC", "KLAC", "LRCX", "MU", "QCOM", "TXN"],
    "Tech - Software": ["ADBE", "ADSK", "MSFT", "ORCL"],
    "Tech - Hardware": ["AAPL", "CSCO", "GLW", "HPQ", "IBM", "XRX"],
    "Banks": ["BAC", "BK", "C", "JPM", "PNC", "STT", "USB", "WFC"],
    "Insurance": ["AFL", "AIG", "ALL", "CB", "LNC", "MMC", "UNM"],
    "Capital Markets & Cards": ["AXP", "COF"],
    "Energy": ["APA", "APC", "COP", "CVX", "HAL", "MRO", "OXY", "SLB", "XOM"],
    "Pharma & Med Device": ["ABT", "BAX", "BDX", "BMY", "JNJ", "LLY", "MRK", "PFE", "SYK", "MDT", "PKI"],
    "Healthcare Services": ["ABC", "CAH", "CVS", "HUM", "MCK", "UNH"],
    "Staples - Food & Beverage": ["CAG", "CPB", "GIS", "HSY", "K", "KO", "PEP", "MO"],
    "Staples - Household": ["CL", "CLX", "ECL", "KMB", "PG"],
    "Retail": ["COST", "GPS", "HD", "JWN", "LOW", "TGT", "TJX", "WMT"],
    "Restaurants": ["MCD", "SBUX", "YUM"],
    "Media & Telecom": ["CMCSA", "DIS", "T", "VZ"],
    "Industrial Conglomerates": ["GE", "HON", "MMM"],
    "Aerospace & Defense": ["BA", "LMT", "NOC", "RTN"],
    "Machinery": ["CAT", "DE", "DOV", "EMR", "ETN", "ITW", "PCAR", "SWK"],
    "Railroads": ["CSX", "NSC", "UNP"],
    "Airlines": ["DAL", "LUV"],
    "Materials": ["APD", "AVY", "FCX", "MAS", "NEM", "NUE", "PPG", "VMC"],
    "Utilities": ["AEP", "D", "DUK", "ED", "ETR", "EXC", "FE", "PEG", "SO"],
    "Apparel & Footwear": ["NKE"],
}


def _load_field(field: str, min_coverage: float = 0.98) -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH, parse_dates=["date"])
    end_date = df["date"].max()
    start_date = end_date - pd.DateOffset(years=WINDOW_YEARS)
    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]

    panel = df.pivot(index="date", columns="Name", values=field).sort_index()
    n_days = panel.index.nunique()
    good_cols = panel.columns[panel.count() >= min_coverage * n_days]
    panel = panel[good_cols]
    panel = panel.ffill(limit=3)
    panel = panel.dropna(axis=0, how="any")
    return panel


def load_universe_prices() -> pd.DataFrame:
    """(date x ticker) daily close panel, 20-year window, ~128 tickers."""
    return _load_field("close")


def load_open_panel() -> pd.DataFrame:
    return _load_field("open")


def load_volume_panel() -> pd.DataFrame:
    return _load_field("volume")


def sector_pairs(tickers_available):
    pairs = []
    avail = set(tickers_available)
    for sector, members in SECTOR_UNIVERSE.items():
        members = [m for m in members if m in avail]
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                pairs.append((members[i], members[j], sector))
    return pairs


def sector_triplets(tickers_available):
    """All within-sector 3-asset combinations, for basket cointegration."""
    from itertools import combinations
    triplets = []
    avail = set(tickers_available)
    for sector, members in SECTOR_UNIVERSE.items():
        members = [m for m in members if m in avail]
        if len(members) < 3:
            continue
        for combo in combinations(members, 3):
            triplets.append((*combo, sector))
    return triplets


if __name__ == "__main__":
    close = load_universe_prices()
    open_ = load_open_panel()
    vol = load_volume_panel()
    print(f"Close panel: {close.shape[0]} trading days x {close.shape[1]} tickers")
    print(f"Date range: {close.index.min().date()} to {close.index.max().date()}  "
          f"(~{(close.index.max()-close.index.min()).days/365.25:.1f} years)")
    pairs = sector_pairs(close.columns)
    triplets = sector_triplets(close.columns)
    print(f"Candidate within-sector pairs: {len(pairs)}")
    print(f"Candidate within-sector triplets (basket candidates): {len(triplets)}")
