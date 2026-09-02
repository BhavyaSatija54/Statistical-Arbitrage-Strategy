"""
regime.py
---------
Market-regime filter based on a rolling Hurst exponent of a broad
equal-weighted index built from the trading universe itself (a real,
in-sample proxy for "the market", since we don't have a separate SPX
series in this dataset).

Why: the v1 backtest's worst rolling windows were concentrated in
low-dispersion, trending periods -- exactly when a mean-reversion
strategy should expect to struggle, since the whole premise (spreads
revert) is weakest when the broad market itself is trending rather than
mean-reverting. The Hurst exponent quantifies this directly:
    H ~ 0.5  : a random walk (no persistence either way)
    H > 0.5  : trending / persistent (autocorrelated in the same direction)
    H < 0.5  : mean-reverting / anti-persistent

We estimate H via the variance-ratio method: Var(k-day log return) should
scale as k^(2H) for a self-similar process, so regressing log(Var(k-day
return)) on log(k) across several lags k and halving the slope gives H.
"""

import numpy as np
import pandas as pd


def hurst_exponent(log_price: pd.Series, lags=(2, 4, 8, 16, 32, 64)) -> float:
    lags = [l for l in lags if l < len(log_price) // 2]
    if len(lags) < 3:
        return np.nan
    variances = []
    for lag in lags:
        diffs = log_price.diff(lag).dropna()
        variances.append(diffs.var())
    log_lags = np.log(lags)
    log_var = np.log(variances)
    slope, _ = np.polyfit(log_lags, log_var, 1)
    return slope / 2.0


def rolling_market_hurst(close_panel: pd.DataFrame, window: int = 126) -> pd.Series:
    """
    Builds an equal-weighted log-price index across the full universe (a
    broad market proxy) and computes a trailing rolling-window Hurst
    exponent of that index at each date.
    """
    log_index = np.log(close_panel).mean(axis=1)   # equal-weighted log-price "index"
    hurst_vals = pd.Series(index=close_panel.index, dtype=float)
    for i in range(window, len(log_index)):
        seg = log_index.iloc[i - window:i]
        hurst_vals.iloc[i] = hurst_exponent(seg)
    return hurst_vals


def regime_scalar(hurst_value: float, trend_threshold: float = 0.45,
                   reverting_threshold: float = 0.30,
                   trend_scale: float = 0.5, reverting_scale: float = 1.25) -> float:
    """
    Maps a Hurst-exponent reading (measured as of the *start* of a trading
    window, using only formation-window/trailing data) to a position-size
    multiplier for that window:
      H > trend_threshold        -> trending regime, scale exposure DOWN
      H < reverting_threshold    -> mean-reverting regime, scale exposure UP
      otherwise                  -> neutral, no adjustment
    """
    if np.isnan(hurst_value):
        return 1.0
    if hurst_value > trend_threshold:
        return trend_scale
    if hurst_value < reverting_threshold:
        return reverting_scale
    return 1.0


if __name__ == "__main__":
    from data_loader import load_universe_prices
    close = load_universe_prices()
    h = rolling_market_hurst(close, window=126)
    print(h.describe())
    print("\nSample regime scalars over time:")
    sample = h.dropna().iloc[::250]
    for d, v in sample.items():
        print(f"  {d.date()}  Hurst={v:.3f}  scalar={regime_scalar(v):.2f}")
