"""
strategy.py
-----------
Mean-reversion signal generation for a single cointegrated pair, trading the
OU-implied z-score of the spread with:
  - entry / exit / stop-loss z-thresholds
  - a maximum holding period (multiple of the estimated half-life), since a
    pair that fails to revert within a reasonable multiple of its half-life
    is treated as a broken (structurally changed) relationship, not a
    trading opportunity
  - one-day execution lag: a signal computed off day-t's close is acted on
    at day t+1's open (no look-ahead / same-bar execution assumption)
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class StrategyParams:
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 3.5
    max_holding_days: int = None  # set dynamically = 4 * half_life if None


def generate_positions(z: pd.Series, params: StrategyParams, half_life: float) -> pd.Series:
    """
    Converts a z-score series into a spread position series in {-1, 0, +1}:
      +1 : long the spread (long asset1 / short hedge_ratio*asset2) -- taken when z < -entry_z
      -1 : short the spread                                          -- taken when z > +entry_z
       0 : flat
    Position is decided using information available at close of day t
    (position is then IMPLEMENTED at the open of day t+1 by the backtester --
    see backtest.py). This function only encodes the *signal*/decision logic.
    """
    max_hold = params.max_holding_days or int(round(4 * half_life))
    pos = np.zeros(len(z))
    state = 0          # current spread position
    days_in_trade = 0
    zvals = z.values

    for i in range(len(zvals)):
        zt = zvals[i]
        if np.isnan(zt):
            pos[i] = state
            continue

        if state == 0:
            if zt > params.entry_z:
                state = -1
                days_in_trade = 0
            elif zt < -params.entry_z:
                state = 1
                days_in_trade = 0
        else:
            days_in_trade += 1
            # exit conditions: reversion to band, stop-loss (blow-through), or max holding period
            if state == 1 and (zt >= -params.exit_z or zt < -params.stop_z or days_in_trade >= max_hold):
                state = 0
            elif state == -1 and (zt <= params.exit_z or zt > params.stop_z or days_in_trade >= max_hold):
                state = 0

        pos[i] = state

    return pd.Series(pos, index=z.index, dtype=float)


if __name__ == "__main__":
    from data_loader import load_universe_prices
    from cointegration import johansen_test_pair
    from ou_process import fit_ou, zscore

    prices = load_universe_prices()
    log_p = np.log(prices)
    jres = johansen_test_pair(log_p["UPS"], log_p["FDX"])
    spread = log_p["UPS"] - jres["hedge_ratio"] * log_p["FDX"]
    params = fit_ou(spread)
    z = zscore(spread, params)
    pos = generate_positions(z, StrategyParams(), params.half_life)
    n_trades = (pos.diff().abs() > 0).sum()
    print(f"half-life={params.half_life:.1f}d, trades={n_trades}, time-in-market={ (pos!=0).mean():.1%}")
