"""
capacity_frontier.py
---------------------
Sweeps total strategy AUM from $1M to $1B, setting capital_per_position =
AUM / top_n (i.e. equal-weighting the target number of concurrent positions
in real dollars), and re-runs the full 20-year walk-forward backtest at each
level. The engine's ADV participation cap (3% of the less-liquid leg's
trailing 20-day dollar volume) increasingly constrains the size of each
day's fill as AUM grows -- realized position size falls short of the
signal's target size, and BOTH the P&L and the cost booked reflect the
capacity-constrained realized fill, not the desired one (see the
`_leg_window_returns` fill/realized-position logic in backtest.py).

At a given AUM level A, run_walkforward returns a dollar P&L series (since
capital_per_position is in real dollars); dividing that series by
capital_per_position (=A/top_n) recovers the ACHIEVED fractional return at
that AUM, capacity drag included -- this is what gets Sharpe'd and plotted.
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from backtest import run_walkforward, BacktestConfig, CostModel
from strategy import StrategyParams
import metrics as M

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def run_one_aum(close, open_, volume, pairs, aum, top_n=4):
    capital_per_position = aum / top_n
    cfg = BacktestConfig(top_n=top_n, capital_per_position=capital_per_position,
                          strategy_params=StrategyParams(entry_z=2.5, exit_z=0.75, stop_z=4.0),
                          cost_model=CostModel(1.0, 2.5, 1.5, 0.03), hedge_mode="static")
    dollar_ret, wl, trades = run_walkforward(close, open_, volume, pairs, cfg)
    frac_ret = dollar_ret / capital_per_position
    s = M.summary(frac_ret)
    fills = pd.DataFrame(trades)
    avg_fill_frac = fills["fill_frac"].mean() if len(fills) else np.nan
    return {"aum": aum, **s, "avg_fill_frac": avg_fill_frac, "n_trades": len(fills)}


if __name__ == "__main__":
    close = pd.read_pickle("cache_close.pkl")
    open_ = pd.read_pickle("cache_open.pkl")
    volume = pd.read_pickle("cache_volume.pkl")
    import pickle
    with open("cache_pairs.pkl", "rb") as f:
        pairs = pickle.load(f)

    aum_levels = [float(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [1e6, 5e6, 25e6]
    rows = []
    for aum in aum_levels:
        t0 = time.time()
        row = run_one_aum(close, open_, volume, pairs, aum)
        rows.append(row)
        print(f"AUM=${aum:,.0f}  Sharpe={row['Sharpe']:.3f}  AnnRet={row['Annualized Return']:.4f}  "
              f"MaxDD={row['Max Drawdown']:.3f}  avg_fill_frac={row['avg_fill_frac']:.3f}  ({time.time()-t0:.0f}s)")

    out_path = os.path.join(RESULTS_DIR, f"capacity_partial_{int(aum_levels[0])}.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print("saved", out_path)
