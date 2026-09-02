"""
kalman_hedge.py
----------------
Dynamic hedge ratio estimation via a Kalman filter, as an alternative to the
formation-window-fixed Johansen beta used in the baseline strategy.

Motivation: the Johansen hedge ratio is estimated once per formation window
and held fixed through the whole out-of-sample trading window. If the true
relationship between the two assets drifts (e.g. relative business mix
shifts, a re-rating, index-weight-driven flow effects), a stale hedge ratio
leaves the "spread" contaminated with a slow-moving trend component that
isn't really mean-reverting -- exactly the failure mode that showed up in
the v1 backtest's trending windows.

Model (standard Ernie-Chan-style pairs-trading state space):
    Observation:  y_t = alpha_t + beta_t * x_t + e_t,      e_t ~ N(0, R)
    State:        [alpha_t, beta_t]' = [alpha_{t-1}, beta_{t-1}]' + w_t,   w_t ~ N(0, Q)

i.e. the intercept and hedge ratio are themselves treated as a random walk,
re-estimated every single day via the standard Kalman recursion, using only
information up to and including day t (no look-ahead: the filter is run
causally, exactly as it would be live).

We use log-prices for x_t, y_t (so alpha_t + beta_t*x_t is the OU spread's
time-varying analogue).
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class KalmanHedgeResult:
    alpha: pd.Series      # filtered intercept, indexed like the input
    beta: pd.Series       # filtered (time-varying) hedge ratio
    beta_std: pd.Series   # filtered std of the beta estimate (sqrt of P[1,1])
    spread: pd.Series     # y_t - alpha_t - beta_t * x_t (filtered residual)
    spread_std: pd.Series  # rolling filtered residual std (Kalman's own uncertainty proxy)


def kalman_hedge_ratio(log_y: pd.Series, log_x: pd.Series,
                        delta: float = 1e-4, ve: float = 1e-3,
                        init_beta: float = 1.0, init_alpha: float = 0.0) -> KalmanHedgeResult:
    """
    Runs a 2-state (alpha, beta) Kalman filter over the pair (log_y, log_x).

    delta : controls Q = delta/(1-delta) * I -- how fast the hedge ratio is
            allowed to drift day to day. Smaller delta = smoother/slower-
            adapting beta (closer to the static Johansen case); larger delta
            = faster adaptation, more responsive but noisier.
    ve    : observation noise variance R (residual/spread noise level).
    """
    idx = log_y.index.intersection(log_x.index)
    y = log_y.loc[idx].values
    x = log_x.loc[idx].values
    n = len(idx)

    Q = delta / (1 - delta) * np.eye(2)   # state transition (process) noise covariance
    R = ve                                  # observation noise variance

    state = np.array([init_alpha, init_beta])   # [alpha, beta]
    P = np.eye(2) * 1.0                          # state covariance

    alphas = np.zeros(n)
    betas = np.zeros(n)
    beta_stds = np.zeros(n)
    spreads = np.zeros(n)
    spread_stds = np.zeros(n)

    for t in range(n):
        # --- predict ---
        state_pred = state  # random walk: no deterministic drift
        P_pred = P + Q

        # --- observe ---
        H = np.array([1.0, x[t]])              # y_t = H . [alpha, beta]
        y_pred = H @ state_pred
        S = H @ P_pred @ H.T + R               # innovation variance
        K = (P_pred @ H) / S                    # Kalman gain

        innovation = y[t] - y_pred
        state = state_pred + K * innovation
        P = P_pred - np.outer(K, H) @ P_pred

        alphas[t] = state[0]
        betas[t] = state[1]
        beta_stds[t] = np.sqrt(max(P[1, 1], 0.0))
        spreads[t] = innovation                 # filtered one-step-ahead residual == spread signal
        spread_stds[t] = np.sqrt(S)

    return KalmanHedgeResult(
        alpha=pd.Series(alphas, index=idx),
        beta=pd.Series(betas, index=idx),
        beta_std=pd.Series(beta_stds, index=idx),
        spread=pd.Series(spreads, index=idx),
        spread_std=pd.Series(spread_stds, index=idx),
    )


if __name__ == "__main__":
    from data_loader import load_universe_prices
    prices = load_universe_prices()
    log_p = np.log(prices)
    res = kalman_hedge_ratio(log_p["JPM"], log_p["BAC"])
    print("Kalman beta (JPM vs BAC) over time -- head/tail:")
    print(res.beta.head())
    print(res.beta.tail())
    print(f"beta drift: min={res.beta.min():.3f} max={res.beta.max():.3f} "
          f"(static Johansen would freeze this at one value)")
