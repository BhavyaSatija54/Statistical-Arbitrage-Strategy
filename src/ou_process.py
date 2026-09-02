"""
ou_process.py
-------------
Models a cointegrated pair's spread as an Ornstein-Uhlenbeck process:

    dS_t = theta * (mu - S_t) dt + sigma dW_t

Discretized (dt = 1 trading day) this is an AR(1):

    S_t = S_{t-1} + theta*(mu - S_{t-1}) + eps_t
        = a + b*S_{t-1} + eps_t,     a = theta*mu,  b = 1 - theta

We estimate (theta, mu, sigma) by OLS on the discretized process (equivalent
to conditional MLE under Gaussian innovations), then derive:
  - half-life of mean reversion   = ln(2) / theta
  - stationary (equilibrium) std  = sigma / sqrt(2*theta)   [continuous-time OU stationary variance]
  - z-score of the spread relative to the fitted stationary distribution,
    used directly as the trading signal.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class OUParams:
    theta: float      # speed of mean reversion
    mu: float          # long-run mean
    sigma: float        # instantaneous volatility of innovations
    half_life: float
    stationary_std: float
    resid_std: float    # residual std of the discrete AR(1) fit (used for z-scoring in practice)


def fit_ou(spread: pd.Series, dt: float = 1.0) -> OUParams:
    """OLS calibration of an OU process from a spread time series."""
    s = spread.dropna().values
    s_lag = s[:-1]
    s_now = s[1:]

    X = np.vstack([np.ones_like(s_lag), s_lag]).T
    coef, *_ = np.linalg.lstsq(X, s_now, rcond=None)
    a, b = coef

    resid = s_now - (a + b * s_lag)
    resid_std = resid.std(ddof=2)

    theta = (1 - b) / dt
    theta = max(theta, 1e-6)  # guard against non-mean-reverting fits
    mu = a / (theta * dt) if theta > 1e-6 else s_lag.mean()

    # sigma of the OU diffusion, back out from discrete residual variance:
    # Var(eps) = sigma^2 * (1 - exp(-2*theta*dt)) / (2*theta)
    denom = (1 - np.exp(-2 * theta * dt))
    sigma = resid_std * np.sqrt(2 * theta / denom) if denom > 1e-8 else resid_std

    half_life = np.log(2) / theta
    stationary_std = sigma / np.sqrt(2 * theta)

    return OUParams(theta=theta, mu=mu, sigma=sigma, half_life=half_life,
                     stationary_std=stationary_std, resid_std=resid_std)


def zscore(spread: pd.Series, params: OUParams) -> pd.Series:
    """
    Z-score of the spread relative to the OU-implied stationary distribution.
    Using the *fitted stationary std* (rather than a rolling sample std) ties
    the trading signal directly to the OU calibration and avoids look-ahead
    bias from a rolling window computed on the same data used for entry.
    """
    return (spread - params.mu) / params.stationary_std


if __name__ == "__main__":
    from data_loader import load_universe_prices
    from cointegration import johansen_test_pair
    prices = load_universe_prices()
    log_p = np.log(prices)
    jres = johansen_test_pair(log_p["UPS"], log_p["FDX"])
    spread = log_p["UPS"] - jres["hedge_ratio"] * log_p["FDX"]
    params = fit_ou(spread)
    print("UPS-FDX OU fit:", params)
    z = zscore(spread, params)
    print("z-score stats:\n", z.describe())
