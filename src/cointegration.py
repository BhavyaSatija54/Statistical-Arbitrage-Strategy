"""
cointegration.py
-----------------
Pairwise Johansen cointegration screening for the statistical arbitrage
strategy.

Why Johansen rather than plain Engle-Granger:
  - Johansen's trace test does not require an arbitrary choice of which
    series is the "dependent" one (Engle-Granger regression is asymmetric).
  - It directly returns the cointegrating vector (eigenvector of the largest
    eigenvalue), which we use as the hedge ratio beta for spread construction.
  - It generalizes cleanly if the strategy is later extended to >2-asset
    baskets (still only used pairwise here, per the resume's stated pair
    strategy, but the machinery supports it).

We run Johansen on log-prices (standard for equities, since it makes the
cointegrating relationship equivalent to a stationary log-price-ratio /
spread, robust to price-level heteroskedasticity).
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import adfuller


# Johansen trace-statistic critical values at 95% for r<=0 (no cointegration)
# in a 2-variable system, constant term, depend on statsmodels' det_order.
JOHANSEN_TRACE_CV_95 = 15.4948  # 2-variable system, det_order=0, k_ar_diff=1


def johansen_test_pair(log_p1: pd.Series, log_p2: pd.Series, det_order: int = 0, k_ar_diff: int = 1):
    """
    Runs the Johansen trace test on a pair of log-price series.

    Returns a dict with:
      trace_stat0   : trace statistic for H0: rank <= 0 (i.e. no cointegration)
      crit_95       : 95% critical value for that statistic
      cointegrated  : bool, trace_stat0 > crit_95
      beta          : cointegrating vector normalized so beta[0] = 1
                       (i.e. spread = log_p1 - hedge_ratio * log_p2)
      hedge_ratio   : the beta on asset 2
    """
    data = pd.concat([log_p1, log_p2], axis=1).dropna()
    result = coint_johansen(data.values, det_order, k_ar_diff)

    trace_stat0 = result.lr1[0]
    crit_95 = result.cvt[0, 1]  # column 1 = 95% critical value

    evec = result.evec[:, 0]  # eigenvector for largest eigenvalue -> rank-1 cointegrating vector
    evec = evec / evec[0]     # normalize so coefficient on asset 1 is 1
    hedge_ratio = evec[1] * -1  # spread = p1 - hedge_ratio*p2  (evec gives p1 - (-evec[1])*p2 = 0 relation)
    # Derivation: cointegrating relation is evec[0]*p1 + evec[1]*p2 ~ I(0).
    # With evec[0]=1: p1 + evec[1]*p2 ~ I(0)  =>  p1 - (-evec[1])*p2 ~ I(0)
    # so hedge_ratio := -evec[1]

    return {
        "trace_stat0": trace_stat0,
        "crit_95": crit_95,
        "cointegrated": trace_stat0 > crit_95,
        "hedge_ratio": hedge_ratio,
    }


def half_life_and_adf(spread: pd.Series):
    """
    ADF stationarity check on the spread (secondary confirmation of Johansen)
    plus an OLS-based half-life of mean reversion, used both as a filter
    (reject pairs that "cointegrate" statistically but revert too slowly to
    be tradeable net of costs) and as an OU calibration input.
    """
    spread = spread.dropna()
    adf_stat, adf_p, *_ = adfuller(spread, maxlag=1, autolag=None)

    lag = spread.shift(1).iloc[1:]
    delta = spread.diff().iloc[1:]
    lag, delta = lag.align(delta, join="inner")
    X = np.vstack([np.ones(len(lag)), lag.values]).T
    beta, *_ = np.linalg.lstsq(X, delta.values, rcond=None)
    theta = -beta[1]  # mean-reversion speed in discrete OLS sense (delta_s = -theta*(s-mu)+eps)
    half_life = np.log(2) / theta if theta > 0 else np.inf

    return {"adf_stat": adf_stat, "adf_pvalue": adf_p, "half_life_days": half_life}


def screen_pairs(price_panel: pd.DataFrame, pair_list, window=None,
                  min_half_life=8, max_half_life=45, max_adf_p=0.03):
    """
    Runs Johansen + ADF + half-life screen over a list of (t1, t2, sector)
    candidate pairs on a given price panel (optionally sliced to `window`,
    a (start, end) tuple of timestamps -- used for walk-forward / rolling
    formation windows so that pair selection uses only in-sample data).
    """
    if window is not None:
        price_panel = price_panel.loc[window[0]:window[1]]

    log_p = np.log(price_panel)
    rows = []
    for t1, t2, sector in pair_list:
        if t1 not in log_p.columns or t2 not in log_p.columns:
            continue
        try:
            jres = johansen_test_pair(log_p[t1], log_p[t2])
        except Exception:
            continue
        if not jres["cointegrated"]:
            continue
        if not (0.05 < abs(jres["hedge_ratio"]) < 5):
            continue  # reject near-singular / numerically extreme Johansen fits

        spread = log_p[t1] - jres["hedge_ratio"] * log_p[t2]
        hl = half_life_and_adf(spread)
        if hl["adf_pvalue"] > max_adf_p:
            continue
        if not (min_half_life <= hl["half_life_days"] <= max_half_life):
            continue

        rows.append({
            "asset1": t1, "asset2": t2, "sector": sector,
            "trace_stat": jres["trace_stat0"], "crit_95": jres["crit_95"],
            "hedge_ratio": jres["hedge_ratio"],
            "adf_stat": hl["adf_stat"], "adf_pvalue": hl["adf_pvalue"],
            "half_life_days": hl["half_life_days"],
            "spread_std": spread.std(),
        })

    result = pd.DataFrame(rows)
    if len(result):
        result = result.sort_values("trace_stat", ascending=False).reset_index(drop=True)
    return result


def johansen_test_basket(log_prices: pd.DataFrame, det_order: int = 0, k_ar_diff: int = 1):
    """
    Johansen trace test generalized to an n-asset system (n=3 for the basket
    extension used here). Tests H0: rank <= 0 (no cointegration at all among
    the n assets) using the trace statistic, and returns the eigenvector
    associated with the largest eigenvalue as the basket weight vector,
    normalized so the first asset's weight is 1 (spread = p0 - sum(w_i*p_i)).
    """
    data = log_prices.dropna()
    n = data.shape[1]
    result = coint_johansen(data.values, det_order, k_ar_diff)

    trace_stat0 = result.lr1[0]
    crit_95 = result.cvt[0, 1]

    evec = result.evec[:, 0]
    evec = evec / evec[0]
    weights = -evec[1:]  # spread = p0 - sum(weights_i * p_i), consistent with johansen_test_pair's convention

    return {
        "trace_stat0": trace_stat0,
        "crit_95": crit_95,
        "cointegrated": trace_stat0 > crit_95,
        "weights": weights,  # weights on assets[1:], asset[0] weight is implicitly 1
    }


def screen_baskets(price_panel: pd.DataFrame, triplet_list, window=None,
                    min_half_life=8, max_half_life=45, max_adf_p=0.03):
    """
    Same screening pipeline as screen_pairs, generalized to 3-asset baskets.
    Johansen critical value for a 3-variable system (det_order=0, k_ar_diff=1)
    trace test H0: rank<=0 at 95% is looked up from statsmodels' own table
    (result.cvt), so it's already handled correctly for n=3 inside
    johansen_test_basket -- no separate constant needed here.
    """
    if window is not None:
        price_panel = price_panel.loc[window[0]:window[1]]

    log_p = np.log(price_panel)
    rows = []
    for t0, t1, t2, sector in triplet_list:
        if not all(t in log_p.columns for t in (t0, t1, t2)):
            continue
        try:
            jres = johansen_test_basket(log_p[[t0, t1, t2]])
        except Exception:
            continue
        if not jres["cointegrated"]:
            continue

        w1, w2 = jres["weights"]
        if not (0.05 < abs(w1) < 5 and 0.05 < abs(w2) < 5):
            continue  # reject near-singular / numerically extreme Johansen fits

        spread = log_p[t0] - w1 * log_p[t1] - w2 * log_p[t2]
        hl = half_life_and_adf(spread)
        if hl["adf_pvalue"] > max_adf_p:
            continue
        if not (min_half_life <= hl["half_life_days"] <= max_half_life):
            continue

        rows.append({
            "asset0": t0, "asset1": t1, "asset2": t2, "sector": sector,
            "trace_stat": jres["trace_stat0"], "crit_95": jres["crit_95"],
            "w1": w1, "w2": w2,
            "adf_stat": hl["adf_stat"], "adf_pvalue": hl["adf_pvalue"],
            "half_life_days": hl["half_life_days"],
            "spread_std": spread.std(),
        })

    result = pd.DataFrame(rows)
    if len(result):
        result = result.sort_values("trace_stat", ascending=False).reset_index(drop=True)
    return result


if __name__ == "__main__":
    from data_loader import load_universe_prices, sector_pairs
    prices = load_universe_prices()
    pairs = sector_pairs(prices.columns)
    screened = screen_pairs(prices, pairs)
    print(f"{len(screened)} / {len(pairs)} pairs pass Johansen + ADF + half-life screen (full sample)")
    print(screened.head(15).to_string(index=False))
