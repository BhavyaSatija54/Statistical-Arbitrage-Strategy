"""metrics.py -- standard performance statistics on a daily returns series."""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def annualized_return(returns: pd.Series) -> float:
    equity = (1 + returns).cumprod()
    n = len(returns)
    if n == 0 or equity.iloc[-1] <= 0:
        return np.nan
    total_return = equity.iloc[-1] - 1
    years = n / TRADING_DAYS
    return (1 + total_return) ** (1 / years) - 1 if years > 0 else np.nan


def annualized_vol(returns: pd.Series) -> float:
    return returns.std() * np.sqrt(TRADING_DAYS)


def sharpe_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    excess = returns - rf / TRADING_DAYS
    if excess.std() == 0:
        return np.nan
    return (excess.mean() / excess.std()) * np.sqrt(TRADING_DAYS)


def sortino_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    excess = returns - rf / TRADING_DAYS
    downside = excess[excess < 0]
    dd_std = downside.std()
    if dd_std == 0 or np.isnan(dd_std):
        return np.nan
    return (excess.mean() / dd_std) * np.sqrt(TRADING_DAYS)


def max_drawdown(returns: pd.Series) -> float:
    equity = (1 + returns).cumprod()
    running_max = equity.cummax()
    dd = equity / running_max - 1
    return dd.min()  # negative number


def calmar_ratio(returns: pd.Series) -> float:
    mdd = max_drawdown(returns)
    if mdd == 0:
        return np.nan
    return annualized_return(returns) / abs(mdd)


def summary(returns: pd.Series) -> dict:
    return {
        "Annualized Return": annualized_return(returns),
        "Annualized Vol": annualized_vol(returns),
        "Sharpe": sharpe_ratio(returns),
        "Sortino": sortino_ratio(returns),
        "Max Drawdown": max_drawdown(returns),
        "Calmar": calmar_ratio(returns),
        "Win Rate (days)": (returns > 0).mean(),
        "N days": len(returns),
    }
