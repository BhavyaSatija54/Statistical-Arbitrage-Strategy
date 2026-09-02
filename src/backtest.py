"""
backtest.py (v2 -- generalized engine)
----------------------------------------
Extends the v1 walk-forward engine along three axes, each independently
selectable via BacktestConfig.hedge_mode / basket flags / regime_filter:

  1. hedge_mode="static"  : formation-window-fixed Johansen hedge ratio
                             (identical mechanics to v1).
  2. hedge_mode="kalman"  : the hedge ratio is re-estimated every day via a
                             causal Kalman filter (src/kalman_hedge.py) run
                             continuously over the *entire* sample (not
                             reset each formation window -- a live trading
                             system wouldn't reset its filter either). The
                             ENTRY-day beta is locked in for the life of
                             each individual trade (realistic: you don't
                             continuously re-hedge for free), but the
                             z-score driving entry/exit decisions uses the
                             filter's live, continuously-updating estimate.
  3. use_baskets=True     : trades 3-asset Johansen-cointegrated baskets
                             (src/cointegration.screen_baskets) instead of
                             2-asset pairs, using the same OU/z-score/
                             execution/cost machinery generalized to n legs.

Regime filter (src/regime.py): a rolling Hurst exponent of a broad
equal-weighted universe index is computed once (causally, trailing window
only) and used to scale each trading window's position size up in
mean-reverting regimes and down in trending regimes.

Cost model, execution lag, ADV participation cap, and the cost-aware
minimum-edge pre-trade filter are unchanged from v1.
"""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from cointegration import screen_pairs, screen_baskets
from ou_process import fit_ou, zscore
from strategy import StrategyParams, generate_positions
from kalman_hedge import kalman_hedge_ratio
from regime import rolling_market_hurst, regime_scalar
from metrics import sharpe_ratio


@dataclass
class CostModel:
    transaction_cost_bps: float = 1.0
    bid_ask_bps: float = 2.5
    slippage_bps: float = 1.5
    adv_participation_cap: float = 0.03


@dataclass
class BacktestConfig:
    formation_days: int = 252
    trading_days: int = 63
    top_n: int = 3
    capital_per_position: float = 1.0 / 3
    strategy_params: StrategyParams = field(default_factory=StrategyParams)
    cost_model: CostModel = field(default_factory=CostModel)
    one_per_sector: bool = True
    min_edge_cost_multiple: float = 3.0
    hedge_mode: str = "static"          # "static" or "kalman"  (kalman only applies to 2-leg pairs)
    use_baskets: bool = False           # trade 3-asset baskets instead of pairs
    use_regime_filter: bool = False
    kalman_delta: float = 1e-4
    kalman_ve: float = 1e-3
    regime_window: int = 126


def _round_trip_cost_bps(cost_model: CostModel) -> float:
    return 4 * (cost_model.transaction_cost_bps + cost_model.bid_ask_bps + cost_model.slippage_bps) / 10000.0


def _select(screened: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    if screened.empty:
        return screened
    if cfg.one_per_sector:
        screened = screened.sort_values("trace_stat", ascending=False).drop_duplicates(subset="sector", keep="first")
    return screened.sort_values("trace_stat", ascending=False).head(cfg.top_n)


def _leg_window_returns(legs, weights_fn, close_panel, open_panel, volume_panel,
                         z_series, half_life, trade_start, trade_end, cfg: BacktestConfig,
                         label: str, size_scalar: float = 1.0):
    """
    Generic n-leg (2 or 3 asset) spread simulator for one trading window.
    legs        : list of tickers, legs[0] is the "long 1 unit" reference asset
    weights_fn  : callable(date) -> array of weights for legs[1:] to use for
                  that day's leg-return calculation (constant for
                  static/basket hedge, entry-day-locked value for kalman)
    z_series    : precomputed z-score series covering at least [trade_start, trade_end]
    size_scalar : regime-filter position-size multiplier for this window
    """
    idx = close_panel.loc[trade_start:trade_end].index
    if len(idx) < 3:
        return pd.Series(dtype=float), [], []

    z = z_series.reindex(idx)
    pos_signal = generate_positions(z, cfg.strategy_params, half_life)
    pos_target = pos_signal.shift(1).fillna(0.0)   # signal-implied target, decided at prior close

    opens = {t: open_panel.loc[idx, t] for t in legs}
    closes = {t: close_panel.loc[idx, t] for t in legs}
    dvols = {t: (volume_panel.loc[idx, t] * close_panel.loc[idx, t]).rolling(20, min_periods=5).mean() for t in legs}

    cost_model = cfg.cost_model
    daily_ret = pd.Series(0.0, index=idx)
    trade_log = []
    round_trips = []
    prev_realized = 0.0     # ACTUAL filled position (may be < target in magnitude if capacity-constrained)
    prev_target = 0.0
    entry_weights = None
    entry_date = None
    entry_z_val = None

    for i in range(1, len(idx)):
        d, d_prev = idx[i], idx[i - 1]
        target_today = pos_target.iloc[i]

        # lock in the hedge weights used for this trade at the moment of (re)entry
        if prev_realized == 0.0 and target_today != 0.0:
            entry_weights = weights_fn(d)
            entry_date = d
            entry_z_val = z.loc[d] if d in z.index else np.nan
        pending_exit = (target_today == 0.0 and prev_target != 0.0 and entry_date is not None)
        if target_today == 0.0:
            entry_weights = None
        w = entry_weights if entry_weights is not None else weights_fn(d)

        realized_today = prev_realized
        if target_today != prev_target:
            # signal wants a position change -- execute it, capacity-constrained.
            # Single-shot fill: whatever fraction of the desired move can't be
            # filled today (ADV participation cap) is NOT carried forward for a
            # later catch-up attempt -- the trade is simply held undersized (or
            # unwound short) for the rest of its life. This is the conservative,
            # simple version of a participation-limited execution algo.
            desired_delta = target_today - prev_realized
            adv_dollar = min(dvols[t].loc[d] if not np.isnan(dvols[t].loc[d]) else np.inf for t in legs)
            notional_wanted = abs(desired_delta) * cfg.capital_per_position * size_scalar
            cap = cost_model.adv_participation_cap * adv_dollar if np.isfinite(adv_dollar) else notional_wanted
            fill_frac = min(1.0, cap / notional_wanted) if notional_wanted > 0 else 1.0
            realized_delta = fill_frac * desired_delta
            realized_today = prev_realized + realized_delta

            size_frac_of_adv = notional_wanted / adv_dollar if np.isfinite(adv_dollar) and adv_dollar > 0 else 0.0
            impact_multiplier = 1.0 + np.sqrt(max(size_frac_of_adv, 0.0))
            eff_bps = (cost_model.transaction_cost_bps + cost_model.bid_ask_bps
                       + cost_model.slippage_bps * impact_multiplier) / 10000.0
            n_legs = len(legs)
            cost = abs(realized_delta) * eff_bps * n_legs * cfg.capital_per_position * size_scalar
            daily_ret.loc[d] -= cost
            trade_log.append({"date": d, "instrument": label, "delta_pos": realized_delta,
                               "target_delta": desired_delta, "cost_bps_eff": eff_bps * 10000,
                               "fill_frac": fill_frac, "n_legs": n_legs})
            prev_target = target_today

        # intraday P&L (REALIZED position held today, open[d] -> close[d])
        if realized_today != 0:
            r0 = (closes[legs[0]].loc[d] / opens[legs[0]].loc[d]) - 1
            spr = r0
            for k, t in enumerate(legs[1:]):
                rk = (closes[t].loc[d] / opens[t].loc[d]) - 1
                spr -= w[k] * rk
            daily_ret.loc[d] += realized_today * spr * cfg.capital_per_position * size_scalar

        # overnight P&L (REALIZED position held coming into today, close[d_prev] -> open[d])
        if prev_realized != 0:
            r0 = (opens[legs[0]].loc[d] / closes[legs[0]].loc[d_prev]) - 1
            spr = r0
            for k, t in enumerate(legs[1:]):
                rk = (opens[t].loc[d] / closes[t].loc[d_prev]) - 1
                spr -= w[k] * rk
            daily_ret.loc[d] += prev_realized * spr * cfg.capital_per_position * size_scalar

        if pending_exit:
            holding_days = idx.get_loc(d) - idx.get_loc(entry_date)
            pnl = daily_ret.loc[entry_date:d].sum()
            round_trips.append({"instrument": label, "entry_date": entry_date, "exit_date": d,
                                 "holding_days": holding_days, "entry_z": entry_z_val, "pnl": pnl})
            entry_date = None

        prev_realized = realized_today

    return daily_ret, trade_log, round_trips


def run_walkforward(close_panel, open_panel, volume_panel, universe, cfg: BacktestConfig):
    dates = close_panel.index
    n = len(dates)

    hurst = rolling_market_hurst(close_panel, window=cfg.regime_window) if cfg.use_regime_filter else None

    # Pre-compute continuous Kalman filters for every candidate pair once, up front,
    # if running in kalman mode (causal / online, exactly like a live system would run it).
    kalman_cache = {}
    if cfg.hedge_mode == "kalman" and not cfg.use_baskets:
        log_p_full = np.log(close_panel)
        for t1, t2, _ in universe:
            if (t1, t2) in kalman_cache:
                continue
            kalman_cache[(t1, t2)] = kalman_hedge_ratio(
                log_p_full[t1], log_p_full[t2], delta=cfg.kalman_delta, ve=cfg.kalman_ve)

    window_returns, window_log, all_trades, all_round_trips = [], [], [], []
    start = 0
    while start + cfg.formation_days + 5 < n:
        form_start, form_end = dates[start], dates[start + cfg.formation_days - 1]
        trade_start_idx = start + cfg.formation_days
        trade_end_idx = min(trade_start_idx + cfg.trading_days - 1, n - 1)
        trade_start, trade_end = dates[trade_start_idx], dates[trade_end_idx]

        size_scalar = 1.0
        regime_h = np.nan
        if cfg.use_regime_filter and hurst is not None:
            h_at_formation_end = hurst.loc[:form_end].dropna()
            if len(h_at_formation_end):
                regime_h = h_at_formation_end.iloc[-1]
                size_scalar = regime_scalar(regime_h)

        instrument_rets, instruments_this_window = [], []

        if cfg.use_baskets:
            screened = screen_baskets(close_panel, universe, window=(form_start, form_end))
            selected = _select(screened, cfg)
            for _, row in selected.iterrows():
                legs = [row["asset0"], row["asset1"], row["asset2"]]
                w = np.array([row["w1"], row["w2"]])
                log_form = (np.log(close_panel.loc[form_start:form_end, legs[0]])
                            - w[0] * np.log(close_panel.loc[form_start:form_end, legs[1]])
                            - w[1] * np.log(close_panel.loc[form_start:form_end, legs[2]]))
                ou = fit_ou(log_form)
                expected_edge = (cfg.strategy_params.entry_z - cfg.strategy_params.exit_z) * ou.stationary_std
                if expected_edge < cfg.min_edge_cost_multiple * _round_trip_cost_bps(cfg.cost_model):
                    continue
                idx = close_panel.loc[trade_start:trade_end].index
                log_c = (np.log(close_panel.loc[idx, legs[0]])
                         - w[0] * np.log(close_panel.loc[idx, legs[1]])
                         - w[1] * np.log(close_panel.loc[idx, legs[2]]))
                z = (log_c - ou.mu) / ou.stationary_std
                r, trades, rts = _leg_window_returns(legs, lambda d, w=w: w, close_panel, open_panel, volume_panel,
                                                 z, ou.half_life, trade_start, trade_end, cfg,
                                                 label="-".join(legs), size_scalar=size_scalar)
                if len(r) == 0:
                    continue
                instrument_rets.append(r)
                for t in trades: t["sector"] = row["sector"]
                for rt in rts: rt["sector"] = row["sector"]
                all_trades.extend(trades)
                all_round_trips.extend(rts)
                instruments_this_window.append("-".join(legs))
        else:
            screened = screen_pairs(close_panel, universe, window=(form_start, form_end))
            selected = _select(screened, cfg)
            for _, row in selected.iterrows():
                t1, t2 = row["asset1"], row["asset2"]
                legs = [t1, t2]

                if cfg.hedge_mode == "kalman":
                    kres = kalman_cache[(t1, t2)]
                    idx = close_panel.loc[trade_start:trade_end].index
                    # Z-score the Kalman innovation against its own trailing realized
                    # std (causal rolling window) rather than the filter's internal
                    # predictive variance S -- S depends sensitively on the delta/ve
                    # hyperparameters and needs separate calibration to be usable
                    # directly as a z-score denominator; a trailing empirical std is
                    # robust to that and is standard practice in Kalman pairs-trading
                    # implementations.
                    trailing_std = kres.spread.rolling(cfg.formation_days, min_periods=40).std()
                    z_full = (kres.spread / trailing_std).reindex(idx)
                    form_innov = kres.spread.loc[form_start:form_end].dropna()
                    ou_hl = fit_ou(form_innov).half_life if len(form_innov) > 30 else 20.0
                    beta_series = kres.beta

                    def weights_fn(d, bseries=beta_series):
                        return np.array([bseries.loc[:d].iloc[-1]])

                    expected_edge = (cfg.strategy_params.entry_z - cfg.strategy_params.exit_z) * form_innov.std()
                    if expected_edge < cfg.min_edge_cost_multiple * _round_trip_cost_bps(cfg.cost_model):
                        continue
                    r, trades, rts = _leg_window_returns(legs, weights_fn, close_panel, open_panel, volume_panel,
                                                     z_full, ou_hl, trade_start, trade_end, cfg,
                                                     label=f"{t1}-{t2}(kalman)", size_scalar=size_scalar)
                else:
                    hedge_ratio = row["hedge_ratio"]
                    log_form_spread = (np.log(close_panel.loc[form_start:form_end, t1])
                                        - hedge_ratio * np.log(close_panel.loc[form_start:form_end, t2]))
                    ou = fit_ou(log_form_spread)
                    expected_edge = (cfg.strategy_params.entry_z - cfg.strategy_params.exit_z) * ou.stationary_std
                    if expected_edge < cfg.min_edge_cost_multiple * _round_trip_cost_bps(cfg.cost_model):
                        continue
                    idx = close_panel.loc[trade_start:trade_end].index
                    log_c = np.log(close_panel.loc[idx, t1]) - hedge_ratio * np.log(close_panel.loc[idx, t2])
                    z = (log_c - ou.mu) / ou.stationary_std
                    r, trades, rts = _leg_window_returns(legs, lambda d, hr=hedge_ratio: np.array([hr]),
                                                     close_panel, open_panel, volume_panel,
                                                     z, ou.half_life, trade_start, trade_end, cfg,
                                                     label=f"{t1}-{t2}", size_scalar=size_scalar)
                if len(r) == 0:
                    continue
                instrument_rets.append(r)
                for t in trades: t["sector"] = row["sector"]
                for rt in rts: rt["sector"] = row["sector"]
                all_trades.extend(trades)
                all_round_trips.extend(rts)
                instruments_this_window.append(legs[0] + "-" + legs[1])

        if instrument_rets:
            window_df = pd.concat(instrument_rets, axis=1).fillna(0.0)
            port_ret = window_df.mean(axis=1)
            avg_pairwise_corr = np.nan
            if window_df.shape[1] >= 2:
                corr = window_df.corr().values
                iu = np.triu_indices_from(corr, k=1)
                avg_pairwise_corr = np.nanmean(corr[iu])
        else:
            port_ret = pd.Series(0.0, index=close_panel.loc[trade_start:trade_end].index)
            avg_pairwise_corr = np.nan

        window_returns.append(port_ret)
        window_log.append({
            "formation_start": form_start, "formation_end": form_end,
            "trade_start": trade_start, "trade_end": trade_end,
            "n_instruments": len(instruments_this_window), "instruments": instruments_this_window,
            "regime_hurst": regime_h, "size_scalar": size_scalar,
            "avg_pairwise_corr": avg_pairwise_corr,
            "window_sharpe": sharpe_ratio(port_ret) if port_ret.std() > 0 else np.nan,
            "window_return": (1 + port_ret).prod() - 1,
        })
        start += cfg.trading_days

    portfolio_returns = pd.concat(window_returns).sort_index()
    portfolio_returns = portfolio_returns[~portfolio_returns.index.duplicated()]
    return portfolio_returns, window_log, all_trades, all_round_trips
