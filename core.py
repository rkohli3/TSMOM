"""
core.py — Pure computation logic for Time Series Momentum (TSMOM).

No plotting. No data downloading. Import from data.py for prices.
"""

import datetime as dt

import numpy as np
import pandas as pd
import statsmodels.api as sm
import empyrical

try:
    import pandas_datareader.data as web
except ImportError:
    web = None

try:
    import arch
except ImportError:
    arch = None


# ── Returns & Price Transforms ────────────────────────────────────────────────

def get_rets(data, kind='arth', freq='m', shift=1):
    """Convert price data to returns, resampled to the desired frequency.

    Parameters
    ----------
    data : Series or DataFrame
        Daily EOD prices.
    kind : str
        'arth' (arithmetic) or 'log'. Default 'arth'.
    freq : str
        'm' monthly (default), 'd' daily, 'w' weekly.
    shift : int
        Lag for pct_change / log diff. Default 1.
    """
    if freq == 'm':
        data_prd = data.resample('BME').last().ffill()
    elif freq == 'd':
        data_prd = data
    elif freq == 'w':
        data_prd = data.resample('W-Fri').last().ffill()
    else:
        raise ValueError(f"Unknown freq '{freq}'. Use 'd', 'w', or 'm'.")

    if kind == 'log':
        return np.log(data_prd / data_prd.shift(shift))
    elif kind == 'arth':
        return data_prd.pct_change(periods=shift)
    else:
        raise ValueError(f"Unknown kind '{kind}'. Use 'arth' or 'log'.")


def get_eq_line(series, data='returns', ret_type='arth', dtime='monthly'):
    """Cumulative growth of $1 (equity line) from a price or return series.

    Parameters
    ----------
    series : Series with DatetimeIndex
    data : 'returns' or 'prices'
    ret_type : 'arth' or 'log'
    dtime : 'daily', 'monthly', or 'weekly'
    """
    if not (isinstance(series, pd.Series) and isinstance(series.index, pd.DatetimeIndex)):
        raise NotImplementedError('Requires a time series (Series with DatetimeIndex)')

    series = series.dropna()

    if data == 'returns':
        cum = (1 + series).cumprod() if ret_type == 'arth' else np.exp(series.cumsum())
        resample_map = {'daily': None, 'monthly': 'BME', 'weekly': 'W-Fri'}
        freq = resample_map.get(dtime)
        cum_prd = cum.resample(freq).last().ffill() if freq else cum
        cum_prd.iloc[0] = 1
    elif data == 'prices':
        cum_prd = series / series.dropna().iloc[0]
        resample_map = {'monthly': 'BME', 'weekly': 'W-Fri'}
        freq = resample_map.get(dtime)
        if freq:
            cum_prd = cum_prd.resample(freq).last().ffill()
    else:
        raise ValueError("data must be 'returns' or 'prices'")

    return cum_prd


def cum_pfmnce(dataframe, data='prices'):
    """Cumulative performance of a panel of prices or returns."""
    if data == 'prices':
        return dataframe.apply(lambda x: x / x[~x.isnull()].iloc[0])
    return dataframe.apply(lambda x: (1 + x).cumprod())


def get_ann_ret(ret_series, dtime='monthly'):
    """Annual returns from a monthly return series."""
    cum_series = get_eq_line(ret_series, dtime='monthly')
    annual = cum_series.resample('A').last()
    annual.loc[ret_series.index[0]] = 1
    annual.sort_index(ascending=True, inplace=True)
    annual_ret = annual.pct_change()
    annual_ret.index = annual_ret.index.to_period('A')
    annual_ret.dropna(inplace=True)
    return annual_ret


def get_ytd(table, year=None):
    """Year-to-date performance.

    Parameters
    ----------
    table : Series or DataFrame
    year : int, optional. Defaults to current year.
    """
    this_year = year or dt.date.today().year
    grouped = table.index.groupby(table.index.year)
    index = grouped[this_year]
    return (table.loc[index].iloc[-1] / table.loc[grouped[this_year]].iloc[0]) - 1


# ── Frequency Conversion ──────────────────────────────────────────────────────

def cnvert_daily_to(index, cnvrt_to='m'):
    """Convert a daily datetime index to empirical period-end dates.

    Unlike resample, this only returns dates that actually exist in the series
    (i.e. the last trading day of each period).

    Parameters
    ----------
    index : DatetimeIndex
    cnvrt_to : str
        'm'/'monthly', 'q'/'quarterly', 'a'/'annually', 'w'/'weekly', 'd'/'daily'
    """
    cnvrt_to = cnvrt_to.lower()
    t_day_index = pd.DatetimeIndex(sorted(index))
    t_years = t_day_index.groupby(t_day_index.year)
    f_date = t_day_index[0]
    ann_dt = [f_date]
    qrter_dt = [f_date]
    mnthly_dt = [f_date]
    weekly_dt = [f_date]

    for yr in t_years.keys():
        iso = t_years[yr].isocalendar()
        yr_end    = pd.DatetimeIndex(t_years[yr]).groupby(iso.year.values)
        qrter_end = pd.DatetimeIndex(t_years[yr]).groupby(iso.quarter.values)
        mnth_end  = pd.DatetimeIndex(t_years[yr]).groupby(t_years[yr].month)
        week_end  = pd.DatetimeIndex(t_years[yr]).groupby(iso.week.values)

        ann_dt.append(max(yr_end[max(yr_end)]))
        for q in qrter_end:
            qrter_dt.append(max(qrter_end[q]))
        for m in mnth_end:
            mnthly_dt.append(max(mnth_end[m]))
        for w, val in week_end.items():
            weekly_dt.append(max(val))

    if cnvrt_to in ('monthly', 'm'):
        return mnthly_dt
    elif cnvrt_to in ('quarterly', 'q'):
        return qrter_dt
    elif cnvrt_to in ('annually', 'a'):
        return ann_dt
    elif cnvrt_to in ('weekly', 'w'):
        return weekly_dt
    return list(index)


# ── Volatility ────────────────────────────────────────────────────────────────

def get_exante_vol(series, alpha=0.05, dtime='monthly', dtype='returns', com=None):
    """Annualized ex-ante (EWMA / Risk Metrics) volatility.

    Parameters
    ----------
    series : Series with DatetimeIndex
    alpha : float
        EWM smoothing factor. Default 0.05.
    com : int, optional
        Center of mass (alternative to alpha).
    dtime : 'daily', 'monthly', or 'weekly'
    dtype : 'returns' or 'prices'
    """
    if dtype == 'prices':
        series = get_rets(series, kind='arth', freq='d')

    vol = series.ewm(alpha=alpha, com=com).std()
    ann_vol = vol * np.sqrt(261)

    resample_map = {'monthly': 'BME', 'weekly': 'W-Fri'}
    freq = resample_map.get(dtime)
    return ann_vol.resample(freq).last().ffill() if freq else ann_vol


def get_inst_vol(y, annualize, x=None, mean='Constant', vol='Garch',
                 dist='normal', data='prices', freq='d'):
    """Conditional volatility via GARCH(1,1).

    Parameters
    ----------
    y : Series
    annualize : str
        'd', 'm', or 'w' — annualization frequency.
    data : 'prices' or 'returns'
    freq : str
        Resample frequency if data='prices'.
    """
    if arch is None:
        raise ImportError("arch package required: pip install arch")

    if data in ('prices', 'price'):
        y = get_rets(y, kind='arth', freq=freq)

    if not isinstance(y, pd.Series):
        raise TypeError('y must be a Series with DatetimeIndex')

    y = y.dropna()
    model = arch.arch_model(y * 100, mean='constant', vol='Garch')
    res = model.fit(update_freq=5)

    ann_map = {'d': np.sqrt(252), 'm': np.sqrt(12), 'w': np.sqrt(52)}
    return res.conditional_volatility * ann_map.get(annualize.lower(), np.sqrt(252)) * 0.01


# ── Excess Returns ────────────────────────────────────────────────────────────

def get_excess_rets(data, freq='d', kind='arth', shift=1, data_type='returns'):
    """Subtract the risk-free rate (Fama-French) from returns.

    Parameters
    ----------
    data : Series or DataFrame of returns or prices
    freq : 'd', 'w', or 'm'
    kind : 'arth' or 'log'
    data_type : 'returns' or 'prices'
    """
    if web is None:
        raise ImportError("pandas_datareader required: pip install pandas-datareader")

    rets = data.copy() if data_type == 'returns' else get_rets(data, kind=kind, freq=freq)
    start_date = rets.index[0]

    if freq == 'm':
        rets.index = rets.index.to_period(freq)

    rets = rets.iloc[1:] if isinstance(rets, pd.Series) else rets.iloc[1:, :]

    ff_map = {
        'd': 'F-F_Research_Data_Factors_daily',
        'w': 'F-F_Research_Data_Factors_weekly',
        'm': 'F-F_Research_Data_Factors',
    }
    rf = web.DataReader(ff_map[freq], 'famafrench', start=start_date)[0]['RF'] / 100
    rf = rf.reindex(rets.index, method='pad')
    return rets.sub(rf, axis=0)


# ── Autocorrelation ───────────────────────────────────────────────────────────

def autocorr(x, t=1):
    """Autocorrelation of a series at lag t."""
    if isinstance(x, np.ndarray):
        return np.corrcoef(x[t:], x[:-t])
    return np.corrcoef(x.iloc[t:], x.shift(t).dropna())


def get_tseries_autocor(series, nlags=40):
    """Full autocorrelation series up to nlags."""
    if isinstance(series, pd.DataFrame):
        raise TypeError('Must be 1-d array or Series')
    if isinstance(series, np.ndarray):
        series = series[~np.isnan(series)]
    else:
        series = series.dropna()

    auto_cor = {i: autocorr(series, i)[0, 1] for i in range(1, nlags + 1)}
    result = pd.Series(auto_cor)
    if hasattr(series, 'name'):
        result.name = series.name
    return result


def get_lagged_params(y, param='t', nlags=24, name=None):
    """T-stats or betas from lagged OLS regressions at each lag.

    Parameters
    ----------
    y : Series or ndarray
    param : 't' (t-statistics) or 'b' (betas)
    nlags : int
    name : str, optional
    """
    if isinstance(y, np.ndarray):
        y = pd.Series(y)

    y = y.ffill().dropna()

    if len(y) <= nlags:
        raise ValueError('Not enough data points for requested lags')

    results = {}
    for lag in range(1, nlags + 1):
        reg = sm.OLS(y.iloc[lag:], y.shift(lag).dropna()).fit()
        results[lag] = reg.tvalues[0] if param == 't' else reg.params[0]

    out = pd.Series(results)
    out.name = name
    return out


def get_ts(df, nlags=48):
    """Lagged t-stats across all columns of a DataFrame."""
    return pd.DataFrame({col: get_lagged_params(df[col], nlags=nlags) for col in df})


# ── TSMOM Core ────────────────────────────────────────────────────────────────

def tsmom(series, mnth_vol, mnth_cum, tolerance=0,
          vol_flag=False, scale=0.4, lookback=12):
    """Single-asset Time Series Momentum.

    Parameters
    ----------
    series : Series
        Used for name lookup only; must match a column in mnth_vol/mnth_cum.
    mnth_vol : DataFrame
        Monthly ex-ante volatility.
    mnth_cum : DataFrame
        Monthly cumulative returns.
    tolerance : float
        Signal threshold. Default 0 (any positive momentum = long).
    vol_flag : bool
        If True, apply volatility scaling. Default False.
    scale : float
        Target volatility for scaling. Default 0.4.
    lookback : int
        Momentum lookback in months. Default 12.

    Returns
    -------
    (pnl_long, pnl_short, leverage) as Series.
    """
    ast = series.name
    df = pd.concat(
        [mnth_vol[ast], mnth_cum[ast], mnth_cum[ast].pct_change(lookback)],
        axis=1,
        keys=[ast + '_vol', ast + '_cum', ast + '_lookback'],
    )
    cum_col = df[ast + '_cum']
    vol_col = df[ast + '_vol']
    lback   = df[ast + '_lookback']

    pnl_long  = {pd.Timestamp(lback.index[lookback]): 0}
    pnl_short = {pd.Timestamp(lback.index[lookback]): 0}
    lev_dict  = {pd.Timestamp(lback.index[lookback]): 1}

    for k, v in enumerate(lback):
        if k <= lookback:
            continue
        leverage = (scale / vol_col.iloc[k - 1]) if vol_flag else 1.0
        period_ret = (cum_col.iloc[k] / float(cum_col.iloc[k - 1])) - 1

        if lback.iloc[k - 1] > tolerance:
            pnl_long[lback.index[k]]  = period_ret * leverage
        elif lback.iloc[k - 1] < tolerance:
            pnl_short[lback.index[k]] = -period_ret * leverage
        lev_dict[lback.index[k]] = leverage

    new_longs  = pd.Series(pnl_long,  name=ast)
    new_shorts = pd.Series(pnl_short, name=ast)
    new_lev    = pd.Series(lev_dict,  name=ast + 'Leverage')
    return new_longs, new_shorts, new_lev


def get_tsmom(mnth_vol, mnth_cum, flag=False, scale=0.20, lookback=12):
    """Apply single-asset TSMOM across all columns and aggregate to portfolio.

    Returns
    -------
    (port_long, port_short, leverage_mean) as Series.
    """
    results   = [tsmom(mnth_cum[col], mnth_vol, mnth_cum,
                       scale=scale, vol_flag=flag, lookback=lookback)
                 for col in mnth_cum.columns]
    pnl_long  = pd.concat([r[0] for r in results], axis=1)
    pnl_short = pd.concat([r[1] for r in results], axis=1)
    lev       = pd.concat([r[2] for r in results], axis=1)

    port_long  = pnl_long.mean(axis=1)
    port_short = pnl_short.mean(axis=1)
    port_long.name  = 'LongPnl VolScale' if flag else 'LongPnl'
    port_short.name = 'ShortPnl VolScale' if flag else 'ShortPnl'

    lev_mean = lev.mean(axis=1).rolling(lookback).mean()
    lev_mean.name = 'Leverage'
    return port_long, port_short, lev_mean


def get_tsmom_port(mnth_vol, mnth_cum, flag=False, scale=0.2, lookback=12):
    """Full TSMOM portfolio: long + short combined with leverage column.

    Returns
    -------
    DataFrame with columns [TSMOM, Leverage].
    """
    port_long, port_short, leverage = get_tsmom(
        mnth_vol, mnth_cum, flag=flag, scale=scale, lookback=lookback
    )
    result = port_long.add(port_short, fill_value=0)
    result.name = 'TSMOM VolScale' if flag else 'TSMOM'
    return pd.concat([result, leverage], axis=1)


def get_long_short(mnth_cum, lookback=12):
    """Count active long/short positions per period.

    Returns
    -------
    DataFrame with columns ['Long Positions', 'Short Positions'].
    """
    lback_ret = mnth_cum.pct_change(lookback).dropna(how='all')
    nlongs  = lback_ret[lback_ret > 0].count(axis=1)
    nshorts = lback_ret[lback_ret < 0].count(axis=1)
    nlongs.name  = 'Long Positions'
    nshorts.name = 'Short Positions'
    return pd.concat([nlongs, nshorts], axis=1)


# ── Performance Analytics ─────────────────────────────────────────────────────

def drawdown(df, data='returns', ret_type='arth', ret_='text'):
    """Maximum drawdown.

    Parameters
    ----------
    df : Series or DataFrame
    data : 'returns' or 'prices'
    ret_type : 'arth' or 'log'
    ret_ : 'text' returns formatted string; anything else returns numeric.
    """
    if data == 'returns':
        eq_line = (1 + df).cumprod() if ret_type == 'arth' else np.exp(df.cumsum())
    else:
        eq_line = df

    max_dd = np.max(1 - eq_line.div(eq_line.cummax()))
    if ret_ != 'text':
        return max_dd
    return 'The maximum drawdown is: {:,.2%}'.format(max_dd)


def rolling_drawdown(df, data='returns', ret_type='arth'):
    """Period-by-period drawdown series."""
    if data == 'returns':
        eq_line = (1 + df).cumprod() if ret_type == 'arth' else np.exp(df.cumsum())
    else:
        eq_line = df
    return eq_line.div(eq_line.cummax()) - 1


def get_stats(returns, dtime='monthly'):
    """Annualized mean, volatility, and Sharpe ratio.

    Returns
    -------
    tuple : (annualized_mean, annualized_std, sharpe_ratio)
    """
    mean = returns.mean()
    std  = returns.std()
    scale = 12 if dtime == 'monthly' else 252
    ann_mean = mean * scale
    ann_std  = std  * np.sqrt(scale)
    return ann_mean, ann_std, ann_mean / ann_std


def get_perf_att(series, benchmark, rf=0.03 / 12, freq='monthly'):
    """Full performance attribution table.

    Parameters
    ----------
    series : Series of returns
    benchmark : Series of benchmark returns (aligned index)
    rf : float
        Monthly risk-free rate. Default 0.03/12.
    freq : 'monthly' or 'daily'

    Returns
    -------
    DataFrame with strategy name as column.
    """
    port_mean, port_std, port_sr = get_stats(series, dtime=freq)
    regs = sm.OLS(series, sm.add_constant(benchmark)).fit()
    alpha, beta = regs.params
    t_alpha, t_beta = regs.tvalues

    perf = pd.Series({
        'Annualized_Mean':       round(port_mean, 5),
        'Annualized_Volatility': round(port_std,  5),
        'Sharpe Ratio':          round(port_sr,   3),
        'Calmar Ratio':          round(empyrical.calmar_ratio(series, period=freq), 3),
        'Alpha':                 round(alpha,  3),
        'Beta':                  round(beta,   3),
        'T Value (Alpha)':       round(t_alpha, 3),
        'T Value (Beta)':        round(t_beta,  3),
        'Max Drawdown':          '{:,.2%}'.format(drawdown(series, ret_='nottext')),
        'Sortino Ratio':         round(empyrical.sortino_ratio(series, required_return=rf, period=freq), 3),
    })
    perf.name = series.name
    return perf.to_frame()


# ── Fama-French Factor Analysis ───────────────────────────────────────────────

def get_ff_rolling_factors(strat, factors=None, rolling_window=36):
    """Rolling Fama-French 5-factor OLS regression.

    Parameters
    ----------
    strat : Series of monthly strategy returns
    factors : DataFrame, optional
        Pre-loaded factor returns. If None, downloads from Ken French's library.
    rolling_window : int
        Rolling window in months. Default 36.

    Returns
    -------
    DataFrame of rolling factor coefficients.
    """
    if web is None:
        raise ImportError("pandas_datareader required: pip install pandas-datareader")

    if factors is None:
        factor_returns = web.DataReader(
            'F-F_Research_Data_5_Factors_2X3', 'famafrench',
            strat.index[0], strat.index[-1]
        )[0]
        factor_returns.index = strat.index
        factor_returns = factor_returns.drop(['RF'], axis=1) / 100
    else:
        factor_returns = factors

    if rolling_window >= len(strat):
        raise ValueError(
            f'rolling_window ({rolling_window}) must be < length of series ({len(strat)})'
        )

    coef_ = {}
    for beg, end in zip(factor_returns.index[:-rolling_window],
                        factor_returns.index[rolling_window:]):
        model = sm.OLS(strat.loc[beg:end], factor_returns.loc[beg:end], hasconst=True).fit()
        coef_[end] = model.params

    return pd.DataFrame(coef_).T


def scaled_rets(data, freq='m'):
    """Returns scaled by conditional (GARCH) volatility."""
    rets = get_rets(data, kind='log', freq=freq)
    cond_vol = rets.apply(lambda x: get_inst_vol(x, annualize=freq))
    scal = rets / cond_vol.shift(-1)
    scal.iloc[-1, :] = rets.mean() / rets.std()
    return scal
