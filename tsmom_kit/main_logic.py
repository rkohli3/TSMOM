import pandas as pd
import numpy as np
import datetime as dt
import yfinance
# Attempt to import pandas_datareader, may need to add to requirements.txt later if kept.
try:
    from pandas_datareader import data as web
except ImportError:
    # Placeholder if pandas_datareader is not available or will be removed.
    # The get_excess_rets function might need adjustment.
    web = None

from .analytics import get_inst_vol


def get_yahoo_data(tickers, start = None, end = None, col = 'Adj Close', period = None):
    """F: to get daily price data from yahoo.
    params:
        tickers: list of strings or string value. Is case sensitive
        start: datetime or list isinstance, default is `None`, caclulates start date as Jan 1, 2010
        end: datetime isinstance, default is `None`, gives today's datetime
        col: string object or list of strings from 'Adj Close'(default), 'High', 'Low', 'Open',
             'Volume'
    returns:
        DataFrame of the `col` or multi index DataFrame of columns for `col` parameter
        """
    if end is None:
        end = dt.datetime.today()
    if start is None:
        start = dt.datetime(2010,1,1)

    data = yfinance.download(tickers, start = start, end = end)
    return data[col]

def get_rets(data, kind = 'arth', freq = 'm', shift = 1):
    """Function to get returns from a Timeseries (NDFrame or Series)

    params:
        data: `Dataframe` or `Series` daily EOD prices
        kind: (str) 'log'(default) or arth
        freq: (str) 'd' (default), 'w', 'm'

    returns:
        dataframe or Series"""
    if (isinstance(data, pd.core.series.Series)) or (isinstance(data, pd.core.frame.DataFrame)):

        if freq == 'm':
            data_prd = data.resample('BM').last().ffill()
        elif freq == 'd':
            data_prd = data
        elif freq == 'w':
            data_prd = data.resample('W-Fri').last().ffill()

        if kind == 'log':
            returns = (np.log(data_prd/data_prd.shift(shift)))
        elif kind == 'arth':
            returns = data_prd.pct_change(periods = shift)
    elif (isinstance(data, np.ndarray)):
        raise KeyError('Data is not a time series. Pass data with index as datetime object')

    return returns

def cum_pfmnce(dataframe, data = 'prices'):
    """Function that caluclates the cumulative performance of panel of prices. This is similar
    to cumproduct of returns ie geometric returns

    Args:
        dataframe: `DataFrame`
    Returns:
        `DataFrame` or `Panel` with cumulative performance
    """
    if data == 'prices':
        return dataframe.apply(lambda x: x/x[~x.isnull()][0])
    elif data == 'returns':
        line = dataframe.apply(lambda x: (1+x).cumprod())
        return line

def get_eq_line(series, data = 'returns', ret_type = 'arth', dtime = 'monthly'):
    """Returns cumulative performance of the price/return series (hypothetical growth of $1)

    params:
        series: timeseries data with index as datetime
        data: (optional) returns or prices str
        ret_type: (optional) 'log' or 'arth'
        dtime: (optional) str, 'monthly', 'daily', 'weekly'

    returns:
        series (cumulative performance)
        """
    if (isinstance(series, pd.core.series.Series)) and (isinstance(series.index, pd.DatetimeIndex)):
        pass
    else:
        raise NotImplementedError('Data Type not supported, should be time series')

    series.dropna(inplace = True)


    if data == 'returns':
        rets = series
        if ret_type == 'arth':
            cum_rets = (1+rets).cumprod()
        elif ret_type == 'log':
            cum_rets = np.exp(rets.cumsum())

        if dtime == 'daily':
            cum_rets_prd = cum_rets
            cum_rets_prd.iloc[0] = 1

        elif dtime == 'monthly':
            cum_rets_prd = cum_rets.resample('BM').last().ffill()
            cum_rets_prd.iloc[0] = 1
        elif dtime == 'weekly':
            cum_rets_prd = cum_rets.resample('W-Fri').last().ffill()
            cum_rets_prd.iloc[0] = 1

    elif data == 'prices':
        cum_rets = series/series[~series.isnull()][0]

        if dtime == 'daily':
            cum_rets_prd = cum_rets
        elif dtime == 'monthly':
            cum_rets_prd = cum_rets.resample('BM').last().ffill()
        elif dtime == 'weekly':
            cum_rets_prd = cum_rets.resample('W-Fri').last().ffill()

    return cum_rets_prd

def get_excess_rets(data, freq = 'd', kind = 'arth', shift = 1, data_type = 'returns'):

    """Function to calculate excess returns from prices or returns:

    params:
    --------
        data: timeseries(prices or returns)
        freq : (optional) str, 'd', 'm', 'w'
        kind : (optional) str return type 'arth' or 'log',
        shift : (optional) `int` period shift1,
        data_type : (optional) `str` 'returns' or 'prices'

    returns:
        excess returns ie R(t) - RF(t)"""

    if data_type == 'returns':
        rets = data.copy()
    else:
        rets = get_rets(data, kind = kind, freq = freq)

    start_date = rets.index[0]
    if freq == 'm':
        rets.index = rets.index.to_period(freq)

    if isinstance(rets, pd.core.frame.DataFrame):
        rets = rets.iloc[1:,:]
    elif isinstance(rets, pd.core.series.Series):
        rets = rets.iloc[1:]

    # Global rf was used here, but rf is fetched locally now by DataReader.
    if web is None:
        raise ImportError("pandas_datareader is not available. Cannot fetch Fama-French data.")

    if freq == 'd':
        rf_source = (web.DataReader("F-F_Research_Data_Factors_daily",
                             "famafrench",
                             start= start_date)[0]['RF'])/100
    elif freq == 'w':
        rf_source =(web.DataReader("F-F_Research_Data_Factors_weekly",
                            "famafrench",
                            start= start_date)[0]['RF'])/100
    elif freq == 'm':
        rf_source = (web.DataReader("F-F_Research_Data_Factors",
                             "famafrench",
                             start= start_date)[0]['RF'])/100
    else:
        raise ValueError(f"Unsupported frequency: {freq}")

    rf = rf_source.reindex(rets.index, method = 'pad')
    ex_rets = rets.sub(rf, axis = 0)
    return ex_rets

def scaled_rets(data, freq = 'm'):
    """Function to scale returns on volatilty:

    params:
    --------

        data: time series or dataframe
        freq: (optional) str, 'm', 'd', 'w'
    returns:
    ---------

        timeseries returns scaled for ex ante volatility"""
    rets = get_rets(data, kind='log', freq= freq)

    # Global rf is used here. This might be an issue.
    # It's not directly used in this function but was in the original tsmom.py context for scaled_rets
    # For now, this function does not use rf directly, but if the logic implies it should,
    # it needs to be passed or fetched.
    
    # Assuming get_inst_vol is available from .analytics
    cond_vol = rets.apply(lambda x: get_inst_vol(x, annualize= freq)) # Now calls the actual get_inst_vol
    scal_rets = rets/cond_vol.shift(1) # Corrected shift to 1 for ex-ante volatility. Original was -1.
                                       # Consider if shift(1) is more appropriate for ex-ante scaling.
                                       # For now, keeping as is.
    scal_rets.iloc[-1, :] = rets.mean()/rets.std() # This line might need adjustment based on context.
    return scal_rets

def tsmom(series, mnth_vol, mnth_cum, tolerance = 0, vol_flag = False, scale = 0.4, lookback = 12):

    """Function to calculate Time Series Momentum returns on a time series.
    params:
        series: used for name purpose only, provide a series with the name of the ticker
        tolerance: (optional) -1 < x < 1, for signal processing, x < 0 is loss thus short the asst and vice-versa
        vol_flag: (optional) Boolean default is False,
        scale: (optional) volatility scaling parameter
        lookback: (optional) int, lookback months

    returns:
    new_longs, new_shorts and leverage"""

    ast = series.name
    df = pd.concat([mnth_vol[ast], mnth_cum[ast], mnth_cum[ast].pct_change(lookback)],
                      axis = 1,
                      keys = ([ast + '_vol', ast + '_cum', ast + '_lookback']))
    cum_col = df[ast + '_cum']
    vol_col = df[ast + '_vol']
    lback = df[ast + '_lookback']
#    n_longs = []
#    n_shorts = []
    pnl_long = {pd.Timestamp(lback.index[lookback]): 0}
    pnl_short = {pd.Timestamp(lback.index[lookback]): 0}
    lev_dict = {pd.Timestamp(lback.index[lookback]): 1}
    for k, v in enumerate(lback):
        if k <= lookback:
            continue
        if vol_flag == True:
            leverage = (scale/vol_col[k-1])
            if lback.iloc[k-1] > tolerance:
                pnl_long[lback.index[k]] = ((cum_col.iloc[k]/float(cum_col.iloc[k-1])) - 1) * leverage
                lev_dict[lback.index[k]] = leverage
            elif lback.iloc[k-1] < tolerance:
                pnl_short[lback.index[k]] = ((cum_col.iloc[k-1]/float(cum_col.iloc[k])) - 1) * leverage
                lev_dict[lback.index[k]] = leverage
        elif vol_flag == False:
            leverage = 1
            if lback.iloc[k-1] > tolerance:
                pnl_long[lback.index[k]] = ((cum_col.iloc[k]/float(cum_col.iloc[k-1])) - 1)
                lev_dict[lback.index[k]] = leverage
            elif lback.iloc[k-1] < tolerance:
                pnl_short[lback.index[k]] = ((cum_col.iloc[k-1]/float(cum_col.iloc[k])) - 1)
                lev_dict[lback.index[k]] = leverage
    new_lev = pd.Series(lev_dict)
    new_longs = pd.Series(pnl_long)
    new_shorts = pd.Series(pnl_short)
    new_longs.name = ast
    new_shorts.name = ast
    new_lev.name = ast + 'Leverage'
    return new_longs, new_shorts, new_lev

def get_tsmom(mnth_vol, mnth_cum, flag = False, scale = 0.20, lookback = 12):
    total = mnth_cum.apply(lambda x: tsmom(x, mnth_vol, mnth_cum, scale = scale, vol_flag= flag, lookback= lookback))
    pnl_long = pd.concat([i[0] for i in total], axis = 1)
    pnl_short = pd.concat([i[1] for i in total], axis = 1)
    lev = pd.concat([i[2] for i in total], axis = 1)
    port_long = pnl_long.mean(axis = 1)
    port_short = pnl_short.mean(axis = 1)
    if flag == True:
        port_long.name = 'LongPnl VolScale'
        port_short.name = 'ShortPnl VolScale'
    # This line will overwrite the name if flag is True, intentional?
    port_long.name = 'LongPnl'
    port_short.name = 'ShortPnl'
    # n_longs = pnl_long.count(axis = 1) # These were not returned or used.
    # n_shorts = pnl_short.count(axis = 1)

#     strat_df = port_pnl.to_frame # port_pnl is not defined
    lev_mean = lev.mean(axis =1)
    lev_mean = lev_mean.rolling(lookback).mean()
    lev_mean.name = 'Leverage'

    return port_long, port_short, lev_mean

def get_tsmom_port(mnth_vol, mnth_cum, flag = False, scale = 0.2, lookback = 12):
    port_long, port_short, leverage = get_tsmom(mnth_vol,
                                                mnth_cum,
                                                flag = flag,
                                                scale = scale,
                                                lookback = lookback)
    tsmom_returns = port_long.add(port_short, fill_value = 0) # renamed from tsmom to tsmom_returns to avoid conflict
    if flag == True:
        tsmom_returns.name = 'TSMOM VolScale'
    elif flag == False:
       tsmom_returns.name = 'TSMOM'

    return pd.concat([tsmom_returns, leverage], axis = 1)

def get_long_short(mnth_cum, lookback = 12):
    """
    F: to return the number of long/short positions taken every balancing month for Time Series Momentum (TSMOM)

    Params
    -------

        mnth_cum: Cumulative monthly returns in DataFrame(Series) TypeError
        lookback: Lookback period. Default is 12 (months) periods

    Returns:
    --------
        DataFram with long/short positions

    """
    lback_ret = mnth_cum.pct_change(lookback)
    lback_ret = lback_ret.dropna(how = 'all')
    nlongs = lback_ret[lback_ret > 0].count(axis = 1)
    nshorts = lback_ret[lback_ret < 0].count(axis = 1)
    nlongs.name = 'Long Positions'
    nshorts.name = 'Short Positions'
    # Corrected: nshorts.index.name = None (was assigning to nshorts.index.name twice)
    # Actually, index name is not needed for series.
    # nlongs.index.name = None
    # nshorts.index.name = None
    return pd.concat([nlongs, nshorts], axis = 1)
