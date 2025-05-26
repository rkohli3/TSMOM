"""Core logic for TSMOM strategy, including data fetching and return calculations."""
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


def get_yahoo_data(tickers, start=None, end=None, col='Adj Close', period=None):
    """Fetches historical financial data from Yahoo Finance.

    Args:
        tickers (list or str): Ticker symbol(s).
        start (datetime.datetime, optional): Start date. Defaults to 2010-01-01.
        end (datetime.datetime, optional): End date. Defaults to today.
        col (str or list, optional): Column(s) to fetch (e.g., 'Adj Close'). 
                                     Defaults to 'Adj Close'.
        period (str, optional): Data period (e.g., '1y', '5d'). 
                                Overrides start/end if provided by yfinance.download. 
                                Note: yfinance.download directly uses start/end if provided,
                                period is an alternative way to specify range. This docstring
                                clarifies original intent vs. yfinance behavior.

    Returns:
        pandas.DataFrame: DataFrame with the requested financial data for the specified column(s).
                          If multiple tickers and one column, tickers are columns.
                          If one ticker and multiple columns, columns are data types.
                          If multiple tickers and multiple columns, MultiIndex columns.
                          If `col` is a single string, it returns a DataFrame (or Series if one ticker).
        """
    # yfinance's download `period` overrides start/end.
    # For clarity, if period is provided, we might want to clear start/end,
    # but current yfinance handles it. The original function signature included `period`
    # but didn't use it directly in its logic, relying on yfinance.download.
    # This docstring assumes yfinance's behavior for `period`.
    if end is None:
        end = dt.datetime.today()
    if start is None:
        start = dt.datetime(2010,1,1)

    data = yfinance.download(tickers, start = start, end = end)
    return data[col]

def get_rets(data, kind='arth', freq='m', shift=1):
    """Calculates returns from a price time series.

    Args:
        data (pandas.Series or pandas.DataFrame): Time series of prices. 
                                                 Index must be datetime-like.
        kind (str, optional): Type of returns to calculate: 'arth' (arithmetic) 
                              or 'log' (logarithmic). Defaults to 'arth'.
        freq (str, optional): Frequency of returns: 'd' (daily), 'w' (weekly), 
                              or 'm' (monthly). Data is resampled to this frequency 
                              before calculating returns. Defaults to 'm'.
        shift (int, optional): Period shift for calculating returns. Defaults to 1.

    Returns:
        pandas.Series or pandas.DataFrame: Time series of calculated returns.

    Raises:
        KeyError: If input `data` is not a pandas Series or DataFrame with a 
                  datetime-like index.
    """
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

def cum_pfmnce(dataframe, data='prices'):
    """Calculates cumulative performance from price or return data.

    If 'prices' data is provided, it normalizes the series to start at 1.
    If 'returns' data is provided, it calculates the cumulative product of (1 + returns).

    Args:
        dataframe (pandas.Series or pandas.DataFrame): Time series of prices or returns.
        data (str, optional): Type of input data: 'prices' or 'returns'. 
                              Defaults to 'prices'.

    Returns:
        pandas.Series or pandas.DataFrame: Time series of cumulative performance.
    """
    if data == 'prices':
        return dataframe.apply(lambda x: x/x[~x.isnull()][0])
    elif data == 'returns':
        line = dataframe.apply(lambda x: (1+x).cumprod())
        return line

def get_eq_line(series, data='returns', ret_type='arth', freq='monthly'):
    """Calculates the equity line (cumulative performance) from price or return series.

    Represents the hypothetical growth of $1 over time.

    Args:
        series (pandas.Series): Time series of prices or returns with a DatetimeIndex.
        data (str, optional): Type of input data: 'returns' or 'prices'. 
                              Defaults to 'returns'.
        ret_type (str, optional): If data is 'returns', specifies the return type: 
                                  'log' (logarithmic) or 'arth' (arithmetic). 
                                  Defaults to 'arth'.
        freq (str, optional): Resamples the equity line to this frequency: 
                              'daily', 'weekly', or 'monthly'. 
                              Defaults to 'monthly'.

    Returns:
        pandas.Series: Time series representing the equity line (cumulative performance).

    Raises:
        NotImplementedError: If `series` is not a pandas Series with a DatetimeIndex.
    """
    if (isinstance(series, pd.core.series.Series)) and (isinstance(series.index, pd.DatetimeIndex)):
        pass
    else:
        raise NotImplementedError('Data Type not supported, should be time series')

    original_index_name = series.index.name
    original_series_name = series.name
    series = series.dropna()

    if series.empty:
        # Return an empty Series with the original index name if possible,
        # or a generic empty series if the original series had no name or index.
        # This helps maintain consistency in DataFrame constructions later.
        # Capture index before it becomes potentially empty Series with default index
        # However, if series is already empty, its index might not be meaningful.
        # Let's try to create an empty series with the same name and an empty index of original type.
        empty_index = pd.Index([], dtype=series.index.dtype if hasattr(series, 'index') else None)
        return pd.Series(dtype=float, index=empty_index, name=original_series_name)


    if data == 'returns':
        rets = series
        if ret_type == 'arth':
            cum_rets = (1+rets).cumprod()
        elif ret_type == 'log':
            cum_rets = np.exp(rets.cumsum())

        if freq == 'daily':
            cum_rets_prd = cum_rets
            if not cum_rets_prd.empty:
                cum_rets_prd.iloc[0] = 1
        elif freq == 'monthly':
            cum_rets_prd = cum_rets.resample('BM').last().ffill()
            if not cum_rets_prd.empty:
                cum_rets_prd.iloc[0] = 1
        elif freq == 'weekly':
            cum_rets_prd = cum_rets.resample('W-Fri').last().ffill()
            if not cum_rets_prd.empty:
                cum_rets_prd.iloc[0] = 1

    elif data == 'prices':
        cum_rets = series/series[~series.isnull()][0]

        if freq == 'daily':
            cum_rets_prd = cum_rets
        elif freq == 'monthly':
            cum_rets_prd = cum_rets.resample('BM').last().ffill()
        elif freq == 'weekly':
            cum_rets_prd = cum_rets.resample('W-Fri').last().ffill()

    return cum_rets_prd

def get_excess_rets(data, freq='d', kind='arth', shift=1, data_type='returns'):
    """Calculates excess returns by subtracting risk-free returns from asset returns.

    Risk-free returns are fetched from Fama-French data factors.

    Args:
        data (pandas.Series or pandas.DataFrame): Time series of asset prices or returns.
        freq (str, optional): Frequency of returns ('d', 'w', 'm'). Defaults to 'd'.
        kind (str, optional): Type of returns if `data` is prices ('arth', 'log'). 
                              Defaults to 'arth'.
        shift (int, optional): Period shift for return calculation if `data` is prices. 
                               Defaults to 1.
        data_type (str, optional): Type of input `data`: 'returns' or 'prices'. 
                                   Defaults to 'returns'.

    Returns:
        pandas.Series or pandas.DataFrame: Time series of excess returns.

    Raises:
        ImportError: If pandas_datareader is not available.
        ValueError: If an unsupported frequency is provided.
    """
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

def scaled_rets(data, freq='m'):
    """Scales returns by their ex-ante volatility.

    Log returns are calculated first, then scaled by the inverse of their 
    conditional volatility (shifted by 1 period to represent ex-ante).

    Args:
        data (pandas.Series or pandas.DataFrame): Time series of prices.
        freq (str, optional): Frequency for return calculation ('m', 'd', 'w'). 
                              Defaults to 'm'.

    Returns:
        pandas.Series or pandas.DataFrame: Time series of volatility-scaled returns.
    """
    rets = get_rets(data, kind='log', freq=freq)

    # Global rf is used here. This might be an issue.
    # It's not directly used in this function but was in the original tsmom.py context for scaled_rets
    # For now, this function does not use rf directly, but if the logic implies it should,
    # it needs to be passed or fetched.
    
    # Note: The following line calculates conditional volatility (e.g., GARCH)
    # per column using .apply(). This can be computationally intensive
    # for DataFrames with many assets.
    cond_vol = rets.apply(lambda x: get_inst_vol(x, annualize=freq)) # Now calls the actual get_inst_vol
    scal_rets = rets/cond_vol.shift(1) 
    # Removed: scal_rets.iloc[-1, :] = rets.mean()/rets.std()
    return scal_rets

def tsmom(series, mnth_vol_df, mnth_cum_df, tolerance=0, vol_flag=False, scale=0.4, lookback=12):
    """
    Calculates Time Series Momentum returns for a single asset using vectorized operations.

    Args:
        series (pd.Series): Cumulative return series for the specific asset.
                            (This is typically a column from mnth_cum_df, passed by .apply()).
        mnth_vol_df (pd.DataFrame): DataFrame of monthly volatilities for all assets.
        mnth_cum_df (pd.DataFrame): DataFrame of monthly cumulative returns for all assets.
                                  (series is one column of this).
        tolerance (float, optional): Signal tolerance. Position is taken if lookback return
                                     is beyond this tolerance. Defaults to 0.
        vol_flag (bool, optional): If True, apply volatility scaling to leverage. Defaults to False.
        scale (float, optional): Volatility scaling target (e.g., 0.4 for 40% annualized vol target
                                 if vol is annualized). Defaults to 0.4.
        lookback (int, optional): Lookback period in months for the momentum signal. Defaults to 12.

    Returns:
        tuple: (pnl_long, pnl_short, effective_leverage)
            pnl_long (pd.Series): P&L from long positions.
            pnl_short (pd.Series): P&L from short positions.
            effective_leverage (pd.Series): Leverage applied.
    """
    asset_name = series.name
    if asset_name is None:
        raise ValueError("Input 'series' must have a name attribute (e.g., when called from DataFrame.apply).")

    asset_vol = mnth_vol_df[asset_name]
    asset_cum_returns = series # This is mnth_cum_df[asset_name]

    # Calculate lookback returns for signal generation
    # .shift(1) to ensure signal is ex-ante (based on previous period's lookback return)
    asset_lookback_signal_source = asset_cum_returns.pct_change(lookback).shift(1)

    # Determine trading signal: 1 for long, -1 for short, 0 or NaN for no position
    signal = pd.Series(np.nan, index=asset_lookback_signal_source.index)
    signal[asset_lookback_signal_source > tolerance] = 1
    signal[asset_lookback_signal_source < tolerance] = -1 # Note: original was strictly < tolerance for short.
                                                        # If tolerance = 0, this is < 0.
                                                        # If tolerance > 0, this means short if below positive tolerance.
                                                        # This matches common TSMOM interpretation.

    # Calculate leverage
    # .shift(1) to ensure leverage is ex-ante (based on previous period's volatility)
    leverage = pd.Series(1.0, index=asset_vol.index)
    if vol_flag:
        shifted_vol = asset_vol.shift(1)
        # Avoid division by zero or by NaN; keep leverage as 1.0 in such cases or handle as NaN.
        # If shifted_vol is 0 or NaN, leverage calculation could result in inf or NaN.
        # Replacing 0 vol with NaN, then NaN leverage means no position/scaling.
        safe_shifted_vol = shifted_vol.replace(0, np.nan)
        leverage = scale / safe_shifted_vol
        leverage.fillna(1.0, inplace=True) # Fallback to 1.0 if vol was NaN/zero. Or could be np.nan to prevent trades.
                                           # Let's make it NaN to prevent trades if vol is undefined.
        leverage = scale / safe_shifted_vol # Recalculate without fillna(1.0)
        # If leverage becomes NaN (e.g. due to NaN vol), P&L will also be NaN, effectively no scaled position.


    # Calculate periodic returns of the asset for P&L calculation
    asset_periodic_returns = asset_cum_returns.pct_change(1)

    # P&L from long positions
    pnl_long = pd.Series(0.0, index=signal.index)
    long_mask = (signal == 1)
    pnl_long[long_mask] = asset_periodic_returns[long_mask] * leverage[long_mask]
    pnl_long[~(long_mask)] = 0.0 # Ensure 0 where no long position

    # P&L from short positions
    pnl_short = pd.Series(0.0, index=signal.index)
    short_mask = (signal == -1)
    # For shorts, P&L is negative of asset return, scaled by leverage
    pnl_short[short_mask] = -asset_periodic_returns[short_mask] * leverage[short_mask]
    pnl_short[~(short_mask)] = 0.0 # Ensure 0 where no short position

    # Effective leverage applied
    # This series shows the actual leverage factor used on days a position was active.
    # It will be NaN if leverage calculation resulted in NaN (e.g. due to NaN vol).
    effective_leverage = pd.Series(np.nan, index=signal.index)
    position_mask = (long_mask | short_mask)
    effective_leverage[position_mask] = leverage[position_mask]


    # Set NaNs at the beginning where lookback or returns are not available
    # This is handled by NaNs from pct_change and shift propagating.
    # If signal is NaN, masks are false. If leverage is NaN, P&L becomes NaN.
    # If asset_periodic_returns is NaN, P&L becomes NaN.
    # Explicitly ensure NaNs where signal is NaN for pnl_long/short.
    pnl_long[signal.isna()] = np.nan
    pnl_short[signal.isna()] = np.nan
    
    # Naming outputs
    pnl_long.name = asset_name # Original code set this, but it's usually handled by .apply()
    pnl_short.name = asset_name
    effective_leverage.name = asset_name + 'Leverage'

    return pnl_long, pnl_short, effective_leverage

def get_tsmom(mnth_vol_df, mnth_cum_df, vol_flag=False, scale=0.20, lookback=12):
    """Aggregates TSMOM results for a portfolio of assets.

    Applies the `tsmom` function to each asset in the `mnth_cum` DataFrame 
    and then aggregates the P&L and leverage.

    Args:
        mnth_vol_df (pandas.DataFrame): DataFrame of monthly volatilities for assets.
                                     Columns should be asset names.
        mnth_cum_df (pandas.DataFrame): DataFrame of monthly cumulative returns for assets.
                                     Columns should be asset names.
        vol_flag (bool, optional): Volatility scaling flag passed to `tsmom`. 
                                   Defaults to False.
        scale (float, optional): Volatility scaling target passed to `tsmom`. 
                                 Defaults to 0.20.
        lookback (int, optional): Lookback period in months passed to `tsmom`. 
                                  Defaults to 12.

    Returns:
        tuple:
            - pandas.Series: Aggregated P&L from long positions for the portfolio.
            - pandas.Series: Aggregated P&L from short positions for the portfolio.
            - pandas.Series: Mean leverage applied across the portfolio, rolled over the lookback period.
    """
    # The 'series' argument to tsmom is mainly for the name.
    # mnth_cum.apply will pass each column (Series) of mnth_cum to tsmom.
    # Ensure column names (asset names) are consistent between mnth_vol and mnth_cum.
    # The 'series' argument to tsmom is mainly for the name.
    # mnth_cum.apply will pass each column (Series) of mnth_cum to tsmom.
    # Ensure column names (asset names) are consistent between mnth_vol and mnth_cum.
    total = mnth_cum_df.apply(lambda asset_series: tsmom(asset_series, mnth_vol_df, mnth_cum_df, 
                                                     scale=scale, vol_flag=vol_flag, lookback=lookback))
    pnl_long = pd.concat([i[0] for i in total], axis = 1)
    pnl_short = pd.concat([i[1] for i in total], axis = 1)
    lev = pd.concat([i[2] for i in total], axis = 1)
    port_long = pnl_long.mean(axis = 1)
    port_short = pnl_short.mean(axis = 1)
    if vol_flag == True:
        port_long.name = 'LongPnl VolScale'
        port_short.name = 'ShortPnl VolScale'
    # This line will overwrite the name if vol_flag is True, intentional?
    # It is intentional as per original logic to provide a default name if not scaled.
    port_long.name = 'LongPnl'
    port_short.name = 'ShortPnl'

    lev_mean = lev.mean(axis =1)
    lev_mean = lev_mean.rolling(lookback).mean()
    lev_mean.name = 'Leverage'

    return port_long, port_short, lev_mean

def get_tsmom_port(mnth_vol_df, mnth_cum_df, vol_flag=False, scale=0.2, lookback=12):
    """Calculates the combined TSMOM portfolio returns and leverage.

    This function calls `get_tsmom` to get the long P&L, short P&L, and leverage,
    then combines the P&Ls to produce the total TSMOM strategy returns.

    Args:
        mnth_vol_df (pandas.DataFrame): DataFrame of monthly volatilities for assets.
        mnth_cum_df (pandas.DataFrame): DataFrame of monthly cumulative returns for assets.
        vol_flag (bool, optional): Volatility scaling flag. Defaults to False.
        scale (float, optional): Volatility scaling target. Defaults to 0.2.
        lookback (int, optional): Lookback period in months. Defaults to 12.

    Returns:
        pandas.DataFrame: DataFrame with two columns: 'TSMOM' (combined returns) 
                          and 'Leverage'.
    """
    port_long, port_short, leverage = get_tsmom(mnth_vol_df,
                                                mnth_cum_df,
                                                vol_flag=vol_flag,
                                                scale=scale,
                                                lookback=lookback)
    tsmom_returns = port_long.add(port_short, fill_value = 0)
    if vol_flag == True:
        tsmom_returns.name = 'TSMOM VolScale'
    elif vol_flag == False:
       tsmom_returns.name = 'TSMOM'

    return pd.concat([tsmom_returns, leverage], axis = 1)

def get_long_short(mnth_cum, lookback=12):
    """Counts the number of long and short positions based on TSMOM signals.

    Calculates lookback returns from cumulative monthly returns and counts how many 
    assets would be longed (positive lookback return) or shorted (negative lookback return)
    each period.

    Args:
        mnth_cum (pandas.DataFrame): DataFrame of monthly cumulative returns for assets.
                                     Columns are asset names, index is DatetimeIndex.
        lookback (int, optional): Lookback period in months for momentum signal. 
                                  Defaults to 12.

    Returns:
        pandas.DataFrame: DataFrame with columns 'Long Positions' and 'Short Positions',
                          indicating the count for each period.
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
