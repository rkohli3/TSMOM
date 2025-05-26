"""Analytics functions for TSMOM, including performance metrics, volatility, and drawdown."""
import pandas as pd
import numpy as np
import statsmodels.api as sm
import empyrical
from arch import arch_model # For get_inst_vol
import datetime as dt # Added for get_ytd

# Import necessary functions from main_logic
# Assuming get_rets might be needed by functions like get_inst_vol
from .main_logic import get_rets, get_eq_line

# If pandas_datareader is used by get_ff_rolling_factors for Fama-French data,
# ensure it's handled similarly to main_logic.py (try/except import).
try:
    from pandas_datareader import data as web
except ImportError:
    web = None # Placeholder

def get_exante_vol(series, alpha=0.05, freq='monthly', input_type='returns', com=None):
    """Calculates annualized ex-ante volatility using Exponentially Weighted Moving Average (EWMA).

    This method, similar to RiskMetrics, estimates volatility based on past volatility
    with a decay factor.

    Args:
        series (pandas.Series): Time series of prices or returns with a DatetimeIndex.
        alpha (float, optional): Smoothing factor for EWMA. Defaults to 0.05.
                                 (Note: `com` is an alternative way to specify decay in pandas EWM)
        freq (str, optional): Target frequency for annualized volatility ('monthly', 'daily', 'weekly').
                              The output is resampled to this frequency. Defaults to 'monthly'.
        input_type (str, optional): Type of input `series` data: 'returns' or 'prices'.
                                    If 'prices', returns are calculated first. Defaults to 'returns'.
        com (float, optional): Center of mass for EWM, an alternative to alpha. 
                               alpha = 1 / (1 + com). Defaults to None.

    Returns:
        pandas.Series: Time series of annualized ex-ante volatility.

    Raises:
        NotImplementedError: If `series` is not a pandas Series with a DatetimeIndex.
        ValueError: If `dtype` is 'prices' and `get_rets` is not available or fails.
    """
    if (isinstance(series, pd.core.series.Series)) and (isinstance(series.index, pd.DatetimeIndex)):
        pass
    else:
        raise NotImplementedError('Data Type not supported, should only be timeseries')
    if input_type == 'prices':
        # Assuming get_rets is available from .main_logic
        series = get_rets(series, kind = 'arth', freq = 'd')

    vol = series.ewm(alpha = alpha, com = com).std()
    ann_vol = vol * np.sqrt(261)

    if freq == 'daily':
        ann_vol_prd = ann_vol
    elif freq == 'monthly':
        ann_vol_prd = ann_vol.resample('BM').last().ffill()
    elif freq == 'weekly':
        ann_vol_prd = ann_vol.resample('W-Fri').last().ffill()

    return ann_vol_prd

def get_inst_vol(y,
                 annualize,
                 x = None,
                 mean = 'Constant',
                 vol = 'Garch',
                 dist = 'normal',
                 input_type = 'prices',
                 freq='d'):
    """Calculates conditional volatility using a GARCH model.

    Args:
        y (pandas.Series or numpy.ndarray): Time series of returns (if `input_type` is 'returns')
                                           or prices (if `input_type` is 'prices').
        annualize (str): Frequency for annualizing the conditional volatility 
                         ('d' for daily, 'm' for monthly, 'w' for weekly).
        x (pandas.Series or numpy.ndarray, optional): Exogenous variables for the mean 
                                                      equation (ARX/HARX models). Defaults to None.
        mean (str, optional): Mean model specification (e.g., 'Constant', 'Zero', 'ARX'). 
                              Defaults to 'Constant'.
        vol (str, optional): Volatility model specification (e.g., 'GARCH', 'EGARCH'). 
                             Defaults to 'GARCH'.
        dist (str, optional): Distribution for the innovations (e.g., 'normal', 't'). 
                              Defaults to 'normal'.
        input_type (str, optional): Type of input `y`: 'prices' or 'returns'. 
                                    Defaults to 'prices'.
        freq (str, optional): Frequency of data if `y` is prices, used for return calculation.
                              Defaults to 'd'.

    Returns:
        pandas.Series: Time series of annualized conditional volatility, scaled by 0.01.

    Raises:
        TypeError: If `y` is not a pandas Series or ndarray when `data` is 'returns',
                   or if `y` cannot be converted to a Series with DatetimeIndex when 
                   `input_type` is 'prices'.
        ValueError: If `annualize` frequency is not recognized.
    """
    if (input_type == 'prices') or (input_type == 'price'):
        # Assuming get_rets is available from .main_logic
        y = get_rets(y, kind = 'arth', freq = freq)

    if isinstance(y, pd.core.series.Series):
        ## remove nan.
        y = y.dropna()
    else:
        raise TypeError('Data should be time series with index as DateTime')

    # provide a model
    model = arch_model(y * 100, mean = mean, vol = vol, dist = dist) # Corrected to use parameters

    # fit the model
    res = model.fit(update_freq= 5)

    # more interested in conditional vol
    if annualize.lower() == 'd':
        ann_cond_vol = res.conditional_volatility * np.sqrt(252)
    elif annualize.lower() == 'm':
        ann_cond_vol = res.conditional_volatility * np.sqrt(12)
    elif annualize.lower() == 'w':
        ann_cond_vol = res.conditional_volatility * np.sqrt(52)
    else:
        # Default or error for unspecified annualization
        raise ValueError("Annualization frequency not recognized: must be 'd', 'm', or 'w'")

    return ann_cond_vol * 0.01

def drawdown(df, input_type='returns', ret_type='arth', output_format='text'):
    """Calculates the drawdown of a time series of prices or returns.

    Drawdown is the percentage decline from a peak.

    Args:
        df (pandas.Series or pandas.DataFrame): Time series of prices or returns.
        input_type (str, optional): Type of input `df`: 'returns' or 'prices'. 
                                    Defaults to 'returns'.
        ret_type (str, optional): If `input_type` is 'returns', specifies the return type: 
                                  'log' or 'arth'. Defaults to 'arth'.
        output_format (str, optional): Output type: 'text' for a formatted string of the max drawdown,
                                       or any other value for the raw max drawdown value(s) (float or Series).
                                       Defaults to 'text'.

    Returns:
        float or pandas.Series or str: 
            - If `output_format` is 'text' and `df` is a Series: Formatted string of max drawdown.
            - If `output_format` is 'text' and `df` is a DataFrame: Dictionary of formatted strings per column.
            - If `output_format` is not 'text': Max drawdown (float for Series, pandas.Series for DataFrame).

    Raises:
        ValueError: If `input_type` parameter is not 'returns' or 'prices'.
    """
    if input_type == 'returns':
        if ret_type == 'arth':
            eq_line = (1 + df).cumprod()
        elif ret_type == 'log':
            eq_line = np.exp(df.cumsum())
    elif input_type == 'prices': # Corrected from 'if' to 'elif' for clarity
        eq_line = df
    else:
        raise ValueError("input_type parameter must be 'returns' or 'prices'")

    draw = 1 - eq_line.div(eq_line.cummax())
    max_drawdown = np.max(draw) # For DataFrames, this will be a Series of max drawdowns per column

    if isinstance(df, pd.DataFrame) and isinstance(max_drawdown, pd.Series):
         if output_format != 'text':
            return max_drawdown # Return Series of max drawdowns
         else:
            # For text output with multiple columns, perhaps return a dict or formatted string
            return {col: "The maximum drawdown is: {0:,.2%}".format(val) for col, val in max_drawdown.items()}

    if output_format != 'text':
        return max_drawdown # single float
    elif output_format =='text':
        return ("The maximum drawdown is: {0:,.2%}").format(max_drawdown) # single string

def rolling_drawdown(df, input_type='returns', ret_type='arth'): # Renamed 'data' to 'input_type' here as well for consistency
    """Calculates the rolling drawdown of a time series.

    The rolling drawdown is the percentage decline from the cumulative peak 
    at each point in time.

    Args:
        df (pandas.Series or pandas.DataFrame): Time series of prices or returns.
        data (str, optional): Type of input `df`: 'returns' or 'prices'. 
                              Defaults to 'returns'.
        ret_type (str, optional): If `data` is 'returns', specifies the return type: 
                                  'log' or 'arth'. Defaults to 'arth'.

    Returns:
        pandas.Series or pandas.DataFrame: Time series of rolling drawdowns.

    Raises:
        ValueError: If `data` parameter is not 'returns' or 'prices'.
    """
    if data == 'returns':
        if ret_type == 'arth':
            eq_line = (1 + df).cumprod()
        elif ret_type == 'log':
            eq_line = np.exp(df.cumsum())
    elif data == 'prices': # Corrected from 'if' to 'elif'
        eq_line = df
    else:
        raise ValueError("data parameter must be 'returns' or 'prices'")
    draw = eq_line.div(eq_line.cummax()) - 1
    return draw

def get_stats(returns, freq='monthly'):
    """Calculates annualized mean return, annualized volatility, and Sharpe ratio.

    Args:
        returns (pandas.Series or pandas.DataFrame or numpy.ndarray): Time series of returns.
        freq (str, optional): Periodicity of input returns ('monthly' or 'daily') 
                              for annualization. Defaults to 'monthly'.

    Returns:
        tuple: Contains:
            - float or pandas.Series: Annualized mean return(s).
            - float or pandas.Series: Annualized volatility(ies).
            - float or pandas.Series: Sharpe ratio(s).

    Raises:
        TypeError: If `returns` cannot be converted to a numpy array.
        ValueError: If `dtime` is not 'monthly' or 'daily'.
    """
    if not (isinstance(returns, pd.Series) or isinstance(returns, pd.DataFrame)):
        try:
            # Attempt to convert to numpy array if not Series/DataFrame
            returns = np.array(returns)
        except:
            raise TypeError("Input must be a pandas Series, DataFrame, or convertible to a numpy array.")

    mean = np.mean(returns, axis=0) # np.mean works for both Series/DF and arrays
    std = np.std(returns, axis=0)   # np.std works for both Series/DF and arrays

    if freq == 'monthly':
        mean = mean * 12
        std = std * np.sqrt(12)
    elif freq == 'daily':
        mean = mean * 252
        std = std * np.sqrt(252)
    else:
        raise ValueError("freq must be 'monthly' or 'daily'")

    # Handle division by zero for Sharpe ratio
    # If std is a Series/array, handle element-wise
    if isinstance(std, (np.ndarray, pd.Series)):
        sr = np.where(std == 0, np.nan, mean / std)
    else: # std is a scalar
        sr = mean / std if std != 0 else np.nan
    return (mean, std, sr)

def get_ytd(table, year=None):
    """Calculates Year-To-Date (YTD) performance for a given price series.

    Args:
        table (pandas.Series or pandas.DataFrame): Time series of prices. 
                                                 Must have a DatetimeIndex.
        year (int, optional): The year for which to calculate YTD performance. 
                              Defaults to the current year.

    Returns:
        float or pandas.Series: YTD performance. Returns np.nan or a Series of np.nan
                                if data for the year is insufficient or not found.

    Raises:
        ValueError: If `table` does not have a DatetimeIndex, or if `year` is in the future.
    """
    if not isinstance(table.index, pd.DatetimeIndex):
        raise ValueError("Table must have a DatetimeIndex.")

    current_year = dt.date.today().year
    if year is None:
        year = current_year
    
    if year > current_year :
        raise ValueError(f"Cannot calculate YTD for a future year: {year}")

    # Filter for the specified year
    year_data = table[table.index.year == year]
    
    if year_data.empty:
        # No data for the specified year, or table doesn't go up to 'year'
        if isinstance(table, pd.DataFrame):
            return pd.Series([np.nan] * len(table.columns), index=table.columns)
        else: # Series
            return np.nan

    # YTD calculation: (last value of the year / first value of the year) - 1
    # Ensure data is sorted by index if not already
    year_data_sorted = year_data.sort_index()
    
    # Get the first available day's data for the year
    first_day_val = year_data_sorted.iloc[0]
    
    # Get the last available day's data for the year (or up to today if it's the current year)
    if year == current_year:
        last_day_val = year_data_sorted[year_data_sorted.index <= pd.Timestamp(dt.date.today())].iloc[-1]
    else: # For past years, use the last data point of that year
        last_day_val = year_data_sorted.iloc[-1]

    # Calculate YTD performance
    ytd_perf = (last_day_val / first_day_val) - 1
    return ytd_perf


def get_perf_att(series, bnchmark, rf=0.03/12, freq='monthly'):
    """Calculates and tabulates various performance attribution statistics.

    Includes annualized mean, volatility, Sharpe ratio, Calmar ratio, alpha, beta,
    t-statistics for alpha/beta, max drawdown, and Sortino ratio.

    Args:
        series (pandas.Series): Time series of strategy returns.
        bnchmark (pandas.Series): Time series of benchmark returns.
        rf (float, optional): Risk-free rate for the period of returns. 
                              Defaults to 0.03/12 (0.25% monthly).
        freq (str, optional): Frequency of returns ('monthly' or 'daily') for
                              annualization and ratio calculations. Defaults to 'monthly'.

    Returns:
        pandas.DataFrame: A single-column DataFrame where the index contains statistic names
                          and the column is the strategy's performance for those statistics.
                          Values are typically formatted as strings.
    """
    port_mean, port_std, port_sr = get_stats(series, dtime=freq)

    # Align series and benchmark indices before regression
    aligned_series, aligned_bnchmark = series.align(bnchmark, join='inner')
    
    if aligned_series.empty or aligned_bnchmark.empty:
        # Cannot perform regression if no overlapping data
        alpha, beta, t_alpha, t_beta = np.nan, np.nan, np.nan, np.nan
    else:
        X = sm.add_constant(aligned_bnchmark)
        model = sm.OLS(aligned_series, X).fit()
        alpha, beta = model.params.iloc[0], model.params.iloc[1] # OLS params might be Series
        t_alpha, t_beta = model.tvalues.iloc[0], model.tvalues.iloc[1]


    max_dd = drawdown(series, data='returns', ret_type='arth', ret_='nottext')
    # If series is a DataFrame, drawdown returns a Series. We need a single value for perf table.
    # Assuming series is a single strategy, so max_dd should be a float.
    # If series can be a DataFrame of multiple strategies, this part needs reconsideration.
    if isinstance(max_dd, pd.Series): # Take the first if multiple, or handle as error/warning
        max_dd_val = max_dd.iloc[0] if not max_dd.empty else np.nan
    else:
        max_dd_val = max_dd

    perf_dict = {
        'Annualized_Mean': port_mean, # Removed formatting for now, can be applied at display time
        'Annualized_Volatility': port_std,
        'Sharpe Ratio': port_sr,
        'Calmar Ratio': empyrical.calmar_ratio(series, period=freq),
        'Alpha': alpha,
        'Beta': beta,
        'T Value (Alpha)': t_alpha,
        'T Value (Beta)': t_beta,
        'Max Drawdown': max_dd_val, # Using the potentially adjusted max_dd_val
        'Sortino Ratio': empyrical.sortino_ratio(series, required_return=rf, period=freq)
    }
    
    # Apply formatting for display if needed, or return raw numbers
    # formatted_perf = {k: '{:,.3f}'.format(v) if isinstance(v, (float, np.float_)) and k not in ['Max Drawdown'] else 
    #                      '{:,.2%}'.format(v) if k == 'Max Drawdown' and isinstance(v, (float, np.float_)) else
    #                      v 
    #                   for k, v in perf_dict.items()}

    # perf_series = pd.Series(formatted_perf)

    # Return raw numerical data
    perf_series = pd.Series(perf_dict)
    perf_series.name = series.name if series.name else "Strategy"
    return perf_series.to_frame()

def get_lagged_params(y, param='t', nlags=24, name=None):
    """Calculates lagged parameters (t-stats or coeffs) of an autoregression.

    Performs an OLS regression of the series `y` against its own lagged values,
    up to `nlags`.

    Args:
        y (pandas.Series or numpy.ndarray): Time series data.
        param (str, optional): Parameter to return: 't' for t-statistic of the lagged variable,
                               'b' for its beta coefficient. Defaults to 't'.
        nlags (int, optional): Number of lags to calculate parameters for. Defaults to 24.
        name (str, optional): Name for the returned Series. Defaults to `y.name` or None.

    Returns:
        pandas.Series: Series of calculated parameters (t-stats or betas) indexed by lag number.
                       Returns an empty Series with a warning if data is insufficient.
    
    Raises:
        ValueError: If `param` is not 't' or 'b'.
    """
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    y = y.ffill().dropna() # Ensure no NaNs after fill, then drop any leading NaNs

    if len(y) <= nlags: # Check if enough data points for any lag
        # raise KeyError('Not enough datapoints for the specified number of lags')
        print(f"Warning: Not enough data points ({len(y)}) for nlags={nlags}. Returning empty Series.")
        return pd.Series(dtype=float, name=name if name else y.name)


    results = {}
    for lag in range(1, nlags + 1):
        if len(y) > lag : # Ensure y has enough elements for current lag
            y_dependent = y.iloc[lag:]
            x_independent = y.shift(lag).iloc[lag:] # Ensure x_independent matches y_dependent length

            if not y_dependent.empty and not x_independent.empty:
                # Add constant for intercept
                x_independent_with_const = sm.add_constant(x_independent, prepend=False) # Using existing values as exog
                reg = sm.OLS(y_dependent, x_independent_with_const).fit()

                if param == 't':
                    results[lag] = reg.tvalues.iloc[0] # t-value of the lagged variable (not intercept)
                elif param == 'b':
                    results[lag] = reg.params.iloc[0] # beta of the lagged variable
                else:
                    raise ValueError("param must be 't' or 'b'")
            else:
                results[lag] = np.nan # Not enough data for this specific lag
        else:
            results[lag] = np.nan


    t_vals = pd.Series(results)
    t_vals.name = name if name else y.name
    return t_vals

def autocorr(x, t=1):
    """Calculates the autocorrelation of a series for a specific lag.

    Args:
        x (pandas.Series or numpy.ndarray): Input time series.
        t (int, optional): Lag for which to calculate autocorrelation. Defaults to 1.

    Returns:
        float: Autocorrelation value for the specified lag. Returns np.nan if
               insufficient data points after handling NaNs.
    """
    if isinstance(x, np.ndarray):
        # Ensure x is 1D
        if x.ndim > 1:
            x = x.flatten() # Or handle error: raise ValueError("Input array must be 1D")
        # Handle NaNs by removing them before correlation
        x_valid = x[~np.isnan(x)]
        if len(x_valid) <= t: return np.nan # Not enough data points
        return np.corrcoef(x_valid[t:], x_valid[:-t])[0, 1]
    elif isinstance(x, pd.Series):
        x_valid = x.dropna()
        if len(x_valid) <= t: # Check if enough non-NaN values for the given lag
            return np.nan
        return x_valid.autocorr(lag=t)


def get_tseries_autocor(series, nlags=40):
    """Calculates autocorrelations for a time series up to a specified number of lags.

    Args:
        series (pandas.Series or numpy.ndarray): Input time series.
        nlags (int, optional): Number of lags to calculate autocorrelations for. 
                               Defaults to 40.

    Returns:
        pandas.Series: Series of autocorrelation values, indexed by lag number.
                       Returns NaNs for lags greater than or equal to series length.

    Raises:
        TypeError: If `series` is a pandas DataFrame (requires 1-D input).
    """
    if isinstance(series, pd.DataFrame):
        raise TypeError('Input must be a 1-D array or Series, not a DataFrame.')
    
    if not isinstance(series, pd.Series):
        series = pd.Series(series) # Convert numpy array to Series for consistent handling

    series_cleaned = series.dropna()
    name = series_cleaned.name

    if len(series_cleaned) <= nlags:
        print(f"Warning: Length of series ({len(series_cleaned)}) is less than or equal to nlags ({nlags}). Autocorrelations may be NaN or unreliable.")
    
    auto_cor = {}
    for i in range(1, nlags + 1):
        if i < len(series_cleaned): # Check if lag is less than series length
             # Using pandas Series' autocorr method is often more robust and simpler
            auto_cor[i] = series_cleaned.autocorr(lag=i)
            # auto_cor[i] = autocorr(series_cleaned, i) # If using the custom autocorr defined above
        else:
            auto_cor[i] = np.nan # Lag is too large for the series length

    auto_series = pd.Series(auto_cor, name=name if name else "Autocorrelation")
    return auto_series

def get_ff_rolling_factors(strat, factors=None, rolling_window=36):
    """Calculates rolling Fama-French factor exposures (betas) for a strategy.

    Fetches Fama-French 5-factor data if not provided and calculates rolling
    OLS regression coefficients for the strategy returns against these factors.

    Args:
        strat (pandas.Series): Time series of strategy returns. Must have a DatetimeIndex.
        factors (pandas.DataFrame, optional): DataFrame of factor returns. If None,
                                              Fama-French 5 factors are fetched.
                                              Defaults to None.
        rolling_window (int, optional): Rolling window length in periods (typically months).
                                        Defaults to 36.

    Returns:
        pandas.DataFrame: DataFrame of rolling factor betas (coefficients), indexed by date.
                          Returns an empty DataFrame if data is insufficient.

    Raises:
        ImportError: If `factors` is None and pandas_datareader is not available.
        ConnectionError: If fetching Fama-French data fails.
        ValueError: If `strat` does not have a DatetimeIndex when factors need fetching,
                    or if `rolling_window` is invalid or too large for the data.
    """
    if web is None and factors is None:
        raise ImportError("pandas_datareader is not available and no factors provided. Cannot fetch Fama-French data.")

    if factors is None:
        # Ensure strat has a DatetimeIndex for Fama-French data fetching
        if not isinstance(strat.index, pd.DatetimeIndex):
            raise ValueError("Strategy series must have a DatetimeIndex to fetch Fama-French factors.")
        
        # Fetch Fama-French 5 factors
        try:
            factor_returns_monthly = web.DataReader('F-F_Research_Data_5_Factors_2x3', 'famafrench', start=strat.index[0], end=strat.index[-1])[0]
        except Exception as e:
            print(f"Error fetching Fama-French data: {e}")
            # Fallback or re-raise if critical
            # For now, let's assume if it fails, we can't proceed without factors
            raise ConnectionError("Failed to fetch Fama-French factors from web.") from e

        # Convert factor returns to numeric and scale (they are usually in percent)
        factor_returns = factor_returns_monthly / 100.0
        # Align factor returns to strategy's index (e.g., if strategy is daily but factors are monthly)
        # This requires careful consideration of how to map monthly factors to daily strategy returns.
        # Common practice: ffill monthly factors.
        # However, strat is usually monthly for FF analysis. If strat is daily, this needs adjustment.
        # For now, assuming 'strat' index is compatible or resampled to monthly if FF factors are monthly.
        # If strat is monthly, ensure its index is PeriodIndex or compatible for alignment.
        if isinstance(strat.index, pd.DatetimeIndex) and isinstance(factor_returns.index, pd.PeriodIndex):
             factor_returns.index = factor_returns.index.to_timestamp(how='end') # Align index types

        # Reindex factors to strat's index, filling missing values (e.g., for days not at month-end)
        # This assumes strat's index is the target frequency.
        factor_returns = factor_returns.reindex(strat.index, method='ffill') # Or 'pad'
        factor_returns = factor_returns.drop(['RF'], axis=1, errors='ignore') # RF usually separate
    else:
        # If factors are provided, use them directly. Assume they are already processed (scaled, RF removed).
        factor_returns = factors

    if rolling_window <= 0:
        raise ValueError("Rolling window must be a positive integer.")
    if rolling_window >= len(strat) or rolling_window >= len(factor_returns):
        raise ValueError(f'The window ({rolling_window}) cannot be greater than or equal to the length of input series (strat: {len(strat)}, factors: {len(factor_returns)}).')

    coef_ = {}
    # Ensure indices are aligned for iteration
    aligned_strat, aligned_factors = strat.align(factor_returns, join='inner')

    if len(aligned_strat) < rolling_window:
        print(f"Warning: Not enough overlapping data ({len(aligned_strat)}) for rolling window ({rolling_window}). Returning empty DataFrame.")
        return pd.DataFrame()

    for i in range(rolling_window, len(aligned_strat) + 1):
        # Window data
        window_strat = aligned_strat.iloc[i-rolling_window:i]
        window_factors = aligned_factors.iloc[i-rolling_window:i]
        
        # Add constant for intercept to the factors
        X = sm.add_constant(window_factors)
        model = sm.OLS(window_strat, X).fit()
        
        # Store coefficients, using the end date of the window as the index
        coef_[aligned_strat.index[i-1]] = model.params # model.params is a Series
        
    return pd.DataFrame.from_dict(coef_, orient='index') # Each dict item becomes a row


def cnvert_daily_to(index, target_freq='m'):
    """Converts a daily DatetimeIndex to other frequencies using actual last observed dates.

    This function is distinct from resample, as it finds the last date present 
    in the input data for each specified period (month, quarter, etc.).

    Args:
        index (pd.DatetimeIndex): Daily DatetimeIndex.
        target_freq (str, optional): Target frequency:
                                     'd' (daily), 
                                     'w' (weekly - ISO), 
                                     'm' (monthly), 
                                     'q' (quarterly), 
                                     'a' (annually). 
                                     Defaults to 'm'.

    Returns:
        pd.DatetimeIndex: DatetimeIndex with actual end-of-period dates found in the data.

    Raises:
        TypeError: If input is not a pandas DatetimeIndex.
        ValueError: If `target_freq` is not a supported frequency string.
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("Input must be a pandas DatetimeIndex.")
    
    if index.empty:
        return pd.DatetimeIndex([])

    # Ensure unique sorted dates for processing
    unique_sorted_index = pd.DatetimeIndex(sorted(list(set(index))))
    
    if unique_sorted_index.empty: # Should not happen if original index wasn't empty
        return pd.DatetimeIndex([])

    s = unique_sorted_index.to_series() # Work with a Series for groupby convenience
    target_freq = target_freq.lower()

    if target_freq == 'd':
        period_end_dates = unique_sorted_index
    elif target_freq in ['monthly', 'm']:
        period_end_dates = s.groupby([s.index.year, s.index.month]).max().values
    elif target_freq in ['quarterly', 'q']:
        period_end_dates = s.groupby([s.index.year, s.index.quarter]).max().values
    elif target_freq in ['annually', 'a']:
        period_end_dates = s.groupby(s.index.year).max().values
    elif target_freq in ['weekly', 'w']:
        # Use ISO calendar year and week for grouping
        isocal = s.index.isocalendar()
        period_end_dates = s.groupby([isocal.year, isocal.week]).max().values
    else:
        raise ValueError(f"target_freq value '{target_freq}' not supported. Use 'd', 'w', 'm', 'q', or 'a'.")
    
    # The .values from groupby().max() will be an array of Timestamps.
    # Convert to DatetimeIndex and sort, as groupby doesn't guarantee sorted output of values.
    return pd.DatetimeIndex(np.sort(period_end_dates))

def get_ann_ret(ret_series, freq='monthly'):
    """Calculates annualized returns from a series of returns.

    The function first calculates an equity line using `get_eq_line`, then
    resamples it annually, and finally computes the percentage change
    to get annualized returns.

    Args:
        ret_series (pandas.Series): Time series of returns. Must have a DatetimeIndex.
        freq (str, optional): The periodicity of the input `ret_series` 
                              (e.g., 'monthly', 'daily', 'weekly'). This is passed to
                              `get_eq_line`. Defaults to 'monthly'.

    Returns:
        pandas.Series: Time series of annualized returns, indexed by year (PeriodIndex).
    """
    cum_series = get_eq_line(ret_series, freq=freq) # Use freq here
    annual = cum_series.resample('A').last()
    # Ensure the first data point is included for pct_change calculation if it's the start of a year
    # or if the series doesn't start at the beginning of a year.
    # The original logic `annual.loc[ret_series.index[0]] = 1` might be problematic if ret_series.index[0]
    # is not an actual year-end, or if it overwrites a valid resampled value.
    # A robust way is to ensure the base for the first pct_change is correct.
    # If the series starts mid-year, the first 'annual' value from resample might be what we want.
    # Let's adjust to ensure the first period's return is calculated correctly.
    
    # Create a base point for the first period's calculation
    # If the series starts at a year boundary, resample handles it.
    # If it starts mid-year, the first resampled value is the cumulative up to that year-end.
    # The original logic might have intended to set the *very first* value of the *cumulative series* to 1,
    # which is usually handled by get_eq_line itself.
    
    # Let's refine the original logic slightly for robustness.
    # If the first index of ret_series is not already in annual.index (i.e., it's not a year-end)
    # and we want to ensure the return calculation starts from the very beginning of ret_series:
    # This typically means the `get_eq_line` should handle the initial base of 1.
    # The resampling for 'A' (annual) takes the *last* value of the year.

    # The original logic:
    # annual.loc[ret_series.index[0]] = 1
    # annual.sort_index(ascending= True, inplace = True)
    # This line above could introduce issues if ret_series.index[0] is not a year end.
    # A common approach for annual returns is to calculate them based on year-end values.
    # If `get_eq_line` correctly gives growth of $1, then `annual.pct_change()` should be fine.
    
    # Consider if `ret_series.index[0]` is a specific date like '2010-03-15'.
    # `annual.loc[pd.Timestamp('2010-03-15')] = 1` would add a non-year-end date.
    # If `get_eq_line` already starts the series at 1 (e.g. (1+ret_series).cumprod()),
    # then `annual = cum_series.resample('A').last()` is correct.
    # The pct_change will then calculate returns from one year-end to the next.

    # Let's assume get_eq_line provides a series that starts with 1 or represents growth of $1.
    # No specific adjustment like `annual.loc[ret_series.index[0]] = 1` should be needed here if
    # get_eq_line's output is consistent.

    annual_ret = annual.pct_change()
    annual_ret.index = annual_ret.index.to_period('A') # Convert DatetimeIndex to PeriodIndex for years
    annual_ret.dropna(inplace = True) # First value will be NaN after pct_change
    return annual_ret

def get_ts(df):
    """Calculates time-series lagged parameters for each column in a DataFrame.

    This function iterates over each column of the input DataFrame and applies
    the `get_lagged_params` function to it, typically to calculate autoregressive
    lagged t-statistics or coefficients.

    Args:
        df (pandas.DataFrame): A DataFrame where each column is a time series.
                               Index should be time-based (e.g., DatetimeIndex).

    Returns:
        pandas.DataFrame: A DataFrame where columns correspond to the input DataFrame's columns,
                          and rows are indexed by the lag number. Values are the
                          calculated lagged parameters (e.g., t-stats or betas from
                          `get_lagged_params`).
    """
    df_ts = {}
    for col_name in df.columns: # Iterate over column names for clarity
        # Assuming get_lagged_params is robust to series with all NaNs or insufficient data for all lags
        df_ts[col_name] = get_lagged_params(df[col_name], nlags=48) # get_lagged_params is in this file
    
    # Consolidate into a DataFrame
    # If some series in df_ts are shorter (e.g. due to insufficient data for all lags),
    # pd.DataFrame will handle this by filling with NaNs where appropriate.
    result_df = pd.DataFrame(df_ts)
    return result_df
