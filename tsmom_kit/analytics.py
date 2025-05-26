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

def get_exante_vol(series, alpha = 0.05, dtime = 'monthly', dtype = 'returns', com = None):
    """F: that provides annualized ex ante volatility based on the method of Exponentially Weighted Average\n
    This method is also know as the Risk Metrics, where the instantaneous volatility is based on past volatility\n
    with some decay

    params:
    -------

        series: pandas series
        com: center of mass (optional) (int)
        dtime: str, (optional), 'monthly', 'daily', 'weekly'

    returns:
        ex-ante volatility with time index"""
    if (isinstance(series, pd.core.series.Series)) and (isinstance(series.index, pd.DatetimeIndex)):
        pass
    else:
        raise NotImplementedError('Data Type not supported, should only be timeseries')
    if dtype == 'prices':
        # Assuming get_rets is available from .main_logic
        series = get_rets(series, kind = 'arth', freq = 'd')

    vol = series.ewm(alpha = alpha, com = com).std()
    ann_vol = vol * np.sqrt(261)

    if dtime == 'daily':
        ann_vol_prd = ann_vol
    elif dtime == 'monthly':
        ann_vol_prd = ann_vol.resample('BM').last().ffill()
    elif dtime == 'weekly':
        ann_vol_prd = ann_vol.resample('W-Fri').last().ffill()

    return ann_vol_prd

def get_inst_vol(y,
                 annualize,
                 x = None,
                 mean = 'Constant',
                 vol = 'Garch',
                 dist = 'normal',
                 data = 'prices',
                 freq = 'd',
                 ):
    """Fn: to calculate conditional volatility of an array using Garch:

    params
    --------------
    y : {numpy array, series, None}
        endogenous array of returns
    x : {numpy array, series, None}
        exogneous
    mean : str, optional
           Name of the mean model.  Currently supported options are: 'Constant',
           'Zero', 'ARX' and  'HARX'
    vol : str, optional
          model, currently supported, 'GARCH' (default),  'EGARCH', 'ARCH' and 'HARCH'
    dist : str, optional
           'normal' (default), 't', 'ged'

    returns
    ----------
    series of conditioanl volatility.
    """
    if (data == 'prices') or (data =='price'):
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

    # get the parameters. Here [1] means number of lags. This is only Garch(1,1)
    # These lines seem illustrative rather than functional for the direct output
    # omega = res.params['omega']
    # alpha = res.params['alpha[1]']
    # beta = res.params['beta[1]']

    # inst_vol = res.conditional_volatility * np.sqrt(252) # This was not used for ann_cond_vol
    # if isinstance(inst_vol, pd.core.series.Series):
    #     inst_vol.name = y.name
    # elif isinstance(inst_vol, np.ndarray):
    #     inst_vol = inst_vol

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

def drawdown(df, data = 'returns', ret_type = 'arth', ret_ = 'text'):
    """
    F: to calculate the drawdown of a timeseries price(s) or returns
    Params:
        df: DataFrame type containing timeseries returns or prices
        data: Either 'returns' or 'prices'. Default is 'returns'
        ret_type: If data is 'returns' then mention the type of rturns. Either 'log' or 'arth'. Default is arth
        ret_: Output type, default is 'text' tickformat
    Returns:
        DataFrame or float/str
         """
    if data == 'returns':
        if ret_type == 'arth':
            eq_line = (1 + df).cumprod()
        elif ret_type == 'log':
            eq_line = np.exp(df.cumsum())
    elif data == 'prices': # Corrected from 'if' to 'elif' for clarity
        eq_line = df
    else:
        raise ValueError("data parameter must be 'returns' or 'prices'")

    draw = 1 - eq_line.div(eq_line.cummax())
    max_drawdown = np.max(draw) # For DataFrames, this will be a Series of max drawdowns per column

    if isinstance(df, pd.DataFrame) and isinstance(max_drawdown, pd.Series):
         if ret_ != 'text':
            return max_drawdown # Return Series of max drawdowns
         else:
            # For text output with multiple columns, perhaps return a dict or formatted string
            return {col: "The maximum drawdown is: {0:,.2%}".format(val) for col, val in max_drawdown.items()}

    if ret_ != 'text':
        return max_drawdown # single float
    elif ret_ =='text':
        return ("The maximum drawdown is: {0:,.2%}").format(max_drawdown) # single string

def rolling_drawdown(df, data = 'returns', ret_type = 'arth'):
    """F: that calculates periodic drawdown.:
    params:
        df: takes in dataframe or pandas series
        data: (optional) str, prices or returns,
        ret_type: (optional) return type, log or arth"""
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

def get_stats(returns, dtime = 'monthly'):
    """Function to calulcte annualized mean, annualized volatility and annualized sharpe ratio
    params:
        returns: series or dataframe of retunrs
        dtime: (optional) 'monthly' or 'daily'
    returns:
        tuple of stats(mean, std and sharpe)"""
    if not (isinstance(returns, pd.Series) or isinstance(returns, pd.DataFrame)):
        try:
            # Attempt to convert to numpy array if not Series/DataFrame
            returns = np.array(returns)
        except:
            raise TypeError("Input must be a pandas Series, DataFrame, or convertible to a numpy array.")

    mean = np.mean(returns, axis=0) # np.mean works for both Series/DF and arrays
    std = np.std(returns, axis=0)   # np.std works for both Series/DF and arrays

    if dtime == 'monthly':
        mean = mean * 12
        std = std * np.sqrt(12)
    elif dtime == 'daily':
        mean = mean * 252
        std = std * np.sqrt(252)
    else:
        raise ValueError("dtime must be 'monthly' or 'daily'")

    # Handle division by zero for Sharpe ratio
    # If std is a Series/array, handle element-wise
    if isinstance(std, (np.ndarray, pd.Series)):
        sr = np.where(std == 0, np.nan, mean / std)
    else: # std is a scalar
        sr = mean / std if std != 0 else np.nan
    return (mean, std, sr)

def get_ytd(table, year = None): # year default to None to use current year
    """Function to calculate year to date performance:
    params:
    --------
    table: pd.series or dataframe with DatetimeIndex
    year: (optional) int, defaults to current year if None
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


def get_perf_att(series, bnchmark, rf = 0.03/12, freq = 'monthly'):
    """F: that provides performance statistic of the returns
    params
    -------
        series: daily or monthly returns (pd.Series)
        bnchmark: benchmark returns (pd.Series)
        rf: risk-free rate (float, period-aligned with series and bnchmark)
        freq: 'monthly' or 'daily'
    returns:
        dataframe of Strategy name and statistics"""
    port_mean, port_std, port_sr = get_stats(series, dtime = freq)

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
    formatted_perf = {k: '{:,.3f}'.format(v) if isinstance(v, (float, np.float_)) and k not in ['Max Drawdown'] else 
                         '{:,.2%}'.format(v) if k == 'Max Drawdown' and isinstance(v, (float, np.float_)) else
                         v 
                      for k, v in perf_dict.items()}

    perf_series = pd.Series(formatted_perf)
    perf_series.name = series.name if series.name else "Strategy"
    return perf_series.to_frame()

def get_lagged_params(y, param = 't', nlags = 24, name = None):
    """Function to calculate lagged parameters of a linear regression:
    params:
    --------
        y: series or numpy array
        param: (optional) `str` parameter to show, either 't' (t-statistic) or 'b' (beta coefficient)
        nlags: (optional) `int`
        name: None (optional) name of the series
    returns:
    ----------
        `pd.Series` of lagged params with index as number of lags"""
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
        if len(x_valid) <= t: return np.nan
        # Use pandas' built-in autocorr for Series, it handles NaNs by default (though we already dropped)
        # return x_valid.autocorr(lag=t) # pandas .autocorr() is simpler and robust
        # Or, stick to np.corrcoef for consistency with ndarray version:
        return np.corrcoef(x_valid.iloc[t:], x_valid.shift(t).dropna().iloc[t-t:])[0,1] # Careful with indexing if using shift
        # A more direct way with Series for np.corrcoef:
        shifted_x = x_valid.shift(t).dropna()
        original_x_aligned = x_valid.loc[shifted_x.index]
        if len(original_x_aligned) < 2 : return np.nan # Need at least 2 points for corrcoef
        return np.corrcoef(original_x_aligned, shifted_x)[0, 1]


def get_tseries_autocor(series, nlags = 40):
    """F: to calculate autocorrelations of a time series
    params:
    --------
        series: numpy array or series
        nlags: number of lags
    returns:
    --------
        pd.Series of autocorrelations"""
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

def get_ff_rolling_factors(strat, factors = None, rolling_window = 36):
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


def cnvert_daily_to(index, cnvrt_to = 'm'):
    """F: to convert a daily time series to monthly, weekly, quarterly, annually. Note this is not same as
    resample, as resample, take last, first, or middle values, even if they are not in the series.
    This function takes the dates witnessed empirically from the actual data.

    params:
    --------
        index: pd.DatetimeIndex (daily)
        cnvrt_to: 'str' (optional), 'm' (monthly), 'q' (quarterly), 'a' (annually), 'w' (weekly), 'd' (daily)
    returns:
    ---------
        pd.DatetimeIndex with the freq as mentioned, containing actual end-of-period dates found in the data."""

    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("Input must be a pandas DatetimeIndex.")
    
    if index.empty:
        return pd.DatetimeIndex([])

    cnvrt_to = cnvrt_to.lower()
    # Sort the index to ensure correct period selection
    t_day_index = pd.DatetimeIndex(sorted(list(set(index)))) # Unique sorted dates

    if t_day_index.empty: # Should not happen if original index was not empty, but good check
        return pd.DatetimeIndex([])
        
    # Group by year, then by the desired period within each year
    grouped_by_year = t_day_index.to_series().groupby(t_day_index.year)
    
    period_end_dates = []

    for year, days_in_year in grouped_by_year:
        if cnvrt_to == 'd': # Daily - just return the unique sorted index
            period_end_dates.extend(days_in_year.index)
            continue

        days_in_year_series = days_in_year.index.to_series() # Series of DatetimeIndex

        if cnvrt_to in ['monthly', 'm']:
            # Group by month, take the last day of each month present in the data
            monthly_groups = days_in_year_series.groupby(days_in_year_series.index.month)
            for _, month_days in monthly_groups:
                period_end_dates.append(month_days.max())
        elif cnvrt_to in ['quarterly', 'q']:
            quarterly_groups = days_in_year_series.groupby(days_in_year_series.index.quarter)
            for _, quarter_days in quarterly_groups:
                period_end_dates.append(quarter_days.max())
        elif cnvrt_to in ['annually', 'a']:
            # The last day of the year present in the data
            period_end_dates.append(days_in_year_series.max())
        elif cnvrt_to in ['weekly', 'w']:
            # For weekly, group by ISO week. Pandas isocalendar().week
            # Need to handle year changes correctly for ISO weeks if week spans year boundary.
            # Simpler: group by year and then week number.
            weekly_groups = days_in_year_series.groupby(days_in_year_series.index.isocalendar().week)
            for _, week_days in weekly_groups:
                period_end_dates.append(week_days.max())
        else:
            raise ValueError(f"cnvrt_to value '{cnvrt_to}' not supported. Use 'd', 'w', 'm', 'q', or 'a'.")

    if cnvrt_to == 'd': # If daily, all unique sorted dates are already collected by year.
        # This path will be taken if original loop was for 'd', just consolidate.
        # However, the logic is structured to build period_end_dates for other freqs.
        # If 'd', we can simply return the unique sorted index directly.
        return t_day_index

    # Remove duplicates that might arise if, e.g., multiple years have same month-end (not typical for this logic)
    # And sort them, as grouping by year and then period might not preserve overall chronological order perfectly.
    return pd.DatetimeIndex(sorted(list(set(period_end_dates))))

def get_ann_ret(ret_series, dtime = 'monthly'):
    cum_series = get_eq_line(ret_series, dtime = dtime) # Corrected: dtime was 'monthly' fixed
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
    """
    Calculates time-series related statistics, typically lagged parameters, for each column in a DataFrame.
    It iterates over each column (assumed to be a time series of a particular asset or factor)
    and applies `get_lagged_params` to it.

    Parameters:
    -----------
    df : pd.DataFrame
        A DataFrame where each column represents a time series for which lagged parameters
        (e.g., t-statistics or beta coefficients of autoregression) are to be calculated.
        The index of the DataFrame should be a DatetimeIndex or similar time-based index.

    Returns:
    --------
    pd.DataFrame
        A DataFrame where each column corresponds to an input column from `df`,
        and the rows are the calculated lagged parameters (e.g., t-stats or betas)
        for different numbers of lags (1 to 48, as per `get_lagged_params` default).
        The index of the output DataFrame represents the number of lags.
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
