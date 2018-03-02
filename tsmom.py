import pandas as pd
import numpy as np
from pandas_datareader import data as web
import plotly.plotly as py
import plotly.tools as tls
from plotly.graph_objs import *
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import datetime as dt
import cufflinks as cf
cf.go_offline()
from jupyterthemes import jtplot
#jtplot.style()
import arch
import statsmodels.tsa.api as smt
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import pyfolio as pf
import pytz
from yahoo import YahooDailyReader
#sns.set_style('white')
import requests
import empyrical
import plotly.figure_factory as ff
import matplotlib

def get_adj_close(tickers,start, end, source = 'yahoo'):
    """F: to get adjusted close columns for a list of tickers using Paython's web data dreader

    params
    -------

        tickers: list of tickers
        start: `str` or `datetime` object
        end: `str` or `datetime` object
        source: (optional) str
                takes input as yahoo, google

    returns:
    ---------
    pandas panel of Adj Close price if input is yahoo. Google has some errors with Adjustd Close
    """
    panel = web.DataReader(tickers, source.lower(), start, end)
    if source == 'yahoo':
        table = panel['Adj Close']
    elif source == 'google':
        table = panel['Close']
    return table.sort_index(ascending = True)


# def drawdown(df_returns, ret_type = 'log'):
#     if ret_type == 'log':
#         cum_returns = np.exp(df_returns.cumsum())

#     elif ret_type == 'arth':
#         cum_returns = (1 + df_returns).cumprod()
#     draw = 1 - cum_returns.div(cum_returns.cummax())
#     max_drawdown = np.max(draw)
# #     print ("The maximum drawdown is:")
# #     print (max_drawdown.apply(lambda x: "{0:,.2%}".format(x)) )
#     return ("The maximum drawdown is: {0:,.2}%").format(max_drawdown)

def drawdown(df, data = 'returns', ret_type = 'arth', ret_ = 'text'):
    if data == 'returns':
        if ret_type == 'arth':
            eq_line = (1 + df).cumprod()
        elif ret_type == 'log':
            eq_line = np.exp(df.cumsum())
    if data == 'prices':
        eq_line = df

    draw = 1 - eq_line.div(eq_line.cummax())
    max_drawdown = np.max(draw)
#     if isinstance(max_drawdown, pd.core.series.Series):
#         if ret_ != 'text':
#             return max_drawdown.apply(lambda x: '{:,.2%}'.format(x))
    if ret_ != 'text':
        return max_drawdown
    elif ret_ =='text':
        return ("The maximum drawdown is: {0:,.2%}").format(max_drawdown)

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
    if data == 'prices':
        eq_line = df

    draw = eq_line.div(eq_line.cummax()) - 1
#     max_drawdown = np.max(draw)
    return draw


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

def get_exante_vol(series, com = 60, dtime = 'monthly', dtype = 'returns'):

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
        series = get_rets(series, kind = 'arth', freq = 'd')

    vol = series.ewm(com = com).std()
    ann_vol = vol * np.sqrt(261)

    if dtime == 'daily':
        ann_vol_prd = ann_vol

    elif dtime == 'monthly':
        ann_vol_prd = ann_vol.resample('BM').last().ffill()

    elif dtime == 'weekly':
        ann_vol_prd = ann_vol.resample('W-Fri').last().ffill()


    return ann_vol_prd

def cnvert_daily_to(index, cnvrt_to = 'm'):
    """F: to convert a daily time series to monthly, weekly, quarterly, annually. Note this is not same as
    resameple, as resample, take last, first, or middle values, even if they are not in the series.
    This function takes the dates witnessed empirically

    params:
    --------

        index: datetime index
        cnvrt_to: 'str' (optional), currenty supported, 'daily', 'monthyl', 'quarterly', 'annually'

    returns:
    ---------
    index with the freq as mentioned"""


    cnvrt_to = cnvrt_to.lower()
    t_day_index = pd.DatetimeIndex(sorted(index))
    t_years = t_day_index.groupby(t_day_index.year)
    f_date = t_day_index[0]
    ann_dt = [f_date]
    qrter_dt = [f_date]
    mnthly_dt = [f_date]
    weekly_dt = [f_date]

    for yr in t_years.keys():
        yr_end = pd.DatetimeIndex(t_years[yr]).groupby(pd.DatetimeIndex(t_years[yr]).month)
        qrter_end = pd.DatetimeIndex(t_years[yr]).groupby(pd.DatetimeIndex(t_years[yr]).quarter)
        week_end = pd.DatetimeIndex(t_years[yr]).groupby(pd.DatetimeIndex(t_years[yr]).week)
        ann_dt.append(max(yr_end[max(yr_end)]))
        for q in qrter_end.keys():
            qrter_dt.append(max(qrter_end[q]))
        for m in yr_end.keys():
            mnthly_dt.append(max(yr_end[m]))
        for w, val in week_end.items():
            weekly_dt.append(max(val))


    if (cnvrt_to == 'monthly')| (cnvrt_to == 'm'):
        return mnthly_dt
    elif (cnvrt_to == 'quarterly')|(cnvrt_to == 'q'):
        return qrter_dt
    elif (cnvrt_to == 'annually')|(cnvrt_to == 'a'):
        return ann_dt
    elif (cnvrt_to == 'weekly')|(cnvrt_to == 'w'):
        return weekly_dt
    elif (cnvrt_to == 'daily')|(cnvrt_to == 'd'):
        return index

def get_ytd(table, year = 2017):
    """Function to calculate year to date performance:
    params:
    --------
    table: pd.series or dataframe:
    year: (optional) int
    """
    this_year = dt.date.today().year
    grouped = table.index.groupby(table.index.year)
    #frst_day = min(grouped[this_year])
    index = grouped[this_year]
    pct = (table.loc[index].iloc[-1]/table.loc[grouped[year]].iloc[0]) - 1
#     return (pct.apply(lambda x: "{0:,.3f}".format(x*100)))
    return pct#.apply(lambda x: "{0:,.3f}".format(x))

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


    rf = rf.reindex(rets.index, fill = 'pad')
    cond_vol = rets.apply(lambda x: get_inst_vol(x, annualize= freq))
    scal_rets = rets/cond_vol.shift(-1)
    scal_rets.iloc[-1, :] = rets.mean()/rets.std()
    return scal_rets

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
        rets = data
    rets = get_rets(data, kind = kind, freq = freq)
    start_date = rets.index[0]
    if freq == 'm':
        rets.index = rets.index.to_period(freq)

    if isinstance(rets, pd.core.frame.DataFrame):
        rets = rets.iloc[1:,:]
    elif isinstance(rets, pd.core.series.Series):
        rets = rets.iloc[1:]


    if freq == 'd':
        rf = (web.DataReader("F-F_Research_Data_Factors_daily",
                             "famafrench",
                             start= start_date)[0]['RF'])/100
    elif freq == 'w':
        rf =(web.DataReader("F-F_Research_Data_Factors_weekly",
                            "famafrench",
                            start= start_date)[0]['RF'])/100
    elif freq == 'm':
        rf = (web.DataReader("F-F_Research_Data_Factors",
                             "famafrench",
                             start= start_date)[0]['RF'])/100
    rf = rf.reindex(rets.index, method = 'pad')
    ex_rets = rets.sub(rf, axis = 0)
    return ex_rets

def get_inst_vol(y,
                 annualize,
                 x = None,
                 mean = 'Constant',
                 vol = 'Garch',
                 dist = 'normal',
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
    if isinstance(y, pd.core.series.Series):
        ## remove nan.
        y = y.dropna()
    elif isinstance(y, np.ndarray):
        y = y[~np.isnan(y)]

    # provide a model
    model = arch.arch_model(y * 100, mean = 'constant', vol = 'Garch')

    # fit the model
    res = model.fit(update_freq= 5)

    # get the parameters. Here [1] means number of lags. This is only Garch(1,1)
    omega = res.params['omega']
    alpha = res.params['alpha[1]']
    beta = res.params['beta[1]']

    inst_vol = res.conditional_volatility * np.sqrt(252)
    if isinstance(inst_vol, pd.core.series.Series):
        inst_vol.name = y.name
    elif isinstance(inst_vol, np.ndarray):
        inst_vol = inst_vol

    # more interested in conditional vol
    if annualize.lower() == 'd':
        ann_cond_vol = res.conditional_volatility * np.sqrt(252)
    elif annualize.lower() == 'm':
        ann_cond_vol = res.conditional_volatility * np.sqrt(12)
    elif annualize.lower() == 'w':
        ann_cond_vol = res.conditional_volatility * np.sqrt(52)
    return ann_cond_vol * 0.01

def get_lagged_params(y, param = 't', nlags = 24, name = None):

    """Function to calculate lagged parameters of a linear regression:

    params:
    --------
        y: series or numpy array
        param: (optiona) `str` parameter to show, either 't' or 'b'
        nlags: (optional) `int`
        name: None (optional) name of the series

    returns:
    ----------
        `pd.series` of lagged params with index as number of lags"""
    if isinstance(y, pd.core.series.Series):
        y = y
    elif isinstance(y, np.ndarray):
        y = pd.Series(y)

    y.fillna(method = 'pad', inplace = True)
    y.dropna(inplace = True)
    if len(y) > nlags:
        t_stats = {}
        betas = {}
        for lag in range(1, nlags + 1):
            reg = sm.OLS(y.iloc[lag:], y.shift(lag).dropna()).fit()
            if param == 't':
                t_stats[lag] = reg.tvalues[0]
            elif param == 'b':
                t_stats[lag] = reg.params[0]
        t_vals = pd.Series(t_stats)
        t_vals.name = name
    else:
        raise KeyError('Not enough datapoints for lags')
    return pd.Series(t_vals)

# def get_lagged_betas(y, nlags = 24)

def autocorr(x, t=1):
    return np.corrcoef(x.iloc[t:], x.shift(t).dropna())


def get_tseries_autocor(series, nlags = 40):
    """F: to calculate autocorrelations of a time series
    params:

        series: numpy array or series
        nlags: number of lags

    returns:
        autocorrelation"""
    if isinstance(series, pd.core.frame.DataFrame):
        raise TypeError('Must be 1-d araay')
    elif isinstance(series, np.ndarray):
        series = series[~np.isnan(series)]
    elif isinstance(series, pd.core.series.Series):
        series.dropna(inplace = True)
        name = series.name

    auto_cor = {}
    for i in range(1, nlags + 1):
        auto_cor[i] = autocorr(series, i)[0, 1]
    auto = pd.Series(auto_cor, name= name)
    return auto

# get_tseries_autocor(logrets['SPY']).plot.bar()


def tsmom(series, mnth_vol, mnth_cum, tolerance = 0, vol_flag = False, scale = 0.4, lookback = 12):

    """Function to calculate Time Series Momentum returns on a time series.
    params:
        series: used for name purpose only, provide a series with the name of the ticker
        tolerance: (optional) -1 < x < 1, for signal processing, x < 0 is loss thus short the asst and vice-versa
        vol_flag: (optional) Boolean default is False,
        scale: (optional) volatility scaling parameter
        lookback: (optional) int, lookback months

    returns:
        time series momentum returns"""
    ast = series.name
    df = pd.concat([mnth_vol[ast], mnth_cum[ast], mnth_cum[ast].pct_change(lookback)],
                      axis = 1,
                      keys = ([ast + '_vol', ast + '_cum', ast + '_lookback']))
    cum_col = df[ast + '_cum']
    vol_col = df[ast + '_vol']
    lback = df[ast + '_lookback']

    pnl_dict = {}
    lev_dict = {}
    for k, v in enumerate(lback):
        if k <= lookback:
            continue
        if vol_flag == True:
            leverage = (scale/vol_col[k-1])
            if lback.iloc[k-1] > tolerance:
                pnl_dict[lback.index[k]] = ((cum_col.iloc[k]/float(cum_col.iloc[k-1])) - 1) * leverage
                lev_dict[lback.index[k]] = leverage
            elif lback.iloc[k-1] < tolerance:
                pnl_dict[lback.index[k]] = ((cum_col.iloc[k-1]/float(cum_col.iloc[k])) - 1) * leverage
                lev_dict[lback.index[k]] = leverage
        elif vol_flag == False:
            leverage = 1
            if lback.iloc[k-1] > tolerance:
                pnl_dict[lback.index[k]] = ((cum_col.iloc[k]/float(cum_col.iloc[k-1])) - 1)
                lev_dict[lback.index[k]] = leverage
            elif lback.iloc[k-1] < tolerance:
                pnl_dict[lback.index[k]] = ((cum_col.iloc[k-1]/float(cum_col.iloc[k])) - 1)
                lev_dict[lback.index[k]] = leverage
    new_lev = pd.Series(lev_dict)
    new_series = pd.Series(pnl_dict)
    new_series.name = ast
    new_lev.name = ast + 'Leverage'
    return new_series, new_lev

def get_stats(returns, dtime = 'monthly'):
    """Function to calulcte annualized mean, annualized volatility and annualized sharpe ratio
    params:

        returns: series or dataframe of retunrs
        dtime: (optional) 'monthly' or 'daily'

    returns:
        tuple of stats(mean, std and sharpe)"""
    if (isinstance(returns, pd.core.series.Series)) | (isinstance(returns, pd.core.frame.DataFrame)):
        mean = returns.mean()
        std = returns.std()
    else:
        try:
            mean = np.mean(returns)
            std = np.std(returns)
        except:
            raise TypeError
    if dtime == 'monthly':
        mean = mean * 12
        std = std * np.sqrt(12)
    elif dtime == 'daily':
        mean = mean * 252
        std = std * np.sqrt(252)

    sr = mean/std

    return (mean, std, sr)

def get_ts(df):
    df_ts = {}
    for i in df:
        df_ts[i] = ((get_lagged_params(df.loc[:, i], nlags = 48)))
    df_ts_df = (pd.DataFrame(df_ts))
    return df_ts_df

def get_tsmom(mnth_vol, mnth_cum, flag = False, scale = 0.20, lookback = 12):
    pnls = mnth_cum.apply(lambda x: tsmom(x, mnth_vol, mnth_cum, scale = scale, vol_flag= flag, lookback= lookback)[0])
    lev = mnth_cum.apply(lambda x: tsmom(x, mnth_vol, mnth_cum, scale = scale, vol_flag= flag, lookback= lookback)[1])
    port_pnl = pnls.mean(axis = 1)
    if flag == False:
        port_pnl.name = 'TSMOM'
    elif flag == True:
        port_pnl.name = 'TSMOM VolScale'

#     strat_df = port_pnl.to_frame
    lev_mean = lev.mean(axis =1)
    lev_mean = lev_mean.rolling(12).mean()
    lev_mean.name = 'Leverage'

    return pd.concat([port_pnl, lev_mean], axis = 1)


# empyrical.alpha(port_pnl, spy_rets, period = 'monthly')

def get_perf_att(series, freq = 'monthly'):
    """F: that provides performance statistic of the returns
    params
    -------

        series: daily or monthly returns

    returns:
        dataframe of Strategy name and statistics"""
    port_mean, port_std, port_sr = (get_stats(series, dtime = freq))
    perf = pd.Series({'Annualized_Mean' : '{:,.2f}'.format(round(port_mean, 3)),
                      'Annualized_Volatility': round(port_std, 3),
                      'Sharpe Ratio' : round(port_sr, 3),
                      'Calmar Ratio' : round(empyrical.calmar_ratio(series,
                                                                    period = freq),
                                             3),
                      'Alpha' : round(empyrical.alpha(series,
                                                      spy_rets,
                                                      period = freq),
                                      3),
                      'Beta':  round(empyrical.beta(series,
                                                    spy_rets),
                                     3),
                      'Max Drawdown':  '{:,.2%}'.format(drawdown(series, ret_ = 'nottext')),
                      'Sortino Ratio': round(empyrical.sortino_ratio(series,
                                                                     required_return= 0.03/12,
                                                                     period = freq
                                                                    ),
                                              3),
                     },
                    )
    perf.name = series.name
    return perf.to_frame()


def matplotlib_to_plotly(cmap):
    """Converts a matplotlib colormap to plotly colormap or colorscale, which is customized

    params:
        cmap: str, valid cmap in matplotlib"""

    pl_entries = 255
    _cmap = matplotlib.cm.get_cmap(cmap)
    h = 1.0/(pl_entries-1)
    pl_colorscale = []

    for k in range(pl_entries):
        C = list(map(np.uint8, np.array(_cmap(k*h)[:3])*pl_entries))
        pl_colorscale.append([k*h, 'rgb'+str((C[0], C[1], C[2]))])

    return pl_colorscale

def get_monthly_heatmap(returns,
                        cmap,
                        font_size = 10,
                        yr_from = None,
                        yr_to = None,
                        cnvrt = 'monthly'):

    """F: to plot heatmap of monthly returns:

    params:

        returns: str, daily or monthly returns, ideally a series with datetime index
        cmap: (optional)str, eg 'RdYlGn'
        font_size: (optional) font_size of annotations
        yr_from: (optional) Heatmap year from
        yr_to: (optional) Heatmap year to
        cnvrt = (optional) str, convert returns to
        """
    cscale = matplotlib_to_plotly(cmap)
    if yr_to is None:
        yr_to = returns.index[-1].year
    if yr_from is None:
        yr_from = returns.index[0].year
    grid = empyrical.aggregate_returns(returns, convert_to = 'monthly').unstack().fillna(0).round(4) * 100
    grid = grid.loc[yr_from:yr_to,:]
    z = grid.as_matrix()
    y = grid.index.values.tolist()
    x = grid.columns.values.tolist()

    z = grid.values.tolist()
    z.reverse()
    z_text = np.round(z, 3)

    fighm = ff.create_annotated_heatmap(z,
                                        x = x,
                                        y= y[::-1],
                                        annotation_text= z_text,
                                        colorscale = cscale,
                                        reversescale = True,
                                        hoverinfo = "y+z",
                                        showscale = True,
                                        font_colors= ['white', 'black'])
    for i in range(len(fighm.layout.annotations)):
        fighm.layout.annotations[i].font.size = font_size

    fighm.layout.title = 'Heatmap of Monthly Returns for {0} from {1} - {2}'.format(returns.name,
                                                                                    y[0],
                                                                                    y[-1])
    fighm['layout']['yaxis']['title'] = 'Years'
    fighm['layout']['yaxis']['dtick'] = 3
    fighm['layout']['yaxis']['tick0'] = 2
    # fighm.layout.xaxis.title = 'Months'
    return iplot(fighm, show_link= False, image_height= 1200)

def get_monthly_hist(series, height = 400, width = 900):
    """F: to plot histogram of monthly returns

    params:
        series: monthyl or daily returns
        height: (optional) int
        width: (optional)

    returns:
        plotly iplotint"""
    if len(series) < 500:
        nbins = int(len(series)/2)
    else:
        nbins = int(len(series)/4)
    hist = series.iplot(kind = 'histogram',
                       colors = '#5f4a52',
                       vline = series.mean(),
                       bins = nbins,
                       asFigure= True,
                       layout_update = {'plot_bgcolor': 'white',
                                        'paper_bgcolor': 'white',
                                        'title': 'Monthly Returns Histogram for {}'.format(series.name),
                                        'margin': dict(t = 40, pad = -40),
                                        'width': width,
                                        'height': height,
                                        'xaxis' : dict(title = 'Returns',
                                                      showgrid = False,
                                                      showticklabels = True,
                                                      zeroline = True,
                                                       zerolinewidth = 3,
                                                      color = 'black',
                                                       range = [-0.06, 0.06],
                                                      hoverformat = '0.2%'
                                                     ),
                                       'yaxis' : dict(title = 'Frequency',
                                                      showgrid = False,
                                                      showticklabels = True,
                                                      zeroline = True,
                                                      zerolinewidth = 1,
                                                      color = 'black'
                                                     ),
                                        'shapes' : [dict(type = 'line',

                                                        x0 = series.mean(),
                                                        x1 = series.mean(),
                                                        y0 = 0,
                                                        y1 = 1,
                                                        yref = 'paper',
                                                        line = dict(dash = 'dash' + 'dot',
                                                                    width = 4,
                                                                    color = 'orange'),
                                                        )
                                                   ],
                                        'showlegend' : True,
                                        'legend' : dict(x = 0.85,
                                                        y = 0.9,
                                                        bgcolor = 'white'),
                                      })
    hist.layout.xaxis.tickformat = '0.00%'
    return iplot(hist, show_link= False)



def underwater(series, s_name = None, width = 900, height = 400, color = 'red'):
    if s_name is not None:
        name = s_name
    name = series.name
    eqspy = (1+series).cumprod()
    dd = (eqspy/eqspy.cummax() - 1) * 100
    dd = dd.apply(lambda x: np.round(x, 2))
    pyfig =  dd.iplot(kind = 'area',
                      fill = 'True',
                      colors = color,
                      asFigure = True,
                      title = 'Underwater plot for {}'.format(name),
                      layout_update = {'plot_bgcolor': 'white',
                                       'paper_bgcolor': 'white',
    #                                    'hovermode': 'closest',
                                       'margin': dict(t = 40, pad = -40),
                                       'width': 900,
                                       'height': 400,
                                       'xaxis' : dict(title = 'Dates',
                                                      showgrid = False,
                                                      showticklabels = True,
                                                      zeroline = True,
                                                      color = 'black',
                                                      hoverformat = '%A, %b %d %Y '
                                                     ),
                                       'yaxis' : dict(title = 'Drawdown',
                                                      showgrid = False,
                                                      showticklabels = True,
                                                      zeroline = True,
                                                      color = 'black'
                                                     )
                                      }
                     )
    return iplot(pyfig, show_link = False)
