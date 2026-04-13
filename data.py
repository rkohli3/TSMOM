"""
data.py — Data downloading for TSMOM.

Uses the stable yfinance-based downloader from AgamiApp (auto_adjust=False required
since yfinance 0.2.54 changed default behaviour).
"""

import datetime as dt

import pandas as pd
import yfinance


def get_yahoo_data(tickers, start=None, end=None, col='Adj Close', actions=True):
    """Download daily price data from Yahoo Finance.

    Parameters
    ----------
    tickers : str or list of str
        Yahoo Finance ticker symbol(s). Case-sensitive.
    start : str or datetime, optional
        Start date. Defaults to 2010-01-01.
    end : str or datetime, optional
        End date. Defaults to today.
    col : str or list of str
        Column(s) to return. Default 'Adj Close'.
    actions : bool
        Whether to include dividend/split data. Default True.

    Returns
    -------
    DataFrame indexed by date.
    """
    if end is None:
        end = dt.datetime.today()
    if start is None:
        start = dt.datetime(2010, 1, 1)

    # auto_adjust=False is required since yfinance 0.2.54
    data = yfinance.download(tickers, start=start, end=end,
                             actions=actions, auto_adjust=False)

    if isinstance(data.columns, pd.MultiIndex):
        result = data.loc[:, col]  # drops the col level, leaves Ticker level
        # Single ticker string → squeeze to Series
        if isinstance(tickers, str) or (hasattr(tickers, '__len__') and len(tickers) == 1):
            result = result.squeeze()
            result.name = tickers if isinstance(tickers, str) else tickers[0]
        return result
    return data[col]


def price_downloader(tickers, start=None, end=None,
                     use_benchmark=True, benchmark='^GSPC',
                     join='inner', col='Adj Close', actions=True):
    """Download prices for a universe and optionally prepend a benchmark.

    Parameters
    ----------
    tickers : str or list of str
    start, end : str or datetime, optional
    use_benchmark : bool
        If True, prepend the benchmark series and inner-join. Default True.
    benchmark : str
        Benchmark ticker. Default '^GSPC' (S&P 500).
    join : str
        How to align benchmark and universe. Default 'inner'.
    col : str
        Price column. Default 'Adj Close'.
    actions : bool

    Returns
    -------
    DataFrame with benchmark as the first column (if use_benchmark=True).
    """
    prices = get_yahoo_data(tickers, start=start, end=end,
                            col=col, actions=actions)
    prices = prices.ffill()

    if use_benchmark:
        bm_prices = get_yahoo_data(benchmark, start=start, end=end,
                                   col=col, actions=actions)
        prices = pd.concat([bm_prices, prices], axis=1, join=join).dropna(axis=1)

    return prices
