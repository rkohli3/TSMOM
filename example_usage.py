import pandas as pd
import numpy as np
import datetime as dt
from tsmom_kit import (
    get_yahoo_data,
    get_rets,
    get_tsmom_port,
    get_stats,
    get_monthly_heatmap,
    get_eq_line # for heatmap example
)

def run_example_analysis():
    print("Running TSMOM Kit Example Analysis...")

    # 1. Fetch data
    tickers = ['SPY', 'AGG'] # Example tickers
    end_date = dt.datetime.today()
    start_date = end_date - dt.timedelta(days=5*365) # Approx 5 years of data
    
    print(f"Fetching data for {tickers} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
    try:
        price_data = get_yahoo_data(tickers, start=start_date, end=end_date)
        if price_data.empty:
            print("No data fetched, exiting example.")
            return
        if isinstance(price_data, pd.DataFrame) and not price_data.columns.nlevels > 1 and len(tickers) > 1:
            # If single column df for multiple tickers, yfinance might return single series.
            # This can happen if only one ticker returns data.
            # For simplicity, we'll just use SPY if that's the case.
            if 'SPY' in price_data.columns:
                price_data = price_data[['SPY']].copy()
                print("Using only SPY data due to single column fetch for multiple tickers.")
            else: # Or if SPY not present, use the first column
                price_data = price_data[[price_data.columns[0]]].copy()
                print(f"Using only {price_data.columns[0]} data due to single column fetch for multiple tickers.")


        print("Data fetched successfully:")
        print(price_data.head())
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    # 2. Calculate monthly returns
    print("\nCalculating monthly returns...")
    monthly_rets = {}
    # Determine if price_data is for a single ticker (Series) or multiple (DataFrame)
    # yfinance behavior:
    # - Single ticker: returns DataFrame with 'Adj Close', 'Close', etc.
    # - Multiple tickers: returns DataFrame with multi-index columns: ('Adj Close', 'SPY'), ('Adj Close', 'AGG'), ...
    # - If get_yahoo_data selects a single column like 'Adj Close' (default):
    #   - Single ticker: returns Series.
    #   - Multiple tickers: returns DataFrame with tickers as columns.

    if isinstance(price_data, pd.Series): # Single ticker, single column selected by get_yahoo_data
        ticker_name = price_data.name if price_data.name else tickers[0] # yfinance might not set name for Series
        monthly_rets[ticker_name] = get_rets(price_data.dropna(), freq='m', kind='arth')
    elif isinstance(price_data, pd.DataFrame):
        if price_data.columns.nlevels > 1: # Multi-index columns, e.g. ('Adj Close', 'SPY')
            # This case occurs if get_yahoo_data was modified to return all columns for multiple tickers
            unique_tickers = price_data.columns.get_level_values(1).unique()
            for ticker in unique_tickers:
                # Assuming 'Adj Close' is the column we want if multiple are present
                if ('Adj Close', ticker) in price_data.columns:
                     monthly_rets[ticker] = get_rets(price_data['Adj Close'][ticker].dropna(), freq='m', kind='arth')
                else: # Fallback if 'Adj Close' is not there for some reason
                    monthly_rets[ticker] = get_rets(price_data.xs(ticker, level=1, axis=1).iloc[:,0].dropna(), freq='m', kind='arth')
        else: # Single-level columns (tickers are columns), e.g. result of get_yahoo_data(col='Adj Close')
            for ticker in price_data.columns:
                monthly_rets[ticker] = get_rets(price_data[ticker].dropna(), freq='m', kind='arth')
    else:
        print("Price data is not in expected format (Series or DataFrame).")
        return


    if not monthly_rets:
        print("Could not calculate monthly returns.")
        return
    
    sample_ticker = list(monthly_rets.keys())[0]
    print(f"Sample monthly returns for {sample_ticker}:")
    print(monthly_rets[sample_ticker].head())

    # 3. Prepare data for TSMOM (requires cumulative returns and volatility)
    dummy_mnth_cum = {}
    dummy_mnth_vol = {}

    for ticker, rets_series in monthly_rets.items():
        if not rets_series.empty:
            dummy_mnth_cum[ticker] = get_eq_line(rets_series, data='returns', dtime='monthly')
            # Dummy volatility: constant 10% annualized for simplicity, converted to monthly
            dummy_mnth_vol[ticker] = pd.Series(0.10 / np.sqrt(12), index=rets_series.index) 
            dummy_mnth_vol[ticker].name = ticker # Ensure series has a name
        else: 
            dummy_mnth_cum[ticker] = pd.Series(dtype=float, name=ticker)
            dummy_mnth_vol[ticker] = pd.Series(dtype=float, name=ticker)

    dummy_mnth_cum_df = pd.DataFrame(dummy_mnth_cum)
    dummy_mnth_vol_df = pd.DataFrame(dummy_mnth_vol)
            
    valid_cols = dummy_mnth_cum_df.columns[dummy_mnth_cum_df.notna().any()].intersection(
                   dummy_mnth_vol_df.columns[dummy_mnth_vol_df.notna().any()])
    
    dummy_mnth_cum_df = dummy_mnth_cum_df[valid_cols]
    dummy_mnth_vol_df = dummy_mnth_vol_df[valid_cols]

    if not dummy_mnth_cum_df.empty and not dummy_mnth_vol_df.empty:
        print("\nRunning TSMOM portfolio calculation (simplified example)...")
        # Ensure mnth_vol and mnth_cum DataFrames passed to get_tsmom_port have actual series for each ticker column
        # The tsmom function inside get_tsmom_port expects series.name and access by [ast]
        
        # Re-construct DataFrames to ensure they contain Series, not just values, if they became single-row/col
        # This is more of a safeguard for how tsmom expects input.
        mnth_vol_for_tsmom = pd.DataFrame({col: dummy_mnth_vol_df[col] for col in dummy_mnth_vol_df.columns})
        mnth_cum_for_tsmom = pd.DataFrame({col: dummy_mnth_cum_df[col] for col in dummy_mnth_cum_df.columns})

        tsmom_portfolio_results = get_tsmom_port(mnth_vol_for_tsmom, mnth_cum_for_tsmom, flag=False, lookback=6)
        print("TSMOM Portfolio Results:")
        print(tsmom_portfolio_results.head())

        # 4. Get stats for the TSMOM strategy
        if not tsmom_portfolio_results.empty and 'TSMOM' in tsmom_portfolio_results.columns:
            print("\nCalculating performance statistics for TSMOM...")
            tsmom_returns = tsmom_portfolio_results['TSMOM'].dropna()
            if not tsmom_returns.empty:
                mean, std, sr = get_stats(tsmom_returns, dtime='monthly')
                print(f"TSMOM Strategy: Ann. Mean: {mean:.2%}, Ann. Std: {std:.2%}, Sharpe: {sr:.2f}")
            else:
                print("TSMOM returns series is empty, cannot calculate stats.")
        else:
            print("TSMOM column not found or results are empty.")
        
        # 5. Example of plotting
        if sample_ticker in monthly_rets and not monthly_rets[sample_ticker].empty:
            print(f"\nGenerating heatmap for {sample_ticker} (this will open in a browser or show in-line depending on environment)...")
            plot_series = monthly_rets[sample_ticker].copy() # Use .copy() to avoid SettingWithCopyWarning
            plot_series.name = sample_ticker 
            # Actual plotting call:
            # get_monthly_heatmap(plot_series, cmap='RdYlGn', plt_type='plot', filename=f'{sample_ticker}_heatmap.html')
            # print(f"Plot saved to {sample_ticker}_heatmap.html (if plt_type='plot')")
            print("Plotting example: get_monthly_heatmap call would be here. Skipping actual plot generation in this script for CI/automated environments.")
        else:
            print(f"Skipping heatmap for {sample_ticker} as its monthly returns are empty.")
    else:
        print("No valid data to run TSMOM portfolio after processing monthly returns and dummy data.")

    print("\nExample Analysis Finished.")

if __name__ == '__main__':
    run_example_analysis()
