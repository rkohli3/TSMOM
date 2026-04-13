"""
tsmom.py — Backward-compatible re-export shim.

All logic has been split into:
  data.py  — Yahoo Finance data downloading
  core.py  — Pure computation (returns, volatility, momentum, performance)
  plots.py — All plotting functions

Import from those modules directly in new code.
This file exists so existing notebooks continue to work without changes:
    import tsmom as tm
    tm.get_yahoo_data(...)   # still works
"""

from data  import get_yahoo_data, price_downloader          # noqa: F401
from core  import (                                          # noqa: F401
    get_rets, get_eq_line, cum_pfmnce, get_ann_ret, get_ytd,
    cnvert_daily_to,
    get_exante_vol, get_inst_vol, scaled_rets,
    get_excess_rets,
    autocorr, get_tseries_autocor, get_lagged_params, get_ts,
    tsmom, get_tsmom, get_tsmom_port, get_long_short,
    drawdown, rolling_drawdown, get_stats, get_perf_att,
    get_ff_rolling_factors,
)
from plots import (                                          # noqa: F401
    matplotlib_to_plotly, plt_cscale,
    get_monthly_heatmap, get_monthly_hist,
    underwater, get_ann_ret_plot, plot_rolling_ff,
)
