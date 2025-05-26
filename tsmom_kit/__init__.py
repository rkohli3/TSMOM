# Expose key functions from main_logic
from .main_logic import (
    get_yahoo_data,
    get_rets,
    cum_pfmnce,
    get_eq_line,
    get_excess_rets,
    scaled_rets,
    tsmom,
    get_tsmom,
    get_tsmom_port,
    get_long_short
)

# Expose key functions from analytics
from .analytics import (
    get_exante_vol,
    get_inst_vol,
    drawdown,
    rolling_drawdown,
    get_stats,
    get_ytd,
    get_perf_att,
    get_lagged_params,
    autocorr,
    get_tseries_autocor,
    get_ff_rolling_factors,
    cnvert_daily_to,
    get_ann_ret,
    get_ts # Added get_ts
)

# Expose key functions from plotting
from .plotting import (
    get_monthly_heatmap,
    get_monthly_hist,
    underwater,
    get_ann_ret_plot,
    plot_rolling_ff,
    matplotlib_to_plotly,
    plt_cscale
)

print("TSMOM Kit initialized.") # Optional: for user feedback on import
