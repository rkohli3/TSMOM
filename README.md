# Time Series Momentum (TSMOM)

A Python implementation of the Time Series Momentum strategy based on Moskowitz, Ooi and Pedersen (2011), with extensions to Indian equities and ETF replication.

---

## What is Time Series Momentum?

Time Series Momentum (TSMOM) is a market anomaly where a security's own past returns predict its future returns. If past 12-month returns are positive, the position is long; if negative, the position is short. Returns are scaled by ex-ante volatility to target a consistent risk level across assets.

This is distinct from **cross-sectional momentum** (Jegadeesh & Titman, 1993), which ranks securities relative to each other. TSMOM only looks at a security's own history.

The strategy is applied across four asset classes:
- **Bonds** — sovereign bond futures
- **Equity Indexes** — index futures (S&P 500, FTSE, Nikkei, etc.)
- **Currencies** — FX futures
- **Commodities** — commodity futures

---

## Repository Structure

```
TSMOM/
├── core.py              # All computation: returns, volatility, momentum, performance, Fama-French
├── data.py              # Yahoo Finance price downloading (yfinance-based)
├── plots.py             # All Plotly visualisations (heatmaps, drawdown, annual returns, FF factors)
├── tsmom.py             # Backward-compatible re-export shim (existing notebooks import this)
├── notebooks/
│   ├── Momentum.ipynb         # Main CTA futures backtest (replicates Moskowitz et al.)
│   ├── MomentumIndia.ipynb    # Extension to Indian equities
│   ├── TSMOMCheck.ipynb       # Strategy validation and checks
│   ├── TSMOM_replicateETF.ipynb # ETF-based replication
│   ├── GARCH.ipynb            # GARCH volatility modelling
│   └── GARCH_code.ipynb       # GARCH implementation details
├── data/
│   ├── futures.csv            # Daily returns for 55 futures contracts (1985–present)
│   ├── futures_list.csv       # Futures metadata (asset class, name)
│   ├── spy_1985.csv           # S&P 500 prices from 1985 (benchmark)
│   └── ETFLists.xlsx          # ETF universe for replication
└── styles/
    └── custom.css             # Notebook styling
```

---

## Module Overview

### `core.py`
Pure computation — no data downloading, no plotting.

| Function | Description |
|---|---|
| `get_rets` | Compute returns from prices or equity line |
| `get_eq_line` | Cumulative equity line from returns |
| `get_exante_vol` | Ex-ante volatility (EWMA) |
| `get_inst_vol` | Instantaneous volatility |
| `scaled_rets` | Volatility-scaled returns |
| `get_excess_rets` | Returns minus risk-free rate (Fama-French RF) |
| `tsmom` | Core TSMOM signal for a single asset |
| `get_tsmom` | TSMOM signals across a universe |
| `get_tsmom_port` | Full portfolio construction (equal-weight or vol-scaled) |
| `get_long_short` | Long/short portfolio split |
| `get_stats` | Annualised mean, volatility, Sharpe ratio |
| `get_perf_att` | Performance attribution vs benchmark |
| `drawdown` | Max drawdown |
| `get_ff_rolling_factors` | Rolling Fama-French 5-factor regression |

### `data.py`
Yahoo Finance downloading via `yfinance` (`auto_adjust=False` for consistency).

| Function | Description |
|---|---|
| `get_yahoo_data` | Download prices for one or more tickers |
| `price_downloader` | Download universe + optional benchmark, forward-fill |

### `plots.py`
All Plotly visualisations.

| Function | Description |
|---|---|
| `get_monthly_heatmap` | Monthly returns heatmap |
| `get_monthly_hist` | Monthly returns histogram |
| `underwater` | Drawdown (underwater) chart |
| `get_ann_ret_plot` | Annual returns bar chart with volatility overlay |
| `plot_rolling_ff` | Rolling Fama-French factor loadings |

---

## Usage

### New code — import directly from modules
```python
from data import price_downloader
from core import get_tsmom_port, get_stats
from plots import get_monthly_heatmap, underwater
```

### Existing notebooks — import via shim (unchanged)
```python
import tsmom as tm
tm.get_tsmom_port(...)   # still works
```

---

## Dependencies

```
pandas
numpy
yfinance
pandas-datareader   # for Fama-French factor data
empyrical
plotly
matplotlib
statsmodels
arch                # for GARCH models
scipy
```

---

## References

- Moskowitz, T., Ooi, Y.H., Pedersen, L.H. (2011). *Time Series Momentum*. Journal of Financial Economics.
- Jegadeesh, N., Titman, S. (1993). *Returns to Buying Winners and Selling Losers*. Journal of Finance.
- Fama, E., French, K. (2015). *A Five-Factor Asset Pricing Model*. Journal of Financial Economics.

---

## Disclaimer

- Nothing here constitutes an offer to sell, a solicitation to buy, or a recommendation for any security or strategy
- Past performance is not indicative of future results
- All investments involve risk, including loss of principal
- Provided for informational and research purposes only
