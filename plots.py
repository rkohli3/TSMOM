"""
plots.py — All plotting functions for TSMOM.

Depends on core.py for helper computations (get_eq_line, get_ann_ret, get_ff_rolling_factors).
No data downloading here.
"""

import calendar

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff
import plotly.tools as tls
from plotly.graph_objs import Bar, Figure, Scatter
from plotly.offline import iplot, plot

try:
    import chart_studio.plotly as py
    _CHART_STUDIO = True
except ImportError:
    _CHART_STUDIO = False

import empyrical

from core import get_eq_line, get_ann_ret, get_ff_rolling_factors


# ── Colorscale Helpers ────────────────────────────────────────────────────────

def matplotlib_to_plotly(cmap, vmin=0, vmax=255):
    """Convert a matplotlib colormap to a Plotly colorscale."""
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    _cmap = matplotlib.cm.get_cmap(cmap)
    pl_entries = 255
    h = 1 / (pl_entries - 1)
    return [
        [k * h, 'rgb' + str(tuple(map(np.uint8, np.array(_cmap(norm(k))[:3]) * pl_entries)))]
        for k in range(pl_entries)
    ]


def plt_cscale(cmap):
    """Alternative matplotlib → Plotly colorscale conversion."""
    _cmap = matplotlib.cm.get_cmap(cmap)
    norm = matplotlib.colors.Normalize(vmin=-100, vmax=100)
    return [matplotlib.colors.colorConverter.to_rgb(_cmap(norm(i))) for i in range(255)]


# ── Monthly Returns Heatmap ───────────────────────────────────────────────────

def get_monthly_heatmap(returns, cmap='RdYlGn', font_size=10,
                        yr_from=None, yr_to=None,
                        width=600, height=600,
                        colors=('white', 'black'),
                        show_scale=False, reversescale=False,
                        vmin=0, vmax=255,
                        plt_type='show', online=False, filename=None):
    """Interactive Plotly heatmap of monthly returns.

    Parameters
    ----------
    returns : Series
        Daily or monthly returns with DatetimeIndex.
    cmap : str
        Matplotlib colormap name. Default 'RdYlGn'.
    yr_from, yr_to : int, optional
        Year range to display.
    plt_type : 'show' (default), 'iplot', or 'plot'
    online : bool
        If True, push to chart_studio (requires credentials).
    filename : str, optional
        Output filename for plt_type='plot'.
    """
    cscale = matplotlib_to_plotly(cmap, vmin=vmin, vmax=vmax)
    yr_to   = yr_to   or returns.index[-1].year
    yr_from = yr_from or returns.index[0].year

    grid = (empyrical.aggregate_returns(returns, convert_to='monthly')
            .unstack().fillna(0).round(4) * 100)
    grid = grid.loc[yr_from:yr_to, :]

    z   = grid.values.tolist()
    z.reverse()
    y   = grid.index.values.tolist()
    x   = [calendar.month_abbr[i] for i in grid.columns.values.tolist()]

    fig = ff.create_annotated_heatmap(
        z,
        x=x,
        y=y[::-1],
        annotation_text=np.round(z, 3),
        colorscale=cscale,
        reversescale=reversescale,
        hoverinfo='y+z',
        showscale=show_scale,
        font_colors=list(colors),
    )
    for ann in fig.layout.annotations:
        ann.font.size = font_size

    fig.layout.title = 'Monthly Returns: {} ({} – {})'.format(
        returns.name, yr_from, yr_to)
    fig.layout.yaxis.title  = 'Years'
    fig.layout.yaxis.dtick  = 3
    fig.layout.yaxis.tick0  = 2
    fig.layout.width  = width
    fig.layout.height = height

    return _render(fig, plt_type, online, filename)


# ── Monthly Returns Histogram ─────────────────────────────────────────────────

def get_monthly_hist(series, height=500, width=700,
                     rng=(-0.1, 0.1), plt_type='show',
                     online=False, filename=None):
    """Histogram of monthly returns with mean marker."""
    hist = px.histogram(series, nbins=40,
                        title='Monthly Returns: {}'.format(series.name),
                        width=width, height=height)

    hist.update_layout(
        plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(t=40),
        xaxis=dict(title='Returns', showgrid=False, zeroline=True,
                   zerolinewidth=3, color='black',
                   range=list(rng), hoverformat='0.2%',
                   tickformat='0.00%'),
        yaxis=dict(title='Frequency', showgrid=False, zeroline=True,
                   zerolinewidth=1, color='black'),
        shapes=[dict(type='line',
                     x0=series.mean(), x1=series.mean(),
                     y0=0, y1=1, yref='paper',
                     line=dict(dash='dashdot', width=4, color='orange'))],
        showlegend=True,
        legend=dict(x=0.85, y=0.9, bgcolor='white'),
    )
    return _render(hist, plt_type, online, filename)


# ── Underwater (Drawdown) Plot ────────────────────────────────────────────────

def underwater(series, benchmark_series=None, width=900, height=400,
               plt_type='show', online=False, filename=None, range_=None):
    """Drawdown (underwater) chart.

    Parameters
    ----------
    series : Series of returns
    benchmark_series : Series of returns, optional
        If provided, overlays benchmark drawdown.
    range_ : list, optional
        y-axis range, e.g. [-60, 5].
    """
    def _dd(s):
        cum = (1 + s).cumprod()
        return ((cum / cum.cummax()) - 1) * 100

    traces = [Scatter(
        x=series.index, y=_dd(series).round(2).values,
        name=series.name,
        mode='lines', fill='tonexty',
        fillcolor='rgba(200, 2, 2, 0.3)',
        line=dict(color='rgba(217, 2, 2, 1)', width=1.3),
    )]

    if benchmark_series is not None:
        traces.append(Scatter(
            x=benchmark_series.index, y=_dd(benchmark_series).round(2).values,
            name=benchmark_series.name,
            mode='lines', fill='tonexty',
            fillcolor='rgba(73, 192, 235, 0.3)',
            line=dict(color='rgba(73, 192, 235, 1)', width=1.3),
        ))

    layout = dict(
        plot_bgcolor='white', paper_bgcolor='white',
        hovermode='x unified',
        width=width, height=height,
        margin=dict(t=70, b=80, l=50, r=50, pad=0),
        xaxis=dict(title='Date', showgrid=False, color='black',
                   hoverformat='%A, %b %d %Y'),
        yaxis=dict(title='Drawdown (%)', showgrid=False, color='black',
                   range=range_),
        legend=dict(bgcolor='white', x=0.85, y=0.2, font=dict(size=9)),
    )

    fig = Figure(data=traces, layout=layout)
    return _render(fig, plt_type, online, filename)


# ── Annual Returns Bar Chart ──────────────────────────────────────────────────

def get_ann_ret_plot(ret_series, height=None, width=None,
                     x2range=None, dtime='monthly', plt_type='show'):
    """Annual returns bar chart with average return and volatility overlays."""
    if dtime == 'monthly':
        av_ann_mean = ret_series.resample('A').mean() * 12
        av_ann_std  = ret_series.resample('A').std()  * np.sqrt(12)
    else:
        av_ann_mean = ret_series.resample('A').mean() * 252
        av_ann_std  = ret_series.resample('A').std()  * np.sqrt(252)

    annual_ret = get_ann_ret(ret_series)

    trace0 = Bar(
        x=np.round(annual_ret.values * 100, 2),
        y=annual_ret.index.year,
        name='Total Annual Returns', orientation='h',
        marker=dict(color='#00FA9A', line=dict(color='#006400', width=1)),
        hoverinfo='x',
    )
    trace1 = Scatter(
        x=np.round(av_ann_mean.values * 100, 2), y=annual_ret.index.year,
        name='Average Annual Returns', mode='lines+markers',
        line=dict(color='black', width=1, dash='dashdot'), hoverinfo='x',
    )
    trace2 = Scatter(
        x=np.round(av_ann_std.values * 100, 2), y=annual_ret.index.year,
        name='Annual Volatility', mode='lines+markers',
        line=dict(color='#944bd2', width=1, dash='longdashdot'), hoverinfo='x',
    )

    annots = []
    for xs, yr in zip(np.round(annual_ret.values * 100, 2), annual_ret.index.year):
        annots.append(dict(
            xref='x1', yref='y1',
            x=xs + 5 if xs > 0 else 5, y=yr,
            text=str(xs) + '%',
            font=dict(family='Arial', size=9, color='#006400'),
            showarrow=False,
        ))

    fig = tls.make_subplots(rows=1, cols=2, shared_xaxes=True,
                            shared_yaxes=False, vertical_spacing=0.001)
    fig.append_trace(trace0, 1, 1)
    fig.append_trace(trace1, 1, 2)
    fig.append_trace(trace2, 1, 2)

    fig['layout'].update(dict(
        height=height, width=width,
        title='Annual Returns & Volatility — {}'.format(ret_series.name),
        hovermode='closest',
        yaxis1=dict(showgrid=False, zeroline=False, domain=[0, 0.85]),
        yaxis2=dict(showgrid=False, showline=True, domain=[0, 0.85], tickangle=90),
        xaxis=dict(zeroline=False, showgrid=True, domain=[0, 0.55]),
        xaxis2=dict(zeroline=False, showgrid=True, domain=[0.58, 1],
                    range=x2range, side='top'),
        legend=dict(x=0.029, y=1.038, font=dict(size=10)),
        margin=dict(l=50, r=50, t=50, b=50),
        paper_bgcolor='rgb(248, 248, 255)',
        plot_bgcolor='rgb(248, 248, 255)',
        annotations=annots,
    ))

    if plt_type == 'show':
        fig.show()
    elif plt_type == 'iplot':
        iplot(fig, show_link=False)
    return fig


# ── Rolling Fama-French Factors ───────────────────────────────────────────────

def plot_rolling_ff(strat, factors=None, rolling_window=36,
                    rng=(-4, 4), width=600, height=400,
                    plt_type='show', online=False):
    """Line chart of rolling Fama-French 5-factor loadings."""
    ff_coefs = get_ff_rolling_factors(strat, factors, rolling_window).round(3)

    fig = px.line(ff_coefs,
                  title='Rolling Fama-French Factors ({}mo)'.format(rolling_window))

    fig.update_layout(
        plot_bgcolor='white', paper_bgcolor='white',
        legend=dict(bgcolor='white'),
        yaxis=dict(range=list(rng), title='Factor Loading'),
        xaxis=dict(title='Date'),
        hovermode='x unified',
        height=height, width=width,
        shapes=[dict(type='line', xref='paper',
                     x0=0, y0=0, x1=1, y1=0,
                     line=dict(color='black', width=1, dash='longdashdot'))],
    )

    if online and _CHART_STUDIO:
        py.iplot(fig, width=width, height=height)
    elif plt_type == 'iplot':
        iplot(fig, show_link=False)
    elif plt_type == 'plot':
        plot(fig, show_link=False, filename='RollingFamaFrench.html')
    else:
        fig.show()
    return fig


# ── Internal Render Helper ────────────────────────────────────────────────────

def _render(fig, plt_type, online, filename):
    """Dispatch figure rendering."""
    if online and _CHART_STUDIO:
        return py.iplot(fig, show_link=False, filename=filename)
    if plt_type == 'iplot':
        return iplot(fig, show_link=False)
    if plt_type == 'plot':
        return plot(fig, show_link=False, filename=filename)
    fig.show()
    return fig
