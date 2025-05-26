import pandas as pd
import numpy as np
import calendar
import matplotlib.cm
import matplotlib.colors
import plotly.figure_factory as ff
from plotly.graph_objs import Scatter, Bar, Figure # Added Figure
from plotly.offline import iplot, plot
import plotly.express as px
# import plotly.tools as tls # Obsolete, replaced by make_subplots from plotly.subplots
from plotly.subplots import make_subplots
import empyrical # for get_monthly_heatmap

# Import necessary functions from other modules
from .analytics import get_ann_ret, get_ff_rolling_factors, drawdown # Added drawdown
from .main_logic import get_eq_line

def matplotlib_to_plotly(cmap, vmin = 0, vmax = 255):
    norm = matplotlib.colors.Normalize(vmin = vmin, vmax = vmax)
    """Converts a matplotlib colormap to plotly colormap or colorscale, which is customized

    params:
        cmap: str, valid cmap in matplotlib"""

    pl_entries = 255
    _cmap = matplotlib.cm.get_cmap(cmap)
    h = 1/(pl_entries-1)
    pl_colorscale = []

    for k in range(pl_entries):
        C = list(map(np.uint8, np.array(_cmap(norm(k))[:3])*(pl_entries)))
        pl_colorscale.append([k*h, 'rgb'+str((C[0], C[1], C[2]))])

    return pl_colorscale

def plt_cscale(cmap):
    _cmap = matplotlib.cm.get_cmap(cmap)
    # The original norm was vmin=-100, vmax=100, but cmap values are 0-255.
    # Using 0-255 to align with typical colormap indexing.
    # If the intent was to map a specific range of values to colors,
    # this function might need adjustment based on how it's used.
    norm = matplotlib.colors.Normalize(vmin = 0, vmax =255) 

    colorscale =[]

    for i in range(255): # Iterate 0-254, as typical for 255 entries
        k = matplotlib.colors.colorConverter.to_rgb(_cmap(norm(i)))
        colorscale.append(k)

    return colorscale

def get_monthly_heatmap(returns,
                        cmap,
                        font_size = 10,
                        yr_from = None,
                        yr_to = None,
                        cnvrt = 'monthly', # This parameter isn't used in the function body
                        width = 600,
                        plt_type = 'iplot',
                        filename = None,
                        colors = ['white', 'black'],
                        online = False,
                        show_scale = False,
                        height = 600,
                        vmin = 0, # Used by matplotlib_to_plotly
                        vmax = 255, # Used by matplotlib_to_plotly
                        reversescale = False,
                        render = 'notebook_connected'):

    """F: to plot heatmap of monthly returns:

    params:

        returns: pd.Series, daily or monthly returns, ideally a series with datetime index
        cmap: (optional)str, eg 'RdYlGn'
        font_size: (optional) font_size of annotations
        yr_from: (optional) Heatmap year from
        yr_to: (optional) Heatmap year to
        cnvrt: (optional) str, convert returns to (currently not used)
        """
    cscale = matplotlib_to_plotly(cmap, vmin = vmin, vmax = vmax)

    if yr_to is None:
        yr_to = returns.index[-1].year
    if yr_from is None:
        yr_from = returns.index[0].year
        
    # Ensure returns is a Series for empyrical
    if not isinstance(returns, pd.Series):
        if isinstance(returns, pd.DataFrame) and returns.shape[1] == 1:
            returns = returns.iloc[:, 0]
        else:
            raise ValueError("Input 'returns' must be a pandas Series or a single-column DataFrame.")
    
    grid = empyrical.aggregate_returns(returns, convert_to = 'monthly').unstack().fillna(0).round(4) * 100
    grid = grid.loc[yr_from:yr_to,:]
    
    z = grid.values.tolist()
    y_labels = grid.index.astype(str).values.tolist() # Ensure y-labels are strings
    x_labels = [calendar.month_abbr[i] for i in grid.columns.values.tolist()]

    # Original code reversed z, which means y labels also need to be reversed for ff.create_annotated_heatmap
    z.reverse() 
    y_labels.reverse()
    
    z_text = np.round(z, 3).tolist() # Convert numpy array to list for annotation_text

    fighm = ff.create_annotated_heatmap(z,
                                        x = x_labels,
                                        y= y_labels,
                                        annotation_text= z_text,
                                        colorscale = cscale,
                                        reversescale = reversescale,
                                        hoverinfo = "x+y+z", # Changed to x+y+z for more info
                                        showscale = show_scale,
                                        font_colors= colors)
    for i in range(len(fighm.layout.annotations)):
        fighm.layout.annotations[i].font.size = font_size

    fighm.layout.title = 'Heatmap for {0} from {1} - {2}'.format(returns.name if returns.name else "Strategy",
                                                                y_labels[0] if y_labels else yr_from, # Use actual first year from data
                                                                y_labels[-1] if y_labels else yr_to) # Use actual last year
    fighm['layout']['yaxis']['title'] = 'Years'
    fighm['layout']['yaxis']['dtick'] = 3
    fighm['layout']['yaxis']['tick0'] = 0 # Adjusted tick0 for consistency with reversed y_labels
    fighm['layout']['width'] = width
    fighm['layout']['height'] = height

    if online == False:
        if plt_type == 'iplot':
            return iplot(fighm,
                         show_link= False) # Removed image_width/height, not standard iplot args
        elif plt_type == 'plot':
            return plot(fighm,
                        show_link= False,
                        filename = filename if filename else "heatmap.html")
        elif plt_type == 'show':
            return fighm.show(renderer=render) # Corrected 'render' to 'renderer' for plotly >= 5
    elif online == True:
        # chart_studio.plotly (py) is not imported here, assuming offline usage for now.
        # If online publishing is needed, py.iplot would be used.
        # For now, defaulting to offline plot for online=True as well.
        return plot(fighm, show_link=False, filename=filename if filename else "heatmap_online.html")


def get_monthly_hist(series,
                     height = 400,
                     width = 900,
                     plt_type = 'iplot',
                     filename = None,
                     online = False, # Not directly used by px.histogram, more for iplot/plot
                     rng = [-0.1, 0.1],
                     render = 'notebook_connected'): # renderer for .show()

    """F: to plot histogram of monthly returns

    params:
        series: pd.Series, monthly or daily returns
        height: (optional) int
        width: (optional)

    returns:
        plotly Figure object or iplot/plot output"""
    # pd.options.plotting.backend = 'plotly' # This is a global option, avoid setting in a library function.

    if not isinstance(series, pd.Series):
        series = pd.Series(series) # Ensure input is a Series

    # Determine nbins based on series length
    if 200 <= len(series) < 500: # Corrected condition
        nbins = int(len(series)/2)
    elif len(series) < 200:
        nbins = int(len(series)) # Potentially too many bins for very small series
        if nbins < 10: nbins = 10 # Ensure a minimum number of bins
    else: # len(series) >= 500
        nbins = int(len(series)/4)
    
    # Ensure nbins is reasonable
    nbins = max(10, min(nbins, 100)) # Cap nbins e.g. between 10 and 100

    hist_fig = px.histogram(series, # Changed variable name to hist_fig
                        nbins = nbins, # Using calculated nbins
                        # title = 'Monthly Returns', # Title is set in layout update
                        width = width, # Using parameter
                        height = height, # Using parameter
                    )
    hist_fig.update_layout( # Changed to hist_fig.update_layout
                     plot_bgcolor = 'white',
                     paper_bgcolor = 'white',
                     title_text = 'Monthly Returns Histogram for {}'.format(series.name if series.name else "Strategy"), # Use title_text
                     margin = dict(t = 40),
                     xaxis = dict(title = 'Returns',
                                    showgrid = False,
                                    showticklabels = True,
                                    zeroline = True,
                                    zerolinewidth = 3,
                                    color = 'black',
                                    range = rng,
                                    hoverformat = '.2%' # Corrected hoverformat
                                 ),
                      yaxis = dict(title = 'Frequency',
                                     showgrid = False,
                                     showticklabels = True,
                                     zeroline = True,
                                     zerolinewidth = 1,
                                     color = 'black'
                                 ),
                      shapes = [dict(type = 'line',
                                       x0 = series.mean(),
                                       x1 = series.mean(),
                                       y0 = 0,
                                       y1 = 1,
                                       yref = 'paper',
                                       line = dict(dash = 'dashdot', # Corrected dash style
                                                   width = 2, # Adjusted width
                                                   color = 'orange'),
                                       name = 'Mean', # Added name for legend
                                       )
                                   ],
                      showlegend = True,
                      legend = dict(x = 0.85,
                                      y = 0.9,
                                      bgcolor = 'rgba(255,255,255,0.5)'), # Slightly transparent bgcolor
                  )
    hist_fig.update_xaxes(tickformat='.2%') # Corrected tickformat application

    if online == False: # online is more about chart_studio publishing
        if plt_type == 'iplot':
            return iplot(hist_fig, show_link= False)
        elif plt_type == 'plot':
            return plot(hist_fig, show_link = False, filename = filename if filename else "histogram.html")
        elif plt_type == 'show':
            return hist_fig.show(renderer = render)
    elif online == True:
        # py.iplot(hist_fig, show_link = False) # Requires chart_studio.plotly (py)
        return plot(hist_fig, show_link=False, filename=filename if filename else "histogram_online.html")
    return hist_fig # Return the figure object by default if no plot type matches

def underwater(series,
               spy_series = None, # benchmark series
               s_name = None, # strategy name
               width = 900,
               height = 400,
               # color = 'red', # This parameter was not used for strategy, hardcoded
               range_y = None, # Renamed from 'range' to avoid conflict with built-in
               plt_type = 'iplot',
               online = False,
               filename = None,
               render = 'notebook_connected'): # renderer for .show()
    
    if not isinstance(series, pd.Series): series = pd.Series(series)
    name = s_name if s_name else (series.name if series.name else "Strategy")
    
    strat_cum = (1+series).cumprod()
    dd = (strat_cum/strat_cum.cummax() - 1) * 100
    dd = dd.round(2) # round after calculation

    trace_strat = Scatter(x = dd.index,
                          y = dd.values,
                          mode = 'lines',
                          name = name,
                          fill = 'tozeroy', # Fill to y=0
                          fillcolor = 'rgba(217, 2, 2, 0.3)', # Red fill
                          line = dict(color = 'rgba(217, 2, 2, 1)', width = 1.3),
                         )
    
    data_traces = [trace_strat]

    if spy_series is not None:
        if not isinstance(spy_series, pd.Series): spy_series = pd.Series(spy_series)
        spy_name = spy_series.name if spy_series.name else "Benchmark"
        spy_cum = (1+spy_series).cumprod()
        dd_spy = (spy_cum/spy_cum.cummax() - 1) * 100
        dd_spy = dd_spy.round(2)
        
        trace_spy = Scatter(x = dd_spy.index,
                            y = dd_spy.values,
                            mode = 'lines',
                            name = spy_name,
                            fill = 'tozeroy', # Fill to y=0
                            fillcolor = 'rgba(73, 192, 235, 0.3)', # Blue fill
                            line = dict(color = 'rgba(73, 192, 235, 1)', width = 1.3),
                           )
        data_traces.append(trace_spy)

    layout =  dict(title_text=f"{name} Underwater Plot", # Added title
                   plot_bgcolor = 'white',
                   paper_bgcolor = 'white',
                   hovermode = 'x unified',
                   margin = dict(t = 70, b = 80, l = 50, r = 50, pad = 0),
                   width = width,
                   height = height,
                   xaxis = dict(title = 'Dates',
                                  showgrid = False,
                                  showticklabels = True,
                                  zeroline = True,
                                  color = 'black',
                                  hoverformat = '%A, %b %d %Y '
                                 ),
                   yaxis = dict(title = 'Drawdown in %',
                                  showgrid = False,
                                  showticklabels = True,
                                  zeroline = True,
                                  color = 'black',
                                  range = range_y, # Use renamed parameter
                                  autorange = True if range_y is None else False,
                                  tickformat = '.2f', # Format y-axis ticks
                                 ),
                   legend = dict(bgcolor = 'rgba(255,255,255,0.5)', # Transparent white
                                   x = 0.01, y = 0.99, # Position top-left
                                   bordercolor = "Black",
                                   borderwidth = 1,
                                   font = dict(size = 9))
                  )

    pyfig = Figure(data = data_traces, layout = layout)
    
    if online == False:
        if plt_type == 'plot':
            plot(pyfig, show_link = False, filename = filename if filename else "underwater_plot.html")
        elif plt_type =='iplot':
            iplot(pyfig, show_link = False)
        elif plt_type == 'show':
            pyfig.show(renderer = render)
    elif online == True:
        # py.iplot(pyfig, show_link = False) # Requires chart_studio
        plot(pyfig, show_link=False, filename=filename if filename else "underwater_plot_online.html")
    return pyfig


def get_ann_ret_plot(ret_series,
                     height = 600, # Default height
                     width = 900,  # Default width
                     x2range = None, # Range for the second x-axis (volatility)
                     # orient = 'h', # This was in original but Bar is explicitly horizontal
                     dtime = 'monthly'):
    
    if not isinstance(ret_series, pd.Series): ret_series = pd.Series(ret_series)
    
    # get_eq_line is imported from main_logic
    # get_ann_ret is imported from analytics
    
    # Calculate necessary stats
    annual_ret_values = get_ann_ret(ret_series, dtime=dtime) # This is already pct returns

    # For average annual mean and std, resample original returns
    if dtime == 'monthly':
        ann_factor = 12
        sqrt_ann_factor = np.sqrt(12)
    elif dtime == 'daily':
        ann_factor = 252
        sqrt_ann_factor = np.sqrt(252)
    else:
        raise ValueError("dtime must be 'monthly' or 'daily'")

    av_ann_mean = ret_series.resample('A').mean() * ann_factor
    av_ann_std = ret_series.resample('A').std() * sqrt_ann_factor
    
    # Align indices - important if ret_series doesn't span full years
    # annual_ret_values is PeriodIndex, av_ann_mean/std might be DatetimeIndex from resample('A')
    # Convert av_ann_mean/std to PeriodIndex for alignment
    av_ann_mean.index = av_ann_mean.index.to_period('A')
    av_ann_std.index = av_ann_std.index.to_period('A')
    
    # Align all three series to the intersection of their indices
    idx_intersect = annual_ret_values.index.intersection(av_ann_mean.index).intersection(av_ann_std.index)
    annual_ret_values = annual_ret_values.loc[idx_intersect]
    av_ann_mean = av_ann_mean.loc[idx_intersect]
    av_ann_std = av_ann_std.loc[idx_intersect]

    y_labels = annual_ret_values.index.year.astype(str).tolist()

    trace0_bar = Bar(y = y_labels, # Year as category
                     x = np.round(annual_ret_values.values * 100,2),
                     name = 'Total Annual Returns',
                     marker = dict(color = '#00FA9A',
                                   line = dict(color = '#006400', width = 1)),
                     orientation = 'h',
                     hoverinfo = 'x'
                    )
    trace1_scatter_mean = Scatter(y = y_labels,
                                  x = np.round(av_ann_mean.values * 100,2),
                                  name = 'Average Annual Returns',
                                  mode = 'lines+markers',
                                  line = dict(color = 'black', width = 1, dash = 'dashdot'),
                                  hoverinfo = 'x'
                                 )

    trace2_scatter_std = Scatter(y = y_labels,
                                 x = np.round(av_ann_std.values * 100,2),
                                 name = 'Annual Volatility',
                                 mode = 'lines+markers',
                                 line = dict(color = '#944bd2', width = 1, dash = 'longdashdot'),
                                 hoverinfo = 'x'
                                )
    
    # Using make_subplots (imported from plotly.subplots)
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, # Share y-axis (years)
                        column_widths=[0.6, 0.4], # Adjust column widths
                        horizontal_spacing=0.05) 

    fig.add_trace(trace0_bar, row=1, col=1)
    fig.add_trace(trace1_scatter_mean, row=1, col=2)
    fig.add_trace(trace2_scatter_std, row=1, col=2)

    fig.update_layout(
        height = height,
        width = width,
        title_text =f'Average Annual Returns and Volatility for {ret_series.name if ret_series.name else "Strategy"}',
        hovermode = 'y unified', # Changed for better hover on horizontal bars/lines
        yaxis=dict( # Common y-axis properties
            showgrid=False,
            zeroline=False,
            showticklabels=True,
            showline=False,
            autorange="reversed", # Show recent years at the top
        ),
        xaxis1=dict( # X-axis for Bar chart (Total Returns)
            title="Total Annual Return (%)",
            zeroline=False,
            showline=False,
            showticklabels=True,
            showgrid=True,
        ),
        xaxis2=dict( # X-axis for Scatter (Mean/Std)
            title="Avg Return / Volatility (%)",
            zeroline=False,
            showline=False,
            showticklabels=True,
            showgrid=True,
            range=x2range, # Optional range for this axis
        ),
        legend=dict(
            x=0.5, y=-0.1, # Position legend below plot
            xanchor='center',
            orientation='h',
            font=dict(size=10),
        ),
        margin=dict(l=70, r=50, t=80, b=100), # Adjusted margins
        paper_bgcolor='rgb(248, 248, 255)',
        plot_bgcolor='rgb(248, 248, 255)',
    )
    
    # Annotations for Bar chart (Total Annual Returns)
    annotations = []
    for y_val, x_val_num in zip(y_labels, annual_ret_values.values * 100):
        x_val = round(x_val_num, 2)
        text_x_pos = x_val + 5 if x_val >= 0 else x_val - 5 # Adjust text position based on value
        annotations.append(dict(xref = 'x1', yref = 'y1', # Refer to first subplot's axes
                               x = text_x_pos, y = y_val,
                               text = str(x_val) + '%',
                               font = dict(family='Arial', size=9, color='#006400' if x_val >=0 else '#FF0000'),
                               showarrow=False,
                               xanchor='left' if x_val >=0 else 'right'
                              ))
    fig.update_layout(annotations=annotations)

    return fig


def plot_rolling_ff(strat,
                    factors = None, # User can provide pre-fetched factors
                    rolling_window = 36,
                    online = False,
                    plt_type = 'iplot', # 'iplot', 'plot', 'show'
                    rng_y = [-2,2], # Renamed from rng to rng_y for y-axis range
                    width = 700, # Adjusted default width
                    height = 500, # Adjusted default height
                    render = 'notebook_connected'):
    
    # get_ff_rolling_factors is imported from analytics
    ff_facs = get_ff_rolling_factors(strat, factors=factors, rolling_window=rolling_window)
    
    if ff_facs.empty:
        print("Cannot plot rolling Fama-French factors: Data is empty.")
        return None # Or an empty Figure

    ff_facs_rounded = ff_facs.round(3) # Round for display if needed, px.line handles data directly

    # Using plotly.express for simplicity
    pyfig = px.line(ff_facs_rounded,
                    title = f'Rolling Fama-French Factors ({rolling_window}mo)',
                    # markers=True # Optionally add markers
                   )

    pyfig.update_layout(
                   plot_bgcolor = 'white',
                   paper_bgcolor = 'white',
                   legend = dict(bgcolor = 'rgba(255,255,255,0.5)',
                                 x=0.01, y=0.99, bordercolor="Black", borderwidth=1),
                   yaxis = dict(range = rng_y,
                                title = 'Factor Beta', # More descriptive title
                                zeroline=True, zerolinewidth=1, zerolinecolor='Gray'
                               ),
                   xaxis = dict(title = 'Date', showgrid=False),
                   hovermode = 'x unified',
                   height = height,
                   width = width,
                   # Adding a horizontal line at y=0 for reference
                   shapes = [
                             dict(type = 'line',
                                 xref = 'paper', x0 = 0, y0 = 0,
                                 x1 = 1, y1 = 0,
                                 line = dict(color = 'black', width = 1, dash = 'dash'),
                                 layer="below" # Ensure line is below data traces
                                 )
                           ]
                   )
    
    if not online: # online is more for chart_studio
        if plt_type == 'iplot':
           return iplot(pyfig, show_link = False)
        elif plt_type == 'plot':
            return plot(pyfig, show_link = False, filename = filename if filename else 'rolling_ff_factors.html')
        elif plt_type == 'show':
            return pyfig.show(renderer = render)
    elif online:
        # py.iplot(pyfig, show_link=False) # Requires chart_studio
        return plot(pyfig, show_link=False, filename=filename if filename else 'rolling_ff_factors_online.html')
    return pyfig # Return figure if no plot type matches
