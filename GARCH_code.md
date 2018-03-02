
#### <center><b> Formula  for GARCH(1,1) </b></center>

$$r_t = \mu_t + \epsilon_t$$ 
$$\epsilon_t = \sigma_t e_t$$ 
$$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$ 
$$e_t\sim N(0,1)$$

<br>
<br>
<br>

Let's start by importing modules
```python
import arch
import pandas as pd
import numpy as np
```

<br>
<br>
<br>

_Define_ what the ``function`` is
<br>
<br>
<br>
<br>
```python
def get_inst_vol(y, 
                 x = None, 
                 mean = 'Constant', 
                 vol = 'Garch', 
                 dist = 'normal'):
    
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
```  



The above function will fit the GARCH(1,1) model to a time series and estimate the instantaneous volatility for the time series.


__params__:


1. **y:** is an array containing data to fit model on. It could either be `Series` or `Numpy Array`
2. **x:** in this case is the ecogenous variable, which is optional
3. **mean:** `str optional` Name of the model. Currently supported options are 'Constant', 'Zero', 'ARX' and 'HARX'
4. **vol:** `str optional` what model to use. 'GARCH' (default), 'EGARCH', 'ARCH' and 'HARCH'
5. **dist:** `str optional` 'normal'(default), 't', 'ged'


__returns__:

Time Series `pd.core.series.Series`

<br>
<br>
<br>
<br>
Let the code begin


``` python


    if isinstance(y, pd.core.series.Series):
        ## remove nan.
        y = y.dropna()
    elif isinstance(y, np.ndarray):
        y = y[~np.isnan(y)]
    
    # provide a model
    model = arch.arch_model(y * 100, mean = 'constant')
    
    # fit the model
    res = model.fit(update_freq= 5)
    
    # get the parameters. Here [1] means number of lags. This is only Garch(1,1)
    omega = res.params['omega']
    alpha = res.params['alpha[1]']
    beta = res.params['beta[1]']
    
    inst_vol = res.conditional_volatility * np.sqrt(252
    # instantaneous variance.
#     inst_var = (omega + 
#                 res.resid ** 2 * alpha + 
#                 res.conditional_volatility ** 2 * beta)
#     inst_vol = 0.01 * np.sqrt(inst_var)
    if isinstance(inst_vol, pd.core.series.Series):
        inst_vol.name = y.name
    elif isinstance(inst_vol, np.ndarray):
        inst_vol = inst_vol
    # more interested in conditional vol
    return inst_vol
```


- The first part of the code checks for the type of data we have entered. Is it in a series or a numpy array. Accordingly it then removes the null values. This is important as data with null value can't have a model fit to it.<br><br><br>
```python
model = arch.arch_model(y * 100, mean = 'constant')
res = model.fit(update_freq = 5)
```
- The second part provides information on what model to fit and does the same<br><br><br><br>

``` python
omega = res.params['omega']
alpha = res.params['alpha[1]']
beta = res.params['beta[1]']
```
- After fitting the model, we then estimate Garch(1,1) parameters i.e. $\alpha, \beta,\omega$ <br><br>

```python
inst_var = (omega + 
            res.resid ** 2 * alpha + 
            res.conditional_volatility ** 2 * beta)
inst_vol = 0.01 * np.sqrt(inst_var)
```

- Next we find instantaneous variance and then find the instantaneous volatility. We multiply with 0.01 because of our earlier mulitplication by 100 to our returns. <br><br>

$$ \sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$
