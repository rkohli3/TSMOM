import pandas as pd
import numpy as np
import tsmom as tm
import datetime as dt

ticks = ['TSLA', 'MSFT', 'QQQ']

data = tm.get_yahoo_data(ticks,
                         start= dt.datetime(2019, 1 , 1),
                         end = dt.datetime.today()
                         )
print(data.head())