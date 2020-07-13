import pandas as pd
class YahooDailyReader():
    def __init__(self, symbol=None, start=None, end=None):
        import datetime as dt, time
        self.symbol = symbol

        # initialize start/end dates if not provided
        if end is None:
            end = dt.datetime.today()
        if start is None:
            start = dt.datetime(2010,1,1)

        self.start = start
        self.end = end

        # convert dates to unix time strings
        unix_start = int(time.mktime(self.start.timetuple()))
        day_end = self.end.replace(hour=23, minute=59, second=59)
        unix_end = int(time.mktime(day_end.timetuple()))

        url = 'https://finance.yahoo.com/quote/{}/history?'
        url += 'period1={}&period2={}'
        url += '&filter=div'
        self.url = url.format(self.symbol, unix_start, unix_end)

    def base(self):
        import requests, re, json
        r = requests.get(self.url)

        ptrn = r'root\.App\.main = (.*?);\n}\(this\)\);'
        txt = re.search(ptrn, r.text, re.DOTALL).group(1)
        jsn = json.loads(txt)
        try:
            stk_data = jsn['context']['dispatcher']['stores']['HistoricalPriceStore']
        except KeyError:
            raise KeyError('No data found for'.format(self.symbol))
        return jsn

    def read(self):
        #retry 3 time
        for x in range(0, 5):
            jsn = self.base()
            df = pd.DataFrame(
                    jsn['context']['dispatcher']['stores']
                    ['HistoricalPriceStore']['prices']
                    )
            if df.empty:
                print('Retrying {}'.format(self.symbol))
                continue
            else:
                break
        df.insert(0, 'symbol', self.symbol)
        df['date'] = pd.to_datetime(df['date'], unit='s').dt.date
        # drop rows that aren't prices
        df = df.dropna(subset=['close'])
        # print(self.symbol, self.start)

        # if 'amount' in df:
        try:
            df = df[['date', 'high', 'low', 'open', 'close',
                         'volume', 'adjclose', 'amount']]
        except KeyError as e:
            print('One of the columns is not availalble')
            pass

        df = df.set_index(pd.DatetimeIndex(df['date']))
        df.sort_index(ascending = True, inplace = True)
        return df

    def read_div(self):
        jsn = self.base()
        df = pd.DataFrame(jsn['context']
                     ['dispatcher']
                     ['stores']
                     ['HistoricalPriceStore']
                     ['eventsData'])
        df.insert(0, 'symbol', self.symbol)
        df['date'] = pd.to_datetime(df.date, unit= 's').dt.date
        df.set_index('date', inplace = True)
        df.sort_index(inplace = True)
        df = df[['amount']]
        return df
