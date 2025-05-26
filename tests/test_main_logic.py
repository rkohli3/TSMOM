import unittest
import pandas as pd
import numpy as np
from tsmom_kit import get_rets, cum_pfmnce

class TestMainLogic(unittest.TestCase):

    def setUp(self):
        self.prices_data = {
            'price': [100, 101, 102, 100, 103, 105]
        }
        self.dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05', '2023-01-06'])
        self.price_series = pd.Series(self.prices_data['price'], index=self.dates, name="TEST_ASSET")

    def test_get_rets_arithmetic(self):
        expected_returns = pd.Series([np.nan, 0.01/1.0, 0.01/1.01, -0.02/1.02, 0.03/1.00, 0.02/1.03], index=self.dates, name="TEST_ASSET")
        # Calculation: (101-100)/100 = 0.01, (102-101)/101 approx 0.0099, (100-102)/102 approx -0.0196
        # (103-100)/100 = 0.03, (105-103)/103 approx 0.0194
        # For simplicity of expected values here, using rougher pct_change logic for expected values
        # More precise: pd.Series([np.nan, 101/100-1, 102/101-1, 100/102-1, 103/100-1, 105/103-1], index=self.dates)
        
        calculated_returns = get_rets(self.price_series, kind='arth', freq='d') # freq='d' to avoid resampling
        
        # Manual expected values for this specific series using pct_change logic
        expected_returns = self.price_series.pct_change()

        pd.testing.assert_series_equal(calculated_returns, expected_returns, check_dtype=False, atol=1e-5)

    def test_cum_pfmnce_prices(self):
        # Expected: [1.0, 1.01, 1.02, 1.00, 1.03, 1.05]
        expected_cum_perf = self.price_series / self.price_series.iloc[0]
        calculated_cum_perf = cum_pfmnce(self.price_series, data='prices')
        pd.testing.assert_series_equal(calculated_cum_perf, expected_cum_perf, check_dtype=False, atol=1e-5)

if __name__ == '__main__':
    unittest.main()
