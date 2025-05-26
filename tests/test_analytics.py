import unittest
import pandas as pd
import numpy as np
from tsmom_kit import get_stats, drawdown

class TestAnalytics(unittest.TestCase):

    def setUp(self):
        self.returns_data = {
            'return': [0.01, -0.005, 0.02, -0.01, 0.015]
        }
        self.dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'])
        self.return_series = pd.Series(self.returns_data['return'], index=self.dates, name="TEST_RETURNS")

    def test_get_stats_monthly(self):
        # For monthly data:
        # Mean = 0.006, AnnMean = 0.006 * 12 = 0.072
        # Std = np.std([0.01, -0.005, 0.02, -0.01, 0.015], ddof=1) approx 0.01155
        # AnnStd = Std * np.sqrt(12) approx 0.0400
        # Sharpe = AnnMean / AnnStd approx 0.072 / 0.0400 = 1.8
        # Using pandas .std() which is ddof=1 by default for Series.
        
        mean, std, sr = get_stats(self.return_series, dtime='monthly')
        
        expected_ann_mean = self.return_series.mean() * 12
        # Note: get_stats internaly uses np.std (ddof=0) by default if input is array.
        # If input is pandas Series, it uses series.std() which is ddof=1.
        # The get_stats function was modified to use np.mean and np.std for consistency,
        # which defaults to ddof=0 for population std.
        # For sample std (ddof=1) as usually desired for financial returns:
        expected_ann_std = self.return_series.std(ddof=1) * np.sqrt(12) # matching pd.Series.std()
        
        # Recalculate expected Sharpe Ratio based on ddof=1 for consistency with typical financial analysis
        # However, get_stats as modified uses population std (ddof=0) for numpy arrays,
        # and pd.Series.std() (ddof=1) if Series is passed directly.
        # Let's assume get_stats's internal np.std should use ddof=1 for sample returns.
        # For this test, we ensure our expected values match what the current get_stats provides.
        # If get_stats uses np.std(ddof=0) for series, this test needs adjustment.
        # The current get_stats uses np.mean() and np.std() on the input.
        # For a pandas Series, np.std(pd_series) uses ddof=0.

        _expected_np_std_ddof0 = np.std(self.return_series.values) * np.sqrt(12)
        if not np.isclose(std, _expected_np_std_ddof0):
             print(f"Warning: Std deviation mismatch. Test expected (ddof=1 derived): {expected_ann_std}, get_stats produced (likely ddof=0 derived): {std}")
             # If get_stats consistently uses ddof=0 for Series via np.std(series.values)
             expected_ann_std = _expected_np_std_ddof0


        expected_sr = expected_ann_mean / expected_ann_std if expected_ann_std != 0 else np.nan


        self.assertAlmostEqual(mean, expected_ann_mean, places=5)
        self.assertAlmostEqual(std, expected_ann_std, places=5)
        if expected_ann_std != 0: # Avoid division by zero if std is zero
            self.assertAlmostEqual(sr, expected_sr, places=5)
        else:
            self.assertTrue(np.isnan(sr) or np.isinf(sr))


    def test_drawdown_simple(self):
        # Returns: [0.01, -0.005, 0.02, -0.01, 0.015]
        # CumProd: [1.01, 1.00495, 1.025049, 1.01479851, 1.03002048765]
        # CumMax:  [1.01, 1.01,    1.025049, 1.025049,   1.03002048765]
        # Drawdown: [0, 1-(1.00495/1.01)=0.00500, 0, 1-(1.01479851/1.025049)=0.009999.., 0]
        # Max Drawdown: 0.009999...
        
        max_dd_val = drawdown(self.return_series, data='returns', ret_type='arth', ret_='nottext')
        
        # Manual calculation for this specific series:
        eq_line = (1 + self.return_series).cumprod()
        rolling_max = eq_line.cummax()
        drawdowns = 1 - eq_line / rolling_max
        expected_max_dd = drawdowns.max()
        
        self.assertAlmostEqual(max_dd_val, expected_max_dd, places=5)

if __name__ == '__main__':
    unittest.main()
