import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest
import numpy as np
import pandas as pd

from backtest.metrics.signal_evaluation import (
    calculate_pearson_ic,
    calculate_rank_ic,
    calculate_ic_series,
    calculate_ic_statistics,
    calculate_forward_returns,
    factor_quantile_analysis,
    calculate_long_short_portfolio,
    calculate_group_turnover,
)


def _make_sample_data(n_stocks=10, n_dates=20, seed=42):
    np.random.seed(seed)
    codes = [f"stock_{i}" for i in range(n_stocks)]
    dates = pd.date_range("2023-01-01", periods=n_dates, freq="B")
    signal_data = {}
    return_data = {}
    for date in dates:
        signals = np.random.randn(n_stocks)
        returns = 0.05 * signals + np.random.randn(n_stocks) * 0.02
        signal_data[date] = pd.Series(signals, index=codes)
        return_data[date] = pd.Series(returns, index=codes)
    signal_df = pd.DataFrame(signal_data)
    return_df = pd.DataFrame(return_data)
    return signal_df, return_df


class TestCalculatePearsonIC(unittest.TestCase):
    def test_perfect_positive(self):
        x = pd.Series([1, 2, 3, 4, 5])
        y = pd.Series([2, 4, 6, 8, 10])
        ic = calculate_pearson_ic(x, y)
        self.assertAlmostEqual(ic, 1.0, places=5)

    def test_perfect_negative(self):
        x = pd.Series([1, 2, 3, 4, 5])
        y = pd.Series([10, 8, 6, 4, 2])
        ic = calculate_pearson_ic(x, y)
        self.assertAlmostEqual(ic, -1.0, places=5)

    def test_zero_correlation(self):
        x = pd.Series([1, 2, 3, 4, 5])
        y = pd.Series([3, 3, 3, 3, 3])
        ic = calculate_pearson_ic(x, y)
        self.assertTrue(np.isnan(ic) or abs(ic) < 0.1)

    def test_too_few_points(self):
        x = pd.Series([1, 2])
        y = pd.Series([3, 4])
        ic = calculate_pearson_ic(x, y)
        self.assertEqual(ic, 0.0)

    def test_with_nan(self):
        x = pd.Series([1, 2, np.nan, 4, 5])
        y = pd.Series([2, 4, 6, np.nan, 10])
        ic = calculate_pearson_ic(x, y)
        self.assertAlmostEqual(ic, 1.0, places=5)

    def test_with_inf(self):
        x = pd.Series([1, 2, np.inf, 4, 5])
        y = pd.Series([2, 4, 6, 8, 10])
        ic = calculate_pearson_ic(x, y)
        self.assertTrue(not np.isnan(ic))


class TestCalculateRankIC(unittest.TestCase):
    def test_perfect_monotonic(self):
        x = pd.Series([1, 2, 3, 4, 5])
        y = pd.Series([1, 4, 9, 16, 25])
        ic = calculate_rank_ic(x, y)
        self.assertAlmostEqual(ic, 1.0, places=5)

    def test_too_few_points(self):
        x = pd.Series([1, 2])
        y = pd.Series([3, 4])
        ic = calculate_rank_ic(x, y)
        self.assertEqual(ic, 0.0)

    def test_realistic_data(self):
        signal_df, return_df = _make_sample_data()
        for date in signal_df.columns:
            ic = calculate_rank_ic(signal_df[date], return_df[date])
            self.assertGreaterEqual(ic, -1.0)
            self.assertLessEqual(ic, 1.0)
            break


class TestCalculateForwardReturns(unittest.TestCase):
    def test_forward_1d(self):
        close_df = pd.DataFrame({
            "2023-01-01": [100, 200],
            "2023-01-02": [110, 210],
            "2023-01-03": [120, 220],
        }, index=["A", "B"])
        close_df.columns = pd.to_datetime(close_df.columns)
        fwd = calculate_forward_returns(close_df, periods=1)
        expected_01 = 110 / 100 - 1
        self.assertAlmostEqual(fwd.loc["A", close_df.columns[0]], expected_01)

    def test_forward_5d(self):
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        np.random.seed(42)
        close_df = pd.DataFrame(
            np.random.randn(3, 10).cumsum(axis=1) + 100,
            index=["A", "B", "C"],
            columns=dates,
        )
        fwd = calculate_forward_returns(close_df, periods=5)
        self.assertEqual(fwd.shape[1], close_df.shape[1])

    def test_last_periods_nan(self):
        close_df = pd.DataFrame({
            "2023-01-01": [100],
            "2023-01-02": [110],
        }, index=["A"])
        close_df.columns = pd.to_datetime(close_df.columns)
        fwd = calculate_forward_returns(close_df, periods=1)
        self.assertTrue(pd.isna(fwd.iloc[0, -1]))


class TestCalculateICSeries(unittest.TestCase):
    def setUp(self):
        self.signal_df, self.return_df = _make_sample_data()

    def test_returns_series(self):
        ic_series = calculate_ic_series(self.signal_df, self.return_df, method="rank")
        self.assertIsInstance(ic_series, pd.Series)
        self.assertGreater(len(ic_series), 0)

    def test_pearson_method(self):
        ic_series = calculate_ic_series(self.signal_df, self.return_df, method="pearson")
        self.assertIsInstance(ic_series, pd.Series)
        self.assertGreater(len(ic_series), 0)

    def test_values_in_range(self):
        ic_series = calculate_ic_series(self.signal_df, self.return_df)
        for val in ic_series.dropna():
            self.assertGreaterEqual(val, -1.0)
            self.assertLessEqual(val, 1.0)


class TestCalculateICStatistics(unittest.TestCase):
    def test_mean_ic(self):
        ic_series = pd.Series([0.1, 0.2, 0.15, 0.05, 0.12])
        stats = calculate_ic_statistics(ic_series)
        self.assertAlmostEqual(stats["mean"], 0.124, places=3)
        self.assertGreater(stats["ir"], 0)

    def test_all_positive(self):
        ic_series = pd.Series([0.01] * 100)
        stats = calculate_ic_statistics(ic_series)
        self.assertAlmostEqual(stats["mean"], 0.01, places=4)

    def test_few_data_points(self):
        ic_series = pd.Series([0.1])
        stats = calculate_ic_statistics(ic_series)
        self.assertEqual(stats["count"], 1)

    def test_output_keys(self):
        ic_series = pd.Series([0.1, 0.2, 0.15, 0.05, 0.12])
        stats = calculate_ic_statistics(ic_series)
        expected_keys = {"mean", "std", "t_stat", "p_value", "ir", "count"}
        self.assertEqual(set(stats.keys()), expected_keys)


class TestFactorQuantileAnalysis(unittest.TestCase):
    def setUp(self):
        self.signal_df, self.return_df = _make_sample_data()

    def test_returns_dataframe(self):
        group_returns = factor_quantile_analysis(self.signal_df, self.return_df, n_quantiles=5)
        self.assertIsInstance(group_returns, pd.DataFrame)
        self.assertEqual(len(group_returns), 5)

    def test_quantile_order(self):
        group_returns = factor_quantile_analysis(self.signal_df, self.return_df, n_quantiles=5)
        self.assertEqual(group_returns.index[0], "Q1")
        self.assertEqual(group_returns.index[-1], "Q5")

    def test_3_quantiles(self):
        group_returns = factor_quantile_analysis(self.signal_df, self.return_df, n_quantiles=3)
        self.assertEqual(len(group_returns), 3)


class TestCalculateLongShortPortfolio(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="B")
        data = {}
        for i in range(5):
            data[f"Q{i+1}"] = np.random.randn(100) * 0.01
        self.group_returns = pd.DataFrame(data, index=dates).T

    def test_returns_dict(self):
        result = calculate_long_short_portfolio(self.group_returns)
        self.assertIsInstance(result, dict)
        expected_keys = {"total_return", "annualized_return", "annualized_volatility",
                          "sharpe_ratio", "max_drawdown", "daily_win_rate"}
        for key in expected_keys:
            self.assertIn(key, result, f"Missing key: {key}")

    def test_empty_input(self):
        result = calculate_long_short_portfolio(pd.DataFrame())
        self.assertEqual(result, {})

    def test_single_row(self):
        df = pd.DataFrame({"2023-01-01": [0.01]}, index=["Q1"])
        result = calculate_long_short_portfolio(df)
        self.assertEqual(result, {})


class TestCalculateGroupTurnover(unittest.TestCase):
    def test_no_change(self):
        dates = pd.date_range("2023-01-01", periods=5, freq="B")
        data = {d: pd.Series([0, 0, 1, 1, 2, 2], index=[f"A{i}" for i in range(6)])
                for d in dates}
        assignments = pd.DataFrame(data)
        turnover = calculate_group_turnover(assignments)
        self.assertEqual(turnover, 0.0)

    def test_complete_change(self):
        dates = pd.date_range("2023-01-01", periods=3, freq="B")
        data = {
            dates[0]: pd.Series([0, 0, 1, 1], index=["A", "B", "C", "D"]),
            dates[1]: pd.Series([1, 1, 0, 0], index=["A", "B", "C", "D"]),
            dates[2]: pd.Series([0, 0, 1, 1], index=["A", "B", "C", "D"]),
        }
        assignments = pd.DataFrame(data)
        turnover = calculate_group_turnover(assignments)
        self.assertAlmostEqual(turnover, 1.0, places=4)

    def test_empty_input(self):
        turnover = calculate_group_turnover(pd.DataFrame())
        self.assertEqual(turnover, 0.0)


if __name__ == "__main__":
    unittest.main()
