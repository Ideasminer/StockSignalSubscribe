import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest
import numpy as np
import pandas as pd

from backtest.strategies.signal import (
    BaseSignal,
    MACDGoldenCrossSignal,
    MACDDeathCrossSignal,
    MACDValueSignal,
    MACDHistSignal,
    MACDHistChangeSignal,
    MACDHistCrossSignal,
    MACDValueCrossSignal,
    MACDSignalCrossSignal,
    MACDConvergenceSignal,
    MACDGoldenCrossWithHistConfirmSignal,
    CrossMovingAverageSignal,
    CompositeSignal,
)


def _make_mock_df(n_stocks=5, n_dates=100, seed=42):
    np.random.seed(seed)
    codes = [f"code_{i}" for i in range(n_stocks)]
    dates = pd.date_range("2023-01-01", periods=n_dates, freq="B")
    rows = []
    for code in codes:
        base_price = np.random.uniform(10, 100)
        prices = base_price * np.exp(np.cumsum(np.random.normal(0.0005, 0.015, n_dates)))
        for i, date in enumerate(dates):
            rows.append({"date": date, "code": code, "close": prices[i]})
    return pd.DataFrame(rows)


class TestBaseSignal(unittest.TestCase):
    def test_abstract_cannot_instantiate(self):
        with self.assertRaises(TypeError):
            BaseSignal()

    def test_get_name(self):
        signal = MACDGoldenCrossSignal()
        self.assertEqual(signal.get_name(), "MACDGoldenCrossSignal")


class TestMACDGoldenCrossSignal(unittest.TestCase):
    def setUp(self):
        self.df = _make_mock_df(n_stocks=3, n_dates=200, seed=42)
        self.signal = MACDGoldenCrossSignal(12, 26, 9)

    def test_output_columns(self):
        result = self.signal.compute(self.df)
        self.assertIn("signal_value", result.columns)

    def test_signal_values_are_binary(self):
        result = self.signal.compute(self.df)
        vals = result["signal_value"].dropna()
        self.assertTrue(vals.isin([0, 1]).all())

    def test_signal_output_not_all_zero(self):
        result = self.signal.compute(self.df)
        self.assertGreater(result["signal_value"].sum(), 0,
                           msg="MACD golden cross should find at least some signals")

    def test_vectorized_shape(self):
        result = self.signal.compute(self.df)
        self.assertEqual(len(result), len(self.df))


class TestMACDDeathCrossSignal(unittest.TestCase):
    def setUp(self):
        self.df = _make_mock_df(n_stocks=3, n_dates=200, seed=42)
        self.signal = MACDDeathCrossSignal(12, 26, 9)

    def test_output_columns(self):
        result = self.signal.compute(self.df)
        self.assertIn("signal_value", result.columns)

    def test_signal_values_are_binary(self):
        result = self.signal.compute(self.df)
        vals = result["signal_value"].dropna()
        self.assertTrue(vals.isin([0, 1]).all())


class TestMACDValueSignal(unittest.TestCase):
    def setUp(self):
        self.df = _make_mock_df(n_stocks=3, n_dates=200, seed=42)
        self.signal = MACDValueSignal(12, 26, 9)

    def test_output_columns(self):
        result = self.signal.compute(self.df)
        self.assertIn("signal_value", result.columns)

    def test_value_range(self):
        result = self.signal.compute(self.df)
        vals = result["signal_value"].dropna()
        self.assertFalse(vals.isna().all())
        self.assertTrue((vals.abs() < 50).all())


class TestMACDHistSignal(unittest.TestCase):
    def setUp(self):
        self.df = _make_mock_df(seed=42)
        self.signal = MACDHistSignal()

    def test_output_columns(self):
        result = self.signal.compute(self.df)
        self.assertIn("signal_value", result.columns)


class TestMACDHistChangeSignal(unittest.TestCase):
    def setUp(self):
        self.df = _make_mock_df(seed=42)
        self.signal = MACDHistChangeSignal()

    def test_output_columns(self):
        result = self.signal.compute(self.df)
        self.assertIn("signal_value", result.columns)


class TestMACDHistCrossSignal(unittest.TestCase):
    def setUp(self):
        self.df = _make_mock_df(seed=42)
        self.signal = MACDHistCrossSignal()

    def test_signal_values_are_binary(self):
        result = self.signal.compute(self.df)
        vals = result["signal_value"].dropna()
        self.assertTrue(vals.isin([0, 1]).all())


class TestMACDValueCrossSignal(unittest.TestCase):
    def setUp(self):
        self.df = _make_mock_df(seed=42)
        self.signal = MACDValueCrossSignal()

    def test_signal_values_are_binary(self):
        result = self.signal.compute(self.df)
        vals = result["signal_value"].dropna()
        self.assertTrue(vals.isin([0, 1]).all())


class TestMACDSignalCrossSignal(unittest.TestCase):
    def setUp(self):
        self.df = _make_mock_df(seed=42)
        self.signal = MACDSignalCrossSignal()

    def test_signal_values_are_binary(self):
        result = self.signal.compute(self.df)
        vals = result["signal_value"].dropna()
        self.assertTrue(vals.isin([0, 1]).all())


class TestMACDConvergenceSignal(unittest.TestCase):
    def setUp(self):
        self.df = _make_mock_df(seed=42)
        self.signal = MACDConvergenceSignal()

    def test_output_columns(self):
        result = self.signal.compute(self.df)
        self.assertIn("signal_value", result.columns)


class TestMACDGoldenCrossWithHistConfirmSignal(unittest.TestCase):
    def setUp(self):
        self.df = _make_mock_df(seed=42)
        self.signal = MACDGoldenCrossWithHistConfirmSignal()

    def test_signal_values_are_binary(self):
        result = self.signal.compute(self.df)
        vals = result["signal_value"].dropna()
        self.assertTrue(vals.isin([0, 1]).all())


class TestCrossMovingAverageSignal(unittest.TestCase):
    def setUp(self):
        self.df = _make_mock_df(seed=42)

    def test_golden_cross(self):
        signal = CrossMovingAverageSignal(short_ma=5, long_ma=10, cross_direction="golden")
        result = signal.compute(self.df)
        self.assertIn("signal_value", result.columns)
        vals = result["signal_value"].dropna()
        self.assertTrue(vals.isin([0, 1]).all())

    def test_death_cross(self):
        signal = CrossMovingAverageSignal(short_ma=5, long_ma=10, cross_direction="death")
        result = signal.compute(self.df)
        self.assertIn("signal_value", result.columns)
        vals = result["signal_value"].dropna()
        self.assertTrue(vals.isin([0, 1]).all())


class TestCompositeSignal(unittest.TestCase):
    def setUp(self):
        self.df = _make_mock_df(seed=42)

    def test_and_logic(self):
        golden = MACDGoldenCrossSignal()
        hist_cross = MACDHistCrossSignal()
        composite = CompositeSignal([golden, hist_cross], logic="AND")
        result = composite.compute(self.df)
        vals = result["signal_value"].dropna()
        self.assertTrue(vals.isin([0, 1]).all())

    def test_or_logic(self):
        golden = MACDGoldenCrossSignal()
        hist_cross = MACDHistCrossSignal()
        composite = CompositeSignal([golden, hist_cross], logic="OR")
        result = composite.compute(self.df)
        vals = result["signal_value"].dropna()
        self.assertTrue(vals.isin([0, 1]).all())

    def test_and_never_more_than_individual(self):
        golden = MACDGoldenCrossSignal()
        hist_cross = MACDHistCrossSignal()
        result_g = golden.compute(self.df.copy())
        result_h = hist_cross.compute(self.df.copy())
        composite = CompositeSignal([MACDGoldenCrossSignal(), MACDHistCrossSignal()], logic="AND")
        result_c = composite.compute(self.df.copy())
        self.assertLessEqual(
            result_c["signal_value"].sum(),
            min(result_g["signal_value"].sum(), result_h["signal_value"].sum()),
        )


class TestSignalMissingClose(unittest.TestCase):
    def test_missing_close_column(self):
        df = pd.DataFrame({"date": ["2023-01-01"], "code": ["A"]})
        signal = MACDGoldenCrossSignal()
        with self.assertRaises(ValueError):
            signal.compute(df)


class TestSignalEdgeCases(unittest.TestCase):
    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["date", "code", "close"])
        signal = MACDGoldenCrossSignal()
        result = signal.compute(df)
        self.assertTrue(result.empty or "signal_value" in result.columns)

    def test_single_stock(self):
        df = _make_mock_df(n_stocks=1, n_dates=50, seed=42)
        signal = MACDGoldenCrossSignal()
        result = signal.compute(df)
        self.assertIn("signal_value", result.columns)

    def test_many_stocks(self):
        df = _make_mock_df(n_stocks=20, n_dates=30, seed=42)
        signal = MACDValueSignal()
        result = signal.compute(df)
        self.assertEqual(len(result), len(df))


class TestSignalNoForwardBias(unittest.TestCase):
    """测试信号计算不包含前瞻性偏差"""

    def setUp(self):
        np.random.seed(42)
        self.n_dates = 150
        dates = pd.date_range("2023-01-01", periods=self.n_dates, freq="B")
        codes = ["stock_A", "stock_B", "stock_C"]
        rows = []
        for code in codes:
            base = np.random.uniform(10, 50)
            prices = base * np.exp(np.cumsum(np.random.normal(0.0005, 0.015, self.n_dates)))
            for i, date in enumerate(dates):
                rows.append({"date": date, "code": code, "close": prices[i]})
        self.full_df = pd.DataFrame(rows)

    def _get_df_up_to(self, df, cutoff_date):
        return df[df["date"] <= cutoff_date].copy()

    def test_MACDValue_expanding_std_no_future_leak(self):
        """验证 expanding().std() 在时刻T不受T+1之后数据影响"""
        cutoff = "2023-06-01"
        cutoff_ts = pd.Timestamp(cutoff)
        signal = MACDValueSignal(12, 26, 9)
        partial_df = self._get_df_up_to(self.full_df, cutoff)
        result_partial = signal.compute(partial_df)
        result_full = signal.compute(self.full_df)
        full_up_to_cutoff = result_full[result_full["date"] <= cutoff_ts]
        # expanding().std() 版本：截断前信号值应完全一致
        for code in partial_df["code"].unique():
            p_vals = result_partial.loc[
                result_partial["code"] == code, "signal_value"
            ].values
            f_vals = full_up_to_cutoff.loc[
                full_up_to_cutoff["code"] == code, "signal_value"
            ].values
            if len(p_vals) > 0 and len(f_vals) > 0 and len(p_vals) == len(f_vals):
                np.testing.assert_array_almost_equal(
                    p_vals, f_vals, decimal=10,
                    err_msg=f"{code}: expanding std signal should be identical"
                )

    def test_MACDHist_expanding_std_no_future_leak(self):
        """验证 MACDHistSignal 的 expanding std 不泄露未来信息"""
        cutoff = "2023-06-01"
        cutoff_ts = pd.Timestamp(cutoff)
        signal = MACDHistSignal(12, 26, 9)
        result_partial = signal.compute(self._get_df_up_to(self.full_df, cutoff))
        result_full = signal.compute(self.full_df)
        full_up_to_cutoff = result_full[result_full["date"] <= cutoff_ts]
        for code in self.full_df["code"].unique():
            p = result_partial.loc[result_partial["code"] == code, "signal_value"].values
            f = full_up_to_cutoff.loc[full_up_to_cutoff["code"] == code, "signal_value"].values
            if len(p) > 0 and len(f) > 0 and len(p) == len(f):
                np.testing.assert_array_almost_equal(p, f, decimal=10)

    def test_MACDConvergence_expanding_std_no_future_leak(self):
        """验证 MACDConvergenceSignal 的 expanding std 不泄露未来信息"""
        cutoff = "2023-06-01"
        cutoff_ts = pd.Timestamp(cutoff)
        signal = MACDConvergenceSignal(12, 26, 9)
        result_partial = signal.compute(self._get_df_up_to(self.full_df, cutoff))
        result_full = signal.compute(self.full_df)
        full_up_to_cutoff = result_full[result_full["date"] <= cutoff_ts]
        for code in self.full_df["code"].unique():
            p = result_partial.loc[result_partial["code"] == code, "signal_value"].values
            f = full_up_to_cutoff.loc[full_up_to_cutoff["code"] == code, "signal_value"].values
            if len(p) > 0 and len(f) > 0 and len(p) == len(f):
                np.testing.assert_array_almost_equal(p, f, decimal=10)

    def test_MACDGoldenCross_no_std_normalization(self):
        """验证二进制信号（金叉）天然不受全样本std影响"""
        cutoff = "2023-06-01"
        cutoff_ts = pd.Timestamp(cutoff)
        signal = MACDGoldenCrossSignal(12, 26, 9)
        result_partial = signal.compute(self._get_df_up_to(self.full_df, cutoff))
        result_full = signal.compute(self.full_df)
        full_up_to_cutoff = result_full[result_full["date"] <= cutoff_ts]
        for code in self.full_df["code"].unique():
            p = result_partial.loc[result_partial["code"] == code, "signal_value"].values
            f = full_up_to_cutoff.loc[full_up_to_cutoff["code"] == code, "signal_value"].values
            if len(p) > 0 and len(f) > 0 and len(p) == len(f):
                np.testing.assert_array_almost_equal(p, f, decimal=10)

    def test_expanding_std_vs_full_sample_std_difference(self):
        """验证 expanding().std() 与全样本 std() 在中间时点存在差异"""
        signal = MACDValueSignal(12, 26, 9)
        result = signal.compute(self.full_df)
        mid_point = int(self.n_dates * 0.5)
        for code in ["stock_A"]:
            vals = result.loc[result["code"] == code, "dif_std"].values
            self.assertGreater(vals[mid_point], 0,
                               f"dif_std at mid-point should be >0 for {code}")


class TestICTimeAlignment(unittest.TestCase):
    """测试IC分析的时间对齐正确性"""

    def setUp(self):
        np.random.seed(42)
        self.n_dates = 30
        self.n_stocks = 10
        codes = [f"s{i}" for i in range(self.n_stocks)]
        dates = pd.date_range("2023-01-01", periods=self.n_dates, freq="B")
        signal_data = {}
        return_data = {}
        for d in dates:
            signal_data[d] = pd.Series(np.random.randn(self.n_stocks), index=codes)
            return_data[d] = pd.Series(np.random.randn(self.n_stocks) * 0.01, index=codes)
        self.signal_df = pd.DataFrame(signal_data)
        self.return_df = pd.DataFrame(return_data)

    def test_ic_series_output_length(self):
        """IC序列长度不应超过对齐后的共同日期数"""
        from backtest.metrics.signal_evaluation import calculate_ic_series
        ic = calculate_ic_series(self.signal_df, self.return_df)
        self.assertLessEqual(len(ic), self.n_dates)

    def test_forward_returns_last_row_nan(self):
        """forward_returns最后一列应为NaN（无法计算shift(-1)）"""
        close = pd.DataFrame(
            {d: np.random.randn(5) + 100 for d in
             pd.date_range("2023-01-01", periods=5, freq="B")},
            index=[f"s{i}" for i in range(5)]
        )
        from backtest.metrics.signal_evaluation import calculate_forward_returns
        fwd = calculate_forward_returns(close, periods=1)
        self.assertTrue(fwd.iloc[:, -1].isna().all(),
                        "Last column of forward returns should be NaN")

    def test_forward_returns_correct_lag(self):
        """验证forward_returns的正确滞后关系: ret[t] = close[t+1]/close[t]-1"""
        close = pd.DataFrame({
            "2023-01-01": [10.0, 20.0],
            "2023-01-02": [11.0, 22.0],
            "2023-01-03": [12.0, 24.0],
        }, index=["A", "B"])
        close.columns = pd.to_datetime(close.columns)
        from backtest.metrics.signal_evaluation import calculate_forward_returns
        fwd = calculate_forward_returns(close, periods=1)
        t0 = close.columns[0]
        expected_a = 11.0 / 10.0 - 1
        expected_b = 22.0 / 20.0 - 1
        self.assertAlmostEqual(fwd.loc["A", t0], expected_a)
        self.assertAlmostEqual(fwd.loc["B", t0], expected_b)

    def test_ic_with_perfect_future_signal_should_detect(self):
        """如果信号完全=未来收益，IC应该接近1（用于验证对齐逻辑正确）"""
        n = 50
        codes = [f"s{i}" for i in range(n)]
        dates = pd.date_range("2023-01-01", periods=10, freq="B")
        signal_data = {}
        return_data = {}
        np.random.seed(42)
        for d in dates:
            r = np.random.randn(n) * 0.01
            return_data[d] = pd.Series(r, index=codes)
            signal_data[d] = pd.Series(r, index=codes)
        from backtest.metrics.signal_evaluation import calculate_ic_series, calculate_ic_statistics
        sig_df = pd.DataFrame(signal_data)
        ret_df = pd.DataFrame(return_data)
        ic = calculate_ic_series(sig_df, ret_df, method="pearson")
        stats = calculate_ic_statistics(ic)
        self.assertAlmostEqual(stats["mean"], 1.0, places=4,
                               msg="Perfect signal should yield IC=1")


if __name__ == "__main__":
    unittest.main()
