"""新增信号类测试"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest
import numpy as np
import pandas as pd

from backtest.strategies.signal import (
    HoldingReturnThresholdSignal,
    RSISignal, CCISignal, VolumeBreakoutSignal,
    PriceMomentumSignal, BollingerBandSignal,
    ATRSignal, VolumeUpDaySignal, VolumeDownDaySignal,
    NewHighBreakoutSignal,
)


def _make_mock_df(n_stocks=5, n_dates=150, seed=42):
    np.random.seed(seed)
    codes = [f"code_{i}" for i in range(n_stocks)]
    dates = pd.date_range("2023-01-01", periods=n_dates, freq="B")
    rows = []
    for code in codes:
        base = np.random.uniform(10, 100)
        prices = base * np.exp(np.cumsum(np.random.normal(0.0005, 0.015, n_dates)))
        for i, date in enumerate(dates):
            close_val = prices[i]
            open_val = close_val * (1 + np.random.uniform(-0.01, 0.01))
            rows.append({
                "date": date, "code": code,
                "open": open_val,
                "high": max(open_val, close_val) * (1 + np.random.uniform(0, 0.01)),
                "low": min(open_val, close_val) * (1 - np.random.uniform(0, 0.01)),
                "close": close_val,
                "volume": np.random.randint(1000000, 50000000),
            })
    return pd.DataFrame(rows)


class TestHoldingReturnThresholdSignal(unittest.TestCase):
    def setUp(self):
        self.df = _make_mock_df(n_stocks=3, n_dates=100, seed=42)

    def test_with_cost_price_map(self):
        signal = HoldingReturnThresholdSignal(
            cost_price_map={"code_0": 50.0, "code_1": 30.0}
        )
        result = signal.compute(self.df)
        self.assertIn("signal_value", result.columns)
        c0_vals = result.loc[result["code"] == "code_0", "signal_value"].dropna()
        self.assertGreater(len(c0_vals), 0)

    def test_empty_cost_price_map_fallback(self):
        signal = HoldingReturnThresholdSignal(cost_price_map={}, period=5)
        result = signal.compute(self.df)
        vals = result["signal_value"].dropna()
        self.assertGreater(len(vals), 0, "fallback period return should produce non-NaN values")


class TestRSISignal(unittest.TestCase):
    def setUp(self):
        self.df = _make_mock_df(seed=42)
        self.signal = RSISignal(period=14)

    def test_output_column(self):
        result = self.signal.compute(self.df)
        self.assertIn("signal_value", result.columns)

    def test_value_range(self):
        result = self.signal.compute(self.df)
        vals = result["signal_value"].dropna()
        self.assertTrue((vals >= -2.0).all())
        self.assertTrue((vals <= 2.0).all())


class TestCCISignal(unittest.TestCase):
    def setUp(self):
        self.df = _make_mock_df(seed=42)
        self.signal = CCISignal(period=20)

    def test_output_column(self):
        result = self.signal.compute(self.df)
        self.assertIn("signal_value", result.columns)

    def test_value_range(self):
        result = self.signal.compute(self.df)
        vals = result["signal_value"].dropna()
        self.assertTrue((vals >= -3.5).all(), f"Min CCI: {vals.min()}")


class TestVolumeBreakoutSignal(unittest.TestCase):
    def setUp(self):
        self.df = _make_mock_df(seed=42)
        self.signal = VolumeBreakoutSignal(period=20, multiplier=1.5)

    def test_output_column(self):
        result = self.signal.compute(self.df)
        self.assertIn("signal_value", result.columns)


class TestPriceMomentumSignal(unittest.TestCase):
    def setUp(self):
        self.df = _make_mock_df(seed=42)
        self.signal = PriceMomentumSignal(period=20)

    def test_output_column(self):
        result = self.signal.compute(self.df)
        self.assertIn("signal_value", result.columns)

    def test_early_periods_nan(self):
        result = self.signal.compute(self.df)
        code_0 = result[result["code"] == "code_0"]
        self.assertTrue(code_0["signal_value"].iloc[:20].isna().all())


class TestBollingerBandSignal(unittest.TestCase):
    def setUp(self):
        self.df = _make_mock_df(seed=42)
        self.signal = BollingerBandSignal(period=20, k=2.0)

    def test_output_column(self):
        result = self.signal.compute(self.df)
        self.assertIn("signal_value", result.columns)

    def test_zscore_range(self):
        result = self.signal.compute(self.df)
        vals = result["signal_value"].dropna()
        self.assertTrue((vals >= -5.1).all())


class TestATRSignal(unittest.TestCase):
    def setUp(self):
        self.df = _make_mock_df(seed=42)
        self.signal = ATRSignal(period=14)

    def test_output_column(self):
        result = self.signal.compute(self.df)
        self.assertIn("signal_value", result.columns)


class TestVolumeUpDaySignal(unittest.TestCase):
    def setUp(self):
        self.df = _make_mock_df(seed=42)
        self.signal = VolumeUpDaySignal(period=20, multiplier=1.2)

    def test_binary_output(self):
        result = self.signal.compute(self.df)
        vals = result["signal_value"].dropna()
        self.assertTrue(vals.isin([0, 1]).all())


class TestVolumeDownDaySignal(unittest.TestCase):
    def setUp(self):
        self.df = _make_mock_df(seed=42)
        self.signal = VolumeDownDaySignal(period=20, multiplier=1.2)

    def test_binary_output(self):
        result = self.signal.compute(self.df)
        vals = result["signal_value"].dropna()
        self.assertTrue(vals.isin([0, 1]).all())


class TestNewHighBreakoutSignal(unittest.TestCase):
    def setUp(self):
        self.df = _make_mock_df(seed=42)
        self.signal = NewHighBreakoutSignal(period=60)

    def test_binary_output(self):
        result = self.signal.compute(self.df)
        vals = result["signal_value"].dropna()
        self.assertTrue(vals.isin([0, 1]).all())


if __name__ == "__main__":
    unittest.main()
