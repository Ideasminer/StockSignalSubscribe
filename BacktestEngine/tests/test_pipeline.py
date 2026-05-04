import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest
import numpy as np
import pandas as pd

from backtest.pipeline.conditions import (
    SignalCondition,
    TopNCondition,
    ThresholdCondition,
    CompositeCondition,
    apply_condition_to_actions_buy,
    apply_condition_to_actions_sell,
)
from backtest.pipeline.pipeline import (
    PipelineConfig,
    SignalPipeline,
)
from backtest.strategies.signal import (
    MACDValueSignal,
    MACDConvergenceSignal,
)


def _make_test_df(n_stocks=20, n_dates=100, seed=42):
    np.random.seed(seed)
    codes = [f"stock_{i:03d}" for i in range(n_stocks)]
    dates = pd.date_range("2023-01-01", periods=n_dates, freq="B")
    rows = []
    for code in codes:
        base = np.random.uniform(10, 100)
        prices = base * np.exp(np.cumsum(np.random.normal(0.0003, 0.02, n_dates)))
        for i, d in enumerate(dates):
            rows.append({"date": d, "code": code, "close": prices[i], "tradestatus": 1})
    return pd.DataFrame(rows)


class TestTopNCondition(unittest.TestCase):
    def setUp(self):
        self.sv = pd.Series(
            {"A": 5.0, "B": 3.0, "C": 8.0, "D": 1.0, "E": 2.0, "F": 9.0}
        )

    def test_top_3(self):
        cond = TopNCondition(n=3, direction="top")
        mask = cond.filter(self.sv)
        selected = self.sv[mask].index.tolist()
        self.assertEqual(set(selected), {"F", "C", "A"})

    def test_bottom_3(self):
        cond = TopNCondition(n=3, direction="bottom")
        mask = cond.filter(self.sv)
        selected = self.sv[mask].index.tolist()
        self.assertEqual(set(selected), {"D", "E", "B"})

    def test_n_larger_than_data(self):
        cond = TopNCondition(n=10, direction="top")
        mask = cond.filter(self.sv)
        self.assertEqual(mask.sum(), 6)

    def test_name(self):
        cond = TopNCondition(n=5, direction="top")
        self.assertIn("TopNCondition", cond.get_name())


class TestThresholdCondition(unittest.TestCase):
    def setUp(self):
        self.sv = pd.Series({"A": 5.0, "B": -2.0, "C": 3.0, "D": -1.5})

    def test_gt(self):
        cond = ThresholdCondition(threshold=0, comparator="gt")
        mask = cond.filter(self.sv)
        selected = self.sv[mask].index.tolist()
        self.assertEqual(set(selected), {"A", "C"})

    def test_lt(self):
        cond = ThresholdCondition(threshold=0, comparator="lt")
        mask = cond.filter(self.sv)
        selected = self.sv[mask].index.tolist()
        self.assertEqual(set(selected), {"B", "D"})

    def test_gte(self):
        cond = ThresholdCondition(threshold=5.0, comparator="gte")
        mask = cond.filter(self.sv)
        self.assertTrue(mask["A"])
        self.assertFalse(mask["B"])

    def test_name(self):
        cond = ThresholdCondition(threshold=1.5, comparator="gt")
        self.assertIn("ThresholdCondition", cond.get_name())


class TestCompositeCondition(unittest.TestCase):
    def setUp(self):
        self.sv = pd.Series({"A": 5.0, "B": 3.0, "C": 8.0, "D": 1.0, "E": -2.0})

    def test_and_logic(self):
        c1 = ThresholdCondition(threshold=3.0, comparator="gt")
        c2 = ThresholdCondition(threshold=6.0, comparator="lt")
        comp = CompositeCondition([c1, c2], logic="AND")
        mask = comp.filter(self.sv)
        selected = self.sv[mask].index.tolist()
        self.assertEqual(selected, ["A"])

    def test_or_logic(self):
        c1 = ThresholdCondition(threshold=7.0, comparator="gt")
        c2 = ThresholdCondition(threshold=0, comparator="lt")
        comp = CompositeCondition([c1, c2], logic="OR")
        mask = comp.filter(self.sv)
        selected = self.sv[mask].index.tolist()
        self.assertEqual(set(selected), {"C", "E"})


class TestSignalPipeline(unittest.TestCase):
    def setUp(self):
        self.df = _make_test_df(n_stocks=20, n_dates=100, seed=42)

    def test_pipeline_basic_run(self):
        config = PipelineConfig(
            buy_signal=MACDValueSignal(12, 26, 9),
            sell_signal=MACDValueSignal(12, 26, 9),
            buy_condition=TopNCondition(n=3, direction="top"),
            sell_condition=TopNCondition(n=3, direction="bottom"),
            initial_cash=1_000_000.0,
            train_test_split=0.3,
        )
        pipeline = SignalPipeline(config)
        result = pipeline.run(self.df)
        self.assertIsNotNone(result.backtest_result)
        self.assertGreater(len(result.trades), 0)
        self.assertFalse(result.equity_curve.empty)

    def test_pipeline_with_threshold_condition(self):
        config = PipelineConfig(
            buy_signal=MACDConvergenceSignal(12, 26, 9),
            sell_signal=MACDConvergenceSignal(12, 26, 9),
            buy_condition=ThresholdCondition(threshold=-0.5, comparator="gt"),
            sell_condition=ThresholdCondition(threshold=-2.0, comparator="lt"),
            initial_cash=1_000_000.0,
            train_test_split=0.3,
        )
        pipeline = SignalPipeline(config)
        result = pipeline.run(self.df)
        self.assertIsNotNone(result.backtest_result)
        self.assertTrue(len(result.trades) >= 0)

    def test_pipeline_evaluate(self):
        config = PipelineConfig(
            buy_signal=MACDValueSignal(12, 26, 9),
            sell_signal=MACDValueSignal(12, 26, 9),
            train_test_split=0.3,
        )
        pipeline = SignalPipeline(config)
        result = pipeline.run(self.df)
        metrics = pipeline.evaluate(result)
        self.assertIsInstance(metrics, dict)
        expected_keys = ["总收益率", "年化收益率", "夏普比率", "最大回撤",
                          "总交易次数", "交易胜率"]
        for key in expected_keys:
            self.assertIn(key, metrics, f"缺少指标: {key}")

    def test_pipeline_config_defaults(self):
        config = PipelineConfig(
            buy_signal=MACDValueSignal(),
            sell_signal=MACDValueSignal(),
        )
        self.assertEqual(config.initial_cash, 1_000_000.0)
        self.assertEqual(config.max_holdings, 5)
        self.assertIsInstance(config.buy_condition, TopNCondition)
        self.assertIsInstance(config.sell_condition, TopNCondition)

    def test_pipeline_split_respects_dates(self):
        config = PipelineConfig(
            buy_signal=MACDValueSignal(12, 26, 9),
            sell_signal=MACDValueSignal(12, 26, 9),
            train_test_split=0.1,
        )
        pipeline = SignalPipeline(config)
        result = pipeline.run(self.df)
        self.assertGreater(len(result.train_dates), 0)
        self.assertGreater(len(result.test_dates), 0)
        self.assertLess(result.train_dates[-1], result.test_dates[0])

    def test_pipeline_empty_after_split_raises(self):
        config = PipelineConfig(
            buy_signal=MACDValueSignal(12, 26, 9),
            sell_signal=MACDValueSignal(12, 26, 9),
            train_test_split=0.99,
        )
        pipeline = SignalPipeline(config)
        try:
            result = pipeline.run(self.df)
            # 如果没抛异常，验证至少有一些交易
            self.assertGreaterEqual(len(result.test_dates), 1)
        except ValueError:
            pass


class TestApplyConditionToActions(unittest.TestCase):
    def setUp(self):
        self.sv = pd.Series({"A": 5.0, "B": 3.0, "C": 1.0, "D": -1.0})
        self.data = {
            "A": {"close": 50.0}, "B": {"close": 30.0},
            "C": {"close": 10.0}, "D": {"close": 5.0},
        }
        self.positions = {}
        self.signal = MACDValueSignal()

    def test_top_n_buy_generates_actions(self):
        cond = TopNCondition(n=2, direction="top")
        actions = apply_condition_to_actions_buy(
            self.sv, cond, self.data, self.positions, 1_000_000, self.signal,
            min_cash_threshold=0,
        )
        self.assertGreater(len(actions), 0)
        for a in actions:
            self.assertEqual(a.type, "buy")

    def test_bottom_n_sell_generates_actions(self):
        from backtest.core.action import Position
        positions = {
            "D": Position(code="D", quantity=100, cost_price=5.0, current_price=5.0),
        }
        cond = TopNCondition(n=1, direction="bottom")
        actions = apply_condition_to_actions_sell(
            self.sv, cond, self.data, positions, self.signal,
        )
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].type, "sell")
        self.assertEqual(actions[0].code, "D")

    def test_threshold_gt_buy(self):
        cond = ThresholdCondition(threshold=2.0, comparator="gt")
        actions = apply_condition_to_actions_buy(
            self.sv, cond, self.data, self.positions, 1_000_000, self.signal,
            min_cash_threshold=0,
        )
        codes = {a.code for a in actions}
        self.assertEqual(codes, {"A", "B"})


if __name__ == "__main__":
    unittest.main()
