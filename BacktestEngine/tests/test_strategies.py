import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest
import pandas as pd
import numpy as np

from backtest.core.data_feed import DataFeed
from backtest.core.action import Action, Position
from backtest.strategies.buy import BaseBuyStrategy, AlwaysBuyStrategy
from backtest.strategies.sell import BaseSellStrategy, AlwaysSellStrategy, StopLossSellStrategy
from backtest.strategies.selector import BaseStockSelector, AllStockSelector, TopNSelector
from backtest.strategies.base import BaseStrategy, CompositeStrategy
from backtest.utils.helpers import generate_mock_data


class TestBaseBuyStrategy(unittest.TestCase):
    def test_abstract_cannot_instantiate(self):
        with self.assertRaises(TypeError):
            BaseBuyStrategy()


class TestAlwaysBuyStrategy(unittest.TestCase):
    def test_always_buy(self):
        strategy = AlwaysBuyStrategy(codes=["000001.SZ"], fixed_quantity=100)
        data = {
            "000001.SZ": {"close": 10.0},
            "399001.SZ": {"close": 20.0},
        }
        actions = strategy.should_buy("2023-01-01", data, {}, 1_000_000)
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].code, "000001.SZ")
        self.assertEqual(actions[0].quantity, 100)

    def test_skip_existing_position(self):
        strategy = AlwaysBuyStrategy(codes=["000001.SZ"], fixed_quantity=100)
        data = {"000001.SZ": {"close": 10.0}}
        pos = {"000001.SZ": Position(code="000001.SZ", quantity=100, cost_price=10.0)}
        actions = strategy.should_buy("2023-01-01", data, pos, 1_000_000)
        self.assertEqual(len(actions), 0)


class TestBaseSellStrategy(unittest.TestCase):
    def test_abstract_cannot_instantiate(self):
        with self.assertRaises(TypeError):
            BaseSellStrategy()


class TestAlwaysSellStrategy(unittest.TestCase):
    def test_always_sell(self):
        strategy = AlwaysSellStrategy()
        data = {"000001.SZ": {"close": 10.0}}
        pos = {"000001.SZ": Position(code="000001.SZ", quantity=100, cost_price=10.0, current_price=10.0)}
        actions = strategy.should_sell("2023-01-01", data, pos, 1_000_000)
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].type, "sell")


class TestStopLossSellStrategy(unittest.TestCase):
    def test_stop_loss_triggered(self):
        strategy = StopLossSellStrategy(stop_loss_pct=-0.05)
        data = {"000001.SZ": {"close": 9.0}}
        pos = {"000001.SZ": Position(code="000001.SZ", quantity=100, cost_price=10.0, current_price=10.0)}
        actions = strategy.should_sell("2023-01-01", data, pos, 1_000_000)
        self.assertEqual(len(actions), 1)
        self.assertIn("stop_loss", actions[0].reason)

    def test_stop_loss_not_triggered(self):
        strategy = StopLossSellStrategy(stop_loss_pct=-0.05)
        data = {"000001.SZ": {"close": 9.6}}
        pos = {"000001.SZ": Position(code="000001.SZ", quantity=100, cost_price=10.0, current_price=10.0)}
        actions = strategy.should_sell("2023-01-01", data, pos, 1_000_000)
        self.assertEqual(len(actions), 0)


class TestAllStockSelector(unittest.TestCase):
    def test_select_all(self):
        selector = AllStockSelector()
        data = {"000001.SZ": {}, "399001.SZ": {}}
        codes = selector.select("2023-01-01", data)
        self.assertEqual(len(codes), 2)
        self.assertIn("000001.SZ", codes)


class TestTopNSelector(unittest.TestCase):
    def test_top_n(self):
        selector = TopNSelector(n=1)
        data = {
            "000001.SZ": {"pctChg": 1.0},
            "399001.SZ": {"pctChg": 2.0},
        }
        codes = selector.select("2023-01-01", data)
        self.assertEqual(len(codes), 1)


class TestCompositeStrategy(unittest.TestCase):
    def setUp(self):
        self.mock_data = generate_mock_data(
            codes=["000001.SZ"],
            start_date="2023-01-01",
            end_date="2023-03-31",
            seed=42,
        )
        self.data_feed = DataFeed(self.mock_data)

    def test_composite_creation(self):
        strategy = CompositeStrategy(
            data_feed=self.data_feed,
            buy_strategy=AlwaysBuyStrategy(codes=["000001.SZ"], fixed_quantity=100),
            sell_strategy=AlwaysSellStrategy(),
            stock_selector=AllStockSelector(),
        )
        self.assertIsNotNone(strategy)
        strategy.init()


class TestSignalBuyStrategyFundValidation(unittest.TestCase):
    def setUp(self):
        from backtest.strategies.signal import MACDValueSignal
        dates = pd.date_range("2023-01-01", periods=10, freq="B")
        np.random.seed(42)
        codes = [f"code_{i}" for i in range(5)]
        data = {}
        for d in dates:
            data[d] = pd.Series(np.random.randn(5), index=codes)
        self.signal_df = pd.DataFrame(data)
        self.signal = MACDValueSignal()
        self.cls = None
        from backtest.strategies.buy import SignalBuyStrategy
        self.cls = SignalBuyStrategy

    def test_min_cash_threshold_rejects(self):
        strategy = self.cls(
            signal=self.signal,
            signal_df=self.signal_df,
            top_n=3,
            min_cash_threshold=100000.0,
        )
        data_dict = {f"code_{i}": {"close": 50.0} for i in range(5)}
        actions = strategy.should_buy("2023-01-05", data_dict, {}, 50000.0)
        self.assertEqual(len(actions), 0,
                         "min_cash_threshold should reject when cash below threshold")

    def test_min_cash_threshold_allows(self):
        strategy = self.cls(
            signal=self.signal,
            signal_df=self.signal_df,
            top_n=3,
            min_cash_threshold=50000.0,
        )
        data_dict = {f"code_{i}": {"close": 50.0} for i in range(5)}
        actions = strategy.should_buy("2023-01-05", data_dict, {}, 100000.0)
        self.assertGreater(len(actions), 0,
                           "min_cash_threshold should allow when cash above threshold")

    def test_max_holdings_blocks_new_buy(self):
        strategy = self.cls(
            signal=self.signal,
            signal_df=self.signal_df,
            top_n=5,
            max_holdings=3,
        )
        data_dict = {f"code_{i}": {"close": 200.0} for i in range(5)}
        positions = {
            f"code_{i}": Position(code=f"code_{i}", quantity=100, cost_price=50.0)
            for i in range(3)
        }
        actions = strategy.should_buy("2023-01-05", data_dict, positions, 1_000_000)
        self.assertEqual(len(actions), 0,
                         "max_holdings should block new buys when at capacity")

    def test_max_holdings_allows_with_room(self):
        strategy = self.cls(
            signal=self.signal,
            signal_df=self.signal_df,
            top_n=5,
            max_holdings=5,
        )
        data_dict = {f"code_{i}": {"close": 200.0} for i in range(5)}
        positions = {
            f"code_{i}": Position(code=f"code_{i}", quantity=100, cost_price=50.0)
            for i in range(2)
        }
        actions = strategy.should_buy("2023-01-05", data_dict, positions, 1_000_000)
        self.assertGreater(len(actions), 0,
                           "max_holdings should allow when under capacity")

    def test_budget_decrements_within_day(self):
        strategy = self.cls(
            signal=self.signal,
            signal_df=self.signal_df,
            top_n=3,
            min_cash_threshold=0,
        )
        data_dict = {f"code_{i}": {"close": 10.0} for i in range(3)}
        actions = strategy.should_buy("2023-01-05", data_dict, {}, 10000.0)
        total_estimated = 0
        for a in actions:
            total_estimated += a.quantity * a.price * 1.001 * 1.003
        self.assertLess(total_estimated, 10000.0 * 1.1,
                        "total estimated cost should not exceed total budget")

    def test_min_stock_value_filter(self):
        strategy = self.cls(
            signal=self.signal,
            signal_df=self.signal_df,
            top_n=3,
            min_stock_value=50000.0,
        )
        data_dict = {f"code_{i}": {"close": 5.0} for i in range(3)}
        actions = strategy.should_buy("2023-01-05", data_dict, {}, 1_000_000)
        for a in actions:
            self.assertGreaterEqual(a.quantity * a.price, 50000.0,
                                    "each trade should meet min_stock_value")


if __name__ == "__main__":
    unittest.main()
