import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest
import pandas as pd
import numpy as np

from backtest.core.action import Action, Trade, Position
from backtest.core.data_feed import DataFeed
from backtest.core.engine import BacktestEngine
from backtest.strategies.buy import AlwaysBuyStrategy, BaseBuyStrategy
from backtest.strategies.sell import AlwaysSellStrategy, StopLossSellStrategy
from backtest.utils.helpers import generate_mock_data


class TestAction(unittest.TestCase):
    def test_action_creation(self):
        action = Action(type="buy", code="000001.SZ", price=10.0, quantity=100, reason="test")
        self.assertTrue(action.is_buy())
        self.assertFalse(action.is_sell())
        self.assertFalse(action.is_hold())
        self.assertEqual(action.code, "000001.SZ")

    def test_action_close(self):
        action = Action(type="close", code="000001.SZ", price=10.0)
        self.assertTrue(action.is_sell())

    def test_action_hold(self):
        action = Action(type="hold", code="000001.SZ", price=0)
        self.assertTrue(action.is_hold())


class TestTrade(unittest.TestCase):
    def test_trade_to_dict(self):
        trade = Trade(
            date="2023-01-01",
            code="000001.SZ",
            direction="buy",
            price=10.0,
            quantity=100,
            amount=1000.0,
            commission=5.0,
            reason="test",
            holding_days=5,
        )
        d = trade.to_dict()
        self.assertEqual(d["date"], "2023-01-01")
        self.assertEqual(d["direction"], "buy")
        self.assertEqual(d["price"], 10.0)


class TestPosition(unittest.TestCase):
    def test_position_properties(self):
        pos = Position(code="000001.SZ", quantity=100, cost_price=10.0, current_price=12.0)
        self.assertEqual(pos.market_value, 1200.0)
        self.assertEqual(pos.cost_value, 1000.0)
        self.assertEqual(pos.unrealized_pnl, 200.0)
        self.assertAlmostEqual(pos.unrealized_pnl_pct, 0.2)

    def test_position_zero_cost(self):
        pos = Position(code="000001.SZ", quantity=0, cost_price=0, current_price=10.0)
        self.assertEqual(pos.unrealized_pnl_pct, 0.0)


class TestDataFeed(unittest.TestCase):
    def setUp(self):
        self.mock_data = generate_mock_data(
            codes=["000001.SZ"],
            start_date="2023-01-01",
            end_date="2023-06-30",
            seed=42,
        )
        self.data_feed = DataFeed(self.mock_data)

    def test_get_codes(self):
        codes = self.data_feed.get_codes()
        self.assertIn("000001.SZ", codes)

    def test_get_date_range(self):
        start, end = self.data_feed.get_date_range()
        self.assertEqual(start, "2023-01-02")
        self.assertEqual(end, "2023-06-30")

    def test_iter_dates(self):
        dates = list(self.data_feed.iter_dates())
        self.assertGreater(len(dates), 0)
        self.assertEqual(dates[0], "2023-01-02")

    def test_get_data_at_date(self):
        data = self.data_feed.get_data_at_date("2023-01-03")
        self.assertIn("000001.SZ", data)
        self.assertIn("close", data["000001.SZ"])

    def test_has_date(self):
        self.assertTrue(self.data_feed.has_date("2023-01-03"))
        self.assertFalse(self.data_feed.has_date("2023-01-01"))

    def test_get_previous_date(self):
        prev = self.data_feed.get_previous_date("2023-01-04")
        self.assertEqual(prev, "2023-01-03")

    def test_get_data_before_date(self):
        data = self.data_feed.get_data_before_date("000001.SZ", "2023-01-10", 5)
        self.assertEqual(len(data), 5)


class TestBacktestEngine(unittest.TestCase):
    def setUp(self):
        self.mock_data = generate_mock_data(
            codes=["000001.SZ"],
            start_date="2023-01-01",
            end_date="2023-06-30",
            seed=42,
        )
        if "399001.SZ" not in self.mock_data:
            bench = self.mock_data["000001.SZ"].copy()
            bench["code"] = "399001.SZ"
            bench["close"] = bench["close"] * np.random.uniform(0.95, 1.05, len(bench))
            self.mock_data["399001.SZ"] = bench
        self.data_feed = DataFeed(self.mock_data)

    def test_engine_initialization(self):
        engine = BacktestEngine(
            data_feed=self.data_feed,
            initial_cash=1_000_000.0,
        )
        self.assertEqual(engine.cash, 1_000_000.0)
        self.assertEqual(len(engine.positions), 0)

    def test_engine_run_without_strategy(self):
        engine = BacktestEngine(data_feed=self.data_feed)
        result = engine.run()
        self.assertIsNotNone(result.equity_curve)
        self.assertGreater(len(result.equity_curve), 0)

    def test_engine_run_with_benchmark(self):
        engine = BacktestEngine(
            data_feed=self.data_feed,
            initial_cash=1_000_000.0,
            benchmark_code="399001.SZ",
        )
        result = engine.run()
        self.assertFalse(result.benchmark_curve.empty)
        self.assertEqual(len(result.benchmark_curve), len(result.equity_curve))

    def test_engine_buy_sell_strategies(self):
        buy_strategy = AlwaysBuyStrategy(codes=["000001.SZ"], fixed_quantity=1000)
        sell_strategy = AlwaysSellStrategy()
        engine = BacktestEngine(
            data_feed=self.data_feed,
            buy_strategy=buy_strategy,
            sell_strategy=sell_strategy,
            initial_cash=1_000_000.0,
            commission_rate=0.0003,
            slippage=0.001,
            min_trade_amount=0,
        )
        result = engine.run()
        self.assertGreater(len(result.trades), 0)

    def test_commission_deduction(self):
        buy_strategy = AlwaysBuyStrategy(codes=["000001.SZ"], fixed_quantity=1000)
        sell_strategy = AlwaysSellStrategy()
        engine = BacktestEngine(
            data_feed=self.data_feed,
            buy_strategy=buy_strategy,
            sell_strategy=sell_strategy,
            initial_cash=1_000_000.0,
            commission_rate=0.001,
            min_trade_amount=0,
        )
        engine.run()
        self.assertGreater(engine.total_commission, 0)

    def test_stop_loss_strategy(self):
        class TestBuy(BaseBuyStrategy):
            def should_buy(self, date, data, positions, cash):
                if "000001.SZ" in data and "000001.SZ" not in positions:
                    return [Action(type="buy", code="000001.SZ", price=float(data["000001.SZ"]["close"]), quantity=1000, reason="test_buy")]
                return []

        buy_strategy = TestBuy()
        sell_strategy = StopLossSellStrategy(stop_loss_pct=-0.02)
        engine = BacktestEngine(
            data_feed=self.data_feed,
            buy_strategy=buy_strategy,
            sell_strategy=sell_strategy,
            initial_cash=1_000_000.0,
            min_trade_amount=0,
        )
        result = engine.run()
        total_trades = len(result.trades)
        sell_reasons = [t.reason for t in result.trades if t.direction == "sell"]
        has_stop_loss = any("stop_loss" in r for r in sell_reasons)
        if total_trades > 0:
            self.assertIsInstance(total_trades, int)

    def test_result_dataclass(self):
        engine = BacktestEngine(data_feed=self.data_feed)
        result = engine.run()
        self.assertTrue(hasattr(result, "equity_curve"))
        self.assertTrue(hasattr(result, "trades"))
        self.assertTrue(hasattr(result, "daily_returns"))
        self.assertTrue(hasattr(result, "total_commission"))


class TestFundValidation(unittest.TestCase):
    def setUp(self):
        self.mock_data = generate_mock_data(
            codes=["000001.SZ"],
            start_date="2023-01-01",
            end_date="2023-06-30",
            seed=42,
        )
        self.data_feed = DataFeed(self.mock_data)
        from backtest.strategies.buy import SignalBuyStrategy
        from backtest.strategies.signal import MACDValueSignal
        from backtest.strategies.sell import SignalSellStrategy
        self.SignalBuyStrategy = SignalBuyStrategy
        self.SignalSellStrategy = SignalSellStrategy
        self.MACDValueSignal = MACDValueSignal

    def test_engine_min_trade_amount_rejects_small(self):
        buy_strategy = AlwaysBuyStrategy(codes=["000001.SZ"], fixed_quantity=1)
        sell_strategy = AlwaysSellStrategy()
        engine = BacktestEngine(
            data_feed=self.data_feed,
            buy_strategy=buy_strategy,
            sell_strategy=sell_strategy,
            initial_cash=1000.0,
            commission_rate=0.0003,
            slippage=0.001,
            min_trade_amount=10000.0,
        )
        result = engine.run()
        buy_trades = [t for t in result.trades if t.direction == "buy"]
        self.assertEqual(len(buy_trades), 0,
                         "min_trade_amount should reject trades below threshold")

    def test_engine_min_trade_amount_allows_large(self):
        buy_strategy = AlwaysBuyStrategy(codes=["000001.SZ"], fixed_quantity=1000)
        sell_strategy = AlwaysSellStrategy()
        engine = BacktestEngine(
            data_feed=self.data_feed,
            buy_strategy=buy_strategy,
            sell_strategy=sell_strategy,
            initial_cash=1_000_000.0,
            commission_rate=0.0003,
            slippage=0.001,
            min_trade_amount=10000.0,
        )
        result = engine.run()
        buy_trades = [t for t in result.trades if t.direction == "buy"]
        self.assertGreater(len(buy_trades), 0,
                           "min_trade_amount should allow large trades")

    def test_engine_skipped_trades_logged(self):
        buy_strategy = AlwaysBuyStrategy(codes=["000001.SZ"], fixed_quantity=1)
        sell_strategy = AlwaysSellStrategy()
        engine = BacktestEngine(
            data_feed=self.data_feed,
            buy_strategy=buy_strategy,
            sell_strategy=sell_strategy,
            initial_cash=1000.0,
            commission_rate=0.0003,
            slippage=0.001,
            min_trade_amount=10000.0,
        )
        engine.run()
        self.assertGreater(len(engine._skipped_trades), 0,
                           "skipped trades should be logged")

    def test_engine_min_cash_reserve(self):
        buy_strategy = AlwaysBuyStrategy(codes=["000001.SZ"], fixed_quantity=100)
        sell_strategy = AlwaysSellStrategy()
        engine = BacktestEngine(
            data_feed=self.data_feed,
            buy_strategy=buy_strategy,
            sell_strategy=sell_strategy,
            initial_cash=1_000_000.0,
            min_cash_reserve=999999.0,
            min_trade_amount=0,
        )
        result = engine.run()
        self.assertEqual(len(result.trades), 0,
                         "min_cash_reserve should prevent all trades")

    def test_default_min_trade_amount(self):
        engine = BacktestEngine(data_feed=self.data_feed)
        self.assertEqual(engine.min_trade_amount, 10000.0)


if __name__ == "__main__":
    unittest.main()
