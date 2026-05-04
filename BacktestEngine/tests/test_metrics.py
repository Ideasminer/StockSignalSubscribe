import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest
import pandas as pd
import numpy as np

from backtest.metrics.returns import (
    calculate_total_return,
    calculate_annualized_return,
    calculate_excess_return,
    calculate_monthly_win_rate,
    calculate_yearly_win_rate,
)
from backtest.metrics.risk import (
    calculate_max_drawdown,
    calculate_max_drawdown_duration,
    calculate_annualized_volatility,
    calculate_downside_volatility,
    calculate_var,
    calculate_cvar,
)
from backtest.metrics.risk_adjusted import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_calmar_ratio,
    calculate_return_drawdown_ratio,
    calculate_information_ratio,
)
from backtest.metrics.trading import (
    calculate_win_rate,
    calculate_profit_loss_ratio,
    calculate_max_consecutive_wins,
    calculate_max_consecutive_losses,
    calculate_total_trades,
    calculate_avg_holding_period,
    calculate_commission_impact,
)
from backtest.core.action import Trade


def _make_equity_curve(values):
    dates = pd.date_range("2023-01-01", periods=len(values), freq="D")
    return pd.Series(values, index=dates)


def _make_daily_returns(values):
    dates = pd.date_range("2023-01-01", periods=len(values), freq="D")
    return pd.Series(values, index=dates)


class TestReturnsMetrics(unittest.TestCase):
    def test_total_return_positive(self):
        eq = _make_equity_curve([1.0, 1.1, 1.2])
        ret = calculate_total_return(eq)
        self.assertAlmostEqual(ret, 0.2)

    def test_total_return_negative(self):
        eq = _make_equity_curve([1.0, 0.9, 0.8])
        ret = calculate_total_return(eq)
        self.assertAlmostEqual(ret, -0.2)

    def test_total_return_zero(self):
        eq = _make_equity_curve([1.0])
        ret = calculate_total_return(eq)
        self.assertEqual(ret, 0.0)

    def test_annualized_return(self):
        eq = _make_equity_curve([1.0, 1.01, 1.02, 1.03])
        ret = calculate_annualized_return(eq, trading_days=252)
        ann_ret = (1 + 0.03) ** (252 / 4) - 1
        self.assertAlmostEqual(ret, ann_ret, places=6)

    def test_excess_return(self):
        excess = calculate_excess_return(0.2, 0.1)
        self.assertAlmostEqual(excess, 0.1)

    def test_monthly_win_rate(self):
        rets = _make_daily_returns([0.001] * 252)
        wr = calculate_monthly_win_rate(rets)
        self.assertGreaterEqual(wr, 0)
        self.assertLessEqual(wr, 1)

    def test_yearly_win_rate(self):
        rets = _make_daily_returns([0.001] * 252)
        wr = calculate_yearly_win_rate(rets)
        self.assertGreaterEqual(wr, 0)
        self.assertLessEqual(wr, 1)

    def test_monthly_win_rate_empty(self):
        wr = calculate_monthly_win_rate(pd.Series(dtype=float))
        self.assertEqual(wr, 0.0)


class TestRiskMetrics(unittest.TestCase):
    def test_max_drawdown(self):
        eq = _make_equity_curve([1.0, 1.2, 0.9, 1.1, 0.8])
        dd, start, end = calculate_max_drawdown(eq)
        self.assertAlmostEqual(dd, -0.3333, places=3)

    def test_max_drawdown_no_drawdown(self):
        eq = _make_equity_curve([1.0, 1.1, 1.2, 1.3])
        dd, start, end = calculate_max_drawdown(eq)
        self.assertAlmostEqual(dd, 0.0)

    def test_drawdown_duration(self):
        eq = _make_equity_curve([1.0, 1.2, 0.9, 0.9, 0.9, 1.2])
        duration = calculate_max_drawdown_duration(eq)
        self.assertEqual(duration, 3)

    def test_annualized_volatility(self):
        rets = _make_daily_returns([0.01, -0.01, 0.02, -0.02])
        vol = calculate_annualized_volatility(rets, trading_days=252)
        self.assertGreater(vol, 0)

    def test_annualized_volatility_short(self):
        rets = _make_daily_returns([0.01])
        vol = calculate_annualized_volatility(rets)
        self.assertEqual(vol, 0.0)

    def test_downside_volatility(self):
        rets = _make_daily_returns([0.01, -0.02, 0.03, -0.01, -0.03])
        dv = calculate_downside_volatility(rets, target_return=0.0, trading_days=252)
        self.assertGreater(dv, 0)

    def test_var(self):
        rets = _make_daily_returns([-0.01, -0.02, -0.03, 0.01, 0.02])
        var_95 = calculate_var(rets, 0.95)
        self.assertLess(var_95, 0)

    def test_cvar(self):
        rets = _make_daily_returns([-0.01, -0.02, -0.03, -0.04, 0.01, 0.02])
        cvar_95 = calculate_cvar(rets, 0.95)
        self.assertLess(cvar_95, 0)


class TestRiskAdjustedMetrics(unittest.TestCase):
    def test_sharpe_ratio(self):
        rets = _make_daily_returns([0.001] * 252)
        sharpe = calculate_sharpe_ratio(rets, risk_free_rate=0.03, trading_days=252)
        self.assertGreater(sharpe, 0)

    def test_sharpe_ratio_negative(self):
        rets = _make_daily_returns([-0.001, -0.002, -0.003, 0.001, 0.002] * 50)
        sharpe = calculate_sharpe_ratio(rets, risk_free_rate=0.03, trading_days=252)
        self.assertLess(sharpe, 0)

    def test_sortino_ratio(self):
        rets = _make_daily_returns([0.01, -0.005, 0.02, -0.003, 0.015])
        sortino = calculate_sortino_ratio(rets, risk_free_rate=0.03, trading_days=252)
        self.assertIsInstance(sortino, float)

    def test_calmar_ratio(self):
        rets = _make_daily_returns([0.001] * 252)
        eq = (1 + rets).cumprod()
        calmar = calculate_calmar_ratio(rets, eq)
        self.assertIsInstance(calmar, float)

    def test_return_drawdown_ratio(self):
        rdd = calculate_return_drawdown_ratio(0.2, -0.1)
        self.assertAlmostEqual(rdd, 2.0)

    def test_return_drawdown_ratio_zero(self):
        rdd = calculate_return_drawdown_ratio(0.2, 0)
        self.assertEqual(rdd, 0.0)

    def test_information_ratio(self):
        strat_rets = _make_daily_returns([0.001] * 100)
        bench_rets = _make_daily_returns([0.0005] * 100)
        ir = calculate_information_ratio(strat_rets, bench_rets, trading_days=252)
        self.assertIsInstance(ir, float)


class TestTradingMetrics(unittest.TestCase):
    def setUp(self):
        self.trades = [
            Trade(date="2023-01-01", code="000001.SZ", direction="buy", price=10.0, quantity=100, amount=1000.0),
            Trade(date="2023-01-10", code="000001.SZ", direction="sell", price=12.0, quantity=100, amount=1200.0),
            Trade(date="2023-02-01", code="000001.SZ", direction="buy", price=11.0, quantity=100, amount=1100.0),
            Trade(date="2023-02-10", code="000001.SZ", direction="sell", price=10.0, quantity=100, amount=1000.0),
        ]

    def test_total_trades(self):
        self.assertEqual(calculate_total_trades(self.trades), 4)

    def test_win_rate(self):
        wr = calculate_win_rate(self.trades)
        self.assertGreater(wr, 0)

    def test_profit_loss_ratio(self):
        plr = calculate_profit_loss_ratio(self.trades)
        self.assertGreater(plr, 0)

    def test_max_consecutive_wins(self):
        mw = calculate_max_consecutive_wins(self.trades)
        self.assertGreaterEqual(mw, 0)

    def test_max_consecutive_losses(self):
        ml = calculate_max_consecutive_losses(self.trades)
        self.assertGreaterEqual(ml, 0)

    def test_avg_holding_period(self):
        trades_with_days = [
            Trade(date="2023-01-10", code="000001.SZ", direction="sell", price=12.0, quantity=100, amount=1200.0, holding_days=10),
        ]
        avg_hold = calculate_avg_holding_period(trades_with_days)
        self.assertAlmostEqual(avg_hold, 10.0)

    def test_commission_impact(self):
        impact = calculate_commission_impact(100.0, 10000.0)
        self.assertAlmostEqual(impact, 0.01)


if __name__ == "__main__":
    unittest.main()
