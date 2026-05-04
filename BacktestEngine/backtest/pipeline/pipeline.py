from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..core.engine import BacktestEngine, BacktestResult
from ..core.data_feed import DataFeed
from ..core.action import Action
from ..strategies.signal import BaseSignal, HoldingReturnThresholdSignal
from ..strategies.buy import BaseBuyStrategy
from ..strategies.sell import BaseSellStrategy
from .conditions import (
    SignalCondition, TopNCondition, ThresholdCondition, CompositeCondition,
    apply_condition_to_actions_buy, apply_condition_to_actions_sell,
    apply_holding_return_sell,
)
from ..metrics.returns import (
    calculate_total_return, calculate_annualized_return,
    calculate_monthly_win_rate, calculate_yearly_win_rate, calculate_excess_return,
)
from ..metrics.risk import (
    calculate_max_drawdown, calculate_max_drawdown_duration,
    calculate_annualized_volatility, calculate_downside_volatility,
)
from ..metrics.risk_adjusted import (
    calculate_sharpe_ratio, calculate_sortino_ratio, calculate_calmar_ratio,
    calculate_return_drawdown_ratio, calculate_information_ratio,
)
from ..metrics.trading import (
    calculate_win_rate, calculate_total_trades, calculate_avg_holding_period,
    calculate_turnover_rate, calculate_total_traded_amount,
)


@dataclass
class PipelineConfig:
    buy_signal: BaseSignal
    sell_signal: BaseSignal
    buy_condition: SignalCondition = field(
        default_factory=lambda: TopNCondition(n=5, direction="top"))
    sell_condition: SignalCondition = field(
        default_factory=lambda: TopNCondition(n=5, direction="bottom"))
    initial_cash: float = 1_000_000.0
    commission_rate: float = 0.0003
    slippage: float = 0.001
    position_pct: float = 0.90
    min_cash_threshold: float = 50000.0
    max_holdings: int = 5
    min_stock_value: float = 10000.0
    trading_days_per_year: int = 252
    train_test_split: float = 0.1
    benchmark_codes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "买入信号": self.buy_signal.get_name(),
            "卖出信号": self.sell_signal.get_name(),
            "买入条件": self.buy_condition.get_name(),
            "卖出条件": self.sell_condition.get_name(),
            "初始资金": self.initial_cash,
            "持仓上限": self.max_holdings,
        }


@dataclass
class PipelineResult:
    config: PipelineConfig
    backtest_result: BacktestResult
    signal_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    pivot_signal: pd.DataFrame = field(default_factory=pd.DataFrame)
    dates_sorted: List = field(default_factory=list)
    train_dates: List = field(default_factory=list)
    test_dates: List = field(default_factory=list)
    benchmark_results: Dict[str, BacktestResult] = field(default_factory=dict)

    @property
    def trades(self):
        return self.backtest_result.trades

    @property
    def equity_curve(self):
        return self.backtest_result.equity_curve

    @property
    def daily_returns(self):
        return self.backtest_result.daily_returns


class SignalPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def run(self, df: pd.DataFrame, show_progress: bool = False) -> PipelineResult:
        df = self._preprocess(df)
        # [FIX] 分别计算买入和卖出信号
        buy_signal_df = self.config.buy_signal.compute(df)
        if self.config.buy_signal is self.config.sell_signal:
            sell_signal_df = buy_signal_df
        else:
            sell_signal_df = self.config.sell_signal.compute(df)

        buy_pivot = buy_signal_df.pivot_table(index="code", columns="date", values="signal_value")
        sell_pivot = sell_signal_df.pivot_table(index="code", columns="date", values="signal_value")

        # 若 sell_pivot 全为NaN → pivot_table 返回空(0,0); 用 buy_pivot 的日期结构 + NaN 填充
        if sell_pivot.empty:
            sell_pivot = pd.DataFrame(np.nan, index=buy_pivot.index, columns=buy_pivot.columns)

        # 取两个 pivot 的公共日期 — 用字符串比较避开不同类型/时区的 Timestamp 差异
        buy_dates_set = {str(d) for d in buy_pivot.columns}
        sell_dates_set = {str(d) for d in sell_pivot.columns}
        common_str = sorted(buy_dates_set & sell_dates_set)
        if not common_str:
            raise ValueError("买入和卖出信号无公共日期，无法回测")

        # 将公共日期还原为 buy_pivot 列的实际类型 (DatetimeIndex 或 Index)
        common_dates = [d for d in buy_pivot.columns if str(d) in common_str]

        split_idx = int(len(common_dates) * self.config.train_test_split)
        train_dates = common_dates[:split_idx]
        test_dates = common_dates[split_idx:]
        if not test_dates:
            raise ValueError("测试集为空")

        backtest_result = self._run_backtest(df, buy_pivot, sell_pivot, test_dates, show_progress)
        benchmark_results = {}
        if self.config.benchmark_codes:
            benchmark_results = self._run_benchmarks(df, test_dates)

        return PipelineResult(
            config=self.config, backtest_result=backtest_result,
            signal_df=buy_signal_df, pivot_signal=buy_pivot,
            dates_sorted=common_dates, train_dates=train_dates,
            test_dates=test_dates, benchmark_results=benchmark_results,
        )

    def evaluate(self, result: PipelineResult) -> Dict[str, float]:
        br = result.backtest_result
        eq = br.equity_curve
        rets = br.daily_returns
        trades = br.trades
        td = self.config.trading_days_per_year
        m: Dict[str, float] = {}

        if not eq.empty and len(eq) > 1:
            m["总收益率"] = calculate_total_return(eq)
            m["年化收益率"] = calculate_annualized_return(eq, td)
        else:
            m["总收益率"] = m["年化收益率"] = 0.0

        if not rets.empty and len(rets) > 1 and rets.std() > 1e-12:
            m["年化波动率"] = calculate_annualized_volatility(rets, td)
            m["下行波动率"] = calculate_downside_volatility(rets, td)
            m["夏普比率"] = calculate_sharpe_ratio(rets, td)
            m["索提诺比率"] = calculate_sortino_ratio(rets, td)
            m["卡玛比率"] = calculate_calmar_ratio(rets, eq, td)
            m["月度胜率"] = calculate_monthly_win_rate(rets)
            m["年度胜率"] = calculate_yearly_win_rate(rets)
        else:
            for k in ["年化波动率","下行波动率","夏普比率","索提诺比率","卡玛比率","月度胜率","年度胜率"]:
                m[k] = 0.0

        if not eq.empty and len(eq) > 1:
            md, mds, mde = calculate_max_drawdown(eq)
            m["最大回撤"] = md
            m["最长回撤修复期"] = float(calculate_max_drawdown_duration(eq))
            rolling_max = eq.expanding().max()
            dd_series = (eq - rolling_max) / rolling_max
            if not dd_series.empty:
                m["年均回撤"] = float(dd_series.groupby(pd.Grouper(freq="YE")).min().mean())
            else:
                m["年均回撤"] = 0.0
        else:
            m["最大回撤"] = m["最长回撤修复期"] = m["年均回撤"] = 0.0

        if trades:
            m["交易胜率"] = calculate_win_rate(trades)
            m["总交易次数"] = float(calculate_total_trades(trades))
            m["平均持仓天数"] = calculate_avg_holding_period(trades)
            m["日均交易次数"] = calculate_total_trades(trades) / max(len(rets.dropna()), 1)
            all_amount = calculate_total_traded_amount(trades)
            m["总成交金额"] = all_amount
        else:
            for k in ["交易胜率","总交易次数","平均持仓天数","日均交易次数","总成交金额"]:
                m[k] = 0.0

        pos_hist = br.positions_history
        if not pos_hist.empty:
            if not isinstance(pos_hist.index, pd.DatetimeIndex):
                pos_hist = pos_hist.copy()
                pos_hist.index = pd.to_datetime(pos_hist.index)
            pos_cols = [c for c in pos_hist.columns if c.endswith("_qty")]
            if pos_cols:
                daily_holdings = (pos_hist[pos_cols] > 0).sum(axis=1)
                m["日均持仓数"] = float(daily_holdings.mean())
                m["最大同时持仓数"] = float(daily_holdings.max())
                m["周均持仓数"] = float(daily_holdings.groupby(pd.Grouper(freq="W")).mean().mean())
                m["月均持仓数"] = float(daily_holdings.groupby(pd.Grouper(freq="ME")).mean().mean())
            if "total_equity" in pos_hist.columns:
                total_equity = pos_hist["total_equity"]
                avg_assets = total_equity.mean() if len(total_equity) > 0 else 0
                if avg_assets > 0 and trades:
                    m["换手率"] = calculate_turnover_rate(
                        calculate_total_traded_amount(trades), avg_assets)
                else:
                    m["换手率"] = 0.0

            m["日均持仓数"] = m.get("日均持仓数", 0.0)
            m["最大同时持仓数"] = m.get("最大同时持仓数", 0.0)
            m["周均持仓数"] = m.get("周均持仓数", 0.0)
            m["月均持仓数"] = m.get("月均持仓数", 0.0)
        else:
            for k in ["日均持仓数","最大同时持仓数","周均持仓数","月均持仓数","换手率"]:
                m[k] = 0.0

        m["总佣金"] = br.total_commission
        m["信号组合"] = f"{self.config.buy_signal.get_name()}×{self.config.sell_signal.get_name()}"

        if result.benchmark_results:
            for bname, bres in result.benchmark_results.items():
                beq = bres.equity_curve
                if not beq.empty and len(beq) > 1:
                    b_total = calculate_total_return(beq)
                    b_ann = calculate_annualized_return(beq, td)
                    b_md, _, _ = calculate_max_drawdown(beq)
                    m[f"基准_{bname}_总收益"] = b_total
                    m[f"基准_{bname}_年化"] = b_ann
                    m[f"基准_{bname}_最大回撤"] = b_md

        return m

    def evaluate_multi_dimension(self, result: PipelineResult) -> Dict[str, Dict[str, float]]:
        """五维度评估"""
        m = self.evaluate(result)
        dims = {}

        # （1）稳定性
        dims["稳定性"] = {
            "年化波动率": m.get("年化波动率", 1.0),
            "下行波动率": m.get("下行波动率", 0.0),
            "月度胜率": m.get("月度胜率", 0.0),
        }
        # （2）收益率
        dims["收益率"] = {
            "总收益率": m.get("总收益率", 0.0),
            "年化收益率": m.get("年化收益率", 0.0),
            "夏普比率": m.get("夏普比率", 0.0),
            "卡玛比率": m.get("卡玛比率", 0.0),
        }
        # （3）抗风险能力
        dims["抗风险能力"] = {
            "最大回撤": abs(m.get("最大回撤", 0.0)),
            "年均回撤": abs(m.get("年均回撤", 0.0)),
            "最长回撤修复期": m.get("最长回撤修复期", 0.0),
        }
        # （4）技术难度（偏好低频）
        dims["技术难度"] = {
            "平均持仓天数": m.get("平均持仓天数", 1.0),
            "日均交易次数": m.get("日均交易次数", 99.0),
            "换手率": m.get("换手率", 99.0),
        }
        # （5）组合难度（偏好少持仓）
        dims["组合难度"] = {
            "日均持仓数": m.get("日均持仓数", 99.0),
            "最大同时持仓数": m.get("最大同时持仓数", 99.0),
            "月均持仓数": m.get("月均持仓数", 99.0),
        }
        return dims

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        for col in ["close", "open", "high", "low", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["close"])
        if "tradestatus" in df.columns:
            df = df[df["tradestatus"] == 1]
        df = df[df["close"] > 0]
        df = df.sort_values(["code", "date"])
        return df

    def _run_backtest(self, df, buy_pivot, sell_pivot, test_dates, show_progress=False):
        first_test = test_dates[0]
        backtest_df = df[df["date"] >= first_test].copy()
        test_codes = list(set(buy_pivot.index) & set(backtest_df["code"].unique()))
        backtest_df = backtest_df[backtest_df["code"].isin(test_codes)]
        buy_test_pivot = buy_pivot[test_dates]
        sell_test_pivot = sell_pivot[test_dates]

        data_dict: Dict[str, pd.DataFrame] = {}
        for code in test_codes:
            cdf = backtest_df[backtest_df["code"] == code].copy()
            if not cdf.empty:
                data_dict[code] = cdf

        cfg = self.config
        is_holding_return = isinstance(cfg.sell_signal, HoldingReturnThresholdSignal)

        class _Buy(BaseBuyStrategy):
            def __init__(self, c, tp):
                self.c = c; self.tp = tp
            def should_buy(self, date, data, positions, cash):
                if date not in self.tp.columns: return []
                return apply_condition_to_actions_buy(
                    self.tp[date], self.c.buy_condition, data, positions, cash,
                    self.c.buy_signal, self.c.position_pct,
                    self.c.min_cash_threshold, self.c.max_holdings,
                    self.c.min_stock_value)

        class _Sell(BaseSellStrategy):
            def __init__(self, c, tp):
                self.c = c; self.tp = tp
            def should_sell(self, date, data, positions, cash):
                if is_holding_return:
                    return apply_holding_return_sell(
                        self.c.sell_condition, data, positions, self.c.sell_signal)
                if date not in self.tp.columns: return []
                return apply_condition_to_actions_sell(
                    self.tp[date], self.c.sell_condition, data, positions,
                    self.c.sell_signal)

        engine = BacktestEngine(
            data_feed=DataFeed(data_dict),
            buy_strategy=_Buy(cfg, buy_test_pivot),
            sell_strategy=_Sell(cfg, sell_test_pivot),
            initial_cash=cfg.initial_cash,
            commission_rate=cfg.commission_rate,
            slippage=cfg.slippage,
        )
        return engine.run(show_progress=show_progress)

    def _run_benchmarks(self, df: pd.DataFrame, test_dates: List):
        """基准: 从df中查找基准code, 若不存在(excluded), 返回空 → 由实验脚本自行加载"""
        benchmarks = {}
        first_test = test_dates[0]
        for bcode in self.config.benchmark_codes:
            if bcode not in df["code"].unique():
                # 基准code不在主df中(excluded)→实验脚本自行加载
                continue
            bdf = df[(df["code"] == bcode) & (df["date"] >= first_test)].copy()
            if bdf.empty:
                continue
            class _HoldBuy(BaseBuyStrategy):
                def should_buy(self, date, data, positions, cash):
                    if not positions and bcode in data:
                        return [Action(type="buy", code=bcode,
                                       price=float(data[bcode]["close"]),
                                       quantity=100, reason="benchmark_hold")]
                    return []
            class _NeverSell(BaseSellStrategy):
                def should_sell(self, date, data, positions, cash): return []

            engine = BacktestEngine(
                data_feed=DataFeed({bcode: bdf}),
                buy_strategy=_HoldBuy(),
                sell_strategy=_NeverSell(),
                initial_cash=self.config.initial_cash,
                commission_rate=self.config.commission_rate,
                slippage=self.config.slippage,
            )
            benchmarks[bcode] = engine.run()
        return benchmarks
