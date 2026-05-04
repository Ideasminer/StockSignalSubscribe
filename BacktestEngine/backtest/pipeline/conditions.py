from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd

from ..core.action import Action, Position


class SignalCondition(ABC):
    """信号过滤条件的抽象基类

    子类实现 filter() 方法，从信号值序列中筛选出符合交易条件的标的。
    返回值为两个选项之一：
    - pd.Series[bool]: 布尔掩码（True=选中）
    - list[str]: 选中标的的 code 列表
    """

    @abstractmethod
    def filter(self, signal_values: pd.Series) -> Union[pd.Series, List[str]]:
        pass

    def get_name(self) -> str:
        return self.__class__.__name__


class TopNCondition(SignalCondition):
    """Top-N / Bottom-N 排名条件
    - direction="top": 取 signal_value 最大的 N 个
    - direction="bottom": 取 signal_value 最小的 N 个
    """

    def __init__(self, n: int = 5, direction: Literal["top", "bottom"] = "top"):
        self.n = n
        self.direction = direction

    def filter(self, signal_values: pd.Series) -> pd.Series:
        sv = signal_values.dropna()
        if sv.empty:
            return pd.Series(False, index=signal_values.index)
        if self.direction == "top":
            selected = sv.nlargest(self.n).index
        else:
            selected = sv.nsmallest(self.n).index
        return pd.Series(signal_values.index.isin(selected), index=signal_values.index)

    def get_name(self) -> str:
        return f"TopNCondition(n={self.n}, dir={self.direction})"


class ThresholdCondition(SignalCondition):
    """基于阈值的条件
    - comparator="gt":  signal_value > threshold
    - comparator="lt":  signal_value < threshold
    - comparator="gte": signal_value >= threshold
    - comparator="lte": signal_value <= threshold
    """

    def __init__(
        self,
        threshold: float = 0.0,
        comparator: Literal["gt", "lt", "gte", "lte"] = "gt",
    ):
        self.threshold = threshold
        self.comparator = comparator

    def filter(self, signal_values: pd.Series) -> pd.Series:
        if self.comparator == "gt":
            return signal_values > self.threshold
        elif self.comparator == "lt":
            return signal_values < self.threshold
        elif self.comparator == "gte":
            return signal_values >= self.threshold
        elif self.comparator == "lte":
            return signal_values <= self.threshold
        return pd.Series(False, index=signal_values.index)

    def get_name(self) -> str:
        return f"ThresholdCondition({self.comparator} {self.threshold})"


class CompositeCondition(SignalCondition):
    """组合条件：AND/OR 逻辑组合多个子条件"""

    def __init__(
        self, conditions: List[SignalCondition], logic: Literal["AND", "OR"] = "AND"
    ):
        self.conditions = conditions
        self.logic = logic

    def filter(self, signal_values: pd.Series) -> pd.Series:
        results = [c.filter(signal_values) for c in self.conditions]
        combined = results[0]
        for r in results[1:]:
            if self.logic == "AND":
                combined = combined & r
            else:
                combined = combined | r
        return combined

    def get_name(self) -> str:
        names = "+".join(c.get_name() for c in self.conditions)
        return f"Composite({self.logic}: {names})"


def apply_condition_to_actions_buy(
    signal_values: pd.Series,
    condition: SignalCondition,
    data: Dict[str, dict],
    positions: Dict[str, Position],
    cash: float,
    signal: object,
    position_pct: float = 0.90,
    min_cash_threshold: float = 50000.0,
    max_holdings: int = 5,
    min_stock_value: float = 10000.0,
) -> List[Action]:
    """通用买入 Action 生成：根据条件筛选信号值 → 生成买入订单

    Args:
        signal_values: 当日所有标的的信号值
        condition: 买入条件
        data: 当日数据
        positions: 当前持仓
        cash: 当前现金
        signal: 信号对象（用于获取名称）
        position_pct: 仓位比例
        min_cash_threshold: 最低现金阈值
        max_holdings: 最大持仓数
        min_stock_value: 最低单笔交易金额

    Returns:
        List[Action]: 买入动作列表
    """
    if cash < min_cash_threshold:
        return []
    current_holding_count = sum(1 for p in positions.values() if p.quantity > 0)
    if current_holding_count >= max_holdings:
        return []

    mask = condition.filter(signal_values)
    if isinstance(mask, list):
        candidates = [c for c in mask if c in signal_values.index]
    else:
        candidates = signal_values[mask].sort_values(ascending=False).index.tolist()

    actions = []
    remaining_budget = cash * position_pct
    remaining_count = len(candidates)

    for code in candidates:
        if code in positions and positions[code].quantity > 0:
            remaining_count -= 1
            continue
        if code not in data:
            remaining_count -= 1
            continue
        current_price = float(data[code]["close"])
        if current_price <= 0:
            remaining_count -= 1
            continue
        if remaining_count <= 0 or remaining_budget <= 0:
            break
        per_stock_budget = remaining_budget / remaining_count
        buy_price = current_price * 1.001
        quantity = int(per_stock_budget / buy_price)
        if quantity < 100:
            remaining_count -= 1
            continue
        quantity = (quantity // 100) * 100
        if quantity <= 0:
            remaining_count -= 1
            continue
        estimated_amount = quantity * current_price
        if estimated_amount < min_stock_value:
            remaining_count -= 1
            continue
        sig_name = getattr(signal, "get_name", lambda: "signal")()
        actions.append(
            Action(
                type="buy",
                code=code,
                price=current_price,
                quantity=quantity,
                reason=f"pipeline_{sig_name}",
            )
        )
        estimated_cost = quantity * current_price * 1.001 + max(quantity * current_price * 1.001 * 0.003, 5.0)
        remaining_budget -= estimated_cost
        remaining_count -= 1

    return actions


def apply_condition_to_actions_sell(
    signal_values: pd.Series,
    condition: SignalCondition,
    data: Dict[str, dict],
    positions: Dict[str, Position],
    signal: object,
) -> List[Action]:
    """通用卖出 Action 生成：根据条件筛选 → 卖出持仓中符合条件的标的

    Args:
        signal_values: 当日所有标的的信号值
        condition: 卖出条件
        data: 当日数据
        positions: 当前持仓
        signal: 信号对象

    Returns:
        List[Action]: 卖出动作列表
    """
    mask = condition.filter(signal_values)
    if isinstance(mask, list):
        sell_codes = set(mask)
    else:
        sell_codes = set(signal_values[mask].index)

    actions = []
    sig_name = getattr(signal, "get_name", lambda: "signal")()
    for code, pos in positions.items():

        if pos.quantity > 0 and code in sell_codes:
            current_price = float(data.get(code, {}).get("close", pos.current_price))
            actions.append(
                Action(
                    type="sell",
                    code=code,
                    price=current_price,
                    quantity=pos.quantity,
                    reason=f"pipeline_{sig_name}",
                )
            )
    return actions


# ============================================================
# 持仓收益率条件 — 使用引擎中的真实 cost_price
# ============================================================


def apply_holding_return_sell(
    condition: SignalCondition,
    data: Dict[str, dict],
    positions: Dict[str, Position],
    signal: object,
) -> List[Action]:
    """基于真实持仓成本计算收益率并执行卖出

    不使用预先计算的 signal_df，而是从引擎实时 positions 中
    提取 cost_price，结合当日 close 计算真实持仓收益率。

    收益率 = close / cost_price - 1

    只有持有仓位(stock)的标的才参与计算; 无仓位的忽略。

    Args:
        condition: 卖出条件(如 ThresholdCondition(0.08, "gt") = 收益率>8%止盈)
        data: 当日行情 {code: {close, open, ...}}
        positions: 当前持仓 {code: Position(cost_price, quantity, ...)}
        signal: 信号对象(用于 reason)

    Returns:
        List[Action]: 卖出动作列表
    """
    # 构建当日持仓收益率 Series
    records = {}
    for code, pos in positions.items():
        if pos.quantity <= 0 or pos.cost_price <= 0:
            continue
        if code not in data or "close" not in data[code]:
            continue
        current_close = float(data[code]["close"])
        holding_return = current_close / pos.cost_price - 1.0
        records[code] = holding_return

    if not records:
        return []

    signal_values = pd.Series(records)
    return apply_condition_to_actions_sell(
        signal_values, condition, data, positions, signal,
    )
