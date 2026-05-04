from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import pandas as pd

from ..core.action import Action, Position
from ..strategies.signal import BaseSignal


class BaseBuyStrategy(ABC):
    @abstractmethod
    def should_buy(
        self,
        date: str,
        data: Dict[str, dict],
        positions: Dict[str, Position],
        cash: float,
    ) -> List[Action]:
        pass


class AlwaysBuyStrategy(BaseBuyStrategy):
    def __init__(self, codes: list[str], fixed_quantity: int = 100):
        self.codes = codes
        self.fixed_quantity = fixed_quantity

    def should_buy(
        self,
        date: str,
        data: Dict[str, dict],
        positions: Dict[str, Position],
        cash: float,
    ) -> List[Action]:
        actions = []
        for code in self.codes:
            if code in data and code not in positions:
                actions.append(
                    Action(
                        type="buy",
                        code=code,
                        price=float(data[code]["close"]),
                        quantity=self.fixed_quantity,
                        reason="always_buy",
                    )
                )
        return actions


class SignalBuyStrategy(BaseBuyStrategy):
    def __init__(
        self,
        signal: BaseSignal,
        signal_df: pd.DataFrame,
        top_n: int = 10,
        position_pct: float = 0.95,
        min_cash_threshold: float = 50000.0,
        max_holdings: Optional[int] = None,
        min_stock_value: float = 10000.0,
    ):
        self.signal = signal
        self.signal_df = signal_df
        self.top_n = top_n
        self.position_pct = position_pct
        self.min_cash_threshold = min_cash_threshold
        self.max_holdings = max_holdings if max_holdings is not None else top_n
        self.min_stock_value = min_stock_value

    def should_buy(
        self,
        date: str,
        data: Dict[str, dict],
        positions: Dict[str, Position],
        cash: float,
    ) -> List[Action]:
        if date not in self.signal_df.columns:
            return []
        if cash < self.min_cash_threshold:
            return []
        current_holding_count = sum(1 for p in positions.values() if p.quantity > 0)
        if current_holding_count >= self.max_holdings:
            return []
        date_signals = self.signal_df[date].dropna().sort_values(ascending=False)
        if date_signals.empty:
            return []
        candidates = date_signals.head(self.top_n)
        actions = []
        remaining_budget = cash * self.position_pct
        remaining_count = len(candidates)

        for code, signal_val in candidates.items():
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
            if estimated_amount < self.min_stock_value:
                remaining_count -= 1
                continue
            actions.append(
                Action(
                    type="buy",
                    code=code,
                    price=current_price,
                    quantity=quantity,
                    reason=f"signal_{self.signal.get_name()}_val={signal_val:.4f}",
                )
            )
            estimated_cost = quantity * current_price * 1.001 * 1.003
            remaining_budget -= estimated_cost
            remaining_count -= 1

        return actions
