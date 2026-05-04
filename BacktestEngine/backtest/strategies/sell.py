from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import pandas as pd

from ..core.action import Action, Position
from ..strategies.signal import BaseSignal


class BaseSellStrategy(ABC):
    @abstractmethod
    def should_sell(
        self,
        date: str,
        data: Dict[str, dict],
        positions: Dict[str, Position],
        cash: float,
    ) -> List[Action]:
        pass


class AlwaysSellStrategy(BaseSellStrategy):
    def should_sell(
        self,
        date: str,
        data: Dict[str, dict],
        positions: Dict[str, Position],
        cash: float,
    ) -> List[Action]:
        actions = []
        for code, pos in positions.items():
            if pos.quantity > 0:
                actions.append(
                    Action(
                        type="sell",
                        code=code,
                        price=float(data.get(code, {}).get("close", 0)),
                        quantity=pos.quantity,
                        reason="always_sell_all",
                    )
                )
        return actions


class StopLossSellStrategy(BaseSellStrategy):
    def __init__(self, stop_loss_pct: float = -0.05):
        self.stop_loss_pct = stop_loss_pct

    def should_sell(
        self,
        date: str,
        data: Dict[str, dict],
        positions: Dict[str, Position],
        cash: float,
    ) -> List[Action]:
        actions = []
        for code, pos in positions.items():
            if pos.quantity > 0 and code in data:
                current_price = float(data[code]["close"])
                pnl_pct = (current_price - pos.cost_price) / pos.cost_price
                if pnl_pct <= self.stop_loss_pct:
                    actions.append(
                        Action(
                            type="sell",
                            code=code,
                            price=current_price,
                            quantity=pos.quantity,
                            reason=f"stop_loss_{self.stop_loss_pct:.0%}",
                        )
                    )
        return actions


class SignalSellStrategy(BaseSellStrategy):
    def __init__(
        self,
        signal: BaseSignal,
        signal_df: pd.DataFrame,
        bottom_n: int = 10,
    ):
        self.signal = signal
        self.signal_df = signal_df
        self.bottom_n = bottom_n

    def should_sell(
        self,
        date: str,
        data: Dict[str, dict],
        positions: Dict[str, Position],
        cash: float,
    ) -> List[Action]:
        if date not in self.signal_df.columns:
            return []
        date_signals = self.signal_df[date].dropna().sort_values(ascending=True)
        if date_signals.empty:
            return []
        weak_stocks = set(date_signals.head(self.bottom_n).index)
        actions = []
        for code, pos in positions.items():
            if pos.quantity > 0 and code in weak_stocks:
                current_price = float(data.get(code, {}).get("close", pos.current_price))
                actions.append(
                    Action(
                        type="sell",
                        code=code,
                        price=current_price,
                        quantity=pos.quantity,
                        reason=f"signal_{self.signal.get_name()}_weak",
                    )
                )
        return actions
