from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Action:
    type: Literal["buy", "sell", "hold", "close"]
    code: str
    price: float
    quantity: int = 0
    reason: str = ""

    def is_buy(self) -> bool:
        return self.type == "buy"

    def is_sell(self) -> bool:
        return self.type in ("sell", "close")

    def is_hold(self) -> bool:
        return self.type == "hold"

    def __post_init__(self):
        if self.type == "close":
            self.quantity = 0


@dataclass
class Trade:
    date: str
    code: str
    direction: Literal["buy", "sell"]
    price: float
    quantity: int
    amount: float
    commission: float = 0.0
    reason: str = ""
    holding_days: int = 0

    @property
    def pnl(self) -> float:
        return 0.0

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "code": self.code,
            "direction": self.direction,
            "price": self.price,
            "quantity": self.quantity,
            "amount": self.amount,
            "commission": self.commission,
            "reason": self.reason,
            "holding_days": self.holding_days,
        }


@dataclass
class Position:
    code: str
    quantity: int = 0
    cost_price: float = 0.0
    current_price: float = 0.0
    buy_dates: list[str] = field(default_factory=list)

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def cost_value(self) -> float:
        return self.quantity * self.cost_price

    @property
    def unrealized_pnl(self) -> float:
        return self.market_value - self.cost_value

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.cost_price == 0:
            return 0.0
        return (self.current_price - self.cost_price) / self.cost_price
