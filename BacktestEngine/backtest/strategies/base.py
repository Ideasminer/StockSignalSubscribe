from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List

from ..core.action import Action
from ..core.data_feed import DataFeed


class BaseStrategy(ABC):
    def __init__(self, data_feed: DataFeed):
        self.data_feed = data_feed

    def init(self):
        pass

    @abstractmethod
    def next(self, date: str, data: Dict[str, dict]) -> List[Action]:
        pass


class CompositeStrategy(BaseStrategy):
    def __init__(
        self,
        data_feed: DataFeed,
        buy_strategy=None,
        sell_strategy=None,
        stock_selector=None,
    ):
        super().__init__(data_feed)
        self.buy_strategy = buy_strategy
        self.sell_strategy = sell_strategy
        self.stock_selector = stock_selector

    def init(self):
        if self.buy_strategy and hasattr(self.buy_strategy, "init"):
            self.buy_strategy.init()
        if self.sell_strategy and hasattr(self.sell_strategy, "init"):
            self.sell_strategy.init()
        if self.stock_selector and hasattr(self.stock_selector, "init"):
            self.stock_selector.init()

    def next(self, date: str, data: Dict[str, dict]) -> List[Action]:
        actions: List[Action] = []

        selected_codes = None
        if self.stock_selector is not None:
            selected_codes = self.stock_selector.select(date, data)
            filtered_data = {k: v for k, v in data.items() if k in selected_codes}
        else:
            filtered_data = data

        if self.buy_strategy is not None:
            buy_actions = self.buy_strategy.should_buy(
                date, filtered_data, self._get_positions_snapshot(), 0
            )
            actions.extend(buy_actions)

        if self.sell_strategy is not None:
            sell_actions = self.sell_strategy.should_sell(
                date, filtered_data, self._get_positions_snapshot(), 0
            )
            actions.extend(sell_actions)

        return actions

    def _get_positions_snapshot(self):
        return {}
