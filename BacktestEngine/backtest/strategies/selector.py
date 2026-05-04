from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List


class BaseStockSelector(ABC):
    @abstractmethod
    def select(self, date: str, data: Dict[str, dict]) -> List[str]:
        pass


class AllStockSelector(BaseStockSelector):
    def select(self, date: str, data: Dict[str, dict]) -> List[str]:
        return list(data.keys())


class TopNSelector(BaseStockSelector):
    def __init__(self, n: int = 1):
        self.n = n

    def select(self, date: str, data: Dict[str, dict]) -> List[str]:
        if not data:
            return []
        sorted_codes = sorted(
            data.keys(),
            key=lambda c: float(data[c].get("pctChg", 0)),
            reverse=True,
        )
        return sorted_codes[: self.n]
