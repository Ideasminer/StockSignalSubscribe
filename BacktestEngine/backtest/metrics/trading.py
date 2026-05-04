from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from ..core.action import Trade


def calculate_win_rate(trades: List[Trade]) -> float:
    closed_trades = _get_closed_trades(trades)
    if len(closed_trades) == 0:
        return 0.0
    wins = sum(1 for t in _iter_trade_pnls(trades) if t > 0)
    return wins / len(closed_trades)


def calculate_profit_loss_ratio(trades: List[Trade]) -> float:
    closed = _get_closed_trades(trades)
    if len(closed) == 0:
        return 0.0
    pnls = list(_iter_trade_pnls(trades))
    total_wins = sum(p for p in pnls if p > 0)
    total_losses = sum(abs(p) for p in pnls if p < 0)
    if total_losses == 0:
        return 0.0
    avg_win = total_wins / max(sum(1 for p in pnls if p > 0), 1)
    avg_loss = total_losses / max(sum(1 for p in pnls if p < 0), 1)
    if avg_loss == 0:
        return 0.0
    return avg_win / avg_loss


def _get_closed_trades(trades: List[Trade]) -> List[Trade]:
    buy_trades = [t for t in trades if t.direction == "buy"]
    return buy_trades


def _iter_trade_pnls(trades: List[Trade]):
    buy_trades = {}
    for t in trades:
        if t.direction == "buy":
            key = t.code
            if key not in buy_trades:
                buy_trades[key] = []
            buy_trades[key].append(t)
        elif t.direction == "sell":
            key = t.code
            if key in buy_trades and buy_trades[key]:
                buy_t = buy_trades[key].pop(0)
                pnl = (t.price - buy_t.price) * t.quantity
                yield pnl


def calculate_max_consecutive_wins(trades: List[Trade]) -> int:
    return _max_consecutive(trades, positive=True)


def calculate_max_consecutive_losses(trades: List[Trade]) -> int:
    return _max_consecutive(trades, positive=False)


def _max_consecutive(trades: List[Trade], positive: bool = True) -> int:
    pnls = list(_iter_trade_pnls(trades))
    max_streak = 0
    current_streak = 0
    for p in pnls:
        if (positive and p > 0) or (not positive and p < 0):
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
    return max_streak


def calculate_total_trades(trades: List[Trade]) -> int:
    return len(trades)


def calculate_daily_avg_trades(trades: List[Trade], trading_days: int) -> float:
    if trading_days <= 0:
        return 0.0
    return len(trades) / trading_days


def calculate_avg_holding_period(trades: List[Trade]) -> float:
    sell_trades = [t for t in trades if t.direction == "sell" and t.holding_days > 0]
    if len(sell_trades) == 0:
        return 0.0
    return float(np.mean([t.holding_days for t in sell_trades]))


def calculate_commission_impact(total_commission: float, final_equity: float) -> float:
    if final_equity == 0:
        return 0.0
    return total_commission / final_equity


def calculate_turnover_rate(
    total_traded_amount: float, avg_total_assets: float
) -> float:
    if avg_total_assets == 0:
        return 0.0
    return total_traded_amount / avg_total_assets


def calculate_total_traded_amount(trades: List[Trade]) -> float:
    return sum(t.amount for t in trades)
