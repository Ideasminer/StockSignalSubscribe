from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def calculate_max_drawdown(equity_curve: pd.Series) -> Tuple[float, str, str]:
    if len(equity_curve) < 2:
        return (0.0, "", "")
    rolling_max = equity_curve.expanding().max()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_dd = drawdown.min()
    if pd.isna(max_dd):
        return (0.0, "", "")
    end_idx = drawdown.idxmin()
    start_idx = rolling_max.loc[:end_idx].idxmax()
    return (
        float(max_dd),
        str(start_idx.strftime("%Y-%m-%d")),
        str(end_idx.strftime("%Y-%m-%d")),
    )


def calculate_max_drawdown_duration(equity_curve: pd.Series) -> int:
    if len(equity_curve) < 2:
        return 0
    rolling_max = equity_curve.expanding().max()
    drawdown = (equity_curve - rolling_max) / rolling_max
    is_underwater = drawdown < 0
    max_duration = 0
    current_duration = 0
    for val in is_underwater:
        if val:
            current_duration += 1
            max_duration = max(max_duration, current_duration)
        else:
            current_duration = 0
    return max_duration


def calculate_annualized_volatility(
    daily_returns: pd.Series, trading_days: int = 252
) -> float:
    if len(daily_returns) < 2:
        return 0.0
    vol = daily_returns.std()
    if pd.isna(vol):
        return 0.0
    return float(vol * np.sqrt(trading_days))


def calculate_downside_volatility(
    daily_returns: pd.Series,
    target_return: float = 0.0,
    trading_days: int = 252,
) -> float:
    if len(daily_returns) < 2:
        return 0.0
    downside = daily_returns[daily_returns < target_return]
    if len(downside) < 2:
        return 0.0
    downside_vol = downside.std()
    if pd.isna(downside_vol):
        return 0.0
    return float(downside_vol * np.sqrt(trading_days))


def calculate_var(
    daily_returns: pd.Series, confidence_level: float = 0.95
) -> float:
    if daily_returns.empty:
        return 0.0
    return float(daily_returns.quantile(1 - confidence_level))


def calculate_cvar(
    daily_returns: pd.Series, confidence_level: float = 0.95
) -> float:
    if daily_returns.empty:
        return 0.0
    var = daily_returns.quantile(1 - confidence_level)
    tail = daily_returns[daily_returns <= var]
    if len(tail) == 0:
        return float(var)
    return float(tail.mean())
