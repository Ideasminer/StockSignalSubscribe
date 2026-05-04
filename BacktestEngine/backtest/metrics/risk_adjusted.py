from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_sharpe_ratio(
    daily_returns: pd.Series,
    risk_free_rate: float = 0.03,
    trading_days: int = 252,
) -> float:
    if len(daily_returns) < 2:
        return 0.0
    daily_rf = risk_free_rate / trading_days
    excess_returns = daily_returns - daily_rf
    if excess_returns.std() == 0:
        return 0.0
    sharpe = np.sqrt(trading_days) * excess_returns.mean() / excess_returns.std()
    return float(sharpe)


def calculate_sortino_ratio(
    daily_returns: pd.Series,
    risk_free_rate: float = 0.03,
    trading_days: int = 252,
) -> float:
    if len(daily_returns) < 2:
        return 0.0
    daily_rf = risk_free_rate / trading_days
    excess_returns = daily_returns - daily_rf
    downside = daily_returns[daily_returns < daily_rf]
    if len(downside) < 2:
        return 0.0
    downside_std = downside.std()
    if downside_std == 0:
        return 0.0
    sortino = np.sqrt(trading_days) * excess_returns.mean() / downside_std
    return float(sortino)


def calculate_calmar_ratio(
    daily_returns: pd.Series,
    equity_curve: pd.Series,
    trading_days: int = 252,
) -> float:
    if daily_returns.empty:
        return 0.0
    n_days = len(daily_returns)
    if n_days < 2:
        return 0.0
    total_ret = (equity_curve.iloc[-1] - equity_curve.iloc[0]) / equity_curve.iloc[0]
    if total_ret <= -1:
        return -1.0
    years = n_days / trading_days
    if years <= 0:
        return 0.0
    annual_ret = (1 + total_ret) ** (1 / years) - 1

    rolling_max = equity_curve.expanding().max()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_dd = abs(drawdown.min())

    if max_dd == 0:
        return 0.0
    return float(annual_ret / max_dd)


def calculate_return_drawdown_ratio(
    total_return: float, max_drawdown: float
) -> float:
    if max_drawdown == 0:
        return 0.0
    return total_return / abs(max_drawdown)


def calculate_information_ratio(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    trading_days: int = 252,
) -> float:
    if strategy_returns.empty or benchmark_returns.empty:
        return 0.0
    aligned = pd.concat(
        [strategy_returns, benchmark_returns], axis=1, join="inner"
    ).dropna()
    if len(aligned) < 2:
        return 0.0
    excess = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    tracking_error = excess.std()
    if tracking_error == 0:
        return 0.0
    ir = np.sqrt(trading_days) * excess.mean() / tracking_error
    return float(ir)
