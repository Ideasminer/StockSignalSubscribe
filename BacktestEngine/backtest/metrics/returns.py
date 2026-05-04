from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_total_return(equity_curve: pd.Series) -> float:
    if len(equity_curve) < 2:
        return 0.0
    total_ret = (equity_curve.iloc[-1] - equity_curve.iloc[0]) / equity_curve.iloc[0]
    return float(total_ret)


def calculate_annualized_return(
    equity_curve: pd.Series, trading_days: int = 252
) -> float:
    total_ret = calculate_total_return(equity_curve)
    if total_ret <= -1:
        return -1.0
    n_days = len(equity_curve)
    if n_days < 2:
        return 0.0
    years = n_days / trading_days
    if years <= 0:
        return 0.0
    annual_ret = (1 + total_ret) ** (1 / years) - 1
    return float(annual_ret)


def calculate_excess_return(
    strategy_return: float, benchmark_return: float
) -> float:
    return strategy_return - benchmark_return


def calculate_monthly_win_rate(daily_returns: pd.Series) -> float:
    if daily_returns.empty:
        return 0.0
    monthly_returns = daily_returns.groupby(
        pd.Grouper(freq="ME")
    ).apply(lambda x: (1 + x).prod() - 1)
    if len(monthly_returns) == 0:
        return 0.0
    win_months = (monthly_returns > 0).sum()
    return float(win_months / len(monthly_returns))


def calculate_yearly_win_rate(daily_returns: pd.Series) -> float:
    if daily_returns.empty:
        return 0.0
    yearly_returns = daily_returns.groupby(
        pd.Grouper(freq="YE")
    ).apply(lambda x: (1 + x).prod() - 1)
    if len(yearly_returns) == 0:
        return 0.0
    win_years = (yearly_returns > 0).sum()
    return float(win_years / len(yearly_returns))


def calculate_benchmark_annualized_return(
    benchmark_curve: pd.Series, trading_days: int = 252
) -> float:
    return calculate_annualized_return(benchmark_curve, trading_days)
