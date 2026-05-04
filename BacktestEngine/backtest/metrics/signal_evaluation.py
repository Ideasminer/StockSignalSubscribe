from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats


def calculate_pearson_ic(signal: pd.Series, fwd_return: pd.Series) -> float:
    mask = signal.notna() & fwd_return.notna() & (~np.isinf(signal)) & (~np.isinf(fwd_return))
    if mask.sum() < 3:
        return 0.0
    result = stats.pearsonr(signal[mask], fwd_return[mask])
    return float(result[0])


def calculate_rank_ic(signal: pd.Series, fwd_return: pd.Series) -> float:
    mask = signal.notna() & fwd_return.notna() & (~np.isinf(signal)) & (~np.isinf(fwd_return))
    if mask.sum() < 3:
        return 0.0
    result = stats.spearmanr(signal[mask], fwd_return[mask])
    return float(result[0])


def calculate_forward_returns(
    close_df: pd.DataFrame, periods: int = 1
) -> pd.DataFrame:
    """计算未来N期收益率矩阵
    时点对齐规则: return_df[:, t] = close[:, t+periods] / close[:, t] - 1
    即: 矩阵列t对应的值是"从t到t+periods"的持有期收益率
    IC分析时：signal[:, t] vs return_df[:, t] = T日信号预测T→T+N收益
    """
    fwd_close = close_df.shift(-periods, axis=1)
    fwd_returns = fwd_close / close_df - 1.0
    return fwd_returns


def calculate_ic_series(
    signal_df: pd.DataFrame,
    return_df: pd.DataFrame,
    method: str = "rank",
) -> pd.Series:
    """逐日计算截面IC值
    时点约束: signal_df与return_df按相同日期列对齐
    调用方必须确保:
    - return_df[:, t] 是 t→t+N 的未来收益（通过calculate_forward_returns获得）
    - signal_df[:, t] 的生成仅使用 ≤t 的数据（不得包含未来信息）
    - 两者取common_dates交集后，column对应关系为 signal[t] ↔ return[t]
    """
    common_dates = signal_df.columns.intersection(return_df.columns)
    common_codes = signal_df.index.intersection(return_df.index)
    signal_df = signal_df.loc[common_codes, common_dates]
    return_df = return_df.loc[common_codes, common_dates]
    ic_values = {}
    for date in common_dates:
        s = signal_df[date]
        r = return_df[date]
        if method == "rank":
            ic = calculate_rank_ic(s, r)
        else:
            ic = calculate_pearson_ic(s, r)
        ic_values[date] = ic
    return pd.Series(ic_values).sort_index()


def calculate_ic_statistics(ic_series: pd.Series) -> Dict[str, float]:
    ic_series = ic_series.dropna()
    if len(ic_series) < 2:
        n = len(ic_series)
        return {"mean": 0.0, "std": 0.0, "t_stat": 0.0, "p_value": 1.0, "ir": 0.0, "count": n}
    mean_ic = float(ic_series.mean())
    std_ic = float(ic_series.std())
    n = len(ic_series)
    if std_ic == 0:
        return {"mean": mean_ic, "std": 0.0, "t_stat": 0.0, "p_value": 1.0, "ir": 0.0, "count": n}
    t_stat = mean_ic / (std_ic / np.sqrt(n))
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1))
    ir = mean_ic / std_ic if std_ic > 0 else 0.0
    return {
        "mean": mean_ic,
        "std": std_ic,
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "ir": ir,
        "count": n,
    }


def factor_quantile_analysis(
    factor_df: pd.DataFrame,
    return_df: pd.DataFrame,
    n_quantiles: int = 5,
) -> pd.DataFrame:
    common_dates = factor_df.columns.intersection(return_df.columns)
    common_codes = factor_df.index.intersection(return_df.index)
    factor_df = factor_df.loc[common_codes, common_dates]
    return_df = return_df.loc[common_codes, common_dates]
    group_returns = {}
    for date in common_dates:
        f = factor_df[date].dropna()
        r = return_df[date]
        if len(f) < n_quantiles:
            continue
        try:
            labels = pd.qcut(f.rank(method="first"), n_quantiles, labels=False)
        except ValueError:
            continue
        for g in range(n_quantiles):
            mask = labels == g
            group_codes = f.index[mask]
            group_mean_ret = r.reindex(group_codes).mean()
            if g not in group_returns:
                group_returns[g] = {}
            group_returns[g][date] = group_mean_ret
    result = pd.DataFrame(group_returns).T
    result.index = [f"Q{i+1}" for i in range(len(result))]
    result = result.sort_index()
    return result


def calculate_long_short_portfolio(
    group_returns: pd.DataFrame,
    trading_days: int = 252,
    risk_free_rate: float = 0.03,
) -> Dict[str, float]:
    if group_returns.empty or len(group_returns) < 2:
        return {}
    top_key = group_returns.index[-1]
    bottom_key = group_returns.index[0]
    ls_returns = group_returns.loc[top_key] - group_returns.loc[bottom_key]
    ls_returns = ls_returns.dropna()
    if len(ls_returns) < 2:
        return {}
    cumulative = (1 + ls_returns).cumprod()
    total_ret = float(cumulative.iloc[-1]) - 1 if len(cumulative) > 0 else 0.0
    n = len(ls_returns)
    years = n / trading_days
    ann_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0.0
    ann_vol = float(ls_returns.std() * np.sqrt(trading_days))
    sharpe = (ann_ret - risk_free_rate) / ann_vol if ann_vol > 0 else 0.0
    rolling_max = cumulative.expanding().max()
    dd = (cumulative - rolling_max) / rolling_max
    max_dd = float(dd.min()) if not dd.empty else 0.0
    monthly = ls_returns.groupby(pd.Grouper(freq="ME")).apply(lambda x: (1 + x).prod() - 1)
    weekly = ls_returns.groupby(pd.Grouper(freq="W")).apply(lambda x: (1 + x).prod() - 1)
    daily_win = (ls_returns > 0).sum() / max(len(ls_returns), 1)
    weekly_win = (weekly > 0).sum() / max(len(weekly), 1)
    monthly_win = (monthly > 0).sum() / max(len(monthly), 1)
    return {
        "total_return": total_ret,
        "annualized_return": ann_ret,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "daily_win_rate": float(daily_win),
        "weekly_win_rate": float(weekly_win) if not np.isnan(weekly_win) else 0.0,
        "monthly_win_rate": float(monthly_win) if not np.isnan(monthly_win) else 0.0,
    }


def calculate_group_turnover(
    group_assignments: pd.DataFrame,
) -> float:
    if group_assignments.empty:
        return 0.0
    turnovers = []
    prev_assignment = None
    for date in sorted(group_assignments.columns):
        curr = group_assignments[date].dropna()
        if prev_assignment is not None:
            common = curr.index.intersection(prev_assignment.index)
            if len(common) > 0:
                changed = (curr.reindex(common) != prev_assignment.reindex(common)).sum()
                turnover = changed / len(common)
                turnovers.append(turnover)
        prev_assignment = curr
    if len(turnovers) == 0:
        return 0.0
    return float(np.mean(turnovers))
