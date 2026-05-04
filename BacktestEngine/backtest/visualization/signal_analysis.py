from __future__ import annotations

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..utils.helpers import setup_chinese_font


def plot_ic_histogram(
    ic_series: pd.Series,
    title: str = "IC 值分布直方图",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6),
):
    setup_chinese_font()
    fig, ax = plt.subplots(figsize=figsize)
    ic_values = ic_series.dropna()
    n, bins, patches = ax.hist(
        ic_values.values,
        bins=30,
        alpha=0.75,
        color="#3498DB",
        edgecolor="white",
        linewidth=0.5,
    )
    mean_ic = ic_values.mean()
    std_ic = ic_values.std()
    ax.axvline(mean_ic, color="red", linestyle="--", linewidth=2, label=f"均值: {mean_ic:.4f}")
    ax.axvline(0, color="gray", linestyle="-", linewidth=0.8)
    ax.axvline(mean_ic + std_ic, color="orange", linestyle=":", linewidth=1.5, label=f"±1σ: {std_ic:.4f}")
    ax.axvline(mean_ic - std_ic, color="orange", linestyle=":", linewidth=1.5)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("IC 值", fontsize=11)
    ax.set_ylabel("频次", fontsize=11)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    stats_text = (
        f"观测数: {len(ic_values)}\n"
        f"均值: {mean_ic:.4f}\n"
        f"标准差: {std_ic:.4f}\n"
        f"IR: {mean_ic / std_ic:.4f}" if std_ic > 0 else "IR: N/A"
    )
    ax.text(
        0.02, 0.95, stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_radar_chart(
    metrics_dicts: list,
    labels: list,
    title: str = "五维多维度评估雷达图",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8),
):
    """五维度雷达图 — 综合评估策略组合

    Args:
        metrics_dicts: [{"dim1": val, "dim2": val}, ...] 多个策略的雷达图数据
        labels: 策略标签列表
        title: 图表标题
    """
    setup_chinese_font()
    dimensions = list(metrics_dicts[0].keys())
    n_dims = len(dimensions)
    angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    colors = plt.cm.tab10(np.linspace(0, 1, len(metrics_dicts)))

    for i, (m_dict, color) in enumerate(zip(metrics_dicts, colors)):
        values = [m_dict.get(d, 0.0) for d in dimensions]
        values += values[:1]
        ax.fill(angles, values, alpha=0.1, color=color)
        ax.plot(angles, values, 'o-', linewidth=2, color=color, label=labels[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dimensions, fontsize=10)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    ax.grid(True)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_equity_with_benchmark(
    equity_curve: pd.Series,
    benchmark_curves: Dict[str, pd.Series],
    title: str = "策略净值 vs 基准对比",
    save_path: Optional[str] = None,
    figsize: tuple = (12, 6),
):
    setup_chinese_font()
    fig, ax = plt.subplots(figsize=figsize)
    norm_eq = equity_curve / equity_curve.iloc[0]
    ax.plot(norm_eq.index, norm_eq.values, color="#E74C3C", linewidth=2,
            label="策略净值")
    for bname, bcurve in benchmark_curves.items():
        if not bcurve.empty:
            norm_b = bcurve / bcurve.iloc[0]
            ax.plot(norm_b.index, norm_b.values, linewidth=1.5,
                    linestyle="--", label=f"基准_{bname}")

    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("日期", fontsize=11)
    ax.set_ylabel("净值", fontsize=11)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ic_time_series(
    ic_series: pd.Series,
    title: str = "IC 时序图",
    save_path: Optional[str] = None,
    figsize: tuple = (12, 6),
):
    setup_chinese_font()
    fig, ax = plt.subplots(figsize=figsize)
    ic_values = ic_series.dropna()
    ax.plot(
        ic_values.index, ic_values.values,
        color="#2C3E50", linewidth=1, alpha=0.7, label="日度 IC",
    )
    rolling_mean = ic_values.rolling(window=20, min_periods=5).mean()
    ax.plot(
        rolling_mean.index, rolling_mean.values,
        color="#E74C3C", linewidth=2, label="20 日滚动均值",
    )
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(y=ic_values.mean(), color="green", linestyle=":", alpha=0.7,
               label=f"均值: {ic_values.mean():.4f}")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("日期", fontsize=11)
    ax.set_ylabel("IC 值", fontsize=11)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_win_rate_trend(
    daily_returns: pd.Series,
    monthly_returns: Optional[pd.Series] = None,
    weekly_returns: Optional[pd.Series] = None,
    title: str = "胜率趋势图",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6),
):
    setup_chinese_font()
    fig, ax = plt.subplots(figsize=figsize)
    categories = []
    values = []
    if daily_returns is not None and len(daily_returns) > 0:
        categories.append("日胜率")
        values.append((daily_returns > 0).sum() / len(daily_returns))
    if weekly_returns is not None and len(weekly_returns) > 0:
        categories.append("周胜率")
        values.append((weekly_returns > 0).sum() / len(weekly_returns))
    if monthly_returns is not None and len(monthly_returns) > 0:
        categories.append("月胜率")
        values.append((monthly_returns > 0).sum() / len(monthly_returns))
    colors = ["#3498DB", "#2ECC71", "#E74C3C"]
    bars = ax.bar(categories, values, color=colors[:len(categories)],
                  alpha=0.8, width=0.5, edgecolor="white")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="50% 基准线")
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.1%}",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("胜率", fontsize=11)
    ax.set_ylim(0, max(values + [0.5]) * 1.2)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_long_short_performance(
    metrics_dict: Dict[str, float],
    title: str = "多空组合绩效指标对比",
    save_path: Optional[str] = None,
    figsize: tuple = (12, 6),
):
    setup_chinese_font()
    fig, ax = plt.subplots(figsize=figsize)
    display_names = {
        "total_return": "累计收益率",
        "annualized_return": "年化收益率",
        "annualized_volatility": "年化波动率",
        "sharpe_ratio": "夏普比率",
        "max_drawdown": "最大回撤",
        "daily_win_rate": "日胜率",
        "weekly_win_rate": "周胜率",
        "monthly_win_rate": "月胜率",
    }
    display = {}
    for k, v in metrics_dict.items():
        name = display_names.get(k, k)
        if k in ("max_drawdown",):
            display[name] = v
        else:
            display[name] = v
    names = list(display.keys())
    values = list(display.values())
    colors = ["#2ECC71" if v >= 0 else "#E74C3C" for v in values]
    bars = ax.bar(names, values, color=colors, alpha=0.8, edgecolor="white")
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (0.02 if val >= 0 else -0.04),
            f"{val:.2%}" if abs(val) < 1 else f"{val:.2f}",
            ha="center", va="bottom" if val >= 0 else "top",
            fontsize=9,
        )
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("指标值", fontsize=11)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_group_returns(
    group_returns: pd.DataFrame,
    title: str = "分组收益对比",
    save_path: Optional[str] = None,
    figsize: tuple = (12, 6),
):
    setup_chinese_font()
    fig, ax = plt.subplots(figsize=figsize)
    mean_returns = group_returns.mean(axis=1)
    groups = mean_returns.index.tolist()
    values = mean_returns.values
    colors = ["#E74C3C" if v < 0 else "#2ECC71" for v in values]
    bars = ax.bar(groups, values, color=colors, alpha=0.8, edgecolor="white", width=0.6)
    for bar, val in zip(bars, values):
        label_y = val + 0.0005 if val >= 0 else val - 0.001
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            label_y,
            f"{val:.4%}",
            ha="center", va="bottom" if val >= 0 else "top",
            fontsize=10, fontweight="bold",
        )
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
    ls_val = values[-1] - values[0] if len(values) >= 2 else 0
    ax.text(
        0.02, 0.95,
        f"多空收益 (Q{len(values)}-Q1): {ls_val:.4%}",
        transform=ax.transAxes,
        fontsize=11, fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("分组", fontsize=11)
    ax.set_ylabel("平均收益率", fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cumulative_return_by_group(
    group_returns: pd.DataFrame,
    title: str = "分组累计收益",
    save_path: Optional[str] = None,
    figsize: tuple = (12, 6),
):
    setup_chinese_font()
    fig, ax = plt.subplots(figsize=figsize)
    dates = pd.to_datetime(group_returns.columns)
    colors = plt.cm.RdYlGn(np.linspace(0, 1, len(group_returns)))
    for i, (group_name, row) in enumerate(group_returns.iterrows()):
        cum_ret = (1 + row).cumprod()
        ax.plot(
            dates, cum_ret.values,
            label=group_name, color=colors[i], linewidth=1.5,
        )
    ax.axhline(y=1, color="gray", linestyle=":", alpha=0.5)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("日期", fontsize=11)
    ax.set_ylabel("累计净值", fontsize=11)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
