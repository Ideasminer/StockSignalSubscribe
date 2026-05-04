from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..metrics.risk_adjusted import calculate_sharpe_ratio
from ..metrics.returns import calculate_annualized_return
from ..utils.helpers import setup_chinese_font


def plot_rolling_sharpe(
    daily_returns: pd.Series,
    window: int = 252,
    title: str = "滚动夏普比率",
    save_path: Optional[str] = None,
    figsize: tuple = (12, 6),
):
    setup_chinese_font()
    fig, ax = plt.subplots(figsize=figsize)

    rolling_sharpe = daily_returns.rolling(window=window).apply(
        lambda x: calculate_sharpe_ratio(x) if len(x) == window else np.nan
    )

    ax.plot(
        rolling_sharpe.index,
        rolling_sharpe.values,
        color="#9B59B6",
        linewidth=1.5,
        label=f"滚动夏普 (窗口={window}天)",
    )
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
    ax.axhline(y=1, color="green", linestyle="--", alpha=0.5, label="夏普=1")
    ax.axhline(y=-1, color="red", linestyle="--", alpha=0.5, label="夏普=-1")

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("日期", fontsize=11)
    ax.set_ylabel("夏普比率", fontsize=11)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_rolling_annualized_return(
    daily_returns: pd.Series,
    window: int = 252,
    title: str = "滚动年化收益率",
    save_path: Optional[str] = None,
    figsize: tuple = (12, 6),
):
    setup_chinese_font()
    fig, ax = plt.subplots(figsize=figsize)

    def rolling_ann_ret(x):
        if len(x) < window:
            return np.nan
        equity = (1 + x).cumprod()
        return calculate_annualized_return(equity, trading_days=window)

    rolling_ret = daily_returns.rolling(window=window).apply(
        lambda x: rolling_ann_ret(x)
    )

    ax.plot(
        rolling_ret.index,
        rolling_ret.values,
        color="#E67E22",
        linewidth=1.5,
        label=f"滚动年化收益 (窗口={window}天)",
    )
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("日期", fontsize=11)
    ax.set_ylabel("年化收益率", fontsize=11)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_annual_returns(
    equity_curve: pd.Series,
    benchmark_curve: Optional[pd.Series] = None,
    title: str = "年度收益对比",
    save_path: Optional[str] = None,
    figsize: tuple = (12, 6),
):
    setup_chinese_font()
    fig, ax = plt.subplots(figsize=figsize)

    strategy_daily = equity_curve.pct_change().dropna()
    strategy_yearly = strategy_daily.groupby(
        pd.Grouper(freq="YE")
    ).apply(lambda x: (1 + x).prod() - 1)

    years = strategy_yearly.index.year
    x = np.arange(len(years))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2,
        strategy_yearly.values * 100,
        width,
        label="策略",
        color="#E74C3C",
        alpha=0.8,
    )

    if benchmark_curve is not None and len(benchmark_curve) > 0:
        bench_daily = benchmark_curve.pct_change().dropna()
        bench_yearly = bench_daily.groupby(
            pd.Grouper(freq="YE")
        ).apply(lambda x: (1 + x).prod() - 1)
        bench_yearly = bench_yearly[bench_yearly.index.year.isin(years)]

        bars2 = ax.bar(
            x + width / 2,
            bench_yearly.values * 100,
            width,
            label="基准",
            color="#3498DB",
            alpha=0.8,
        )

    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("年份", fontsize=11)
    ax.set_ylabel("收益率 (%)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(y)) for y in years], fontsize=10)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    for bar in bars1:
        height = bar.get_height()
        label_y = height + 0.5 if height >= 0 else height - 2.5
        ax.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(bar.get_x() + bar.get_width() / 2, label_y),
            ha="center", va="bottom",
            fontsize=8, color="black",
        )

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
