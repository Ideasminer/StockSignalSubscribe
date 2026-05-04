from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from ..utils.helpers import setup_chinese_font


def plot_equity_curve(
    equity_curve: pd.Series,
    benchmark_curve: Optional[pd.Series] = None,
    title: str = "策略净值曲线",
    save_path: Optional[str] = None,
    figsize: tuple = (12, 6),
):
    setup_chinese_font()
    fig, ax = plt.subplots(figsize=figsize)

    normalized_equity = equity_curve / equity_curve.iloc[0]
    ax.plot(
        normalized_equity.index,
        normalized_equity.values,
        label="策略净值",
        color="#E74C3C",
        linewidth=1.5,
    )

    if benchmark_curve is not None and len(benchmark_curve) > 0:
        normalized_benchmark = benchmark_curve / benchmark_curve.iloc[0]
        ax.plot(
            normalized_benchmark.index,
            normalized_benchmark.values,
            label="基准净值",
            color="#3498DB",
            linewidth=1.5,
            linestyle="--",
        )

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


def plot_drawdown_curve(
    equity_curve: pd.Series,
    title: str = "回撤曲线（水下曲线）",
    save_path: Optional[str] = None,
    figsize: tuple = (12, 6),
):
    setup_chinese_font()
    fig, ax = plt.subplots(figsize=figsize)

    rolling_max = equity_curve.expanding().max()
    drawdown = (equity_curve - rolling_max) / rolling_max

    ax.fill_between(
        drawdown.index,
        0,
        drawdown.values,
        color="#E74C3C",
        alpha=0.6,
        label="回撤",
    )
    ax.plot(
        drawdown.index,
        drawdown.values,
        color="#C0392B",
        linewidth=1,
    )
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("日期", fontsize=11)
    ax.set_ylabel("回撤比例", fontsize=11)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    max_dd = drawdown.min()
    if not pd.isna(max_dd):
        max_dd_idx = drawdown.idxmin()
        ax.annotate(
            f"最大回撤: {max_dd:.2%}",
            xy=(max_dd_idx, max_dd),
            xytext=(max_dd_idx, max_dd * 1.3),
            arrowprops=dict(arrowstyle="->", color="black", lw=1),
            fontsize=10,
            color="black",
        )

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
