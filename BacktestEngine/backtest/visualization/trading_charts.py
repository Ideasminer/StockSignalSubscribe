from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..core.action import Trade
from ..utils.helpers import setup_chinese_font


def plot_signals_on_equity(
    equity_curve: pd.Series,
    trades: List[Trade],
    data: Dict[str, pd.DataFrame],
    title: str = "买卖信号标注图",
    save_path: Optional[str] = None,
    figsize: tuple = (14, 8),
):
    setup_chinese_font()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={"height_ratios": [3, 1]})

    normalized_equity = equity_curve / equity_curve.iloc[0]
    ax1.plot(
        normalized_equity.index,
        normalized_equity.values,
        color="#2C3E50",
        linewidth=1.5,
        label="策略净值",
    )

    buy_signals = [t for t in trades if t.direction == "buy"]
    sell_signals = [t for t in trades if t.direction == "sell"]

    buy_dates = [pd.Timestamp(t.date) for t in buy_signals if pd.Timestamp(t.date) in normalized_equity.index]
    sell_dates = [pd.Timestamp(t.date) for t in sell_signals if pd.Timestamp(t.date) in normalized_equity.index]

    buy_values = [normalized_equity.loc[d] for d in buy_dates]
    sell_values = [normalized_equity.loc[d] for d in sell_dates]

    if buy_dates:
        ax1.scatter(
            buy_dates, buy_values,
            marker="^", color="red", s=100, alpha=0.8,
            label="买入信号", zorder=5,
        )
    if sell_dates:
        ax1.scatter(
            sell_dates, sell_values,
            marker="v", color="green", s=100, alpha=0.8,
            label="卖出信号", zorder=5,
        )

    ax1.set_title(title, fontsize=14, fontweight="bold")
    ax1.set_ylabel("净值", fontsize=11)
    ax1.legend(loc="best", fontsize=10)
    ax1.grid(True, alpha=0.3)

    holding_data = []
    for i, t in enumerate(trades):
        if t.direction == "sell":
            buy_t = None
            for bt in reversed(trades[:i]):
                if bt.direction == "buy" and bt.code == t.code:
                    buy_t = bt
                    break
            if buy_t:
                holding_days = (pd.Timestamp(t.date) - pd.Timestamp(buy_t.date)).days
                holding_data.append((pd.Timestamp(t.date), holding_days))

    if holding_data:
        dates, days = zip(*holding_data)
        ax2.bar(dates, days, color="#3498DB", alpha=0.7, width=3)
        ax2.set_ylabel("持仓天数", fontsize=11)
        ax2.set_xlabel("日期", fontsize=11)
    else:
        ax2.text(0.5, 0.5, "暂无持仓数据", ha="center", va="center", fontsize=12)
        ax2.set_xlabel("日期", fontsize=11)

    ax2.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_position_changes(
    positions_history: pd.DataFrame,
    title: str = "仓位/杠杆变化图",
    save_path: Optional[str] = None,
    figsize: tuple = (12, 6),
):
    setup_chinese_font()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    if positions_history.empty:
        ax1.text(0.5, 0.5, "暂无持仓数据", ha="center", va="center", transform=ax1.transAxes, fontsize=12)
        ax2.text(0.5, 0.5, "暂无持仓数据", ha="center", va="center", transform=ax2.transAxes, fontsize=12)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    dates = pd.to_datetime(positions_history.index)
    total_equity = positions_history["total_equity"].values
    cash = positions_history["cash"].values

    position_pct = (total_equity - cash) / total_equity * 100
    cash_pct = cash / total_equity * 100

    ax1.fill_between(dates, 0, position_pct, label="持仓比例", color="#E74C3C", alpha=0.6)
    ax1.fill_between(dates, position_pct, 100, label="现金比例", color="#3498DB", alpha=0.4)
    ax1.set_ylabel("仓位比例 (%)", fontsize=11)
    ax1.set_title(title, fontsize=14, fontweight="bold")
    ax1.legend(loc="best", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 105)

    mv_columns = [c for c in positions_history.columns
                   if c.endswith("_mv") and not c.startswith("total_")]
    if mv_columns:
        colors = plt.cm.Set3(np.linspace(0, 1, len(mv_columns)))
        bottom = np.zeros(len(positions_history))
        for col, color in zip(mv_columns, colors):
            values = positions_history[col].values / total_equity * 100
            code_name = col.replace("_mv", "")
            ax2.bar(dates, values, bottom=bottom, label=code_name,
                    color=color, alpha=0.7, width=1)
            bottom += values

    ax2.set_ylabel("各标的仓位 (%)", fontsize=11)
    ax2.set_xlabel("日期", fontsize=11)
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
