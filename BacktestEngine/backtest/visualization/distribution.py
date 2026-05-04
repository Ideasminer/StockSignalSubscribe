from __future__ import annotations

from typing import Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..utils.helpers import setup_chinese_font


def plot_monthly_returns_heatmap(
    daily_returns: pd.Series,
    title: str = "月度收益热力图",
    save_path: Optional[str] = None,
    figsize: tuple = (12, 8),
):
    setup_chinese_font()
    monthly_ret = daily_returns.groupby(
        pd.Grouper(freq="ME")
    ).apply(lambda x: (1 + x).prod() - 1)

    monthly_ret.index = pd.to_datetime(monthly_ret.index)
    years = monthly_ret.index.year
    months = monthly_ret.index.month

    pivot = pd.DataFrame(
        {"year": years, "month": months, "return": monthly_ret.values}
    )
    pivot = pivot.pivot_table(
        index="year", columns="month", values="return", aggfunc="first"
    )
    pivot = pivot * 100

    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.cm.RdYlGn
    norm = mcolors.TwoSlopeNorm(
        vmin=-max(abs(pivot.min().min()), abs(pivot.max().max())),
        vcenter=0,
        vmax=max(abs(pivot.min().min()), abs(pivot.max().max())),
    )

    im = ax.imshow(pivot.values, cmap=cmap, norm=norm, aspect="auto")

    month_labels = ["1月", "2月", "3月", "4月", "5月", "6月",
                     "7月", "8月", "9月", "10月", "11月", "12月"]
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([month_labels[i - 1] for i in pivot.columns], fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(int(y)) for y in pivot.index], fontsize=9)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                text_color = "white" if abs(val) > abs(pivot.min().min()) * 0.6 else "black"
                ax.text(
                    j, i, f"{val:.1f}%",
                    ha="center", va="center",
                    fontsize=8, color=text_color,
                )

    ax.set_title(title, fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8, label="收益率 (%)")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_yearly_returns_heatmap(
    daily_returns: pd.Series,
    title: str = "年度收益热力图",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6),
):
    setup_chinese_font()
    daily_returns.index = pd.to_datetime(daily_returns.index)
    yearly_ret = daily_returns.groupby(
        pd.Grouper(freq="YE")
    ).apply(lambda x: (1 + x).prod() - 1)

    years = yearly_ret.index.year

    monthly_by_year = {}
    for year in years:
        year_data = daily_returns[daily_returns.index.year == year]
        monthly = year_data.groupby(
            pd.Grouper(freq="ME")
        ).apply(lambda x: (1 + x).prod() - 1)
        monthly_by_year[year] = monthly.values * 100

    df = pd.DataFrame(monthly_by_year, index=range(1, 13)).T

    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.cm.RdYlGn
    norm = mcolors.TwoSlopeNorm(
        vmin=-max(abs(df.min().min()), abs(df.max().max())),
        vcenter=0,
        vmax=max(abs(df.min().min()), abs(df.max().max())),
    )

    im = ax.imshow(df.values, cmap=cmap, norm=norm, aspect="auto")

    month_labels = ["1月", "2月", "3月", "4月", "5月", "6月",
                     "7月", "8月", "9月", "10月", "11月", "12月"]
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(month_labels, fontsize=9)
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels([str(int(y)) for y in df.index], fontsize=9)

    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            val = df.values[i, j]
            if not np.isnan(val):
                text_color = "white" if abs(val) > abs(df.min().min()) * 0.6 else "black"
                ax.text(
                    j, i, f"{val:.1f}%",
                    ha="center", va="center",
                    fontsize=8, color=text_color,
                )

    ax.set_title(title, fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8, label="月度收益率 (%)")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_daily_returns_histogram(
    daily_returns: pd.Series,
    title: str = "日度收益分布直方图",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6),
):
    setup_chinese_font()
    fig, ax = plt.subplots(figsize=figsize)

    returns = daily_returns.dropna() * 100
    mean_ret = returns.mean()
    std_ret = returns.std()

    n, bins, patches = ax.hist(
        returns.values,
        bins=50,
        alpha=0.75,
        color="#3498DB",
        edgecolor="white",
        linewidth=0.5,
    )

    for bin_edge, patch in zip(bins[:-1], patches):
        if bin_edge < 0:
            patch.set_facecolor("#E74C3C")
            patch.set_alpha(0.75)

    ax.axvline(mean_ret, color="green", linestyle="--", linewidth=1.5,
               label=f"均值: {mean_ret:.2f}%")
    ax.axvline(0, color="gray", linestyle="-", linewidth=0.8)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("日收益率 (%)", fontsize=11)
    ax.set_ylabel("频次", fontsize=11)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    stats_text = (
        f"均值: {mean_ret:.2f}%\n"
        f"标准差: {std_ret:.2f}%\n"
        f"偏度: {returns.skew():.2f}\n"
        f"峰度: {returns.kurtosis():.2f}"
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
