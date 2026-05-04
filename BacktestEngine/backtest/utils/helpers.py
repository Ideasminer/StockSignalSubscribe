from __future__ import annotations

import os
import platform
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def setup_chinese_font():
    system = platform.system()
    try:
        if system == "Windows":
            plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
        elif system == "Darwin":
            plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS"]
        else:
            plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "Noto Sans CJK SC", "SimHei"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False


def generate_mock_data(
    codes: Optional[list[str]] = None,
    start_date: str = "2023-01-01",
    end_date: str = "2024-12-31",
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    if codes is None:
        codes = ["000001.SZ", "399001.SZ"]

    np.random.seed(seed)
    date_range = pd.bdate_range(start=start_date, end=end_date)
    result: Dict[str, pd.DataFrame] = {}

    for code in codes:
        n = len(date_range)
        base_price = np.random.uniform(10, 50)
        returns = np.random.normal(0.0005, 0.02, n)
        prices = base_price * np.exp(np.cumsum(returns))

        df = pd.DataFrame({"date": date_range, "code": code})
        df["close"] = prices
        df["open"] = df["close"] * (1 + np.random.uniform(-0.01, 0.01, n))
        df["high"] = df[["open", "close"]].max(axis=1) * (1 + np.random.uniform(0, 0.015, n))
        df["low"] = df[["open", "close"]].min(axis=1) * (1 - np.random.uniform(0, 0.015, n))
        df["preclose"] = df["close"].shift(1).fillna(df["open"])
        df["volume"] = np.random.randint(1_000_000, 50_000_000, n)
        df["amount"] = df["volume"] * df["close"]
        df["adjustflag"] = 2
        df["turn"] = np.random.uniform(0.5, 5.0, n)
        df["tradestatus"] = 1
        df["pctChg"] = df["close"].pct_change().fillna(0) * 100
        df["isST"] = 0
        df["code_name"] = f"Mock_{code[:6]}"
        df["ipoDate"] = "2000-01-01"
        df["outDate"] = "9999-12-31"
        df["type"] = "stock"
        df["status"] = 1

        result[code] = df

    return result
