from __future__ import annotations

import os
import glob
import gc
from typing import List, Optional

import numpy as np
import pandas as pd


def merge_data_getter_csvs(
    data_dir: str,
    start_year: int = 2023,
    end_year: int = 2026,
    exclude_codes_pattern: Optional[List[str]] = None,
    max_stocks: Optional[int] = None,
) -> pd.DataFrame:
    """合并 DataGetter 目录下所有 CSV 文件为单个 DataFrame

    遍历 data_dir 下所有 202*-*.csv 文件（排除 merged.csv），
    合并为单个大数据集，保留全部原始数据不做抽样。

    Args:
        data_dir: CSV 文件所在目录
        start_year: 起始年份
        end_year: 结束年份
        exclude_codes_pattern: 排除的股票代码模式列表，如 ['sh.000', 'sz.399'] 排除指数

    Returns:
        pd.DataFrame: 合并后的完整数据集
    """
    pattern = os.path.join(data_dir, "202*-*.csv")
    files = sorted(glob.glob(pattern))
    files = [f for f in files if "merged" not in os.path.basename(f).lower()]
    if not files:
        raise FileNotFoundError(f"未找到匹配的 CSV 文件: {pattern}")

    print(f"发现 {len(files)} 个 CSV 文件，开始合并...")
    frames: List[pd.DataFrame] = []
    dtypes = {
        "date": str, "code": str, "open": float, "high": float,
        "low": float, "close": float, "preclose": float,
        "volume": float, "amount": float, "adjustflag": int,
        "turn": float, "tradestatus": int, "pctChg": float,
        "isST": int, "code_name": str, "ipoDate": str,
        "outDate": str, "type": str, "status": int,
    }
    cols_needed = list(dtypes.keys())

    for i, f in enumerate(files):
        df = pd.read_csv(
            f, usecols=cols_needed, dtype=dtypes,
            low_memory=False,
        )
        df["date"] = pd.to_datetime(df["date"])
        year_val = df["date"].dt.year.iloc[0] if len(df) > 0 else 0
        if start_year <= year_val <= end_year:
            frames.append(df)
        if (i + 1) % 5 == 0:
            print(f"  已加载 {i+1}/{len(files)} 个文件...")

    if not frames:
        raise ValueError("没有符合年份范围的数据文件")

    merged = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()

    if exclude_codes_pattern:
        for pat in exclude_codes_pattern:
            merged = merged[~merged["code"].str.startswith(pat)]

    merged = merged.sort_values(["code", "date"]).reset_index(drop=True)
    merged = merged.drop_duplicates(subset=["date", "code"], keep="first")
    merged = merged[merged["tradestatus"] == 1]
    merged = merged[merged["close"] > 0]
    merged = merged[merged["isST"] == 0]

    codes = merged["code"].unique()
    dates = merged["date"].unique()

    if max_stocks and max_stocks < len(codes):
        np.random.seed(42)
        sampled = np.random.choice(codes, size=max_stocks, replace=False)
        merged = merged[merged["code"].isin(sampled)]
        codes = sampled

    print(f"合并完成: {len(merged):,} 行, {len(codes):,} 只股票, "
          f"{dates[0].strftime('%Y-%m-%d')} ~ {dates[-1].strftime('%Y-%m-%d')}")

    return merged
