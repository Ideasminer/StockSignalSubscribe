"""基本面数据获取 — 基于 baostock 季频盈利能力
参照: https://baostock.com/mainContent?file=seasonProfit.md

返回字段: npMargin(净利率), epsTTM(每股收益TTM)
给出判断: 收益波动 / 收益向上 / 收益劣化
"""
from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def fetch_quarterly_profit(code: str, year: int, quarter: int) -> Optional[dict]:
    """单只股票的单季度盈利数据"""
    from data_fetcher import _ensure_bs_login
    _ensure_bs_login()
    import baostock as bs
    rs = bs.query_profit_data(code, year=year, quarter=quarter)
    if rs.error_code != '0':
        return None
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    if not rows:
        return None
    return dict(zip(rs.fields, rows[0]))


def fetch_recent_4_quarters(code: str) -> pd.DataFrame:
    """获取最近4个季度的盈利数据

    返回 DataFrame 含列: code, year, quarter, npMargin, epsTTM, roe
    """
    from data_fetcher import _ensure_bs_login
    _ensure_bs_login()

    results = []
    now = pd.Timestamp.now()
    current_quarter = (now.month - 1) // 3 + 1
    current_year = now.year

    for offset in range(4):
        q = current_quarter - offset
        y = current_year
        while q <= 0:
            q += 4
            y -= 1
        try:
            row = fetch_quarterly_profit(code, y, q)
            if row:
                row["query_year"] = y
                row["query_quarter"] = q
                results.append(row)
            time.sleep(0.05)
        except Exception:
            continue

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    for col in ["npMargin", "epsTTM", "roe"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def assess_profit_trend(df: pd.DataFrame) -> dict:
    """基于近4季度盈利能力给出判断

    Returns:
        {"trend": "收益向上"|"收益劣化"|"收益波动"|"数据不足",
         "np_margins": [...],
         "eps_ttms": [...],
         "detail": str}
    """
    if df.empty or "npMargin" not in df.columns or "epsTTM" not in df.columns:
        return {"trend": "数据不足", "np_margins": [], "eps_ttms": [], "detail": "无可用数据"}

    df = df.sort_values(["query_year", "query_quarter"])
    margins = df["npMargin"].dropna().tolist()
    epss = df["epsTTM"].dropna().tolist()

    if len(margins) < 2:
        return {"trend": "数据不足", "np_margins": margins, "eps_ttms": epss,
                "detail": f"仅{len(margins)}个季度数据"}

    # 2个方向判断: margin + eps 各加权50%
    score = 0
    detail_parts = []

    # npMargin 趋势
    if len(margins) >= 2:
        changes = np.diff(margins)
        pos = sum(1 for c in changes if c > 0)
        neg = sum(1 for c in changes if c < 0)
        if pos >= len(changes) * 0.67:
            score += 1
            detail_parts.append("净利率持续走强")
        elif neg >= len(changes) * 0.67:
            score -= 1
            detail_parts.append("净利率持续走弱")
        else:
            detail_parts.append("净利率波动")

    # epsTTM 趋势
    if len(epss) >= 2:
        changes = np.diff(epss)
        pos = sum(1 for c in changes if c > 0)
        neg = sum(1 for c in changes if c < 0)
        if pos >= len(changes) * 0.67:
            score += 1
            detail_parts.append("每股收益持续走强")
        elif neg >= len(changes) * 0.67:
            score -= 1
            detail_parts.append("每股收益持续走弱")
        else:
            detail_parts.append("每股收益波动")

    if score >= 2:
        trend = "收益向上"
    elif score <= -2:
        trend = "收益劣化"
    else:
        trend = "收益波动"

    return {
        "trend": trend,
        "np_margins": margins,
        "eps_ttms": epss,
        "detail": "; ".join(detail_parts),
    }


def fetch_fundamentals_batch(codes: List[str], delay: float = 0.1) -> pd.DataFrame:
    """批量获取基本面数据

    Returns DataFrame: code, trend, np_margins, eps_ttms, detail
    """
    rows = []
    for i, code in enumerate(codes):
        try:
            df = fetch_recent_4_quarters(code)
            assess = assess_profit_trend(df)
            rows.append({
                "code": code,
                "trend": assess["trend"],
                "np_margins": ",".join(f"{v:.2f}" for v in assess["np_margins"]),
                "eps_ttms": ",".join(f"{v:.4f}" for v in assess["eps_ttms"]),
                "detail": assess["detail"],
            })
        except Exception:
            rows.append({
                "code": code, "trend": "数据不足",
                "np_margins": "", "eps_ttms": "", "detail": "查询失败",
            })
        time.sleep(delay)
        if (i + 1) % 50 == 0:
            print(f"  基本面进度: {i+1}/{len(codes)}", flush=True)

    return pd.DataFrame(rows)
