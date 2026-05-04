"""数据获取模块 — 基于 baostock 拉取全量A股历史K线

两种获取模式:
  fetch_all_daily_data()   — 全量拉取 (首次使用或重建)
  merge_latest_daily_data() — 增量拼接 (日常更新: 历史179天 + 最新1天 = 180天)
"""
from __future__ import annotations

import os
import re
import time
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

_BS_LOGGED_IN = False

REQUIRED_COLUMNS = ["date", "code", "close", "tradestatus"]


def _ensure_bs_login():
    global _BS_LOGGED_IN
    if _BS_LOGGED_IN:
        return
    import baostock as bs
    lg = bs.login()
    if lg.error_code != '0':
        raise ConnectionError(f"baostock login失败: {lg.error_msg}")
    _BS_LOGGED_IN = True


def _disable_progress():
    return {"disable": not os.isatty(0)}


# ============================================================
# 股票代码获取
# ============================================================


def fetch_all_stock_codes(batch_size: int = 0) -> pd.DataFrame:
    """获取当前全部A股代码列表

    baostock query_all_stock 在非交易日可能返回空数据,
    此时依次回退 1~7 天重试。
    """
    _ensure_bs_login()
    import baostock as bs

    for offset in range(7):
        query_date = (datetime.now() - timedelta(days=offset)).strftime("%Y-%m-%d")
        rs = bs.query_all_stock(query_date)
        df = rs.get_data()
        if df is not None and not df.empty and "code" in df.columns:
            codes = df[df["code"].str.startswith(("sh.", "sz."))].copy()
            if not codes.empty:
                if batch_size > 0:
                    codes = codes.head(batch_size)
                return codes.reset_index(drop=True)

    raise RuntimeError("baostock query_all_stock 连续7天返回空数据, 请检查网络或 baostock 服务状态")


def fetch_daily_kline(
    code: str,
    start_date: str,
    end_date: str,
    fields: str = "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST",
) -> pd.DataFrame:
    """拉取单只股票日K线"""
    _ensure_bs_login()
    import baostock as bs
    rs = bs.query_history_k_data_plus(code, fields, start_date=start_date, end_date=end_date)
    if rs.error_code != '0':
        return pd.DataFrame()
    return rs.get_data()


# ============================================================
# 日期计算
# ============================================================


def trading_days_to_calendar_days(trading_days: int) -> int:
    """交易日数 → 自然日数 (保守估计: trading * 1.5, 下限30天)"""
    return max(trading_days * 3 // 2, 30)


def _calc_trade_date_range(lookback_trading_days: int) -> Tuple[str, str]:
    """根据目标交易日数计算 baostock 查询的起止自然日

    Args:
        lookback_trading_days: 希望覆盖的交易日数 (如 180)

    Returns:
        (start_date, end_date): YYYY-MM-DD 格式

    例: lookback_trading_days=15  → ~23自然日区间
        lookback_trading_days=180 → ~270自然日区间
    """
    end = datetime.now()
    calendar_days = trading_days_to_calendar_days(lookback_trading_days)
    start = end - timedelta(days=calendar_days)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


# ============================================================
# 文件检索与校验
# ============================================================


_DATE_FILE_PATTERN = re.compile(r"^(\d{8})\.csv$")


def find_latest_data_file(data_dir: str) -> Tuple[Optional[str], Optional[str]]:
    """在 data_dir 下检索 yyyyMMdd.csv 文件, 返回 (最新日期串, 完整路径)

    排序规则: 按文件名中的日期字符串降序 → 取第一个
    """
    os.makedirs(data_dir, exist_ok=True)
    candidates = []
    for fname in os.listdir(data_dir):
        m = _DATE_FILE_PATTERN.match(fname)
        if m:
            candidates.append((m.group(1), os.path.join(data_dir, fname)))

    if not candidates:
        return None, None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0]


def validate_data_file(filepath: str) -> Tuple[bool, str]:
    """校验数据文件的完整性和格式

    检查项:
      - 文件存在且可读
      - 包含必须列 (date, code, close, tradestatus)
      - 行数 > 0
      - date 列可解析为日期

    Returns:
        (is_valid, reason)
    """
    if not os.path.exists(filepath):
        return False, f"文件不存在: {filepath}"

    try:
        df = pd.read_csv(filepath, nrows=5)
    except Exception as e:
        return False, f"CSV读取失败: {e}"

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return False, f"缺少必要列: {missing}"

    # 校验列不为全空
    if df["date"].isna().all():
        return False, "date列为空"
    if df["code"].isna().all():
        return False, "code列为空"

    try:
        pd.to_datetime(df["date"].dropna().iloc[0])
    except Exception:
        return False, "date列格式不可解析"

    return True, "OK"


# ============================================================
# 全量拉取
# ============================================================


def fetch_all_daily_data(
    data_dir: str,
    lookback_trading_days: int = 180,
    stock_batch_size: int = 0,
    delay: float = 0.2,
) -> str:
    """全量拉取: 从 baostock 获取全量A股过去 N 个交易日的日K线

    适用于首次运行或数据重建, 不依赖历史文件。

    Args:
        data_dir: 数据存储目录
        lookback_trading_days: 目标交易日数 (默认180)
        stock_batch_size: 股票数量限制 (0=全部)
        delay: 请求间隔秒数

    Returns:
        保存路径 subscribe/data/yyyyMMdd.csv
    """
    print(f"全量更新|开始拉取 {lookback_trading_days} 个交易日的日K线", flush=True)
    os.makedirs(data_dir, exist_ok=True)
    start_date, end_date = _calc_trade_date_range(lookback_trading_days)

    codes_df = fetch_all_stock_codes(batch_size=stock_batch_size)
    all_codes = codes_df["code"].tolist()
    calendar_span = trading_days_to_calendar_days(lookback_trading_days)
    print(f"共 {len(all_codes)} 只A股 | 目标 {lookback_trading_days} 个交易日 "
          f"({calendar_span} 自然日) | 日期 {start_date} ~ {end_date}", flush=True)

    frames = []
    failed = 0
    for code in tqdm(all_codes, desc="拉取K线", **_disable_progress()):
        try:
            kdf = fetch_daily_kline(code, start_date, end_date)
            if kdf.empty:
                failed += 1
                continue
            basic = codes_df[codes_df["code"] == code]
            if not basic.empty and "code_name" in basic.columns:
                kdf["code_name"] = basic["code_name"].iloc[0]
            frames.append(kdf)
            time.sleep(delay)
        except Exception:
            failed += 1

    if not frames:
        raise RuntimeError(f"所有{len(all_codes)}只股票拉取失败")

    merged = pd.concat(frames, ignore_index=True)
    merged = merged[merged["tradestatus"] == "1"]
    merged["close"] = pd.to_numeric(merged["close"], errors="coerce")
    merged = merged[merged["close"] > 0]
    merged["date"] = pd.to_datetime(merged["date"])

    date_str = datetime.now().strftime("%Y%m%d")
    save_path = os.path.join(data_dir, f"{date_str}.csv")
    merged.to_csv(save_path, index=False, encoding="utf-8-sig")

    actual_trading_days = merged["date"].nunique()
    print(f"保存: {save_path} | {merged.code.nunique()} stocks × "
          f"{actual_trading_days} trading dates | 失败={failed}", flush=True)
    return save_path


# ============================================================
# 增量拼接 (日常更新)
# ============================================================


def fetch_latest_trading_day(
    codes: pd.DataFrame,
    delay: float = 0.15,
) -> pd.DataFrame:
    """获取所有A股最新一个交易日的数据

    基于 baostock 拉取过去7自然日的数据, 然后取最大日期。
    这样可以覆盖非交易日场景 (周末/假日 → 自动取最近的交易日)。

    Returns:
        DataFrame with latest trading day for all codes
    """
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    all_codes = codes["code"].tolist()

    print(f"获取最新交易日数据: {start_date} ~ {end_date} | {len(all_codes)} 只股票", flush=True)

    frames = []
    failed = 0
    for code in tqdm(all_codes, desc="最新交易日", **_disable_progress()):
        try:
            kdf = fetch_daily_kline(code, start_date, end_date)
            if kdf.empty:
                failed += 1
                continue
            frames.append(kdf)
            time.sleep(delay)
        except Exception:
            failed += 1

    if not frames:
        raise RuntimeError("所有股票最新交易日拉取失败")

    merged = pd.concat(frames, ignore_index=True)
    merged = merged[merged["tradestatus"] == "1"]
    merged["close"] = pd.to_numeric(merged["close"], errors="coerce")
    merged = merged[merged["close"] > 0]
    merged["date"] = pd.to_datetime(merged["date"])

    latest_date = merged["date"].max()
    latest_df = merged[merged["date"] == latest_date].copy()
    print(f"  最新交易日: {latest_date.strftime('%Y-%m-%d')} | {latest_df.code.nunique()} 只股票 | 失败={failed}", flush=True)
    return latest_df


def _detect_new_trading_day(hist_latest_date: pd.Timestamp) -> bool:
    """探测自 hist_latest_date 以来是否有新的交易日

    用 baostock 查询 sh.000001 在 (hist_latest_date+1天, 今天] 区间内
    是否有交易数据。有 → True (需要增量拉取), 无 → False (跳过)。

    这只需要 1 次 API 调用, 避免对全部股票做无效拉取。
    """
    start = (hist_latest_date + timedelta(days=1)).strftime("%Y-%m-%d")
    end = datetime.now().strftime("%Y-%m-%d")

    # 如果起始日期 >= 今天, 肯定没有新交易日
    if start >= end:
        return False

    kdf = fetch_daily_kline("sh.000001", start, end)
    if kdf.empty:
        return False

    kdf = kdf[kdf["tradestatus"] == "1"]
    kdf["close"] = pd.to_numeric(kdf["close"], errors="coerce")
    kdf = kdf[kdf["close"] > 0]
    return not kdf.empty


def _copy_as_today(src_path: str, data_dir: str) -> str:
    """将历史数据文件复制为当日命名的新文件"""
    today_str = datetime.now().strftime("%Y%m%d")
    dst_path = os.path.join(data_dir, f"{today_str}.csv")
    hist_df = pd.read_csv(src_path, low_memory=False)
    hist_df.to_csv(dst_path, index=False, encoding="utf-8-sig")
    print(f"复制: {os.path.basename(src_path)} → {os.path.basename(dst_path)} (无新交易日)", flush=True)
    return dst_path


def merge_latest_daily_data(
    data_dir: str,
    target_trading_days: int = 180,
    history_days: int = 179,
    stock_batch_size: int = 0,
    delay: float = 0.15,
) -> str:
    """增量拼接: 历史 (N-1) 天 + 最新 1 天 = 完整 N 天数据集

    流程:
      1. 检索 data_dir 下最新的 yyyyMMdd.csv → 校验
      2. 探测自历史最新日期以来是否有新交易日
         - 无 → 复制历史文件为当日文件, 直接返回
         - 有 → 继续
      3. 从历史文件中截取最近 history_days 个交易日
      4. 通过 baostock 获取最新 1 个交易日全量A股数据
      5. 拼接 历史 + 最新, 按 code+date 去重, 保留最新
      6. 保存为新文件 yyyyMMdd.csv
    """
    os.makedirs(data_dir, exist_ok=True)

    # Step 1: 检索 + 校验最新历史文件
    date_str, hist_path = find_latest_data_file(data_dir)
    if date_str is None:
        print("未找到历史数据文件, 回退到全量拉取模式", flush=True)
        return fetch_all_daily_data(data_dir, lookback_trading_days=target_trading_days,
                                    stock_batch_size=stock_batch_size, delay=delay)

    valid, reason = validate_data_file(hist_path)
    if not valid:
        print(f"历史文件校验失败 ({reason}), 回退到全量拉取模式", flush=True)
        return fetch_all_daily_data(data_dir, lookback_trading_days=target_trading_days,
                                    stock_batch_size=stock_batch_size, delay=delay)

    print(f"历史文件: {os.path.basename(hist_path)} (日期标识 {date_str})", flush=True)

    # Step 2: 读取历史, 截取最近 history_days 个交易日
    hist_df = pd.read_csv(hist_path, low_memory=False)
    hist_df["date"] = pd.to_datetime(hist_df["date"])
    hist_df = hist_df[hist_df["tradestatus"].astype(str) == "1"]
    hist_df["close"] = pd.to_numeric(hist_df["close"], errors="coerce")
    hist_df = hist_df[hist_df["close"] > 0]

    all_hist_dates = sorted(hist_df["date"].unique())
    hist_latest = all_hist_dates[-1]

    # Step 3: 探测新交易日 — 无则跳过拉取, 直接复制
    if not _detect_new_trading_day(hist_latest):
        return _copy_as_today(hist_path, data_dir)

    print(f"检测到新交易日 (历史最新: {hist_latest.strftime('%Y-%m-%d')}), 开始增量拉取", flush=True)

    if len(all_hist_dates) <= history_days:
        print(f"历史文件仅有 {len(all_hist_dates)} 个交易日 (不足 {history_days}), "
              f"使用全部历史数据", flush=True)
        recent_dates = all_hist_dates
    else:
        recent_dates = all_hist_dates[-history_days:]

    hist_recent = hist_df[hist_df["date"].isin(recent_dates)].copy()
    old_codes = set(hist_recent["code"].unique())
    print(f"历史截取: {len(recent_dates)} 交易日 ({recent_dates[0].strftime('%Y-%m-%d')} ~ "
          f"{recent_dates[-1].strftime('%Y-%m-%d')}) | {len(old_codes)} 只股票", flush=True)

    # Step 4: 获取最新 1 个交易日
    codes_df = fetch_all_stock_codes(batch_size=stock_batch_size)
    latest_df = fetch_latest_trading_day(codes_df, delay=delay)

    # Step 5: 拼接 + 去重
    combined = pd.concat([hist_recent, latest_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["date", "code"], keep="last")
    combined = combined.sort_values(["code", "date"]).reset_index(drop=True)

    final_dates = sorted(combined["date"].unique())
    final_codes = combined["code"].nunique()
    print(f"拼接完成: {final_codes} stocks × {len(final_dates)} dates "
          f"({final_dates[0].strftime('%Y-%m-%d')} ~ {final_dates[-1].strftime('%Y-%m-%d')})", flush=True)

    # Step 6: 保存
    today_str = datetime.now().strftime("%Y%m%d")
    save_path = os.path.join(data_dir, f"{today_str}.csv")
    combined.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"保存: {save_path}", flush=True)
    return save_path
