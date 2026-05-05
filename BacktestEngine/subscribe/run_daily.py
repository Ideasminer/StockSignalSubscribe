#!/usr/bin/env python
"""信号订阅 — 每日运行脚本

完整流程:
  1. 数据获取: baostock 拉取全量A股 ~N日K线 → data/yyyyMMdd.csv
  2. 信号计算: 对14个注册频道逐一计算信号值 → 阈值筛选命中
  3. 基本面: 命中标的近4季度 npMargin + epsTTM + 趋势判定
  4. 生成报告: HTML 信号日报 → output/yyyyMMdd_信号日报.html
  5. 邮件发送: SMTP 发送至指定邮箱

用法:
  python run_daily.py                          # 完整流程
  python run_daily.py --skip-fetch             # 跳过数据拉取(用已有csv)
  python run_daily.py --skip-mail              # 跳过邮件发送
  python run_daily.py --max-stocks 50          # 仅处理前50只(测试)
  python run_daily.py --max-pct-change 3.0     # 剔除|涨跌幅|>3%的标的
  python run_daily.py --max-targets 50         # 筛选后目标≤50只
  python run_daily.py --min-trading-days 60    # 剔除上市<60日的次新股(默认60)
  python run_daily.py --llm                    # 启用大模型解读(需联网)
"""
import argparse
import os
import re
import sys
import time
from datetime import datetime
from typing import List, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from subscribe.signal_registry import list_channels, SignalChannel
from subscribe.data_fetcher import fetch_all_daily_data, merge_latest_daily_data
from subscribe.fundamentals import fetch_fundamentals_batch
from subscribe.report_generator import generate_report
from subscribe.mail_sender import send_report_email
from subscribe.llm_infer import infer_batch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")


def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_local_csv() -> pd.DataFrame:
    """加载最新本地K线数据"""
    csvs = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".csv")])
    if not csvs:
        raise FileNotFoundError(f"{DATA_DIR} 下未找到K线数据CSV文件")
    latest = csvs[-1]
    path = os.path.join(DATA_DIR, latest)
    print(f"加载本地数据: {path}", flush=True)
    df = pd.read_csv(path, low_memory=False)
    for col in ["open", "high", "low", "close", "preclose", "volume", "pctChg"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["tradestatus"].astype(str) == "1"]
    df = df[df["close"] > 0]
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    print(f"  {df.code.nunique()} stocks × {df.date.nunique()} dates", flush=True)
    return df


def compute_signals(df: pd.DataFrame, channels: List[SignalChannel]) -> Dict[str, pd.DataFrame]:
    """对每个频道计算信号值并筛选命中"""
    results = {}
    # 进度条
    from tqdm import tqdm
    for ch in tqdm(channels, desc="信号计算"):  
        try:
            signal_df = ch.compute(df)
            if signal_df.empty or "signal_value" not in signal_df.columns:
                results[ch.name] = pd.DataFrame()
                continue

            latest = signal_df.groupby("code").last().reset_index()
            sv = latest.set_index("code")["signal_value"].dropna()
            if sv.empty:
                results[ch.name] = pd.DataFrame()
                continue

            mask = ch.filter(sv)
            if isinstance(mask, list):
                hit_codes = set(mask)
            else:
                hit_codes = set(sv[mask].index) if hasattr(mask, "__getitem__") else set()

            if hit_codes:
                hit_vals = latest[latest["code"].isin(hit_codes)][["code", "signal_value"]].copy()
                hit_vals["channel"] = ch.name
                hit_vals["category"] = ch.category
                results[ch.name] = hit_vals
            else:
                results[ch.name] = pd.DataFrame()
        except Exception as e:
            print(f"  [WARN] {ch.name}: {e}", flush=True)
            results[ch.name] = pd.DataFrame()

    return results


def get_stock_info(df: pd.DataFrame, codes: List[str]) -> pd.DataFrame:
    """获取股票最新基本信息: code_name, close, pctChg"""
    latest = df.sort_values("date").groupby("code").last().reset_index()
    info_cols = ["code"]
    for col in ["code_name", "close", "pctChg"]:
        if col in latest.columns:
            info_cols.append(col)
    return latest[latest["code"].isin(codes)][info_cols].copy()


def main():
    parser = argparse.ArgumentParser(description="信号订阅 — 每日运行")
    parser.add_argument("--skip-fetch", action="store_true", help="跳过数据拉取")
    parser.add_argument("--skip-mail", action="store_true", help="跳过邮件发送")
    parser.add_argument("--skip-fundamentals", action="store_true", help="跳过基本面查询(加速测试)")
    parser.add_argument("--max-stocks", type=int, default=0, help="限制拉取股数(0=全部)")
    parser.add_argument("--full-fetch", action="store_true", help="强制全量拉取(不使用增量拼接)")
    parser.add_argument("--lookback", type=int, default=180, help="目标交易日数(仅全量模式)")
    parser.add_argument("--max-pct-change", type=float, default=5.0, help="涨跌幅上限(绝对值, %%), 命中后按此过滤")
    parser.add_argument("--max-targets", type=int, default=100, help="多信号筛选后目标股数上限")
    parser.add_argument("--min-trading-days", type=int, default=60, help="上市不满N个交易日的次新股剔除(0=不剔除)")
    parser.add_argument("--llm", action="store_true", help="启用大模型解读(需联网, 默认关闭)")
    args = parser.parse_args()

    ensure_dirs()
    channels = list_channels()
    print(f"信号订阅系统启动 | {len(channels)}个频道 | {datetime.now():%Y-%m-%d %H:%M}", flush=True)

    # Step 1: 数据获取 — 增量拼接模式 (历史179天 + 最新1天 = 180天)
    if not args.skip_fetch:
        print("\n[1/5] 数据获取...", flush=True)
        t0 = time.time()
        if args.full_fetch:
            fetch_all_daily_data(DATA_DIR, lookback_trading_days=args.lookback,
                                 stock_batch_size=args.max_stocks, delay=0.15)
        else:
            merge_latest_daily_data(DATA_DIR, target_trading_days=args.lookback,
                                    history_days=args.lookback - 1,
                                    stock_batch_size=args.max_stocks, delay=0.15)
        print(f"  完成 ({time.time()-t0:.0f}s)", flush=True)

    # Step 2: 信号计算
    print("\n[2/5] 信号计算...", flush=True)
    t0 = time.time()
    df = load_local_csv()
    signal_results = compute_signals(df, channels)

    all_hits = []
    for ch_name, hits_df in signal_results.items():
        if not hits_df.empty:
            all_hits.append(hits_df)
    hits = pd.concat(all_hits, ignore_index=True) if all_hits else pd.DataFrame()

    channel_summary = []
    for ch in channels:
        count = len(signal_results.get(ch.name, pd.DataFrame()))
        channel_summary.append({"channel": ch.name, "category": ch.category, "count": count})

    total_hits = len(hits)
    n_stocks_raw = hits['code'].nunique() if not hits.empty else 0
    print(f"  原始命中: {total_hits} 条 | {n_stocks_raw} 只标的 | {time.time()-t0:.0f}s", flush=True)
    for cs in sorted(channel_summary, key=lambda x: x["count"], reverse=True):
        if cs["count"] > 0:
            print(f"    {cs['channel']:15s} {cs['count']:4d}", flush=True)

    if hits.empty:
        print("  无命中信号，仍生成空报告。", flush=True)

    # Step 2.5: 涨跌幅过滤 — 剔除指数、保留股票+ETF，再按涨跌幅过滤
    if not hits.empty:
        index_prefix = re.compile(r'^(sh\.000|sz\.399)')
        index_codes = set(c for c in hits["code"].unique() if index_prefix.match(c))
        if index_codes:
            hits = hits[~hits["code"].isin(index_codes)].copy()
            print(f"  指数过滤: 剔除{len(index_codes)}只指数标的", flush=True)
    if hits.empty:
        print("  指数过滤后无命中信号，仍生成空报告。", flush=True)

    if not hits.empty and args.max_pct_change > 0:
        latest_info = df.sort_values("date").groupby("code").last()
        hit_codes = hits["code"].unique()
        if "pctChg" in latest_info.columns:
            pct_map = latest_info.loc[latest_info.index.isin(hit_codes), "pctChg"]
            extreme = pct_map[abs(pct_map) > args.max_pct_change].index
            if len(extreme) > 0:
                hits = hits[~hits["code"].isin(extreme)].copy()
                print(f"  涨跌幅过滤: |pctChg|>{args.max_pct_change}% → 剔除{len(extreme)}只标的", flush=True)
    if hits.empty:
        print("  涨跌幅过滤后无命中信号，仍生成空报告。", flush=True)

    # Step 2.6: 次新股过滤 — 剔除上市不满 min_trading_days 个交易日的标的
    if not hits.empty and args.min_trading_days > 0:
        code_trading_days = df.groupby("code")["date"].nunique()
        hit_codes = hits["code"].unique()
        short_codes = [c for c in hit_codes
                       if code_trading_days.get(c, 0) < args.min_trading_days]
        if short_codes:
            hits = hits[~hits["code"].isin(short_codes)].copy()
            print(f"  次新股过滤: 上市<{args.min_trading_days}交易日 → 剔除{len(short_codes)}只标的", flush=True)
    if hits.empty:
        print("  次新股过滤后无命中信号，仍生成空报告。", flush=True)

    # Step 2.7: 多信号组合筛选 — 只保留同时被 ≥N 个频道命中的股票
    min_signals = 1
    filtered_codes = set()
    if not hits.empty:
        code_signal_count = hits.groupby("code")["channel"].nunique()
        for n in range(2, 15):
            codes_n = set(code_signal_count[code_signal_count >= n].index)
            if len(codes_n) <= args.max_targets:
                min_signals = n
                filtered_codes = codes_n
                break
        else:
            min_signals = 14
            filtered_codes = set(code_signal_count[code_signal_count >= 14].index)

        if min_signals > 1:
            pre_count = hits['code'].nunique()
            hits = hits[hits["code"].isin(filtered_codes)].copy()
            post_count = hits['code'].nunique()
            print(f"  多信号筛选: ≥{min_signals}个信号 → {post_count}只标的 (从{pre_count}过滤, 阈值≤{args.max_targets})", flush=True)

    if hits.empty:
        print("  筛选后无命中信号，仍生成空报告。", flush=True)

    # Step 3: 获取股票基本信息
    if not hits.empty:
        hit_codes = hits["code"].unique().tolist()
        info = get_stock_info(df, hit_codes)
        hits = hits.merge(info, on="code", how="left")

    # Step 4: 基本面
    fundamentals = pd.DataFrame()
    if not args.skip_fundamentals and not hits.empty:
        print("\n[3/5] 基本面查询...", flush=True)
        t0 = time.time()
        hit_codes = hits["code"].unique().tolist()
        fundamentals = fetch_fundamentals_batch(hit_codes, delay=0.12)
        print(f"  完成 ({time.time()-t0:.0f}s)", flush=True)

    # Step 4.5: 大模型解读 (仅在 --llm 启用且存在命中标的时运行)
    llm_conclusions = None
    if args.llm and not hits.empty:
        print("\n[LLM] 大模型解读...", flush=True)
        try:
            llm_conclusions = infer_batch(hits, fundamentals)
            if llm_conclusions:
                hits["llm_conclusion"] = hits["code"].map(llm_conclusions)
                print(f"  已拼接 {len(llm_conclusions)} 条解读结果到数据表", flush=True)
            else:
                print("  大模型未返回有效结果，跳过", flush=True)
        except Exception as e:
            print(f"  [LLM] 错误: {e}, 跳过解读步骤", flush=True)

    # Step 5: 生成报告
    print("\n[4/5] 生成HTML日报...", flush=True)
    report_path = generate_report(hits, fundamentals, OUTPUT_DIR, channel_summary, min_signals,
                                  llm_conclusions=llm_conclusions)
    print(f"  报告: {report_path}", flush=True)

    # Step 6: 邮件发送
    if not args.skip_mail:
        print("\n[5/5] 邮件发送...", flush=True)
        send_report_email(report_path)
    else:
        print("\n[5/5] 跳过邮件发送", flush=True)

    print(f"\n{'='*60}")
    print(f"信号订阅完成 | {total_hits}条命中 | 报告: {report_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
