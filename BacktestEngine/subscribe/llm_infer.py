"""大模型解读模块 — 基于股票信号+基本面信息，联网搜索后进行买入研判

用法:
  from subscribe.llm_infer import infer_batch
  conclusions = infer_batch(stocks_info, fundamentals, max_concurrency=3)

stocks_info: DataFrame, 需含列 code, code_name, channel, category, signal_value
fundamentals: DataFrame, 需含列 code, trend, np_margins, eps_ttms
返回: dict {code: conclusion_string}
"""
from __future__ import annotations

import json
import os
import ssl
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ─── 配置区 ──────────────────────────────────────────
LLM_BASE_URL = "" # 请填写您的大模型API地址
LLM_MODEL = "" # 请填写您的大模型模型名称
LLM_API_KEY = "" # 请填写您的大模型API密钥
LLM_MAX_CONCURRENCY = 3
LLM_REQUEST_TIMEOUT = 60
# ─────────────────────────────────────────────────────


def _build_stock_prompt(
    code: str,
    name: str,
    signals: str,
    trend: str,
    np_margins: str,
    eps_ttms: str,
) -> str:
    return f"""请以A股分析师身份，对以下标的做简短研判（≤200字），结构：估值、信号解读、财报核心信息、风险提示、总体结论。

标的: {code} {name}
触发信号: {signals}
利润趋势: {trend}
近4季净利润率: {np_margins}
近4季每股收益: {eps_ttms}

请联网搜索该公司最新财报、近7日重要新闻、机构评级，结合上述信号数据，给出买入建议。"""


SYSTEM_PROMPT = (
    "你是A股量化分析师。请开启联网搜索，结合实时信息与提供的信号数据做研判。"
    "每只标的输出≤200字，结构固定为5段，用 * 开头分隔，格式如下：\n"
    "* 估值: ...\n"
    "* 信号解读: ...\n"
    "* 财报核心信息: ...\n"
    "* 风险提示: ...\n"
    "* 总体结论: ...\n"
    "禁止使用markdown格式（**、#、-、>、`等），只允许纯文本 + * 分段。"
    "用中文输出。"
)


def _call_llm(prompt: str) -> Optional[str]:
    body = json.dumps({
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 600,
        "temperature": 0.5,
    }).encode("utf-8")

    url = f"{LLM_BASE_URL}/v1/chat/completions"
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {LLM_API_KEY}")

    try:
        ctx = ssl.create_default_context()
        with urllib.request.urlopen(req, timeout=LLM_REQUEST_TIMEOUT, context=ctx) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"  [LLM] API错误: {e}", flush=True)
        return None


def infer_single(
    code: str,
    name: str,
    signals: str,
    trend: str,
    np_margins: str,
    eps_ttms: str,
) -> tuple:
    prompt = _build_stock_prompt(code, name, signals, trend, np_margins, eps_ttms)
    result = _call_llm(prompt)
    return code, result


def infer_batch(
    hits: pd.DataFrame,
    fundamentals: pd.DataFrame,
    max_concurrency: int = LLM_MAX_CONCURRENCY,
) -> Dict[str, str]:
    if hits.empty:
        return {}

    fund_map = {}
    if not fundamentals.empty:
        for _, row in fundamentals.iterrows():
            fund_map[row["code"]] = {
                "trend": row.get("trend", "未知"),
                "np_margins": row.get("np_margins", ""),
                "eps_ttms": row.get("eps_ttms", ""),
            }

    tasks = []
    for code, grp in hits.groupby("code"):
        row0 = grp.iloc[0]
        name = row0.get("code_name", code)
        signal_list = []
        for _, r in grp.iterrows():
            sv = r.get("signal_value", np.nan)
            if not np.isnan(sv):
                signal_list.append(f"{r['channel']}({sv:.3f})")
            else:
                signal_list.append(str(r["channel"]))
        signals_str = "; ".join(signal_list)

        f = fund_map.get(code, {})
        trend = f.get("trend", "未评估")
        np_margins = f.get("np_margins", "")
        eps_ttms = f.get("eps_ttms", "")

        tasks.append((code, name, signals_str, trend, np_margins, eps_ttms))

    print(f"  大模型解读: {len(tasks)} 只标的, 并发={max_concurrency}", flush=True)
    t0 = time.time()

    results: Dict[str, str] = {}
    completed = 0

    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        futures = {
            executor.submit(infer_single, *task): task[0]
            for task in tasks
        }
        for future in as_completed(futures):
            completed += 1
            try:
                code, content = future.result()
                if content:
                    results[code] = content
            except Exception as e:
                code = futures[future]
                print(f"  [LLM] {code} 失败: {e}", flush=True)

            if completed % 5 == 0:
                print(f"    LLM进度: {completed}/{len(tasks)}", flush=True)

    elapsed = time.time() - t0
    print(f"  大模型解读完成: {len(results)}/{len(tasks)} 成功 ({elapsed:.0f}s)", flush=True)
    return results
