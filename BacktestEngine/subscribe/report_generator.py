"""HTML 信号日报生成器

整理信号类型 + 股票基本信息 + 信号值 + 近4季度利润曲线 + 盈利判断
输出: subscribe/output/yyyyMMdd_信号日报.html
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd


_CSS = """
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
  * { box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Microsoft YaHei', sans-serif; margin: 0; padding: 12px; background: #f5f6fa; color: #2c3e50; }
  .header { background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%); color: white; padding: 20px 16px; border-radius: 10px; margin-bottom: 16px; }
  .header h1 { margin: 0 0 4px 0; font-size: 20px; }
  .header .date { opacity: 0.85; font-size: 12px; }
  .header .filter-info { margin-top: 6px; font-size: 12px; opacity: 0.9; }
  .summary-bar { display: flex; gap: 8px; margin-bottom: 20px; flex-wrap: wrap; }
  .summary-card { flex: 1 1 80px; min-width: 70px; background: white; padding: 12px 8px; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,0.06); text-align: center; }
  .summary-card .value { font-size: 22px; font-weight: bold; color: #2c3e50; }
  .summary-card .label { font-size: 11px; color: #7f8c8d; margin-top: 2px; }
  .section-title { font-size: 16px; font-weight: bold; margin: 20px 0 10px 0; border-left: 3px solid #3498db; padding-left: 10px; }
  .table-wrap { overflow-x: auto; -webkit-overflow-scrolling: touch; background: white; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,0.06); margin-bottom: 16px; }
  table { width: 100%; border-collapse: collapse; min-width: 700px; }
  th { background: #ecf0f1; padding: 10px 6px; font-size: 12px; text-align: left; font-weight: 600; white-space: nowrap; }
  td { padding: 8px 6px; font-size: 12px; border-bottom: 1px solid #f0f0f0; white-space: nowrap; }
  td.wrap { white-space: normal; max-width: 120px; }
  tr:hover { background: #f8f9fa; }
  .trend-up { color: #27ae60; font-weight: bold; }
  .trend-down { color: #e74c3c; font-weight: bold; }
  .trend-mixed { color: #f39c12; font-weight: bold; }
  .trend-na { color: #95a5a6; }
  .signal-tag { display: inline-block; padding: 2px 6px; border-radius: 3px; font-size: 10px; font-weight: 600; white-space: nowrap; }
  .tag-MACD { background: #eaf2f8; color: #2980b9; }
  .tag-动量 { background: #fef9e7; color: #d4ac0d; }
  .tag-反转 { background: #e8f8f5; color: #1abc9c; }
  .tag-量价 { background: #f5eef8; color: #8e44ad; }
  .tag-突破 { background: #fdedec; color: #c0392b; }
  .tag-趋势 { background: #ebf5fb; color: #2471a3; }
  .chart-bar { display: flex; align-items: flex-end; gap: 1px; height: 30px; }
  .chart-bar div { background: #3498db; border-radius: 1px 1px 0 0; min-width: 3px; }
  .footer { text-align: center; margin-top: 24px; padding: 12px; color: #95a5a6; font-size: 11px; }
  .code-display { font-family: 'SF Mono', 'Consolas', monospace; font-size: 11px; }
  .notes { font-size: 12px; color: #7f8c8d; line-height: 1.8; background: white; padding: 12px; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }

  @media screen and (max-width: 480px) {
    body { padding: 8px; }
    .header { padding: 14px 12px; border-radius: 8px; }
    .header h1 { font-size: 18px; }
    .summary-card { padding: 10px 6px; }
    .summary-card .value { font-size: 18px; }
    .section-title { font-size: 15px; }
    table { min-width: 600px; font-size: 11px; }
    th, td { padding: 6px 4px; font-size: 11px; }
  }
</style>
"""

_CATEGORY_TAG = {
    "MACD": "tag-MACD", "动量": "tag-动量", "反转": "tag-反转",
    "量价": "tag-量价", "突破": "tag-突破", "趋势": "tag-趋势",
}


def _trend_css(trend: str) -> str:
    m = {"收益向上": "trend-up", "收益劣化": "trend-down", "收益波动": "trend-mixed"}
    return m.get(trend, "trend-na")


def _mini_bar(values_str: str) -> str:
    if not values_str:
        return ""
    try:
        vals = [float(v) for v in values_str.split(",")]
    except ValueError:
        return ""
    if not vals or max(vals) == 0:
        return ""
    bars = "".join(
        f'<div style="height:{max(4, int(v / max(abs(max(vals)), 1e-8) * 40))}px"></div>'
        for v in vals
    )
    return f'<div class="chart-bar">{bars}</div>'


def generate_report(
    hits: pd.DataFrame,
    fundamentals: pd.DataFrame,
    output_dir: str,
    channel_summary: List[dict],
    min_signals: int = 1,
) -> str:
    """生成 HTML 信号日报

    Args:
        hits: 信号命中记录 [code, code_name, channel, signal_value, category, close, pctChg, ...]
        fundamentals: 基本面 [code, trend, np_margins, eps_ttms, detail]
        output_dir: 输出目录
        channel_summary: 每个频道的命中数 [{channel, category, count}]
        min_signals: 多信号筛选阈值 (0或1表示未启用)
    """
    today_str = datetime.now().strftime("%Y%m%d")
    today_display = datetime.now().strftime("%Y-%m-%d %H:%M")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{today_str}_信号日报.html")

    fundamental_map = {}
    if not fundamentals.empty:
        for _, row in fundamentals.iterrows():
            fundamental_map[row["code"]] = row

    # ── 汇总卡片 ──
    total_hits = len(hits)
    n_codes = hits["code"].nunique() if not hits.empty else 0
    n_channels = len(hits["channel"].unique()) if not hits.empty else 0

    filter_info_html = ""
    if min_signals > 1:
        filter_info_html = f'<div class="filter-info">筛选: 同时被 ≥{min_signals} 个频道命中 | 标的数 ≤100</div>'

    signal_names = "MACD金叉/柱转正/零轴上穿/金叉+柱确认, RSI超卖, CCI超卖, 价格动量, 布林带下轨, MACD收敛, 放量突破, 放量阳线, 60日新高, 均线金叉, DIF领先"

    # ── 表行: 按 code 聚合, 同股多信号合并到一行 ──
    table_rows = ""
    if not hits.empty:
        grouped = []
        for code, grp in hits.groupby("code"):
            row0 = grp.iloc[0]
            channels = grp["channel"].tolist()
            categories = grp["category"].tolist()
            sig_vals = grp["signal_value"].tolist()

            name = row0.get("code_name", "—")
            close = row0.get("close", np.nan)
            pct_chg = row0.get("pctChg", np.nan)

            fund = fundamental_map.get(code, {})
            trend = fund.get("trend", "未评估")
            np_margins = fund.get("np_margins", "")
            eps_ttms = fund.get("eps_ttms", "")
            detail = fund.get("detail", "")

            close_display = f"{close:.2f}" if not np.isnan(close) else "—"
            pct_display = f'{pct_chg:+.2f}%' if not np.isnan(pct_chg) else "—"

            signal_tags_html = "<br>".join(
                f'<span class="signal-tag {_CATEGORY_TAG.get(categories[i], "")}">{ch}</span>'
                for i, ch in enumerate(channels)
            )
            sig_vals_html = "<br>".join(
                f"{v:.4f}" if not np.isnan(v) else "—"
                for v in sig_vals
            )

            trend_cls = _trend_css(trend)
            margin_bar = _mini_bar(np_margins)

            table_rows += f"""
            <tr>
                <td><span class="code-display">{code}</span></td>
                <td>{name}</td>
                <td class="wrap">{signal_tags_html}</td>
                <td class="wrap" style="text-align:right">{sig_vals_html}</td>
                <td style="text-align:right">{close_display}</td>
                <td style="text-align:right">{pct_display}</td>
                <td style="text-align:right">{np_margins}</td>
                <td style="text-align:right">{eps_ttms}</td>
                <td>{margin_bar}</td>
                <td class="{trend_cls}">{trend}</td>
                <td class="wrap">{detail}</td>
            </tr>"""

    # ── 频道汇总表 ──
    ch_rows = ""
    for cs in sorted(channel_summary, key=lambda x: x["count"], reverse=True):
        tag_cls = _CATEGORY_TAG.get(cs.get("category", ""), "")
        ch_rows += f"""
            <tr>
                <td><span class="signal-tag {tag_cls}">{cs['channel']}</span></td>
                <td>{cs['category']}</td>
                <td style="font-weight:bold;text-align:center">{cs['count']}</td>
            </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><title>{today_str} 信号日报</title>
{_CSS}
</head>
<body>

<div class="header">
    <h1>A股信号日报</h1>
    <div class="date">{today_display} | 信号订阅系统 v2.0</div>
    {filter_info_html}
</div>

<div class="summary-bar">
    <div class="summary-card"><div class="value">{total_hits}</div><div class="label">信号命中</div></div>
    <div class="summary-card"><div class="value">{n_codes}</div><div class="label">入选标的</div></div>
    <div class="summary-card"><div class="value">{n_channels}</div><div class="label">触发频道</div></div>
    <div class="summary-card"><div class="value">14</div><div class="label">监控频道</div></div>
</div>

<div class="section-title">频道命中汇总</div>
<div class="table-wrap">
<table>
    <thead><tr><th>频道</th><th>分类</th><th style="text-align:center">命中数</th></tr></thead>
    <tbody>{ch_rows}</tbody>
</table>
</div>

<div class="section-title">信号明细 (共 {total_hits} 条)</div>
<div class="table-wrap">
<table>
    <thead><tr>
        <th>代码</th><th>名称</th><th>信号</th>
        <th style="text-align:right">信号值</th><th style="text-align:right">收盘价</th><th style="text-align:right">涨跌幅</th>
        <th style="text-align:right">净利率(Q1-Q4)</th><th style="text-align:right">EPS(Q1-Q4)</th>
        <th>利润趋势</th><th>盈利判断</th><th>详情</th>
    </tr></thead>
    <tbody>{table_rows}</tbody>
</table>
</div>

<div class="section-title">说明</div>
<div class="notes">
    <strong>信号来源</strong>: {signal_names}<br>
    <strong>筛选规则</strong>: 仅保留同时被 ≥{min_signals} 个频道命中的标的<br>
    <strong>基本面</strong>: 基于 baostock seasonProfit 近4个季度 npMargin(净利率) + epsTTM(每股收益TTM) 趋势判定<br>
    <strong>趋势判据</strong>: 连续≥67%季度向上 → 收益向上; 连续≥67%季度向下 → 收益劣化; 其余 → 收益波动<br>
    <strong>免责</strong>: 本报告仅供研究参考，不构成投资建议。
</div>

<div class="footer">Auto-generated by Signal Subscribe System · {today_display}</div>
</body></html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    return output_path
