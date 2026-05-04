"""信号订阅 — 信号/条件注册中心

基于 BacktestEngine 的 signal.py 和 conditions.py，
将买入信号 + 阈值条件绑定为可订阅的"信号频道"。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Callable

import numpy as np
import pandas as pd

from backtest.strategies.signal import (
    BaseSignal,
    MACDGoldenCrossSignal, MACDHistCrossSignal, MACDValueCrossSignal,
    MACDSignalCrossSignal, MACDGoldenCrossWithHistConfirmSignal,
    MACDValueSignal, MACDHistSignal, MACDHistChangeSignal, MACDConvergenceSignal,
    RSISignal, CCISignal, VolumeBreakoutSignal, PriceMomentumSignal,
    BollingerBandSignal, VolumeUpDaySignal, NewHighBreakoutSignal,
    CrossMovingAverageSignal,
)
from backtest.pipeline.conditions import (
    SignalCondition, ThresholdCondition,
)


class SignalChannel:
    """信号频道：信号 + 阈值条件 + 元数据"""

    def __init__(
        self,
        name: str,
        signal: BaseSignal,
        condition: SignalCondition,
        category: str = "动量",
        description: str = "",
    ):
        self.name = name
        self.signal = signal
        self.condition = condition
        self.category = category
        self.description = description

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """在原始 DataFrame 上计算信号值"""
        return self.signal.compute(df)

    def filter(self, signal_values: pd.Series) -> pd.Series:
        """对信号值应用条件筛选"""
        return self.condition.filter(signal_values)


# ── 注册表 ──────────────────────────────────────────
SUBSCRIBED_CHANNELS: Dict[str, SignalChannel] = {}


def register(channel: SignalChannel):
    SUBSCRIBED_CHANNELS[channel.name] = channel


def list_channels() -> List[SignalChannel]:
    return list(SUBSCRIBED_CHANNELS.values())


def get_channel(name: str) -> Optional[SignalChannel]:
    return SUBSCRIBED_CHANNELS.get(name)


# ── 内置注册 14 个买入频道 ───────────────────────

def _default_registry():
    if SUBSCRIBED_CHANNELS:
        return

    _channels = [
        # === MACD 系列 (二元: 0/1, 自然稀疏) ===
        SignalChannel(
            "MACD金叉", MACDGoldenCrossSignal(12, 26, 9),
            ThresholdCondition(0.9, "gt"), "MACD",
            "DIF上穿DEA，短期动能增强"),
        SignalChannel(
            "MACD柱转正", MACDHistCrossSignal(12, 26, 9),
            ThresholdCondition(0.9, "gt"), "MACD",
            "MACD柱由负转正，动能从空转多"),
        SignalChannel(
            "MACD零轴上穿", MACDValueCrossSignal(12, 26, 9),
            ThresholdCondition(0.9, "gt"), "MACD",
            "DIF从负值穿越到正值，中期趋势转多"),
        SignalChannel(
            "MACD金叉+柱确认", MACDGoldenCrossWithHistConfirmSignal(12, 26, 9),
            ThresholdCondition(0.9, "gt"), "MACD",
            "金叉且柱为正，双重确认"),

        # === 动量/摆动 ===
        SignalChannel(
            "RSI超卖", RSISignal(14),
            ThresholdCondition(-0.2, "lt"), "动量",
            "signal=(RSI-50)/50; <-0.2 → RSI<40，超卖区间"),
        SignalChannel(
            "CCI超卖", CCISignal(20),
            ThresholdCondition(-0.4, "lt"), "动量",
            "signal=CCI/200; <-0.4 → CCI<-80，超卖区间"),
        SignalChannel(
            "价格动量", PriceMomentumSignal(20),
            ThresholdCondition(0.10, "gt"), "动量",
            "20日收益>10%，强动量效应"),

        # === 波动/反转 ===
        SignalChannel(
            "布林带下轨", BollingerBandSignal(20, 2.0),
            ThresholdCondition(-0.98, "lt"), "反转",
            "价格触及2σ布林带下轨(signal<-0.98 → 距均线<-1.96σ)"),
        SignalChannel(
            "MACD收敛", MACDConvergenceSignal(12, 26, 9),
            ThresholdCondition(-0.5, "gt"), "反转",
            "|DIF-DEA|<0.5σ，柱体极端收窄，趋势衰竭"),

        # === 量价 ===
        SignalChannel(
            "放量突破", VolumeBreakoutSignal(20, 1.5),
            ThresholdCondition(1.5, "gt"), "量价",
            "成交量≥20日均量的3.75倍，显著放量"),
        SignalChannel(
            "放量阳线", VolumeUpDaySignal(20, 1.2),
            ThresholdCondition(0.5, "gt"), "量价",
            "收盘>开盘且放量≥1.2倍均量"),

        # === 趋势/突破 ===
        SignalChannel(
            "60日新高", NewHighBreakoutSignal(60),
            ThresholdCondition(0.5, "gt"), "突破",
            "创60日收盘新高"),
        SignalChannel(
            "均线金叉", CrossMovingAverageSignal(5, 10, "golden"),
            ThresholdCondition(0.5, "gt"), "趋势",
            "5日EMA上穿10日EMA"),

        # === 连续值 ===
        SignalChannel(
            "DIF领先", MACDValueSignal(12, 26, 9),
            ThresholdCondition(3.0, "gt"), "MACD",
            "标准化DIF>3.0σ，极端动量领先"),
    ]

    for ch in _channels:
        register(ch)


_default_registry()
