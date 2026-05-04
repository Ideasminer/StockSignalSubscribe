from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Literal, Optional

import numpy as np
import pandas as pd


class BaseSignal(ABC):
    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def get_name(self) -> str:
        return self.__class__.__name__


def _compute_macd_components(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    macd_hist = 2 * (dif - dea)
    return dif, dea, macd_hist


def _compute_macd_signal(df: pd.DataFrame, fast: int, slow: int, signal: int) -> pd.DataFrame:
    if "close" not in df.columns:
        raise ValueError("DataFrame must contain 'close' column")
    df = df.copy()
    df = df.sort_values(["code", "date"])
    codes = df["code"].unique()
    if len(codes) == 0:
        df["dif"] = np.nan
        df["dea"] = np.nan
        df["macd_hist"] = np.nan
        return df
    all_results = []
    for code in codes:
        mask = df["code"] == code
        idx = df.index[mask]
        close = df.loc[idx, "close"]
        dif, dea, hist = _compute_macd_components(close, fast, slow, signal)
        tmp = pd.DataFrame({"dif": dif, "dea": dea, "macd_hist": hist}, index=idx)
        all_results.append(tmp)
    result = pd.concat(all_results).sort_index()
    df["dif"] = result["dif"]
    df["dea"] = result["dea"]
    df["macd_hist"] = result["macd_hist"]
    return df


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _standardize(series: pd.Series) -> pd.Series:
    mean = series.mean()
    std = series.std()
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=series.index)
    return (series - mean) / std


class MACDGoldenCrossSignal(BaseSignal):
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = _compute_macd_signal(df, self.fast, self.slow, self.signal)
        prev_dif = df.groupby("code")["dif"].shift(1)
        prev_dea = df.groupby("code")["dea"].shift(1)
        df["signal_value"] = ((prev_dif <= prev_dea) & (df["dif"] > df["dea"])).astype(int)
        return df


class MACDDeathCrossSignal(BaseSignal):
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = _compute_macd_signal(df, self.fast, self.slow, self.signal)
        prev_dif = df.groupby("code")["dif"].shift(1)
        prev_dea = df.groupby("code")["dea"].shift(1)
        df["signal_value"] = ((prev_dif >= prev_dea) & (df["dif"] < df["dea"])).astype(int)
        return df


class MACDValueSignal(BaseSignal):
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = _compute_macd_signal(df, self.fast, self.slow, self.signal)
        # [原始代码 — 前瞻性偏差]: dif_std = df.groupby("code")["dif"].transform(lambda x: x.std())
        # 问题: 使用全样本std()标准化，T时刻的信号值包含了T+1、T+2等未来数据
        # 影响: 导致IC分析结果虚高、回测信号被未来信息污染
        # 修复: 改用expanding().std()，确保T时刻仅使用[0..T]的数据
        dif_std = df.groupby("code")["dif"].transform(lambda x: x.expanding(min_periods=20).std())
        df["dif_std"] = dif_std
        df["signal_value"] = np.where(
            dif_std > 0, df["dif"] / dif_std, 0.0
        )
        return df


class MACDHistSignal(BaseSignal):
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = _compute_macd_signal(df, self.fast, self.slow, self.signal)
        # [原始代码 — 前瞻性偏差]: hist_std = df.groupby("code")["macd_hist"].transform(lambda x: x.std())
        # 问题: 使用全样本std()标准化，T时刻的信号值包含了T+1、T+2等未来数据
        # 影响: 导致IC分析结果虚高、回测信号被未来信息污染
        # 修复: 改用expanding().std()，确保T时刻仅使用[0..T]的数据
        hist_std = df.groupby("code")["macd_hist"].transform(lambda x: x.expanding(min_periods=20).std())
        df["hist_std"] = hist_std
        df["signal_value"] = np.where(
            hist_std > 0, df["macd_hist"] / hist_std, 0.0
        )
        return df


class MACDHistChangeSignal(BaseSignal):
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = _compute_macd_signal(df, self.fast, self.slow, self.signal)
        hist_change = df.groupby("code")["macd_hist"].diff()
        # [原始代码 — 前瞻性偏差]: change_std = df.groupby("code")["macd_hist"].transform(lambda x: x.std())
        # 问题: 使用全样本std()标准化，T时刻的信号值包含了T+1、T+2等未来数据
        # 影响: 导致IC分析结果虚高、回测信号被未来信息污染
        # 修复: 改用expanding().std()，确保T时刻仅使用[0..T]的数据
        change_std = df.groupby("code")["macd_hist"].transform(lambda x: x.expanding(min_periods=20).std())
        df["change_std"] = change_std
        df["signal_value"] = np.where(
            change_std > 0, hist_change / change_std, 0.0
        )
        return df


class MACDHistCrossSignal(BaseSignal):
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = _compute_macd_signal(df, self.fast, self.slow, self.signal)
        prev_hist = df.groupby("code")["macd_hist"].shift(1)
        df["signal_value"] = (
            (prev_hist <= 0) & (df["macd_hist"] > 0)
        ).astype(int)
        return df


class MACDValueCrossSignal(BaseSignal):
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = _compute_macd_signal(df, self.fast, self.slow, self.signal)
        prev_dif = df.groupby("code")["dif"].shift(1)
        df["signal_value"] = (
            (prev_dif <= 0) & (df["dif"] > 0)
        ).astype(int)
        return df


class MACDSignalCrossSignal(BaseSignal):
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = _compute_macd_signal(df, self.fast, self.slow, self.signal)
        prev_dea = df.groupby("code")["dea"].shift(1)
        df["signal_value"] = (
            (prev_dea <= 0) & (df["dea"] > 0)
        ).astype(int)
        return df


class MACDConvergenceSignal(BaseSignal):
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = _compute_macd_signal(df, self.fast, self.slow, self.signal)
        spread = (df["dif"] - df["dea"]).abs()
        # [原始代码 — 前瞻性偏差]: spread_std = df.groupby("code")["macd_hist"].transform(lambda x: x.std())
        # 问题: 使用全样本std()标准化，T时刻的信号值包含了T+1、T+2等未来数据
        # 影响: 导致IC分析结果虚高、回测信号被未来信息污染
        # 修复: 改用expanding().std()，确保T时刻仅使用[0..T]的数据
        spread_std = df.groupby("code")["macd_hist"].transform(lambda x: x.expanding(min_periods=20).std())
        df["spread_std"] = spread_std
        df["signal_value"] = -spread / spread_std.replace(0, np.nan)
        df["signal_value"] = df["signal_value"].fillna(0.0)
        return df


class MACDGoldenCrossWithHistConfirmSignal(BaseSignal):
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = _compute_macd_signal(df, self.fast, self.slow, self.signal)
        prev_dif = df.groupby("code")["dif"].shift(1)
        prev_dea = df.groupby("code")["dea"].shift(1)
        golden_cross = (prev_dif <= prev_dea) & (df["dif"] > df["dea"])
        df["signal_value"] = (golden_cross & (df["macd_hist"] > 0)).astype(int)
        return df


class CrossMovingAverageSignal(BaseSignal):
    def __init__(
        self,
        short_ma: int = 5,
        long_ma: int = 10,
        cross_direction: Literal["golden", "death"] = "golden",
    ):
        self.short_ma = short_ma
        self.long_ma = long_ma
        self.cross_direction = cross_direction

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.sort_values(["code", "date"])
        codes = df["code"].unique()
        all_results = []
        for code in codes:
            mask = df["code"] == code
            idx = df.index[mask]
            close = df.loc[idx, "close"]
            ma_short = _ema(close, self.short_ma)
            ma_long = _ema(close, self.long_ma)
            prev_short = ma_short.shift(1)
            prev_long = ma_long.shift(1)
            if self.cross_direction == "golden":
                signal = ((prev_short <= prev_long) & (ma_short > ma_long)).astype(int)
            else:
                signal = ((prev_short >= prev_long) & (ma_short < ma_long)).astype(int)
            all_results.append(pd.DataFrame({"signal_value": signal}, index=idx))
        result = pd.concat(all_results).sort_index()
        df["signal_value"] = result["signal_value"]
        return df


class CompositeSignal(BaseSignal):
    def __init__(self, signals: List[BaseSignal], logic: Literal["AND", "OR"] = "AND"):
        self.signals = signals
        self.logic = logic

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        combined = None
        for sig in self.signals:
            result = sig.compute(df)
            col_name = f"signal_{sig.get_name()}"
            df[col_name] = result["signal_value"]
            if combined is None:
                combined = result["signal_value"]
            elif self.logic == "AND":
                combined = combined & result["signal_value"]
            else:
                combined = combined | result["signal_value"]
        df["signal_value"] = combined.astype(int)
        return df

    def get_name(self) -> str:
        names = "_".join(s.get_name() for s in self.signals)
        return f"{self.logic}({names})"


# ============================================================
# 持仓收益率阈值信号
# ============================================================


class HoldingReturnThresholdSignal(BaseSignal):
    """持仓收益率阈值卖出信号

    计算每个标的的"持仓收益率"作为信号值，配合 ThresholdCondition 触发卖出。

    收益率计算优先级:
    1. 若传入 cost_price_map → 直接使用 cost_price 计算真实持仓收益率
    2. 若 df 中有 'cost_price' 列 → 使用该列
    3. 否则 → 使用 period 日回溯收益率作为代理信号
       signal(t) = close(t) / close(t - period) - 1
       正值=近期有浮盈(潜在止盈卖出)

    数学表达:
        收益率 = close / cost_price - 1   (模式1/2)
        收益率 = close / close_lag(N) - 1  (模式3, 代理信号)

    使用示例:
        signal = HoldingReturnThresholdSignal(period=20)
        # 返回连续收益率值，供 ThresholdCondition(threshold=0.1, "gt") 筛选
    """

    def __init__(self, cost_price_map: dict = None, period: int = 20):
        """Args:
            cost_price_map: {code: cost_price} 平均持仓成本, 若为空则使用 代理信号
            period: 回溯期数(仅cost_price_map为空时生效), 默认20日
        """
        self.cost_price_map = cost_price_map or {}
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.sort_values(["code", "date"])
        df["signal_value"] = np.nan

        if self.cost_price_map:
            for code, cost_price in self.cost_price_map.items():
                mask = df["code"] == code
                if cost_price > 0:
                    df.loc[mask, "signal_value"] = (
                        df.loc[mask, "close"] / cost_price - 1.0
                    )
        elif "cost_price" in df.columns:
            df["signal_value"] = np.where(
                df["cost_price"] > 0,
                df["close"] / df["cost_price"] - 1.0,
                np.nan,
            )
        else:
            # [FIX] 无 cost_price 数据时, 使用 period 日回溯收益率作为代理
            codes = df["code"].unique()
            all_results = []
            for code in codes:
                mask = df["code"] == code
                idx = df.index[mask]
                close = df.loc[idx, "close"].astype(float)
                lag_close = close.shift(self.period)
                returns = close / lag_close.replace(0, np.nan) - 1.0
                returns = returns.fillna(0.0)
                all_results.append(pd.DataFrame({"signal_value": returns}, index=idx))
            result = pd.concat(all_results).sort_index()
            df["signal_value"] = result["signal_value"].values

        return df


# ============================================================
# RSI 相对强弱指数信号
# ============================================================


class RSISignal(BaseSignal):
    """RSI 相对强弱指数信号

    业务逻辑：
    - RSI = 100 - 100 / (1 + RS)
    - RS = 周期内平均上涨幅度 / 周期内平均下跌幅度
    - RSI > 70 表示超买，RSI < 30 表示超卖
    - 信号值 = (RSI - 50) / 50，标准化到 [-1, 1] 区间
    - 正值表示偏强（潜在买入方向），负值表示偏弱（潜在卖出方向）

    参考：Wilder, J. Welles. "New Concepts in Technical Trading Systems." (1978)
    """

    def __init__(self, period: int = 14):
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.sort_values(["code", "date"])
        codes = df["code"].unique()
        all_results = []
        for code in codes:
            mask = df["code"] == code
            idx = df.index[mask]
            close = df.loc[idx, "close"].astype(float)
            delta = close.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.ewm(span=self.period, adjust=False).mean()
            avg_loss = loss.ewm(span=self.period, adjust=False).mean()
            rs = avg_gain / avg_loss.replace(0, 1e-9)
            rsi = 100.0 - 100.0 / (1.0 + rs)
            signal = (rsi - 50.0) / 50.0
            all_results.append(pd.DataFrame({"signal_value": signal}, index=idx))
        result = pd.concat(all_results).sort_index()
        df["signal_value"] = result["signal_value"]
        return df


# ============================================================
# CCI 商品通道指数信号
# ============================================================


class CCISignal(BaseSignal):
    """CCI 商品通道指数信号

    业务逻辑：
    - TP = (High + Low + Close) / 3 (典型价格)
    - CCI = (TP - SMA(TP, N)) / (0.015 * MeanDeviation)
    - CCI > 100 表示超买，CCI < -100 表示超卖
    - 信号值 = CCI / 200，标准化到约 [-2, 2] 区间

    参考：Lambert, Donald. "Commodity Channel Index." (1980)
    """

    def __init__(self, period: int = 20):
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.sort_values(["code", "date"])
        codes = df["code"].unique()
        all_results = []
        for code in codes:
            mask = df["code"] == code
            idx = df.index[mask]
            high = df.loc[idx, "high"].astype(float)
            low = df.loc[idx, "low"].astype(float)
            close = df.loc[idx, "close"].astype(float)
            tp = (high + low + close) / 3.0
            sma_tp = tp.rolling(window=self.period, min_periods=1).mean()
            mad = tp.rolling(window=self.period, min_periods=1).apply(
                lambda x: np.abs(x - x.mean()).mean()
            )
            cci = np.where(mad > 0, (tp - sma_tp) / (0.015 * mad), 0.0)
            signal = np.clip(cci / 200.0, -3.0, 3.0)
            all_results.append(pd.DataFrame({"signal_value": signal}, index=idx))
        result = pd.concat(all_results).sort_index()
        df["signal_value"] = result["signal_value"].values
        return df


# ============================================================
# 成交量突破信号
# ============================================================


class VolumeBreakoutSignal(BaseSignal):
    """成交量突破信号

    业务逻辑：
    - 计算过去N日成交量的移动平均
    - 当今日成交量超过均值的M倍时触发信号
    - 信号值 = volume / (avg_volume * multiplier) - 1（连续值）
    - 正值表示放量突破，可用于辅助判断趋势强度

    参考：Shaleen, Kenneth. "Volume and Open Interest." (1991)
    """

    def __init__(self, period: int = 20, multiplier: float = 1.5):
        self.period = period
        self.multiplier = multiplier

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.sort_values(["code", "date"])
        if "volume" not in df.columns:
            df["volume"] = 1.0
        codes = df["code"].unique()
        all_results = []
        for code in codes:
            mask = df["code"] == code
            idx = df.index[mask]
            vol = df.loc[idx, "volume"].astype(float)
            avg_vol = vol.rolling(window=self.period, min_periods=1).mean()
            threshold = avg_vol * self.multiplier
            signal = (vol / threshold.replace(0, 1e-9) - 1.0)
            all_results.append(pd.DataFrame({"signal_value": signal}, index=idx))
        result = pd.concat(all_results).sort_index()
        df["signal_value"] = result["signal_value"].values
        return df


# ============================================================
# 价格动量信号
# ============================================================


class PriceMomentumSignal(BaseSignal):
    """价格动量信号

    业务逻辑：
    - 计算过去N日的收益率（动量因子）
    - signal_value = (close(t) - close(t-N)) / close(t-N)
    - 正值表示上涨动量（买入方向），负值表示下跌动量（卖出方向）

    参考：Jegadeesh & Titman. "Returns to Buying Winners and Selling Losers." (1993)
    """

    def __init__(self, period: int = 20):
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.sort_values(["code", "date"])
        all_results = []
        for code in df["code"].unique():
            mask = df["code"] == code
            idx = df.index[mask]
            close = df.loc[idx, "close"].astype(float)
            momentum = close.pct_change(periods=self.period)
            all_results.append(pd.DataFrame({"signal_value": momentum}, index=idx))
        result = pd.concat(all_results).sort_index()
        df["signal_value"] = result["signal_value"].values
        return df


# ============================================================
# 布林带信号
# ============================================================


class BollingerBandSignal(BaseSignal):
    """布林带信号

    业务逻辑：
    - 中轨 = SMA(close, N)
    - 上轨 = 中轨 + K * std(close, N)
    - 下轨 = 中轨 - K * std(close, N)
    - 信号值 = (close - 中轨) / (K * std)（Z-score 位置）
    - > 1 表示突破上轨，< -1 表示突破下轨
    - 突破上轨后可能回调（卖出信号），突破下轨后可能反弹（买入信号）

    参考：Bollinger, John. "Bollinger on Bollinger Bands." (2001)
    """

    def __init__(self, period: int = 20, k: float = 2.0):
        self.period = period
        self.k = k

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.sort_values(["code", "date"])
        all_results = []
        for code in df["code"].unique():
            mask = df["code"] == code
            idx = df.index[mask]
            close = df.loc[idx, "close"].astype(float)
            sma = close.rolling(window=self.period, min_periods=1).mean()
            std = close.rolling(window=self.period, min_periods=1).std()
            band_width = self.k * std.replace(0, 1e-9)
            z_score = (close - sma) / band_width.replace(0, 1e-9)
            z_score = z_score.fillna(0.0).clip(-5.0, 5.0)
            all_results.append(pd.DataFrame({"signal_value": z_score}, index=idx))
        result = pd.concat(all_results).sort_index()
        df["signal_value"] = result["signal_value"].values
        return df


# ============================================================
# ATR 平均真实波幅信号
# ============================================================


class ATRSignal(BaseSignal):
    """ATR 平均真实波幅信号

    业务逻辑：
    - TR = max(H-L, |H-昨日C|, |L-昨日C|)
    - ATR = EMA(TR, N)
    - 信号值 = (TR / ATR - 1)，正值表示波动加大
    - 高波动信号值可能预示趋势启动或结束

    参考：Wilder, J. Welles. "New Concepts in Technical Trading Systems." (1978)
    """

    def __init__(self, period: int = 14):
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.sort_values(["code", "date"])
        all_results = []
        for code in df["code"].unique():
            mask = df["code"] == code
            idx = df.index[mask]
            high = df.loc[idx, "high"].astype(float)
            low = df.loc[idx, "low"].astype(float)
            close = df.loc[idx, "close"].astype(float)
            prev_close = close.shift(1)
            tr1 = high - low
            tr2 = (high - prev_close).abs()
            tr3 = (low - prev_close).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.ewm(span=self.period, adjust=False).mean()
            signal = tr / atr.replace(0, 1e-9) - 1.0
            signal = signal.fillna(0.0).clip(-1.0, 10.0)
            all_results.append(pd.DataFrame({"signal_value": signal}, index=idx))
        result = pd.concat(all_results).sort_index()
        df["signal_value"] = result["signal_value"].values
        return df


# ============================================================
# 放量阳线信号
# ============================================================


class VolumeUpDaySignal(BaseSignal):
    """放量阳线信号

    业务逻辑：
    - 阳线: close > open
    - 放量: volume > avg_volume(N) * multiplier
    - 同时满足两者 → signal_value = 1，否则为 0
    - 用于捕捉放量上涨突破的时机

    参考：Elder, Alexander. "Trading for a Living." (1993)
    """

    def __init__(self, period: int = 20, multiplier: float = 1.2):
        self.period = period
        self.multiplier = multiplier

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.sort_values(["code", "date"])
        if "volume" not in df.columns:
            df["volume"] = 1.0
        if "open" not in df.columns:
            df["open"] = df["close"] * 0.99
        codes = df["code"].unique()
        all_results = []
        for code in codes:
            mask = df["code"] == code
            idx = df.index[mask]
            vol = df.loc[idx, "volume"].astype(float)
            opn = df.loc[idx, "open"].astype(float)
            close = df.loc[idx, "close"].astype(float)
            avg_vol = vol.rolling(window=self.period, min_periods=1).mean()
            up_day = close > opn
            vol_break = vol > (avg_vol * self.multiplier)
            signal = (up_day & vol_break).astype(int)
            all_results.append(pd.DataFrame({"signal_value": signal}, index=idx))
        result = pd.concat(all_results).sort_index()
        df["signal_value"] = result["signal_value"].values
        return df


# ============================================================
# 放量阴线信号
# ============================================================


class VolumeDownDaySignal(BaseSignal):
    """放量阴线信号

    业务逻辑：
    - 阴线: close < open
    - 放量: volume > avg_volume(N) * multiplier
    - 同时满足两者 → signal_value = 1，否则为 0
    - 用于捕捉放量下跌的卖出时机

    参考：Elder, Alexander. "Trading for a Living." (1993)
    """

    def __init__(self, period: int = 20, multiplier: float = 1.2):
        self.period = period
        self.multiplier = multiplier

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.sort_values(["code", "date"])
        if "volume" not in df.columns:
            df["volume"] = 1.0
        if "open" not in df.columns:
            df["open"] = df["close"] * 0.99
        codes = df["code"].unique()
        all_results = []
        for code in codes:
            mask = df["code"] == code
            idx = df.index[mask]
            vol = df.loc[idx, "volume"].astype(float)
            opn = df.loc[idx, "open"].astype(float)
            close = df.loc[idx, "close"].astype(float)
            avg_vol = vol.rolling(window=self.period, min_periods=1).mean()
            down_day = close < opn
            vol_break = vol > (avg_vol * self.multiplier)
            signal = (down_day & vol_break).astype(int)
            all_results.append(pd.DataFrame({"signal_value": signal}, index=idx))
        result = pd.concat(all_results).sort_index()
        df["signal_value"] = result["signal_value"].values
        return df


# ============================================================
# 新高新低信号
# ============================================================


class NewHighBreakoutSignal(BaseSignal):
    """N日新高突破信号

    业务逻辑：
    - 当日收盘价创过去N日新高 → signal_value = 1
    - 否则为 0
    - 用于捕捉趋势突破的买入时机

    参考：O'Neil, William. "How to Make Money in Stocks." (1988)
    """

    def __init__(self, period: int = 60):
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.sort_values(["code", "date"])
        all_results = []
        for code in df["code"].unique():
            mask = df["code"] == code
            idx = df.index[mask]
            close = df.loc[idx, "close"].astype(float)
            rolling_max = close.rolling(window=self.period, min_periods=1).max()
            prev_max = rolling_max.shift(1)
            signal = (close > prev_max).astype(int)
            all_results.append(pd.DataFrame({"signal_value": signal}, index=idx))
        result = pd.concat(all_results).sort_index()
        df["signal_value"] = result["signal_value"].values
        return df