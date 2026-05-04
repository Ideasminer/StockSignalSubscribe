# 股票/ETF 回测系统 — BacktestEngine

![Stars](https://img.shields.io/github/stars/Ideasminer/StockSignalSubscribe?style=flat&logo=github)
![Forks](https://img.shields.io/github/forks/Ideasminer/StockSignalSubscribe?style=flat&logo=github)
![Issues](https://img.shields.io/github/issues/Ideasminer/StockSignalSubscribe)
![Pull Requests](https://img.shields.io/github/issues-pr/Ideasminer/StockSignalSubscribe)
![License](https://img.shields.io/badge/license-MIT-blue)
![Last Commit](https://img.shields.io/github/last-commit/Ideasminer/StockSignalSubscribe)

## 目录

- [项目简介](#项目简介)
- [核心功能](#核心功能)
- [系统架构](#系统架构)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [使用指南](#使用指南)
- [订阅日报模块](#订阅日报模块)
- [测试](#测试)
- [依赖](#依赖)
- [许可证](#许可证)
- [安全提醒](#安全提醒)

## 项目简介

**BacktestEngine** 是一个灵活、可扩展的 Python 回测框架，专为 **A 股日线级别** 策略开发与验证而设计。系统支持股票与 ETF 的策略回测、绩效评估、信号评估与可视化分析，并提供基于实时数据的每日信号日报订阅功能。

### 核心设计理念

- **可插拔**：策略与引擎解耦，通过 `Action` 对象通信
- **向量化计算**：所有信号算法采用向量化实现，支持数千只股票的高效处理
- **模块化**：核心引擎、策略体系、指标计算、可视化、信号订阅各自独立

## 核心功能

### 回测引擎
- 逐日推进的事件驱动回测引擎
- 多标的持仓、资金管理与订单执行（支持滑点与手续费）
- 资金校验机制：最低交易金额、现金预留、持仓上限等多重保障
- 完整的交易记录与持仓历史追踪

### 策略体系
| 模块 | 说明 |
|------|------|
| 信号信号 (Signal) | 22 种技术指标信号：MACD 系列 (11 种)、RSI、CCI、布林带、成交量突破、价格动量、均线交叉、新高突破、ATR、量比、上涨家数等 |
| 买入策略 (Buy) | `AlwaysBuy`、`SignalBuy`（含资金预算管理） |
| 卖出策略 (Sell) | `AlwaysSell`、`StopLoss`、`SignalSell` |
| 选股策略 (Selector) | `AllStockSelector`、`TopNSelector` |
| 策略组合 (Composite) | 将多个子策略组合为完整策略 |

### 信号评估系统
- **IC 分析**：Pearson IC / Rank IC / IC_IR
- **多空组合绩效**：年化收益、夏普比率、最大回撤、胜率
- **分组收益分析**：分 5/10 组排序、换手率统计

### 指标体系（共 25+ 项）
| 类别 | 指标 |
|------|------|
| 收益类 | 累计收益率、年化收益率、超额收益、月度/年度胜率 |
| 风险类 | 最大回撤、最长回撤修复期、年化波动率、下行波动率、VaR/CVaR |
| 风险调整 | 夏普比率、索提诺比率、卡玛比率、收益回撤比、信息比率 |
| 交易运作 | 胜率/盈亏比、最大连胜/连亏、平均持仓时间、换手率 |

### 可视化图表
- 净值曲线 & 回撤曲线（策略 vs 基准叠加）
- 月度/年度收益热力图、日度收益分布直方图
- 滚动指标图（夏普、年化收益）
- 买卖信号标注图、仓位/杠杆变化图
- IC 分布直方图、分组收益对比柱状图、雷达图

### 订阅日报模块
数据拉取 → 信号计算 → 基本面分析 → HTML 日报生成 → 邮件推送的完整自动化流程。详见[订阅日报模块](#订阅日报模块)。

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                      回测引擎 (Engine)                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────┐  │
│  │ 数据供给  │  │ 策略调度  │  │ 订单执行  │  │ 绩效评估   │  │
│  │ DataFeed  │  │ Strategy  │  │ Broker   │  │ Metrics    │  │
│  └──────────┘  └──────────┘  └──────────┘  └────────────┘  │
│                      │                                        │
│                      ▼                                        │
│              ┌──────────────┐                                 │
│              │  Action 对象  │  (策略输出，引擎消费)           │
│              └──────────────┘                                 │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                      信号评估系统                            │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ 信号生成  │  │ IC 分析       │  │ 多空组合绩效         │  │
│  │ Signal   │  │ Pearson/Rank  │  │ Long-Short Portfolio │  │
│  └──────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                      订阅日报模块                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────┐  │
│  │ 数据获取  │  │ 信号计算  │  │ 基本面   │  │ 邮件推送   │  │
│  │ Baostock │  │ Registry │  │ 盈利分析  │  │ SMTP      │  │
│  └──────────┘  └──────────┘  └──────────┘  └────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

**数据流**：
```
历史数据 (CSV/Mock) → DataFeed → Engine (逐Bar推进)
                                    ↓
                            调用 Strategy.next()
                                    ↓
                            策略返回 Action[]
                                    ↓
                            Engine 执行交易 & 更新持仓
                                    ↓
                            BacktestResult (净值/交易/持仓)
```

## 项目结构

```
BacktestEngine/
├── backtest/                          # 核心回测包
│   ├── core/                          # 回测引擎核心
│   │   ├── action.py                  # Action/Trade/Position 数据类
│   │   ├── data_feed.py               # 多标的数据供给
│   │   └── engine.py                  # 回测引擎（逐 Bar 执行）
│   ├── strategies/                    # 策略模块
│   │   ├── base.py                    # BaseStrategy / CompositeStrategy
│   │   ├── buy.py                     # 买入策略（BaseBuyStrategy 等）
│   │   ├── sell.py                    # 卖出策略（StopLoss / SignalSell 等）
│   │   ├── selector.py                # 选股策略（TopNSelector 等）
│   │   └── signal.py                  # 信号模块（22 种技术指标信号）
│   ├── pipeline/                      # 可复用流水线模块
│   │   ├── conditions.py              # SignalCondition 条件体系
│   │   └── pipeline.py                # SignalPipeline 流水线
│   ├── metrics/                       # 指标模块
│   │   ├── returns.py                 # 收益类指标（5 项）
│   │   ├── risk.py                    # 风险类指标（6 项）
│   │   ├── risk_adjusted.py           # 风险调整收益（5 项）
│   │   ├── trading.py                 # 交易运作指标（9 项）
│   │   └── signal_evaluation.py       # 信号评估指标（8 项）
│   ├── visualization/                 # 可视化模块
│   │   ├── charts.py                  # 净值曲线、回撤曲线
│   │   ├── distribution.py            # 月度/年度热力图、日度分布
│   │   ├── attribution.py             # 滚动指标、年度对比
│   │   ├── trading_charts.py          # 买卖信号、仓位变化
│   │   └── signal_analysis.py         # IC 直方图、分组收益、雷达图
│   └── utils/                         # 工具函数
│       ├── helpers.py                 # 中文字体、Mock 数据生成
│       └── data_loader.py             # 多 CSV 合并工具
├── subscribe/                         # 订阅日报模块
│   ├── data_fetcher.py                # 数据获取（Baostock 全量 A 股 K 线）
│   ├── signal_registry.py             # 信号/条件注册中心（14 个频道）
│   ├── fundamentals.py                # 基本面数据获取（季度盈利）
│   ├── report_generator.py            # HTML 信号日报生成
│   ├── mail_sender.py                 # SMTP 邮件发送
│   ├── run_daily.py                   # 一键每日运行脚本
│   ├── tests/test_subscribe.py        # 订阅模块测试
│   ├── data/                          # 本地 K 线数据缓存（运行自动创建）
│   └── output/                        # HTML 日报输出（运行自动创建）
├── tests/                             # 单元测试
│   ├── test_core.py                   # 核心模块测试
│   ├── test_strategies.py             # 策略测试
│   ├── test_metrics.py                # 指标测试
│   ├── test_signal.py                 # 信号测试
│   ├── test_new_signals.py            # 新增信号测试
│   ├── test_signal_evaluation.py      # 信号评估测试
│   └── test_pipeline.py               # Pipeline 测试
├── README.md                          # 本文件
└── requirements.txt                   # Python 依赖
```

## 快速开始

### 环境要求

- Python 3.7+
- 操作系统：Windows / macOS / Linux

### 安装

```bash
# 克隆项目后进入目录
cd BacktestEngine

# 安装依赖
pip install -r requirements.txt
```

### 运行测试

```bash
# 运行全部单元测试
python -m unittest discover -s tests -v
```

### 运行信号日报（订阅模块）

```bash
cd subscribe

# 完整流程（数据拉取 + 信号计算 + 报告生成 + 邮件发送）
python run_daily.py

# 跳过数据拉取（使用已有缓存）
python run_daily.py --skip-fetch

# 跳过邮件发送
python run_daily.py --skip-mail

# 测试模式（仅处理前 50 只股票）
python run_daily.py --max-stocks 50
```

## 使用指南

### 策略开发

添加新信号：继承 `BaseSignal`，实现 `compute()` 方法。

```python
from backtest.strategies.signal import BaseSignal

class MyCustomSignal(BaseSignal):
    def compute(self, df):
        # 返回包含 signal 列的 DataFrame
        ...
```

添加新买入策略：继承 `BaseBuyStrategy`。

```python
from backtest.strategies.buy import BaseBuyStrategy

class MyBuyStrategy(BaseBuyStrategy):
    def generate_actions(self, date, context):
        # 返回 Action 列表
        ...
```

### 关键文件路径指引

| 用途 | 文件路径 |
|------|----------|
| 信号实现 | `backtest/strategies/signal.py` |
| 买入策略 | `backtest/strategies/buy.py` |
| 卖出策略 | `backtest/strategies/sell.py` |
| 条件系统 | `backtest/pipeline/conditions.py` |
| 收益指标 | `backtest/metrics/returns.py` |
| 信号评估指标 | `backtest/metrics/signal_evaluation.py` |
| 可视化图表 | `backtest/visualization/` |

> 新增指标函数后需在 `backtest/metrics/__init__.py` 中注册。

## 订阅日报模块

`subscribe/` 模块实现了一套完整的 **信号日报自动化流程**：

```
Baostock K线数据 → 14个信号频道计算 → 阈值筛选 → 基本面分析 → HTML报告 → 邮件推送
```

### 已注册信号频道（14 个）

| 频道名称 | 类别 | 说明 |
|----------|------|------|
| MACD 金叉 | MACD | DIF 上穿 DEA |
| MACD 柱交叉 | MACD | MACD 柱由负转正 |
| MACD 值交叉 | MACD | MACD 值上穿零轴 |
| MACD 信号交叉 | MACD | 信号线交叉 |
| MACD 金叉+柱确认 | MACD | 金叉且柱为正 |
| MACD 值信号 | MACD | 连续 MACD 值 |
| MACD 柱信号 | MACD | MACD 柱连续值 |
| MACD 柱变化 | MACD | 柱变化率 |
| MACD 收敛 | MACD | 收敛度信号 |
| RSI 超卖 | 动量 | RSI 低于阈值 |
| CCI 超卖 | 反转 | CCI 低于阈值 |
| 成交量突破 | 量价 | 成交量放大 |
| 价格动量 | 动量 | 价格动量信号 |
| 布林带 | 反转 | 触及下轨 |

### 运行

```bash
cd subscribe
python run_daily.py              # 完整流程
python run_daily.py --skip-mail  # 跳过邮件发送
```

> **注意**：`mail_sender.py` 中的 SMTP 配置（服务器地址、账号、授权码、收件人）需要在首次使用前替换为个人配置。

## 测试

测试框架：`unittest`（配合 `pytest` 运行）

| 测试文件 | 覆盖模块 |
|----------|----------|
| `test_core.py` | Action/Trade/Position、DataFeed、BacktestEngine、资金校验 |
| `test_strategies.py` | Buy/Sell 基类、Selector、Composite、资金校验 |
| `test_metrics.py` | 收益类/风险类/风险调整/交易运作指标 |
| `test_signal.py` | 11 种 MACD 信号、均线交叉、组合信号、前向偏差隔离、IC 对齐 |
| `test_new_signals.py` | RSI、CCI、布林带、成交量突破、价格动量、ATR、新高突破等信号 |
| `test_signal_evaluation.py` | Pearson/Rank IC、前向收益、分组分析、多空组合、换手率 |
| `test_pipeline.py` | 条件系统、Pipeline 执行、评估指标 |
| `subscribe/tests/test_subscribe.py` | 信号注册中心、基本面分析、报告生成、数据获取工具函数 |

```bash
# 运行全部单元测试
python -m unittest discover -s tests -v

# 运行订阅模块测试
python -m pytest subscribe/tests/test_subscribe.py -v
```

## 依赖

| 包 | 最低版本 | 用途 |
|----|----------|------|
| numpy | 1.20.0 | 数值计算与向量化操作 |
| pandas | 1.3.0 | 数据处理与时间序列 |
| matplotlib | 3.5.0 | 可视化图表绘制 |
| seaborn | 0.11.0 | 统计可视化增强 |
| scipy | 1.7.0 | 科学计算（统计检验等） |
| baostock | - | A 股数据源（订阅模块） |
| tqdm | - | 进度条（订阅模块） |

> `baostock` 和 `tqdm` 为订阅日报模块的额外依赖，通过代码中的 `import` 引用，不在 `requirements.txt` 中。请手动安装：
> ```bash
> pip install baostock tqdm
> ```

## 许可证

本项目基于 [MIT License](LICENSE) 开源。

## 安全提醒

> ⚠️ **重要**：切勿将包含真实邮箱地址、SMTP 授权码或其他敏感凭证的配置文件提交到代码仓库。
>
> 本项目 `subscribe/mail_sender.py` 中的 `SMTP_SERVER`、`SENDER`、`AUTH_CODE`、`DEFAULT_RECEIVERS` 等配置项为示例占位值，使用前请替换为个人配置并确保不被泄露。
>
> 建议做法：
> 1. 将敏感配置移至环境变量或 `.env` 文件
> 2. 将 `.env` 加入 `.gitignore`
> 3. 提供 `.env.example` 模板文件供参考
