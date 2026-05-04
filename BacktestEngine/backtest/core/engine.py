from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

import pandas as pd

from .action import Action, Position, Trade
from .data_feed import DataFeed

if TYPE_CHECKING:
    from ..strategies.base import BaseStrategy
    from ..strategies.buy import BaseBuyStrategy
    from ..strategies.sell import BaseSellStrategy


@dataclass
class BacktestResult:
    equity_curve: pd.Series = field(default_factory=pd.Series)
    benchmark_curve: pd.Series = field(default_factory=pd.Series)
    trades: List[Trade] = field(default_factory=list)
    positions_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    daily_returns: pd.Series = field(default_factory=pd.Series)
    benchmark_daily_returns: pd.Series = field(default_factory=pd.Series)
    total_commission: float = 0.0


class BacktestEngine:
    def __init__(
        self,
        data_feed: DataFeed,
        strategy: Optional[BaseStrategy] = None,
        buy_strategy: Optional[BaseBuyStrategy] = None,
        sell_strategy: Optional[BaseSellStrategy] = None,
        initial_cash: float = 1_000_000.0,
        commission_rate: float = 0.0003,
        slippage: float = 0.001,
        benchmark_code: Optional[str] = None,
        min_trade_amount: float = 10000.0,
        min_cash_reserve: float = 0.0,
    ):
        self.data_feed = data_feed
        self.strategy = strategy
        self.buy_strategy = buy_strategy
        self.sell_strategy = sell_strategy
        self.initial_cash = initial_cash
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.benchmark_code = benchmark_code
        self.min_trade_amount = min_trade_amount
        self.min_cash_reserve = min_cash_reserve

        self.cash: float = initial_cash
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: pd.Series = pd.Series(dtype=float)
        self.benchmark_curve: pd.Series = pd.Series(dtype=float)
        self.daily_returns: pd.Series = pd.Series(dtype=float)
        self.benchmark_daily_returns: pd.Series = pd.Series(dtype=float)
        self.positions_history: pd.DataFrame = pd.DataFrame()
        self.total_commission: float = 0.0
        self._previous_equity: float = initial_cash
        self._previous_benchmark: float = 1.0
        self._skipped_trades: List[str] = []

    def run(self, show_progress: bool = False) -> BacktestResult:
        if self.strategy is not None:
            self.strategy.init()

        equity_records: List[float] = []
        benchmark_records: List[float] = []
        date_labels: List[str] = []
        position_records: List[dict] = []
        prev_equity = self.initial_cash
        prev_benchmark = 1.0
        first_date = True

        all_dates = list(self.data_feed.iter_dates())
        total_dates = len(all_dates)
        for di, date in enumerate(all_dates):
            if show_progress and total_dates > 0:
                pct = (di + 1) / total_dates
                bl = 25
                f = int(bl * pct)
                bar = "█" * f + "░" * (bl - f)
                print(f"\r  [{bar}] {di+1:4d}/{total_dates} ({pct*100:5.1f}%)", end="", flush=True)
            data = self.data_feed.get_data_at_date(date)
            if not data:
                continue

            actions: List[Action] = []

            if self.strategy is not None:
                actions = self.strategy.next(date, data)
            else:
                if self.buy_strategy is not None:
                    actions.extend(self.buy_strategy.should_buy(date, data, self.positions, self.cash))
                if self.sell_strategy is not None:
                    actions.extend(self.sell_strategy.should_sell(date, data, self.positions, self.cash))

            for action in actions:
                self._execute_action(action, date, data)

            self._update_positions_price(data)

            total_equity = self.cash + sum(
                pos.market_value for pos in self.positions.values()
            )
            equity_records.append(total_equity)
            date_labels.append(date)

            if not first_date:
                daily_ret = (total_equity - prev_equity) / prev_equity
                dt = pd.Timestamp(date)
                self.daily_returns = pd.concat([
                    self.daily_returns,
                    pd.Series({dt: daily_ret})
                ])
            prev_equity = total_equity

            if self.benchmark_code and self.benchmark_code in data:
                bench_close = float(data[self.benchmark_code]["close"])
                first_close = float(
                    self.data_feed.get_bars(self.benchmark_code).iloc[0]["close"]
                )
                benchmark_value = bench_close / first_close
                benchmark_records.append(benchmark_value)
                if not first_date:
                    bench_ret = (benchmark_value - prev_benchmark) / prev_benchmark
                    dt = pd.Timestamp(date)
                    self.benchmark_daily_returns = pd.concat([
                        self.benchmark_daily_returns,
                        pd.Series({dt: bench_ret})
                    ])
                prev_benchmark = benchmark_value

            pos_record = {"date": date}
            total_mv = 0
            for code, pos in self.positions.items():
                pos_record[f"{code}_qty"] = pos.quantity
                pos_record[f"{code}_mv"] = pos.market_value
                total_mv += pos.market_value
            pos_record["cash"] = self.cash
            pos_record["total_mv"] = total_mv
            pos_record["total_equity"] = total_equity
            position_records.append(pos_record)

            first_date = False

        if show_progress:
            print()  # 换行结束进度条

        self.equity_curve = pd.Series(
            equity_records, index=pd.to_datetime(date_labels)
        )
        if benchmark_records:
            self.benchmark_curve = pd.Series(
                benchmark_records, index=pd.to_datetime(date_labels)
            )

        self.daily_returns.index = pd.to_datetime(self.daily_returns.index)
        if not self.benchmark_daily_returns.empty:
            self.benchmark_daily_returns.index = pd.to_datetime(self.benchmark_daily_returns.index)

        self.positions_history = pd.DataFrame(position_records)
        if not self.positions_history.empty:
            self.positions_history = self.positions_history.set_index("date")

        return BacktestResult(
            equity_curve=self.equity_curve,
            benchmark_curve=self.benchmark_curve,
            trades=self.trades,
            positions_history=self.positions_history,
            daily_returns=self.daily_returns,
            benchmark_daily_returns=self.benchmark_daily_returns,
            total_commission=self.total_commission,
        )

    def _execute_action(self, action: Action, date: str, data: dict):
        if action.is_hold():
            return

        if action.code not in data:
            return

        bar = data[action.code]
        raw_price = action.price if action.price > 0 else float(bar["close"])

        if action.is_buy():
            self._execute_buy(action, date, raw_price, bar)
        elif action.is_sell():
            self._execute_sell(action, date, raw_price, bar)

    def _execute_buy(self, action: Action, date: str, raw_price: float, bar):
        buy_price = raw_price * (1 + self.slippage)
        position = self.positions.get(action.code)

        max_qty_by_cash = int(self.cash / (buy_price * (1 + self.commission_rate)))
        if max_qty_by_cash <= 0:
            self._skipped_trades.append(f"{date} {action.code}: 现金不足以购买任何数量")
            return

        target_qty = action.quantity if action.quantity > 0 else max_qty_by_cash
        quantity = min(target_qty, max_qty_by_cash)

        if quantity <= 0:
            self._skipped_trades.append(f"{date} {action.code}: 计算数量为0")
            return

        amount = quantity * buy_price
        commission = max(amount * self.commission_rate, 5.0)
        total_cost = amount + commission

        if total_cost > self.cash:
            adjusted_quantity = int((self.cash - commission) / buy_price)
            if adjusted_quantity <= 0:
                self._skipped_trades.append(f"{date} {action.code}: 资金不足以覆盖最低佣金")
                return
            adjusted_amount = adjusted_quantity * buy_price
            if adjusted_amount < self.min_trade_amount:
                self._skipped_trades.append(
                    f"{date} {action.code}: 缩减后交易额{adjusted_amount:.0f}<最低{self.min_trade_amount:.0f}"
                )
                return
            quantity = adjusted_quantity
            amount = adjusted_amount
            commission = max(amount * self.commission_rate, 5.0)
            total_cost = amount + commission

        if self.min_trade_amount > 0 and amount < self.min_trade_amount:
            self._skipped_trades.append(
                f"{date} {action.code}: 交易额{amount:.0f}<最低{self.min_trade_amount:.0f}"
            )
            return

        if self.min_cash_reserve > 0 and (self.cash - total_cost) < self.min_cash_reserve:
            self._skipped_trades.append(
                f"{date} {action.code}: 执行后现金{self.cash - total_cost:.0f}<保留{self.min_cash_reserve:.0f}"
            )
            return

        self.cash -= total_cost
        self.total_commission += commission

        if action.code in self.positions:
            old_pos = self.positions[action.code]
            total_shares = old_pos.quantity + quantity
            total_cost_value = old_pos.cost_value + amount
            old_pos.cost_price = total_cost_value / total_shares if total_shares > 0 else 0
            old_pos.quantity = total_shares
            old_pos.buy_dates.append(date)
        else:
            self.positions[action.code] = Position(
                code=action.code,
                quantity=quantity,
                cost_price=buy_price,
                current_price=buy_price,
                buy_dates=[date],
            )

        trade = Trade(
            date=date,
            code=action.code,
            direction="buy",
            price=buy_price,
            quantity=quantity,
            amount=amount,
            commission=commission,
            reason=action.reason,
        )
        self.trades.append(trade)

    def _execute_sell(self, action: Action, date: str, raw_price: float, bar):
        if action.code not in self.positions:
            return

        position = self.positions[action.code]
        if position.quantity <= 0:
            return

        sell_price = raw_price * (1 - self.slippage)
        quantity = (
            position.quantity if action.quantity == 0 else min(action.quantity, position.quantity)
        )

        if quantity <= 0:
            return

        amount = quantity * sell_price
        commission = max(amount * self.commission_rate, 5.0)
        total_proceed = amount - commission

        self.cash += total_proceed
        self.total_commission += commission

        pnl = (sell_price - position.cost_price) * quantity

        holding_days = 0
        if position.buy_dates:
            buy_date = pd.Timestamp(position.buy_dates[-1])
            sell_date = pd.Timestamp(date)
            holding_days = (sell_date - buy_date).days

        position.quantity -= quantity

        trade = Trade(
            date=date,
            code=action.code,
            direction="sell",
            price=sell_price,
            quantity=quantity,
            amount=amount,
            commission=commission,
            reason=action.reason,
            holding_days=holding_days,
        )
        self.trades.append(trade)

        if position.quantity == 0:
            del self.positions[action.code]

    def _update_positions_price(self, data: dict):
        for code, position in self.positions.items():
            if code in data:
                position.current_price = float(data[code]["close"])
