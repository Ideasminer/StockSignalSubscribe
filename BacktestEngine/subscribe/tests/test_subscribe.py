import os
import sys
import unittest
import tempfile
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from subscribe.signal_registry import (
    list_channels, get_channel, SUBSCRIBED_CHANNELS, _default_registry,
)
from subscribe.fundamentals import assess_profit_trend
from subscribe.report_generator import generate_report
from subscribe.data_fetcher import (
    _calc_trade_date_range, trading_days_to_calendar_days,
    find_latest_data_file, validate_data_file,
    _detect_new_trading_day, _copy_as_today,
)


class TestSignalRegistry(unittest.TestCase):
    def test_channels_registered(self):
        channels = list_channels()
        self.assertGreaterEqual(len(channels), 10, "至少注册10个频道")
        names = [ch.name for ch in channels]
        self.assertIn("MACD金叉", names)
        self.assertIn("RSI超卖", names)

    def test_channel_has_compute(self):
        ch = get_channel("MACD金叉")
        self.assertIsNotNone(ch)
        n = 50
        df = pd.DataFrame({
            "code": ["sz.000001"] * n,
            "date": pd.date_range("2024-01-01", periods=n, freq="B"),
            "close": np.linspace(10, 15, n),
            "tradestatus": [1] * n,
        })
        result = ch.compute(df)
        self.assertIn("signal_value", result.columns)

    def test_channel_threshold_filter(self):
        ch = get_channel("RSI超卖")
        self.assertIsNotNone(ch)
        sv = pd.Series([-0.85, 0.2, -0.5, -0.9, 0.1], index=[f"c{i}" for i in range(5)])
        mask = ch.filter(sv)
        hit = set(sv[mask].index)
        self.assertIn("c0", hit)
        self.assertIn("c3", hit)
        self.assertNotIn("c1", hit)
        self.assertNotIn("c2", hit)

    def test_threshold_channel(self):
        ch = get_channel("价格动量")
        self.assertIsNotNone(ch)
        sv = pd.Series([0.15, 0.19, 0.18, -0.02, 0.22, -0.01, 0.10],
                       index=["a","b","c","d","e","f","g"])
        mask = ch.filter(sv)
        hit = set(sv[mask].index)
        self.assertIn("b", hit)
        self.assertIn("e", hit)


class TestFundamentals(unittest.TestCase):
    def test_assess_profit_trend_up(self):
        df = pd.DataFrame({
            "npMargin": [8.0, 9.0, 10.0, 11.0],
            "epsTTM": [1.2, 1.3, 1.5, 1.6],
            "query_year": [2025, 2025, 2025, 2025],
            "query_quarter": [1, 2, 3, 4],
        })
        r = assess_profit_trend(df)
        self.assertEqual(r["trend"], "收益向上")

    def test_assess_profit_trend_down(self):
        df = pd.DataFrame({
            "npMargin": [10.0, 8.0, 6.0, 4.0],
            "epsTTM": [1.5, 1.3, 1.0, 0.8],
            "query_year": [2025] * 4,
            "query_quarter": [1, 2, 3, 4],
        })
        r = assess_profit_trend(df)
        self.assertEqual(r["trend"], "收益劣化")

    def test_assess_mixed(self):
        df = pd.DataFrame({
            "npMargin": [8.0, 9.0, 7.0, 10.0],
            "epsTTM": [1.2, 1.5, 1.1, 1.3],
            "query_year": [2025] * 4,
            "query_quarter": [1, 2, 3, 4],
        })
        r = assess_profit_trend(df)
        self.assertEqual(r["trend"], "收益波动")

    def test_assess_insufficient(self):
        r = assess_profit_trend(pd.DataFrame())
        self.assertEqual(r["trend"], "数据不足")


class TestReportGenerator(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_generate_empty_report(self):
        hits = pd.DataFrame()
        fundamentals = pd.DataFrame()
        summary = []
        path = generate_report(hits, fundamentals, self.tmp, summary)
        self.assertTrue(os.path.exists(path))
        with open(path, "r", encoding="utf-8") as f:
            html = f.read()
        self.assertIn("信号日报", html)

    def test_generate_report_with_data(self):
        hits = pd.DataFrame({
            "code": ["sz.000001", "sh.600000"],
            "code_name": ["平安银行", "浦发银行"],
            "channel": ["MACD金叉", "RSI超卖"],
            "category": ["MACD", "动量"],
            "signal_value": [0.5, -0.35],
            "close": [12.50, 9.80],
            "pctChg": [2.3, -1.5],
        })
        fundamentals = pd.DataFrame({
            "code": ["sz.000001", "sh.600000"],
            "trend": ["收益向上", "收益波动"],
            "np_margins": ["8.5,9.2,10.1,11.0", "6.0,5.5,7.0,6.8"],
            "eps_ttms": ["1.20,1.30,1.45,1.55", "0.90,0.85,1.00,0.95"],
            "detail": ["净利率持续走强", "净利率波动; 每股收益波动"],
        })
        summary = [{"channel":"MACD金叉","category":"MACD","count":1},
                   {"channel":"RSI超卖","category":"动量","count":1}]
        path = generate_report(hits, fundamentals, self.tmp, summary)
        self.assertTrue(os.path.exists(path))
        with open(path, "r", encoding="utf-8") as f:
            html = f.read()
        self.assertIn("平安银行", html)
        self.assertIn("MACD金叉", html)
        self.assertIn("收益向上", html)


class TestDataFetcher(unittest.TestCase):
    def test_trading_days_conversion(self):
        self.assertEqual(trading_days_to_calendar_days(15), 30)
        self.assertEqual(trading_days_to_calendar_days(180), 270)
        self.assertEqual(trading_days_to_calendar_days(5), 30)

    def test_date_range_180(self):
        start, end = _calc_trade_date_range(180)
        self.assertRegex(start, r"\d{4}-\d{2}-\d{2}")
        self.assertRegex(end, r"\d{4}-\d{2}-\d{2}")
        s = datetime.strptime(start, "%Y-%m-%d")
        e = datetime.strptime(end, "%Y-%m-%d")
        self.assertGreater(e, s)
        self.assertGreater((e - s).days, 250)

    def test_date_range_small(self):
        start, end = _calc_trade_date_range(15)
        s = datetime.strptime(start, "%Y-%m-%d")
        e = datetime.strptime(end, "%Y-%m-%d")
        self.assertLess((e - s).days, 50)

    def test_date_range_different_lookbacks(self):
        _, end = _calc_trade_date_range(15)
        start180, _ = _calc_trade_date_range(180)
        self.assertLess(start180, _calc_trade_date_range(15)[0])

    def test_find_latest_empty_dir(self):
        with tempfile.TemporaryDirectory() as td:
            date_str, path = find_latest_data_file(td)
            self.assertIsNone(date_str)
            self.assertIsNone(path)

    def test_find_latest_picks_newest(self):
        with tempfile.TemporaryDirectory() as td:
            for d in ["20260101", "20260115", "20260201"]:
                p = os.path.join(td, f"{d}.csv")
                pd.DataFrame({"date": ["2026-01-01"], "code": ["sz.000001"],
                              "close": [10.0], "tradestatus": ["1"]}).to_csv(p, index=False)
            date_str, path = find_latest_data_file(td)
            self.assertEqual(date_str, "20260201")

    def test_validate_good_file(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "test.csv")
            pd.DataFrame({"date": ["2026-01-01"], "code": ["sz.000001"],
                          "close": [10.0], "tradestatus": ["1"]}).to_csv(p, index=False)
            ok, reason = validate_data_file(p)
            self.assertTrue(ok, reason)

    def test_validate_missing_column(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "test.csv")
            pd.DataFrame({"date": ["2026-01-01"], "code": ["sz.000001"]}).to_csv(p, index=False)
            ok, reason = validate_data_file(p)
            self.assertFalse(ok)
            self.assertIn("缺少必要列", reason)

    def test_validate_nonexistent(self):
        ok, reason = validate_data_file("/nonexistent/file.csv")
        self.assertFalse(ok)

    def test_detect_new_trading_day_no_gap(self):
        today = pd.Timestamp.now().normalize()
        self.assertFalse(_detect_new_trading_day(today),
                         "历史日期=今天 → 无新交易日")

    def test_copy_as_today(self):
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "20260430.csv")
            pd.DataFrame({
                "date": ["2026-04-30"], "code": ["sz.000001"],
                "close": [10.0], "tradestatus": ["1"],
            }).to_csv(src, index=False)
            dst = _copy_as_today(src, td)
            today_str = datetime.now().strftime("%Y%m%d")
            expected_name = f"{today_str}.csv"
            self.assertIn(expected_name, dst)
            self.assertTrue(os.path.exists(dst))


if __name__ == "__main__":
    unittest.main()
