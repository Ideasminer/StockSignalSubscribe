from __future__ import annotations

from typing import Dict, Generator, List, Optional, Tuple

import pandas as pd


class DataFeed:
    def __init__(self, data: Dict[str, pd.DataFrame]):
        self._data: Dict[str, pd.DataFrame] = {}
        for code, df in data.items():
            df = df.copy()
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date").sort_index()
            self._data[code] = df

        self._all_dates: pd.DatetimeIndex = self._compute_all_dates()

    def _compute_all_dates(self) -> pd.DatetimeIndex:
        all_dates = pd.DatetimeIndex([])
        for df in self._data.values():
            all_dates = all_dates.union(df.index)
        return all_dates.sort_values()

    def get_bars(self, code: str) -> pd.DataFrame:
        return self._data.get(code, pd.DataFrame())

    def get_codes(self) -> List[str]:
        return list(self._data.keys())

    def get_date_range(self) -> Tuple[str, str]:
        if len(self._all_dates) == 0:
            return ("", "")
        return (
            str(self._all_dates[0].strftime("%Y-%m-%d")),
            str(self._all_dates[-1].strftime("%Y-%m-%d")),
        )

    def iter_dates(self) -> Generator[str, None, None]:
        for dt in self._all_dates:
            yield str(dt.strftime("%Y-%m-%d"))

    def get_data_at_date(self, date: str) -> Dict[str, pd.Series]:
        dt = pd.Timestamp(date)
        result: Dict[str, pd.Series] = {}
        for code, df in self._data.items():
            if dt in df.index:
                row = df.loc[dt]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                result[code] = row
        return result

    def has_date(self, date: str) -> bool:
        dt = pd.Timestamp(date)
        return dt in self._all_dates

    def get_previous_date(self, date: str) -> Optional[str]:
        dt = pd.Timestamp(date)
        if dt not in self._all_dates:
            return None
        idx = self._all_dates.get_loc(dt)
        if idx > 0:
            return str(self._all_dates[idx - 1].strftime("%Y-%m-%d"))
        return None

    def get_data_before_date(self, code: str, date: str, n: int) -> pd.DataFrame:
        df = self._data.get(code, pd.DataFrame())
        if df.empty:
            return pd.DataFrame()
        dt = pd.Timestamp(date)
        before = df.index[df.index < dt]
        if len(before) < n:
            return pd.DataFrame()
        return df.loc[before[-n:]]
