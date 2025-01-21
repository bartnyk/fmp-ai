import re
from datetime import datetime, timedelta
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

now = datetime.now()
year_ago = now - timedelta(weeks=10)


class DataAdapter:
    __columns: list[str] = []

    def __init__(self, data: dict | list):
        if not self.columns:
            raise ValueError("Columns attribute must be set.")

        self._df = pd.DataFrame(data)

    @property
    def df(self):
        return self._df

    @property
    def columns(self):
        return getattr(self, f"_{self.__class__.__name__}__columns")

    def _format(self) -> None:
        self._add_new_values()
        self._cut_columns()
        self._df.round(8)

    def _add_new_values(self): ...

    @classmethod
    def format(cls, data: dict | list) -> "DataAdapter":
        formatter = cls(data)
        formatter._format()

        return formatter

    def _cut_columns(self):
        self._df = self._df[self.columns]

    def convert_currency(self, value: str) -> int:
        codes = {
            "USD": 1,
            "EUR": 2,
            "JPY": 3,
            "GBP": 4,
            "AUD": 5,
            "CAD": 6,
            "CHF": 7,
            "CNY": 8,
            "MXN": 9,
            "INR": 10,
            "RUB": 11,
            "TRY": 12,
            "PLN": 13,
        }
        return codes.get(value, -1)


class ForexDataAdapter(DataAdapter):
    __columns = [
        "ticker",
        "close",
        "high",
        "low",
        "open",
        "timestamp",
        "base_currency",
        "quote_currency",
        "base_currency_id",
        "quote_currency_id",
        "change",
        "change_pct",
        "range",
        "range_pct",
    ]

    def _format(self) -> None:
        self.break_the_ticker()

        super()._format()

    def break_the_ticker(self) -> None:
        self._df["base_currency"] = self._df["ticker"].apply(lambda x: x[:3])
        self._df["quote_currency"] = self._df["ticker"].apply(lambda x: x[3:])
        self._df["base_currency_id"] = self._df["base_currency"].apply(
            self.convert_currency
        )
        self._df["quote_currency_id"] = self._df["quote_currency"].apply(
            self.convert_currency
        )

    def _add_new_values(self) -> None:
        self._df["change"] = self._df["close"] - self._df["open"]
        self._df["change_pct"] = (self._df["change"] / self._df["open"]) * 100
        self._df["range"] = self._df["high"] - self._df["low"]
        self._df["range_pct"] = (self._df["range"] / self._df["open"]) * 100

    @property
    def df_daily(self):
        return self.df[self.df["timestamp"].dt.time == pd.Timestamp("00:00").time()]


class EconomicEventsAdapter(DataAdapter):
    __columns = [
        "title",
        "timestamp",
        "actual",
        "consensus",
        "forecast",
        "previous",
        "sentiment",
        "currency",
        "currency_id",
        "title_encoded",
        "title_code",
        "consensus_diff",
        "forecast_diff",
        "change",
        "actual_change_pct",
        "consensus_diff_pct",
        "forecast_diff_pct",
        "impact_range",
        "impact_range_pct",
    ]

    def _format(self) -> None:
        self.convert_result_values()
        self.map_currencies()
        self.convert_event_titles()
        super()._format()

    def map_currencies(self):
        self._df["currency"] = self._df["subject"].apply(
            lambda x: x.get("currency") if isinstance(x, dict) else None
        )
        self._df["currency_id"] = self._df["currency"].apply(self.convert_currency)

    def convert_event_titles(self) -> None:
        encoder = LabelEncoder()
        self._df["title_encoded"] = encoder.fit_transform(self._df["title"])

        vocab_size = len(encoder.classes_)
        embedding_dim = 8
        embedding_matrix = np.random.rand(vocab_size, embedding_dim)  # Losowe wagi

        self._df["title_code"] = self._df["title_encoded"].apply(
            lambda x: embedding_matrix[x].tolist()
        )

    def _add_new_values(self):
        self.add_consensus_diff()
        self.add_forecast_diff()
        self.add_change()
        self.add_actual_change_pct()
        self.add_consensus_diff_pct()
        self.add_forecast_diff_pct()
        self.add_impact_range()
        self.add_impact_range_pct()

    def add_consensus_diff(self):
        self._df["consensus_diff"] = (self._df["actual"] - self._df["consensus"]).where(
            self._df["actual"].notna() & self._df["consensus"].notna(), np.nan
        )

    def add_forecast_diff(self):
        self._df["forecast_diff"] = (self._df["actual"] - self._df["forecast"]).where(
            self._df["actual"].notna() & self._df["forecast"].notna(), np.nan
        )

    def add_change(self):
        self._df["change"] = (self._df["actual"] - self._df["previous"]).where(
            self._df["actual"].notna() & self._df["previous"].notna(), np.nan
        )

    def add_actual_change_pct(self):
        self._df["actual_change_pct"] = (
            (self._df["actual"] - self._df["previous"]) / self._df["previous"]
        ).where(self._df["previous"] != 0, np.nan) * 100

    def add_consensus_diff_pct(self):
        self._df["consensus_diff_pct"] = (
            self._df["consensus_diff"] / self._df["consensus"]
        ).where(self._df["consensus"] != 0, np.nan) * 100

    def add_forecast_diff_pct(self):
        self._df["forecast_diff_pct"] = (
            self._df["forecast_diff"] / self._df["forecast"]
        ).where(self._df["forecast"] != 0, np.nan) * 100

    def add_impact_range(self):
        self._df["impact_range"] = self._df[
            ["forecast_diff", "consensus_diff", "change"]
        ].max(axis=1) - self._df[["forecast_diff", "consensus_diff", "change"]].min(
            axis=1
        )

    def add_impact_range_pct(self):
        self._df["impact_range_pct"] = (
            self._df["impact_range"] / self._df["previous"]
        ).where(self._df["previous"] != 0, np.nan) * 100

    def break_the_value(self, value: str) -> Optional[tuple[str, str, str]]:
        re_pattern = re.compile(
            r"^(?P<prefix>[^\d-]*)(?P<number>-?\d*\.?\d+)(?P<suffix>[^\d]*)$"
        )
        match = re_pattern.match(value)
        if match:
            return (
                match.group("prefix"),
                match.group("number"),
                match.group("suffix").replace(" ®", "").strip(),
            )

    def convert_value(self, value: str) -> Union[float, np.nan]:
        if value_elements := self.break_the_value(value):
            prefix, number, suffix = value_elements
        else:
            return np.nan

        suffix_map = {
            "%": 0.01,
            "K": 1e3,
            "B": 1e9,
            "T": 1e12,
            "M": 1e6,
            "f": 1e15,
            "million": 1e6,
            "Bcf": 1e9,
            "": 1,
        }
        try:
            multiplier = suffix_map[suffix]
        except KeyError:
            return float(number)

        return float(number) * multiplier

    def convert_result_values(self):
        for col in ["actual", "consensus", "forecast", "previous"]:
            if col in self._df.columns:
                self._df[col] = self._df[col].apply(self.convert_value)
