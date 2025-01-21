from datetime import datetime
from typing import Type

from fmp.repository.mongo import (ForexDataRepository,
                                  ForexEconomicEventsRepository)

min_date, max_date = datetime(2018, 1, 1), datetime(2025, 1, 1)


class DataSupplier:
    def __init__(
        self,
        economic_events_repository: Type[ForexEconomicEventsRepository],
        stock_data_repository: Type[ForexDataRepository],
    ):
        self._ee_repository = economic_events_repository()
        self._sd_repository = stock_data_repository()

    async def get_events_data(
        self,
        query: dict = {
            "timestamp": {
                "$gte": min_date,
                "$lt": max_date,
            }
        },
    ):
        data = await self._ee_repository.find(query)
        return await data.to_list(length=None)

    async def get_forex_data(
        self,
        query: dict = {
            "timestamp": {
                "$gt": min_date,
                "$lte": max_date,
            }
        },
    ):
        data = await self._sd_repository.find(query)
        return await data.to_list(length=None)

    async def get_forex_data_for_today(self):
        date_today = datetime.now().date()
        start_of_today = datetime.combine(date_today, datetime.min.time())
        end_of_today = datetime.combine(date_today, datetime.max.time())
        return self.get_forex_data(
            {
                "timestamp": {
                    "$gte": start_of_today,
                    "$lte": end_of_today,
                }
            },
        )
