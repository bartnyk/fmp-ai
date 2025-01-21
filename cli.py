import asyncio

import typer
from fmp.repository.mongo import (ForexDataRepository,
                                  ForexEconomicEventsRepository)

from core.adapters import EconomicEventsAdapter, ForexDataAdapter
from core.ai import DailyForexPredictor, ShortTermForexPredictor
from core.config import cfg
from core.judge import ForexTradingSystem
from core.repository import DataSupplier

app = typer.Typer(pretty_exceptions_enable=False)


@app.command(name="start-training")
async def train_model2():
    supplier = DataSupplier(
        economic_events_repository=ForexEconomicEventsRepository,
        stock_data_repository=ForexDataRepository,
    )

    events_data = await supplier.get_events_data()
    forex_data = await supplier.get_forex_data()

    events = EconomicEventsAdapter.format(events_data)
    forex = ForexDataAdapter.format(forex_data)

    events_df = events.df
    forex_df = forex.df

    short_model = ShortTermForexPredictor.train_model(events_df, forex_df)
    daily_model = DailyForexPredictor.train_model(events_df, forex_df)

    short_model.save(cfg.project_path.short_term_model_path)
    daily_model.save(cfg.project_path.daily_model_path)


@app.command(name="create-forecast")
async def create_forecast():
    supplier = DataSupplier(
        economic_events_repository=ForexEconomicEventsRepository,
        stock_data_repository=ForexDataRepository,
    )

    events_data = await supplier.get_events_data()
    forex_data = await supplier.get_forex_data()

    events = EconomicEventsAdapter.format(events_data)
    forex = ForexDataAdapter.format(forex_data)

    events_df = events.df
    forex_df = forex.df

    short_model, daily_model = (
        ForexTradingSystem.load_model(cfg.project_path.short_term_model_path),
        ForexTradingSystem.load_model(cfg.project_path.daily_model_path),
    )
    ForexTradingSystem.make_decisions(events_df, forex_df, short_model, daily_model)


if __name__ == "__main__":
    asyncio.run(app())
