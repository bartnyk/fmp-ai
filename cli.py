import asyncio
from functools import wraps
from typing import Callable

import typer

from core.adapters import EconomicEventsAdapter, ForexDataAdapter
from core.ai import DailyForexPredictor, ShortTermForexPredictor
from core.config import cfg
from core.judge import ForexTradingSystem
from core.repository import DataSupplier
from fmp.repository.mongo import ForexDataRepository, ForexEconomicEventsRepository

app = typer.Typer(pretty_exceptions_enable=False)


def async_command(func: Callable) -> Callable:
    """
    Decorator to run a command asynchronously.

    Parameters
    ----------
    func : Callable
        The function to be decorated.

    Returns
    -------
    Callable
        The decorated function.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        asyncio.run(func(*args, **kwargs))

    return wrapper


@app.command(name="start-training")
@async_command
async def train_model():
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

    short_model = await ShortTermForexPredictor.train_model(events_df, forex_df)
    daily_model = await DailyForexPredictor.train_model(events_df, forex_df)

    short_model.save(cfg.project_path.short_term_model_path)
    daily_model.save(cfg.project_path.daily_model_path)


@app.command(name="create-forecast")
@async_command
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
    app()
