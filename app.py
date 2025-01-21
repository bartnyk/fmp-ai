import asyncio

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

from cli import create_forecast

trigger = IntervalTrigger(minutes=10)
scheduler = BackgroundScheduler()


if __name__ == "__main__":
    scheduler.add_job(
        lambda: asyncio.run(create_forecast()),
        trigger,
        max_instances=1,
        replace_existing=True,
    )
    scheduler.start()

    try:
        while True:
            ...
    except:
        scheduler.shutdown()
