"""
serving/scheduler.py

Responsible for running the full Crypto AI Intelligence pipeline
automatically on a recurring hourly schedule.

Every hour, this scheduler triggers run_pipeline() which orchestrates
the full system from ingestion through to model inference — fetching
fresh data, writing to storage, computing features, and generating
predictions across all 10 assets.

Takes no inputs. Runs indefinitely until manually stopped.

Run manually:
    python serving/scheduler.py
"""

import signal
import sys
import time
from datetime import datetime

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from loguru import logger

from main import run_pipeline


def start_scheduler() -> None:
    """
    Initialise and start the scheduler.
    Runs run_pipeline() immediately on startup, then every hour.
    Shuts down cleanly on Ctrl+C.
    """
    scheduler = BackgroundScheduler()

    scheduler.add_job(
        run_pipeline,
        IntervalTrigger(hours=1),
        next_run_time=datetime.now(),
    )

    scheduler.start()
    logger.info("Scheduler started — pipeline will run every hour")

    try:
        while True:
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutdown signal received — stopping scheduler")
        scheduler.shutdown(wait=False)
        logger.success("Scheduler stopped cleanly")


if __name__ == "__main__":
    start_scheduler()