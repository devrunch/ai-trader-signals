"""`asyncio.run` with a bounded default executor.

Shared by every scheduled job. `main.py` bounds the executor for the API
process in its lifespan hook; a job process never runs `main.py`, so it has
to be done here too.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

# Same bound as main.py's: blocking vendor calls (yfinance, boto3) are
# offloaded with `to_thread` / `run_in_executor(None, ...)`, and a fresh
# `asyncio.run()` loop otherwise gets the interpreter default --
# `min(32, cpu_count + 4)` threads, unbounded from the caller's point of view.
MAX_WORKERS = 16


def run_async(coro):
    async def main():
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="worker-io") as pool:
            loop.set_default_executor(pool)
            return await coro

    return asyncio.run(main())
