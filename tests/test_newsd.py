"""newsd, the news engine's own process.

What matters here: it runs the news job and nothing else, its Redis job store
cannot collide with the terminal's scheduler, and the job it stores is one a
fresh process can actually resolve.
"""
import importlib

from apscheduler.triggers.cron import CronTrigger

from app.worker import newsd
from app.worker import scheduler as terminal_scheduler


async def _jobs():
    # start(paused=True): APScheduler applies job defaults only on start, and
    # paused means nothing fires.
    sched = newsd.build_scheduler(jobstore=None)
    sched.start(paused=True)
    try:
        return {job.id: job for job in sched.get_jobs()}
    finally:
        sched.shutdown(wait=False)


async def test_it_runs_the_desk_jobs_and_none_of_the_terminals():
    # newsd owns the reading and the event desk; the terminal's jobs stay in
    # the terminal's process, which is what lets this one be memory-capped
    # and killed on its own.
    assert set(await _jobs()) == {"news-analysis", "event-agenda", "event-watchdog",
                                  "event-watches"}
    assert set(newsd.SCHEDULE) & set(terminal_scheduler.SCHEDULE) == set()


async def test_it_runs_hourly_on_the_hour_india_time():
    job = (await _jobs())["news-analysis"]
    assert isinstance(job.trigger, CronTrigger)
    assert {f.name: str(f) for f in job.trigger.fields}["minute"] == "0"
    assert str(job.trigger.timezone) == "Asia/Kolkata"


async def test_a_run_missed_during_a_restart_still_happens():
    job = (await _jobs())["news-analysis"]
    assert job.max_instances == 1          # an LLM pass can outrun the hour
    assert job.coalesce is True
    assert job.misfire_grace_time and job.misfire_grace_time >= 300


def test_its_job_store_is_not_the_terminals():
    # Both processes point at the same Redis. Sharing a key would have each
    # scheduler load, and run, the other's jobs.
    assert newsd.JOBS_KEY != "apscheduler.jobs"
    assert newsd.RUN_TIMES_KEY != "apscheduler.run_times"


def test_the_stored_job_path_resolves():
    # The store keeps an import path and resolves it on load, so a typo here
    # only surfaces at the top of the hour.
    module, func = newsd.SCHEDULE["news-analysis"][0].split(":")
    assert callable(getattr(importlib.import_module(module), func))
