"""The in-process scheduler that replaces Celery beat.

What matters here: every job it claims to run is scheduled with the right
cadence, a slow run can't overlap itself, and a job missed while the process
was restarting still runs instead of being silently skipped.
"""
from apscheduler.triggers.cron import CronTrigger

from app.worker import scheduler as scheduler_mod


async def _jobs():
    # start(paused=True): APScheduler only applies job defaults (max_instances,
    # misfire grace) when the scheduler starts, and paused means nothing fires.
    sched = scheduler_mod.build_scheduler(jobstore=None)
    sched.start(paused=True)
    try:
        return {job.id: job for job in sched.get_jobs()}
    finally:
        sched.shutdown(wait=False)


async def test_every_cheap_job_is_scheduled():
    assert set(await _jobs()) == {
        "drift-check", "reddit-sentiment", "market-overview-morning",
        "market-overview-evening", "kite-session-refresh", "paper-square-off",
    }


async def test_the_cadences_match_what_beat_ran():
    jobs = await _jobs()
    expected = {
        "drift-check": {"minute": "30"},
        "reddit-sentiment": {"minute": "45", "hour": "1,3,5,7,9,11,13,15,17,19,21,23"},
        "market-overview-morning": {"minute": "30", "hour": "6", "day_of_week": "mon-fri"},
        "market-overview-evening": {"minute": "0", "hour": "18", "day_of_week": "mon-fri"},
        "kite-session-refresh": {"minute": "0", "hour": "6"},
        "paper-square-off": {"minute": "20", "hour": "15", "day_of_week": "mon-fri"},
    }
    for job_id, fields in expected.items():
        trigger = jobs[job_id].trigger
        assert isinstance(trigger, CronTrigger)
        actual = {f.name: str(f) for f in trigger.fields}
        for name, value in fields.items():
            assert actual[name] == value, f"{job_id}.{name}: {actual[name]} != {value}"


async def test_everything_runs_on_india_time():
    # Square-off at 15:20 means 15:20 IST; a UTC scheduler would fire it after
    # the market has already closed.
    for job in (await _jobs()).values():
        assert str(job.trigger.timezone) == "Asia/Kolkata"


async def test_a_slow_run_cannot_overlap_itself():
    # The hourly news/drift work can outrun its own interval; a second copy
    # would double-spend on the LLM and double-publish alerts.
    for job in (await _jobs()).values():
        assert job.max_instances == 1


async def test_a_job_missed_during_a_restart_still_runs():
    # Deploys restart this process. Without a grace period, a job due during
    # the restart is dropped -- the 15:20 square-off is the one that matters.
    for job in (await _jobs()).values():
        assert job.misfire_grace_time and job.misfire_grace_time >= 300
        assert job.coalesce is True   # one catch-up run, not a burst


async def test_jobs_are_stored_as_import_paths_so_a_restart_can_resolve_them():
    for job in (await _jobs()).values():
        assert job.func_ref.startswith("app.worker.jobs:"), job.func_ref
