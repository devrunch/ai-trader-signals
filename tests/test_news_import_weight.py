"""The news path must not drag pandas in.

Importing yfinance for one HTTP call pulled pandas + numpy with it: 65 MB of
resident memory, measured in the running container, for a JSON fetch that never
touches a DataFrame. That is also what the news engine will be split out on
(a separate process that never loads the terminal's maths stack), so this is
worth a test rather than discipline.
"""
import subprocess
import sys
import textwrap

HEAVY = ("pandas", "numpy", "yfinance", "pandas_ta")


def _heavy_modules_after_importing(module: str) -> list[str]:
    # A subprocess, because pytest has already imported the world by now.
    script = textwrap.dedent(
        f"""
        import sys
        import {module}  # noqa: F401
        loaded = [m for m in {HEAVY!r} if m in sys.modules]
        print(",".join(loaded))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return [m for m in result.stdout.strip().split(",") if m]


def test_importing_the_news_pipeline_does_not_load_pandas():
    loaded = _heavy_modules_after_importing("app.market.news")
    assert loaded == [], f"the news path imported: {loaded}"


def test_the_news_process_does_not_load_the_terminals_stack():
    # newsd is memory-capped below what pandas + numpy alone cost, so an
    # import that reaches back into jobs.py (SignalService) OOMs the process
    # rather than failing a test.
    loaded = _heavy_modules_after_importing("app.worker.newsd")
    assert loaded == [], f"newsd imported: {loaded}"
