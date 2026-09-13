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


def test_importing_the_news_pipeline_does_not_load_pandas():
    # A subprocess, because pytest has already imported the world by now.
    script = textwrap.dedent(
        """
        import sys
        import app.market.news  # noqa: F401
        loaded = [m for m in %r if m in sys.modules]
        print(",".join(loaded))
        """ % (HEAVY,)
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    loaded = [m for m in result.stdout.strip().split(",") if m]
    assert loaded == [], f"the news path imported: {loaded}"
