import pytest

from app.signals.pine.sandbox import run_pine_script

BARS = [
    {"open": 100 + i, "high": 101 + i, "low": 99 + i, "close": 100.5 + i, "volume": 1000, "openTime": 1767000900000 + i * 60000}
    for i in range(30)
]


@pytest.mark.asyncio
async def test_run_pine_script_returns_plot_data():
    result = await run_pine_script('//@version=5\nindicator("t")\nplot(ta.sma(close, 5), "SMA5")', BARS)
    assert result["ok"] is True
    assert isinstance(result["plots"]["SMA5"], list)


@pytest.mark.asyncio
async def test_run_pine_script_reports_a_timeout_as_a_structured_error_not_a_hang():
    source = "//@version=5\nindicator(\"t\")\nvar x = 0\nwhile true\n    x := x + 1"
    result = await run_pine_script(source, BARS, timeout_s=0.5)
    assert result["ok"] is False
    assert result["error"]
