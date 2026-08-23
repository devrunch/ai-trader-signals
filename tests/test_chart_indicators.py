"""Reactive chart-indicator tools -- list_chart_indicators, set_indicator_params,
edit_indicator_source, remove_chart_indicator (chart_indicators.py).

Replaces the old add_chart_indicator/CHART_INDICATORS tool (chart.py), which
toggled names from a fixed klinecharts-era catalog the frontend's
applyChartIndicators has silently discarded since the Pine migration --
confirmed live, see chart_indicators.py's own module docstring.

set_indicator_params and edit_indicator_source both run their change against
the real sandbox (app/pine_sandbox/) before accepting it -- these tests prove
that gate actually rejects a bad change, not just that the mocks were wired
right.
"""
from __future__ import annotations

import pytest

from app.config import get_settings
from app.signals.agent.context import StaticTradingContext
from app.signals.agent.toolbox import AgentToolbox


class _StaticMarket:
    def __init__(self, df):
        self.df = df

    async def get_historical_df(self, symbol, exchange="NSE", interval="15m", days=30):
        return self.df


SMA_SOURCE = (
    '//@version=5\nindicator("t")\n'
    'length = input.int(20, minval=1, title="Length")\n'
    'plot(ta.sma(close, length), "SMA")'
)

ATTACHED = [
    {"id": "ind1", "source": SMA_SOURCE, "label": "SMA", "pane": "main", "params": {}},
    {"id": "ind2", "source": '//@version=5\nindicator("t")\nplot(ta.rsi(close, 14), "RSI")',
     "label": "RSI", "pane": "sub"},
]


def _toolbox(trending_frame, chart_indicators=None, chart_interval=None):
    return AgentToolbox(
        "RELIANCE", "NSE", trending_frame, context=StaticTradingContext({}),
        market=_StaticMarket(trending_frame), settings=get_settings(),
        chart_indicators=chart_indicators, chart_interval=chart_interval,
    )


class TestListChartIndicators:
    @pytest.mark.asyncio
    async def test_reports_exactly_what_was_pushed_from_the_browser(self, trending_frame):
        box = _toolbox(trending_frame, chart_indicators=ATTACHED, chart_interval="5m")
        result = await box.execute("list_chart_indicators", {})
        assert result["count"] == 2
        assert result["interval"] == "5m"
        assert [i["id"] for i in result["indicators"]] == ["ind1", "ind2"]

    @pytest.mark.asyncio
    async def test_empty_when_nothing_was_pushed_yet(self, trending_frame):
        box = _toolbox(trending_frame)
        result = await box.execute("list_chart_indicators", {})
        assert result["count"] == 0
        assert result["indicators"] == []


class TestSetIndicatorParams:
    @pytest.mark.asyncio
    async def test_real_override_is_accepted_and_recorded(self, trending_frame):
        box = _toolbox(trending_frame, chart_indicators=ATTACHED)
        result = await box.execute("set_indicator_params", {"id": "ind1", "params": {"length": 5}})
        assert result["updated"] == "ind1"
        assert box.results["indicator_changes"]["update"] == [{"id": "ind1", "params": {"length": 5}}]

    @pytest.mark.asyncio
    async def test_unknown_id_is_rejected_without_touching_results(self, trending_frame):
        box = _toolbox(trending_frame, chart_indicators=ATTACHED)
        result = await box.execute("set_indicator_params", {"id": "nope", "params": {"length": 5}})
        assert "error" in result
        assert "indicator_changes" not in box.results

    @pytest.mark.asyncio
    async def test_missing_params_is_rejected(self, trending_frame):
        box = _toolbox(trending_frame, chart_indicators=ATTACHED)
        result = await box.execute("set_indicator_params", {"id": "ind1"})
        assert "error" in result
        assert "indicator_changes" not in box.results


class TestEditIndicatorSource:
    @pytest.mark.asyncio
    async def test_valid_edit_is_accepted_and_recorded(self, trending_frame):
        box = _toolbox(trending_frame, chart_indicators=ATTACHED)
        new_source = '//@version=5\nindicator("t")\nplot(ta.ema(close, 20), "EMA20")'
        result = await box.execute("edit_indicator_source", {"id": "ind1", "source": new_source})
        assert result["edited"] == "ind1"
        assert box.results["indicator_changes"]["edit_source"] == [{"id": "ind1", "source": new_source}]

    @pytest.mark.asyncio
    async def test_invalid_source_is_rejected_without_touching_results(self, trending_frame):
        box = _toolbox(trending_frame, chart_indicators=ATTACHED)
        result = await box.execute("edit_indicator_source", {"id": "ind1", "source": "this is not pine @#$%"})
        assert "error" in result
        assert "indicator_changes" not in box.results

    @pytest.mark.asyncio
    async def test_still_forbidden_call_is_rejected(self, trending_frame):
        box = _toolbox(trending_frame, chart_indicators=ATTACHED)
        bad = '//@version=5\nindicator("t")\nbgcolor(close > open ? color.green : na)'
        result = await box.execute("edit_indicator_source", {"id": "ind1", "source": bad})
        assert "error" in result
        assert "bgcolor(" in result["error"]

    @pytest.mark.asyncio
    async def test_unknown_id_is_rejected(self, trending_frame):
        box = _toolbox(trending_frame, chart_indicators=ATTACHED)
        result = await box.execute("edit_indicator_source", {"id": "nope", "source": SMA_SOURCE})
        assert "error" in result
        assert "indicator_changes" not in box.results


class TestRemoveChartIndicator:
    @pytest.mark.asyncio
    async def test_known_id_is_recorded(self, trending_frame):
        box = _toolbox(trending_frame, chart_indicators=ATTACHED)
        result = await box.execute("remove_chart_indicator", {"id": "ind2"})
        assert result["removed"] == "ind2"
        assert result["label"] == "RSI"
        assert box.results["indicator_changes"]["remove"] == ["ind2"]

    @pytest.mark.asyncio
    async def test_unknown_id_is_rejected(self, trending_frame):
        box = _toolbox(trending_frame, chart_indicators=ATTACHED)
        result = await box.execute("remove_chart_indicator", {"id": "nope"})
        assert "error" in result
        assert "indicator_changes" not in box.results
