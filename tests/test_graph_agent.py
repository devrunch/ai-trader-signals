from __future__ import annotations

from unittest.mock import patch

import pytest

from app.config import get_settings
from app.signals.agent.context import StaticTradingContext
from app.signals.agent.toolbox import AgentToolbox


class _StaticMarket:
    def __init__(self, df): self.df = df
    async def get_historical_df(self, symbol, exchange="NSE", interval="15m", days=30):
        return self.df


def _toolbox(trending_frame):
    return AgentToolbox("RELIANCE", "NSE", trending_frame, context=StaticTradingContext({}),
                        market=_StaticMarket(trending_frame), settings=get_settings())


@pytest.mark.asyncio
async def test_valid_on_first_try(trending_frame):
    box = _toolbox(trending_frame)
    with (
        patch("app.signals.agent.tools.graph_agent._write_formula",
              return_value="result = line(ema(close, 20))"),
        patch("app.signals.agent.tools.graph_agent._validate_via_node",
              return_value={"valid": True, "outputType": "line"}),
    ):
        result = await box.execute("generate_custom_indicator", {"description": "20 EMA", "label": "EMA 20"})

    assert result["created"].startswith("DIA_CUSTOM_")
    assert box.results["custom_indicators"][0]["source"] == "result = line(ema(close, 20))"
    assert box.results["custom_indicators"][0]["outputName"] == "result"
    assert box.results["custom_indicators"][0]["displayLabel"] == "EMA 20"


@pytest.mark.asyncio
async def test_invalid_then_valid_on_retry(trending_frame):
    box = _toolbox(trending_frame)
    write_calls = []

    async def fake_write(description: str, feedback: str | None = None) -> str:
        write_calls.append(feedback)
        return "result = line(ema(close, 20))" if feedback else "not diascript at all"

    async def fake_validate(source: str, output_name: str) -> dict:
        if source == "not diascript at all":
            return {"valid": False, "error": {"message": "unexpected token"}}
        return {"valid": True, "outputType": "line"}

    with (
        patch("app.signals.agent.tools.graph_agent._write_formula", new=fake_write),
        patch("app.signals.agent.tools.graph_agent._validate_via_node", new=fake_validate),
    ):
        result = await box.execute("generate_custom_indicator", {"description": "20 EMA"})

    assert len(write_calls) == 2
    assert write_calls[0] is None
    assert write_calls[1] == "unexpected token"
    assert "created" in result


@pytest.mark.asyncio
async def test_invalid_twice_returns_error_without_touching_results(trending_frame):
    box = _toolbox(trending_frame)

    async def fake_write(description: str, feedback: str | None = None) -> str:
        return "still not diascript"

    async def fake_validate(source: str, output_name: str) -> dict:
        return {"valid": False, "error": {"message": "unexpected token"}}

    with (
        patch("app.signals.agent.tools.graph_agent._write_formula", new=fake_write),
        patch("app.signals.agent.tools.graph_agent._validate_via_node", new=fake_validate),
    ):
        result = await box.execute("generate_custom_indicator", {"description": "bad request"})

    assert "error" in result
    assert "custom_indicators" not in box.results


@pytest.mark.asyncio
async def test_missing_description_short_circuits(trending_frame):
    box = _toolbox(trending_frame)
    result = await box.execute("generate_custom_indicator", {})
    assert "error" in result
    assert "custom_indicators" not in box.results
