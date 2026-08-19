from __future__ import annotations

from app.signals.prompts import chat_system_prompt


def test_chat_system_prompt_forbids_describing_chart_changes_beyond_the_tool_result():
    """Live bug: after a real generate_custom_indicator call that produced
    exactly one marker, the analyst's final answer still claimed "3 blocks
    marked (green rectangles)" and "BOS marked (red arrows)" -- richer,
    more specific chart detail than any tool reported creating. The prompt
    must forbid inventing that detail, the same way it already forbids
    inventing a price level or statistic."""
    prompt = chat_system_prompt("RELIANCE", "NSE", 1314.1)
    assert "describe ONLY" in prompt
    assert "Never invent additional visual detail" in prompt
