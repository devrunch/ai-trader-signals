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


def test_chat_system_prompt_forbids_contradicting_a_user_who_says_something_isnt_showing():
    """Live bug: the user said "its not there" about a Gaussian filter
    indicator, and the analyst replied "I can see the Gaussian filter
    indicator is now displayed... it's working correctly" -- a claim about
    the user's own browser it has no channel to verify. The prompt must
    say plainly that it cannot see the user's screen and must not insist
    otherwise."""
    prompt = chat_system_prompt("RELIANCE", "NSE", 1314.1)
    assert "no visibility into what is currently rendering in the user's browser" in prompt
    assert "never insist it is there or working" in prompt


def test_chat_system_prompt_requires_trying_generate_custom_indicator_before_declining():
    """Live bug: asked for "a wavelet transform indicator that decomposes
    price into trend and cycle", the analyst answered in one round with no
    tool call at all -- "I don't currently have access to a wavelet
    transform indicator in my available tools" -- and suggested EMA/MACD/
    Bollinger instead, despite generate_custom_indicator existing
    specifically for requests like this (and despite the tool actually
    being able to build one, see graph_agent.py's own worked example). The
    prompt must say plainly to try that tool before declaring a gap."""
    prompt = chat_system_prompt("RELIANCE", "NSE", 1314.1)
    assert "call it with the request before concluding you cannot do this" in prompt
