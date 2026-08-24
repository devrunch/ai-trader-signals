from __future__ import annotations

from app.signals.prompts import _currency_for, chat_system_prompt


def test_currency_for_derives_forex_quote_currency_from_the_symbol():
    """A flat exchange->currency map would be wrong for most of this app's
    known FOREX pairs -- only a minority are USD-quoted; AUDCAD prices in
    CAD, EURJPY in JPY, not USD."""
    assert _currency_for("XAUUSD", "FOREX") == "USD"
    assert _currency_for("AUDCAD", "FOREX") == "CAD"
    assert _currency_for("EURJPY", "FOREX") == "JPY"


def test_currency_for_still_uses_the_flat_map_for_every_other_exchange():
    assert _currency_for("RELIANCE", "NSE") == "INR"
    assert _currency_for("AAPL", "NASDAQ") == "USD"


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


def test_chat_system_prompt_points_to_list_chart_indicators_and_still_forbids_overclaiming():
    """Live bug: the user said "its not there" about a Gaussian filter
    indicator, and the analyst replied "I can see the Gaussian filter
    indicator is now displayed... it's working correctly" -- at the time, a
    claim about the user's own browser it had no channel to verify at all.
    The agent now has real (if partial) visibility via list_chart_indicators
    -- the prompt must point to it, but still must not claim more than that
    tool actually covers (manual drawings and exact on-screen rendering
    stay genuinely unverifiable)."""
    prompt = chat_system_prompt("RELIANCE", "NSE", 1314.1)
    assert "list_chart_indicators" in prompt
    assert "not manual drawings" in prompt


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
