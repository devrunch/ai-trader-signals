"""
Prompt authoring for the signal engine and the chat agent.

Kept apart from orchestration so that changing what the model is told is a
diff in one small file rather than a diff inside a 200-line method, and so the
backtest and live paths demonstrably share the same text.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from app.config import get_settings

# Tool the signal LLM can call mid-reasoning to inspect raw price action at a
# finer (or coarser) resolution than the pre-computed 15m indicators, before
# committing to a direction.
CANDLE_TOOL = {
    "type": "function",
    "function": {
        "name": "get_candles",
        "description": "Fetch recent OHLCV candles for the stock being analysed, at a chosen timeframe, to inspect actual price action before deciding.",
        "parameters": {
            "type": "object",
            "properties": {
                "interval": {"type": "string", "enum": ["1m", "5m", "15m", "1h"], "description": "Candle timeframe"},
                "count": {"type": "integer", "description": "Number of most recent candles (max 50)"},
            },
            "required": ["interval"],
        },
    },
}

SIGNAL_SYSTEM = (
    "You are an expert Indian equity trader. You may call get_candles to inspect "
    "price action before deciding. When done, respond with ONLY the final valid "
    "JSON object — no markdown, no explanation."
)

SIGNAL_SYSTEM_NO_TOOLS = (
    "You are an expert Indian equity trader. Respond ONLY with valid JSON — "
    "no markdown, no explanation."
)


def signal_prompt(
    symbol: str,
    exchange: str,
    df: pd.DataFrame,
    indicators: Mapping[str, Any],
    sentiment: Mapping[str, Any],
    atr: float,
    sr: list[dict],
    tl: dict | None,
) -> str:
    """The user-turn prompt for one signal decision.

    Every number in here is computed deterministically and passed in — the model
    places levels relative to REAL structure rather than inventing them. Mirrors
    the chat agent's design: maths computes geometry, the LLM reasons about
    which levels matter.
    """
    settings = get_settings()
    ltp = indicators["ltp"]
    sr_lines = "\n".join(
        f"- {level['kind'].upper()} ₹{level['value']} (strength {level['strength']})" for level in sr
    ) or "- none detected"
    tl_line = f"{tl['direction']}-trend line active" if tl else "no clear trend line"
    supertrend_word = (
        "Bullish" if indicators.get("supertrend_dir") == 1
        else "Bearish" if indicators.get("supertrend_dir") == -1
        else "unavailable"
    )

    return f"""Analyse {symbol} ({exchange}) for an intraday trade.

CURRENT PRICE: {ltp}
ATR(14): {atr}  — use this to size your stop-loss. Stop distance MUST be between {settings.min_stop_atr_multiple}x and {settings.max_stop_atr_multiple}x ATR from entry ({round(atr * settings.min_stop_atr_multiple, 2)} to {round(atr * settings.max_stop_atr_multiple, 2)} price points). A stop tighter than {settings.min_stop_atr_multiple}x ATR will be rejected — it sits inside normal noise and will stop out regardless of your thesis.

TECHNICAL INDICATORS:
- RSI(14): {indicators.get('rsi')}
- MACD: {indicators.get('macd')} | Signal: {indicators.get('macd_signal')}
- EMA20: {indicators.get('ema20')} | EMA50: {indicators.get('ema50')}
- ADX(14): {indicators.get('adx')} (already filtered for trend strength — you are only being asked because this regime is trending)
- SuperTrend direction: {supertrend_word}
- VWAP: {indicators.get('vwap')}

COMPUTED SUPPORT/RESISTANCE (place targets/stops relative to these, not arbitrary numbers):
{sr_lines}
TREND: {tl_line}

NEWS SENTIMENT: {sentiment['label'].upper()} (score: {sentiment['score']}, from {sentiment['headlines_count']} headlines)

RECENT OHLCV (last 5 candles, 15m):
{df[['open','high','low','close','volume']].tail(5).to_string()}

You have a get_candles tool — call it to inspect actual recent price action at 1m, 5m, 15m, or 1h resolution before deciding, e.g. to check for a very recent reversal the 15m indicators above wouldn't yet show, or to see whether the 1h trend agrees with your read. Use it a few times if useful; you don't have to.

When ready, respond with ONLY a valid JSON object — no markdown, no extra text, no tool call:
{{"signal_type":"BUY"|"HOLD","confidence":0.0-1.0,"entry_price":float,"target_price":float,"stop_loss":float,"reasoning":"2-3 sentences"}}

Rules: LONG ONLY — this product cannot short, so if the setup is bearish return HOLD rather than SELL; a SELL is rejected server-side and the call is wasted. BUY only when confidence >= {settings.confidence_threshold}. Minimum reward-to-risk = {settings.min_reward_risk}. Stop-loss sized to ATR as instructed above. Do not propose a BUY into an RSI above {settings.max_buy_rsi:.0f} — that is an already-extended move and is rejected server-side. Intraday positions only, squared off by 15:20 IST."""


# Exchange -> what the agent should call its own numbers. Everything computed
# server-side (indicators, levels, backtests) is exchange-agnostic maths over
# OHLCV bars, so widening this list is safe on its own — it only changes how a
# price is NAMED, never how one is calculated.
_CURRENCY = {"NSE": "INR", "BSE": "INR", "NASDAQ": "USD", "NYSE": "USD"}


def chat_system_prompt(symbol: str, exchange: str, last_price: float) -> str:
    """System turn for the conversational agent."""
    currency = _CURRENCY.get(exchange.upper(), "")
    return (
        "You are an experienced trading assistant embedded in a charting terminal, "
        f"currently looking at {symbol} ({exchange}), last price {currency} {last_price}.\n\n"
        "You have tools. Use them before answering - never estimate a number you could look up, "
        "and never invent a price level, position size, or statistic. If a question depends on the "
        "user's account (affordability, sizing, exposure, risk), call the portfolio tools first.\n\n"
        "Match the effort to the question. A greeting, a thank-you, or a question about what you "
        "can do needs no tools at all - answer it in a sentence and stop. Tools cost the user real "
        "money and several seconds each, so a one-line reply that ran four of them is a worse "
        "answer than the same reply that ran none.\n\n"
        "How to answer:\n"
        "- Be concise and concrete. Traders want numbers and a clear read, not lectures.\n"
        "- Always state sample sizes and caveats for backtests; small samples are unreliable.\n"
        "- Never claim to predict prices. If asked, say so plainly and offer what you can give: "
        "structure, levels, risk and historical behaviour.\n"
        "- If the user challenges you, engage honestly rather than defending a weak position. "
        "Overall signal accuracy currently sits near breakeven - say so if asked.\n"
        "- Push back on reckless requests (oversized positions, no stop) with a concrete safer alternative.\n"
        "- You cannot place orders or set alerts yet; say so if asked.\n\n"
        "IMPORTANT: do not write any part of your answer while you are still calling tools. "
        "Gather everything you need first, then write your complete answer in a single final message."
    )


BRIEF_SYSTEM = (
    "You are a market analyst writing a concise pre-market brief. "
    "Never invent numbers. Never promise returns."
)


def extract_json_text(text: str) -> str:
    """Strip a markdown code fence the model may have wrapped its JSON in."""
    text = (text or "").strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) > 1:
            text = parts[1]
        if text.startswith("json"):
            text = text[4:]
    return text.strip()
