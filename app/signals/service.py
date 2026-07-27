"""
SignalService — orchestration, and only orchestration.

Flow per symbol:
  1. Fetch OHLCV data (via the provider router)
  2. Compute technical indicators           -> app/signals/indicators.py
  3. Regime filter (ADX)
  4. News sentiment                         -> app/signals/sentiment.py
  5. Ask the LLM for a structured decision  -> app/signals/prompts.py + app/llm
  6. Validate it                            -> app/signals/validation.py
  7. Publish                                -> app/signals/publisher.py

This class used to be 759 lines holding seven unrelated responsibilities,
including its own copy of the indicator engine and its own NewsAPI/FinBERT
clients. Both copies had already drifted from the originals next door. It is
now ~200 lines and takes its collaborators as constructor arguments, so it can
be tested against fakes with no AWS account and no LLM endpoint.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any, NamedTuple

import pandas as pd

from app.config import get_settings
from app.llm.client import LlmClient, get_llm
from app.market.intervals import default_days
from app.market.providers.registry import market_data_router
from app.signals import analysis, prompts, validation
from app.signals import indicators as ind_mod
from app.signals import sentiment as sentiment_mod
from app.signals.publisher import NullPublisher, SqsSignalPublisher
from app.signals.types import GeneratedSignal, SignalType

logger = logging.getLogger(__name__)

# Re-exported for callers that imported them from here historically.
__all__ = ["SignalService", "GeneratedSignal", "SignalType", "SignalResult"]

# Bars the signal engine reasons over. 60 days of 15m is ~1,500 bars, enough
# warm-up for every indicator in the set including EMA200.
SIGNAL_INTERVAL = "15m"
SIGNAL_DAYS = 60


class SignalResult(NamedTuple):
    """A signal, or a machine-readable reason there isn't one.

    `None` used to be the return value for "insufficient data", "ADX
    unavailable", "below confidence threshold", "rejected by a gate" AND "the
    LLM endpoint is down". The screener's counter therefore could not tell
    "market was quiet" from "Bedrock is broken" — both showed as `signals: 0`.
    """
    signal: GeneratedSignal | None
    reason: str | None = None


class SignalService:
    def __init__(self, llm: LlmClient | None = None, publisher=None, market=market_data_router, settings=None):
        self.settings = settings or get_settings()
        self.llm = llm or get_llm()
        self.publisher = publisher if publisher is not None else SqsSignalPublisher(self.settings)
        self.market = market

    # ------------------------------------------------------------------
    # Chat agent
    # ------------------------------------------------------------------
    async def chat(
        self, symbol: str, exchange: str, message: str,
        history: list[dict] | None = None, user_id: str | None = None,
        recorder=None, store=None,
    ) -> dict:
        """`recorder` is optional: pass a TurnRecorder wired to a live emitter to
        stream progress (see POST /signals/chat/stream). Without one the turn
        still records its events, into a recorder nobody is listening to."""
        from app.signals.agent.orchestrator import run_chat
        from app.signals.agent.turn import new_turn_id

        df = await self._fetch_ohlcv(symbol, exchange)
        if df is None or len(df) < 20:
            # Same shape as a completed turn — a caller should never have to
            # branch on which kind of answer it received.
            return {"turn_id": new_turn_id(), "symbol": symbol, "exchange": exchange,
                    "message": f"I don't have enough chart data for {symbol} to analyse right now.",
                    "drawings": [], "results": {}, "events": [],
                    "usage": {}, "stop_reason": "no_data"}
        return await run_chat(
            self.llm, symbol, exchange, message, df,
            history=history, user_id=user_id, settings=self.settings,
            recorder=recorder, store=store,
        )

    # ------------------------------------------------------------------
    # Public entry point — Celery tasks call only this
    # ------------------------------------------------------------------
    async def generate(self, symbol: str, exchange: str = "NSE", publish: bool = True) -> SignalResult:
        """Generate a signal, or return the reason there isn't one.

        `publish=False` keeps it out of the live SQS feed — used by the
        pre-market brief, whose 06:30 signals are priced off the PREVIOUS
        session's close and would otherwise be scored as live intraday signals
        with the entire overnight gap folded into their P&L.
        """
        df = await self._fetch_ohlcv(symbol, exchange)
        if df is None or len(df) < 50:
            return SignalResult(None, "insufficient_data")
        if newest_bar_is_stale(df):
            logger.info("Signal for %s skipped — newest closed bar is stale", symbol)
            return SignalResult(None, "stale_data")

        try:
            indicators = ind_mod.compute(df, ind_mod.SIGNAL_SET)
        except Exception:
            logger.exception("Indicator computation failed for %s", symbol)
            return SignalResult(None, "indicator_error")

        # Regime filter — skip choppy/range-bound conditions before paying for
        # FinBERT + the LLM call. Weak ADX means no reliable trend to trade.
        # Fail CLOSED: a risk filter that disables itself when its input is
        # missing is worse than no filter, because it is invisible.
        adx = indicators.get("adx")
        if adx is None:
            logger.info("Signal for %s skipped — ADX unavailable, cannot confirm regime", symbol)
            return SignalResult(None, "adx_unavailable")
        if adx < self.settings.min_adx_trend:
            logger.info("Signal for %s skipped — ADX %.2f below trend threshold %.2f",
                        symbol, adx, self.settings.min_adx_trend)
            return SignalResult(None, "regime_filtered")

        news_sentiment = await sentiment_mod.symbol_sentiment(symbol)

        try:
            signal, reason = await self._ask_llm(symbol, exchange, df, indicators, news_sentiment, live=True)
        except Exception:
            logger.exception("LLM call failed for %s", symbol)
            return SignalResult(None, "llm_error")

        if signal is None:
            return SignalResult(None, reason)
        if signal.confidence < self.settings.confidence_threshold:
            logger.info("Signal for %s below threshold (%.2f)", symbol, signal.confidence)
            return SignalResult(None, "low_confidence")

        if publish:
            try:
                await self.publisher.publish(signal)
            except Exception:
                # A publish failure must not discard a valid signal — the caller
                # still gets it, and the failure is loud.
                logger.exception("Publishing signal for %s failed", symbol)
        return SignalResult(signal, None)

    async def generate_signal(self, symbol: str, exchange: str = "NSE",
                              publish: bool = True) -> GeneratedSignal | None:
        """Backwards-compatible wrapper returning just the signal or None."""
        return (await self.generate(symbol, exchange, publish)).signal

    # ------------------------------------------------------------------
    # OHLCV fetch
    # ------------------------------------------------------------------
    async def _fetch_ohlcv(self, symbol: str, exchange: str) -> pd.DataFrame | None:
        df = await self.market.get_historical_df(
            symbol, exchange, interval=SIGNAL_INTERVAL, days=SIGNAL_DAYS
        )
        return drop_forming_bar(df)

    # ------------------------------------------------------------------
    # LLM decision
    # ------------------------------------------------------------------
    async def _candles_tool_result(
        self, symbol: str, exchange: str, base_df: pd.DataFrame,
        interval: str, count: int, live: bool,
    ) -> dict[str, object]:
        count = max(1, min(int(count or 20), 50))
        if not live:
            # Backtesting must never fetch fresh data — that would leak information
            # from beyond the frozen point-in-time slice into the model's decision,
            # invalidating every backtest result. Serve only from the already-frozen
            # base_df, regardless of what interval was requested.
            return {
                "note": "Backtest mode: only the 15m data already frozen at this historical point is available, regardless of requested interval.",
                "interval_served": SIGNAL_INTERVAL,
                "candles": _candles(base_df.tail(count)),
            }

        if interval == SIGNAL_INTERVAL:
            tail = base_df.tail(count)
        else:
            fresh = await self.market.get_historical_df(
                symbol, exchange, interval=interval, days=default_days(interval)
            )
            if fresh is None or fresh.empty:
                return {"error": f"No {interval} data available for {symbol}"}
            tail = fresh.tail(count)
        return {"interval_served": interval, "candles": _candles(tail)}

    async def _ask_llm(
        self, symbol: str, exchange: str, df: pd.DataFrame,
        indicators: Mapping[str, Any], news_sentiment: dict, live: bool = True,
    ) -> tuple[GeneratedSignal | None, str | None]:
        ltp = indicators["ltp"]
        atr = indicators.get("atr") or (ltp * 0.005)  # fallback: 0.5% if ATR unavailable

        user_prompt = prompts.signal_prompt(
            symbol, exchange, df, indicators, news_sentiment, atr,
            sr=analysis.support_resistance(df),
            tl=analysis.trendline(df),
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": prompts.SIGNAL_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]

        data = None
        try:
            for _ in range(self.settings.signal_max_tool_rounds):  # cap tool-call rounds — never loop unbounded
                resp = self.llm.chat(
                    temperature=0, max_tokens=700,
                    messages=messages, tools=[prompts.CANDLE_TOOL], tool_choice="auto",
                )
                msg = resp.choices[0].message
                tool_calls = getattr(msg, "tool_calls", None)

                if not tool_calls:
                    data = json.loads(prompts.extract_json_text(msg.content))
                    break

                messages.append({"role": "assistant", "content": msg.content, "tool_calls": [
                    {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in tool_calls
                ]})
                for tc in tool_calls:
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    result = await self._candles_tool_result(
                        symbol, exchange, df, args.get("interval", SIGNAL_INTERVAL), args.get("count", 20), live
                    )
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
        except Exception as e:
            # Tool calling may not be supported by the underlying model — fall back
            # to a single plain call rather than failing the signal entirely.
            logger.warning("Tool-calling LLM call failed for %s (%s) — retrying without tools", symbol, e)
            try:
                resp = self.llm.chat(
                    temperature=0, max_tokens=512,
                    messages=[
                        {"role": "system", "content": prompts.SIGNAL_SYSTEM_NO_TOOLS},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                data = json.loads(prompts.extract_json_text(resp.choices[0].message.content))
            except Exception as e2:
                logger.warning("Fallback LLM call also failed for %s: %s", symbol, e2)
                return None, "llm_error"

        if data is None:
            logger.warning("LLM tool-call loop for %s exhausted without a final answer", symbol)
            return None, "llm_no_answer"

        ok, reason = validation.validate(data, indicators, atr, ltp, self.settings)
        if not ok:
            logger.info("Signal for %s rejected — %s", symbol, reason)
            return None, f"rejected:{reason}"

        try:
            return GeneratedSignal(
                symbol=symbol,
                exchange=exchange,
                signal_type=SignalType(data["signal_type"]),
                confidence=float(data["confidence"]),
                entry_price=float(data["entry_price"]),
                target_price=float(data["target_price"]),
                stop_loss=float(data["stop_loss"]),
                reasoning=data["reasoning"],
                indicators=dict(indicators),
            ), None
        except (KeyError, ValueError) as e:
            logger.warning("LLM response parse error for %s: %s", symbol, e)
            return None, "parse_error"

    # ------------------------------------------------------------------
    # Backtest — delegates to app/signals/backtest/
    # ------------------------------------------------------------------
    async def backtest_walkforward(
        self, symbols: list[str], exchange: str = "NSE",
        points_per_symbol: int = 8, warmup_bars: int = 60, forward_bars: int = 40,
    ) -> dict:
        from app.signals.backtest.runner import backtest_walkforward
        return await backtest_walkforward(
            self, symbols, exchange, points_per_symbol, warmup_bars, forward_bars
        )


INTERVAL_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "1d": 1440}


def drop_forming_bar(df: pd.DataFrame | None, interval: str = SIGNAL_INTERVAL) -> pd.DataFrame | None:
    """Drop the final bar unless it is provably closed.

    Indicators read `iloc[-1]`. On an intraday feed that row is a
    PARTIALLY-FORMED bar: a 15-minute candle that may be four minutes old, on a
    feed that is itself 15-20 minutes delayed. Its high, low and close will all
    still change. Computing RSI, MACD or SuperTrend on it is the intraday
    equivalent of reading a partially-written file, and a signal can fire on a
    "breakout" that unwinds before the bar completes.

    A bar is treated as closed only once a full interval has elapsed since it
    opened. On a non-datetime index we cannot tell, so we drop the last bar
    anyway — losing one bar of history is cheap; acting on a forming one is not.
    """
    if df is None or df.empty:
        return df

    # Must be a genuine DatetimeIndex. `pd.Timestamp(4)` on an integer index
    # succeeds — it reads 4 as nanoseconds since the epoch — so the bar looks
    # decades old, is judged closed, and the forming bar is kept. That is the
    # precise silent-wrong-answer this function exists to prevent.
    if not isinstance(df.index, pd.DatetimeIndex):
        return df.iloc[:-1]

    minutes = INTERVAL_MINUTES.get(interval, 15)
    try:
        last_open = df.index[-1]
        now = pd.Timestamp.now(tz=last_open.tz) if last_open.tz is not None else pd.Timestamp.now()
        closed = (now - last_open) >= pd.Timedelta(minutes=minutes)
    except (TypeError, ValueError, AttributeError):
        closed = False

    return df if closed else df.iloc[:-1]


def newest_bar_is_stale(df: pd.DataFrame, interval: str = SIGNAL_INTERVAL,
                        max_intervals: int = 2) -> bool:
    """True when even the newest CLOSED bar is too old to act on.

    Guards the case the delayed feed goes quiet: without it the engine happily
    computes a fresh-looking signal from prices that stopped updating an hour
    ago. Returns False on a non-datetime index rather than blocking synthetic
    frames used in tests and backtests.
    """
    if df is None or df.empty:
        return True
    if not isinstance(df.index, pd.DatetimeIndex):
        return False
    minutes = INTERVAL_MINUTES.get(interval, 15)
    try:
        last_open = df.index[-1]
        if last_open.tz is None:
            return False
        age = pd.Timestamp.now(tz=last_open.tz) - last_open
    except (TypeError, ValueError, AttributeError):
        return False
    return age > pd.Timedelta(minutes=minutes * (max_intervals + 1))


def _candles(frame: pd.DataFrame) -> list[dict]:
    return [
        {"time": str(idx), "open": round(float(r.open), 2), "high": round(float(r.high), 2),
         "low": round(float(r.low), 2), "close": round(float(r.close), 2), "volume": round(float(r.volume), 0)}
        for idx, r in frame.iterrows()
    ]


def backtest_publisher() -> NullPublisher:
    """A publisher that discards — for the brief and the backtest."""
    return NullPublisher()
