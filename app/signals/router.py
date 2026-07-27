"""
Signals HTTP surface.

Convention: every response body from this service is snake_case, matching the
language. The NestJS side maps to camelCase at its boundary. One endpoint used
to return camelCase keys mixed with snake_case ones in the same object.
"""
from __future__ import annotations

import logging
from functools import lru_cache

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.signals.backtest.evaluator import evaluate
from app.signals.service import SignalService

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Streaming limits. Module level so a test can tighten them — a heartbeat that
# only fires after 15 real seconds is not something a test suite can wait for.
# ---------------------------------------------------------------------------

# An unbounded queue with a slow reader — a browser on a phone, a tab in the
# background — holds every event of every concurrent turn in memory. Well above
# what a normal turn produces (a dozen or two), so it only bites when something
# is genuinely wrong.
SSE_MAX_PENDING = 200

# Long enough that a proxy will not call the connection idle, short enough that
# a browser sees something before it gives up. Most idle timeouts are 30–60s,
# which is inside a normal turn's 55-second budget.
SSE_HEARTBEAT_SECONDS = 15.0


@lru_cache(maxsize=1)
def get_service() -> SignalService:
    """Single service instance, resolved on first request rather than at import.

    A module-level `SignalService()` meant importing the router built an LLM
    client and an SQS client — which also made the router unimportable in a
    test.
    """
    return SignalService()


@router.post("/generate/{symbol}")
async def generate_signal(symbol: str, exchange: str = "NSE",
                          service: SignalService = Depends(get_service)):
    """Manually trigger signal generation for a symbol (dev/testing)."""
    result = await service.generate(symbol.upper(), exchange.upper())
    if result.signal is None:
        # 200 with a typed reason rather than an error: "no signal" is a normal,
        # expected outcome, and the reason is now machine-readable instead of
        # collapsing every cause into one message.
        return {"signal": None, "reason": result.reason}
    return {"signal": result.signal.__dict__, "reason": None}


class ChatBody(BaseModel):
    symbol: str
    exchange: str = "NSE"
    message: str = Field(max_length=4000)
    history: list[dict] | None = Field(default=None, max_length=20)
    user_id: str | None = None


@router.post("/chat")
async def chat(body: ChatBody, service: SignalService = Depends(get_service)):
    """Agentic analysis — the model calls tools (market data, indicators, levels,
    the user's portfolio, risk sizing, backtests, chart drawing) before answering."""
    return await service.chat(
        body.symbol.upper(), body.exchange.upper(), body.message, body.history, body.user_id
    )


@router.post("/chat/stream")
async def chat_stream(body: ChatBody, service: SignalService = Depends(get_service)):
    """The same turn as POST /chat, streamed as Server-Sent Events.

    The buffered endpoint stays — it is simpler for any caller that just wants
    the answer, and it is what the non-streaming clients use. This one exists
    because a turn can legitimately run for the better part of a minute (up to
    `agent_max_tool_rounds` model calls, each able to trigger market-data
    fetches), and a spinner for 55 seconds is indistinguishable from a hang.

    Every event the agent already records is forwarded as it happens; the final
    `result` event carries exactly what POST /chat would have returned, so a
    client can ignore the progress entirely and still be correct.
    """
    import asyncio
    import json as _json

    from fastapi.responses import StreamingResponse

    from app.signals.agent.events import EventKind, TurnRecorder

    queue: asyncio.Queue = asyncio.Queue(maxsize=SSE_MAX_PENDING)
    loop = asyncio.get_running_loop()
    dropped = 0

    def put(payload: dict) -> None:
        """Enqueue, shedding progress rather than blocking the turn.

        The turn must never wait on the reader: a slow client would otherwise
        slow the analysis it is watching. Progress events are the ones worth
        dropping — they are already superseded by the next one — and the final
        `result` never comes through here, so it cannot be lost this way.
        """
        nonlocal dropped
        try:
            queue.put_nowait(payload)
        except asyncio.QueueFull:
            dropped += 1
            logger.warning("SSE consumer is not keeping up; dropped a progress event")

    def emit(event) -> None:
        # Ordering matters: a progress event must reach the queue before the
        # result that follows it. `call_soon_threadsafe` DEFERS the put to the
        # next loop iteration, so on a fast turn the final result overtook every
        # progress event and the client saw only the answer.
        #
        # So: put directly when already on the loop thread (the normal case —
        # the turn is a coroutine), and fall back to the thread-safe path only
        # when genuinely called from another thread, which a tool offloaded to
        # the executor could be. `put_nowait` is not thread-safe, hence both.
        payload = event.to_dict()
        try:
            on_loop = asyncio.get_running_loop() is loop
        except RuntimeError:
            on_loop = False
        if on_loop:
            put(payload)
        else:
            loop.call_soon_threadsafe(put, payload)

    recorder = TurnRecorder(emit)

    async def run() -> None:
        try:
            result = await service.chat(
                body.symbol.upper(), body.exchange.upper(), body.message,
                body.history, body.user_id, recorder=recorder,
            )
            await queue.put({"kind": "result", "at_ms": recorder.elapsed_ms(), "detail": result})
        except Exception as exc:  # noqa: BLE001 - reported to the client, then logged
            logger.exception("Streamed chat failed for %s", body.symbol)
            await queue.put({
                "kind": EventKind.ERROR.value, "at_ms": recorder.elapsed_ms(),
                "label": "The analysis could not be completed", "detail": {"reason": str(exc)[:200]},
            })
        finally:
            await queue.put(None)   # sentinel: the turn is over

    async def stream():
        task = asyncio.create_task(run())
        try:
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=SSE_HEARTBEAT_SECONDS)
                except TimeoutError:
                    # A quiet stretch is normal — one LLM round can take tens of
                    # seconds with nothing to report. Idle proxies close a silent
                    # connection at 30–60s, which is inside a normal turn's
                    # budget, so say something. A comment frame is ignored by
                    # every SSE client, so no consumer needs to know about it.
                    yield ": keep-alive\n\n"
                    continue

                if item is None:
                    break
                yield "data: " + _json.dumps(item, default=str) + "\n\n"
        finally:
            # A browser that navigates away cancels this generator; the turn
            # itself should not outlive the reader it was running for.
            if not task.done():
                task.cancel()

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # nginx and several proxies buffer by default, which turns a live
            # stream back into one delayed blob.
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/brief/generate")
async def generate_brief(publish: bool = True, max_candidates: int = 5):
    from app.signals import brief as brief_mod
    doc = await brief_mod.generate(max_candidates=max_candidates)
    if publish:
        doc["published"] = await brief_mod.publish(doc)
    return doc


@router.get("/global-cues")
async def get_global_cues():
    """Overnight global market cues and the computed market read."""
    from app.market import global_cues
    return await global_cues.collect()


class BacktestBody(BaseModel):
    symbols: list[str] = Field(min_length=1, max_length=100)
    exchange: str = "NSE"
    points_per_symbol: int = Field(default=8, ge=1, le=50)


@router.post("/backtest")
async def backtest(body: BacktestBody, service: SignalService = Depends(get_service)):
    """Walk-forward backtest: runs production signal logic against frozen historical
    slices, evaluated against the real bars that followed. Diagnostic/tuning tool."""
    symbols = [s.upper() for s in body.symbols]
    return await service.backtest_walkforward(symbols, body.exchange.upper(), body.points_per_symbol)


# ---------------------------------------------------------------------------
# Signal outcome evaluation — the single implementation.
#
# The NestJS performance endpoint used to carry its own copy of this logic in
# TypeScript, with different rules (exact stop fills, zero costs). Two numbers
# describing the same thing, disagreeing systematically, both shown to users.
# This endpoint is the one owner; NestJS calls it and keeps only the
# aggregation and presentation, which are legitimately its job.
# ---------------------------------------------------------------------------

class SignalToEvaluate(BaseModel):
    id: str | None = None
    symbol: str
    exchange: str = "NSE"
    direction: str
    entry_price: float
    target_price: float
    stop_loss: float
    generated_at: str


class EvaluateBody(BaseModel):
    signals: list[SignalToEvaluate] = Field(min_length=1, max_length=500)
    interval: str = "5m"
    days: int = Field(default=58, ge=1, le=720)


@router.post("/evaluate")
async def evaluate_signals(body: EvaluateBody, service: SignalService = Depends(get_service)):
    """Resolve stored signals against the bars that actually followed them.

    Bars are fetched once per (symbol, exchange) and reused across every signal
    on that symbol.
    """
    bars_cache: dict[tuple[str, str], pd.DataFrame | None] = {}
    results: list[dict] = []

    for s in body.signals:
        key = (s.symbol.upper(), s.exchange.upper())
        if key not in bars_cache:
            bars_cache[key] = await service.market.get_historical_df(
                key[0], key[1], interval=body.interval, days=body.days
            )
        bars = bars_cache[key]
        if bars is None or bars.empty:
            results.append({"id": s.id, "symbol": s.symbol, "outcome": "NO_DATA",
                            "exit_price": None, "pnl_pct": None})
            continue

        try:
            generated_at = pd.to_datetime(s.generated_at, utc=True)
        except (ValueError, TypeError) as e:
            raise HTTPException(
                status_code=422, detail=f"Unparseable generated_at for {s.symbol}"
            ) from e

        forward = _bars_after(bars, generated_at)
        if forward.empty:
            results.append({"id": s.id, "symbol": s.symbol, "outcome": "OPEN",
                            "exit_price": None, "pnl_pct": 0.0})
            continue

        outcome, exit_price, pnl_pct = evaluate(
            s.direction, s.entry_price, s.target_price, s.stop_loss, forward
        )
        results.append({"id": s.id, "symbol": s.symbol, "outcome": outcome,
                        "exit_price": exit_price, "pnl_pct": pnl_pct})

    resolved = [r for r in results if r["outcome"] in ("TARGET_HIT", "STOP_HIT")]
    wins = [r for r in resolved if r["outcome"] == "TARGET_HIT"]
    return {
        "results": results,
        "summary": {
            "evaluated": len(results),
            "resolved": len(resolved),
            "wins": len(wins),
            "losses": len(resolved) - len(wins),
            "win_rate": round(len(wins) / len(resolved) * 100, 1) if resolved else 0.0,
            "avg_pnl_pct": round(sum(r["pnl_pct"] for r in resolved) / len(resolved), 3) if resolved else 0.0,
        },
    }


def _bars_after(bars: pd.DataFrame, when) -> pd.DataFrame:
    """Bars strictly after `when`. Timezone-naive frames are treated as UTC."""
    idx = bars.index
    if getattr(idx, "tz", None) is None:
        try:
            idx = idx.tz_localize("UTC")
        except (TypeError, AttributeError):
            return bars.iloc[0:0]
    else:
        idx = idx.tz_convert("UTC")
    return bars[idx > when]
