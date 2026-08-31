"""
Morning Brief generation.

Runs as an overnight batch (~06:30 IST) — after the US close, before the Indian
open. Produces the day's plan as a stored document so the user opens the app to
a finished answer rather than a prompt.

Division of labour, deliberately:
  * All numbers (cues, betas, entries, stops, targets) come from data and maths.
  * The LLM writes only the narrative connecting them.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime

import httpx

from app.config import get_settings
from app.llm.client import LlmClient, get_llm
from app.market import global_cues, macro_events
from app.market.global_cues import IST
from app.signals import prompts

logger = logging.getLogger(__name__)

# Liquid, widely-held names — a shared brief should be the same for everyone.
DEFAULT_UNIVERSE = [
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK",
    "KOTAKBANK", "ITC", "LT", "BPCL", "IOC", "ONGC", "NTPC", "POWERGRID",
    "MARUTI", "TATAMOTORS", "TATASTEEL", "JSWSTEEL", "HINDALCO",
    "SUNPHARMA", "CIPLA", "WIPRO", "HCLTECH", "TITAN",
]

# Which driver each cue maps to when explaining a candidate
_DRIVER_LABEL = {"^IXIC": "NASDAQ", "BZ=F": "crude", "USDINR=X": "USD/INR"}


async def generate(
    universe: list[str] | None = None,
    max_candidates: int = 5,
    service=None,
    llm: LlmClient | None = None,
) -> dict:
    """Build today's morning brief.

    Collaborators are parameters, not imports. `generate` used to construct a
    whole `SignalService` — building an SQS client it never used — purely to
    reach into it for `_get_llm()`, a private method on another module's class.
    """
    settings = get_settings()
    if service is None:
        from app.signals.publisher import NullPublisher
        from app.signals.service import SignalService
        # NullPublisher, not publish=False: brief levels are derived from the
        # previous session's close and must be revalidated at the open, never
        # scored as live intraday signals.
        service = SignalService(publisher=NullPublisher())
    llm = llm or get_llm()

    universe = universe or _universe(settings)

    cues = await global_cues.collect()
    summary = cues["summary"]

    # The WHY behind the cues above -- real US macro releases (FRED) and
    # general macro headlines (yfinance), run concurrently with the
    # per-candidate work below rather than blocking ahead of it. Both
    # degrade to an empty list on any failure (no key, vendor down) rather
    # than raising -- a missing "why" is a worse brief, not a broken one.
    events_task = asyncio.gather(
        macro_events.fred_releases(), macro_events.yfinance_headlines(),
        return_exceptions=True,
    )

    # Which overnight driver actually moved enough to matter today?
    movers = _dominant_drivers(cues["cues"], settings.brief_driver_move_threshold)

    # Bounded concurrency. Sequentially this loop was 25 x (fetch + sentiment +
    # up to 4 LLM rounds + a sensitivity call that itself does 3 more
    # downloads) — 5-10 minutes, against a 06:30 start and an 07:00 deadline.
    # Bounded rather than unbounded because yfinance rate-limits aggressively.
    sem = asyncio.Semaphore(settings.brief_concurrency)

    async def candidate_for(symbol: str) -> dict | None:
        async with sem:
            try:
                signal = await service.generate_signal(symbol, "NSE", publish=False)
                if signal is None:
                    return None
                sens = await global_cues.sensitivity(symbol, "NSE", drivers=list(_DRIVER_LABEL))
                return {
                    "symbol": symbol,
                    "direction": signal.signal_type.value,
                    "confidence": round(signal.confidence, 2),
                    "entry": signal.entry_price,
                    "target": signal.target_price,
                    "stop": signal.stop_loss,
                    "reward_risk": _rr(signal),
                    "reasoning": signal.reasoning,
                    "indicators": signal.indicators,
                    "global_context": _global_context(
                        sens, movers, signal.signal_type.value, settings.brief_min_expected_move
                    ),
                }
            except Exception:
                logger.exception("Brief candidate failed for %s", symbol)
                return None

    candidates = [c for c in await asyncio.gather(*(candidate_for(s) for s in universe)) if c]

    # Rank by a WEIGHTED BLEND of conviction and global alignment.
    #
    # This used to sort lexicographically on alignment then confidence. A
    # lexicographic sort on floats means the first key decides outright — the
    # second is consulted only on an exact tie, which floats never produce. So
    # ranking was entirely the alignment score, which is the *least* reliable
    # input in the system: it rests on correlations measured over ~25 days that
    # are not statistically distinguishable from zero. The brief was led by its
    # noisiest signal.
    #
    # Alignment still matters — it is the only thing that catches a trade
    # fighting the overnight tape, and a negative score (CONFLICT) still pushes
    # a candidate down. It is just no longer allowed to outvote conviction.
    candidates.sort(key=lambda c: -_rank_score(c))
    candidates = _cap_per_sector(candidates, settings.brief_max_per_sector)[:max_candidates]

    fred_result, yf_news_result = await events_task
    if isinstance(fred_result, BaseException):
        logger.exception("fred_releases raised", exc_info=fred_result)
        fred_result = None
    if isinstance(yf_news_result, BaseException):
        logger.exception("yfinance_headlines raised", exc_info=yf_news_result)
        yf_news_result = []
    # fred_result stays None (not []) when FRED wasn't configured/reachable --
    # collapsing it to [] here would make "couldn't check" indistinguishable
    # from "checked, nothing new" for anything reading the stored brief
    # document, the exact ambiguity _events_block() exists to avoid in the
    # LLM's own prompt.
    events = {"fred_releases": fred_result, "headlines": yf_news_result}

    narrative = await _narrative(llm, cues, candidates, events)

    return {
        "date": datetime.now(IST).strftime("%Y-%m-%d"),
        "generated_at": datetime.now(IST).isoformat(),
        "global_cues": cues["cues"],
        "market_read": {
            "bias": summary["bias"],
            "label": summary["label"],
            # "signal_strength", not "confidence" — it measures how strongly
            # the overnight cues point one way, not how often they are right.
            # Nothing has ever measured the latter.
            "signal_strength": summary["signal_strength"],
            "confidence": summary["confidence"],  # legacy key, same value
            "notes": summary["notes"],
            "us_avg_pct": summary["us_avg_pct"],
            "asia_avg_pct": summary["asia_avg_pct"],
        },
        "narrative": narrative,
        "macro_events": events,
        "candidates": candidates,
        "universe_size": len(universe),
        "disclaimer": (
            "Analysis and candidates, not a promise of returns. Levels are derived from "
            "the previous session's close and must be revalidated after the open — "
            "treat them as reference, not as live orders. Costs and slippage are not "
            "modelled. Signal accuracy is measured and published — see the performance "
            "view before sizing up."
        ),
    }


def _universe(settings) -> list[str]:
    """DEFAULT_UNIVERSE unless BRIEF_UNIVERSE overrides it — a product decision
    that changes often should not need a rebuild of two containers."""
    raw = (settings.brief_universe or "").strip()
    if not raw:
        return list(DEFAULT_UNIVERSE)
    return [s.strip().upper() for s in raw.split(",") if s.strip()]


# Sector map for the default universe. Hardcoded rather than fetched: it covers
# 25 known large caps, it changes about never, and a wrong lookup here is more
# damaging than a missing dependency. Symbols absent from this map are treated
# as their own sector, so an unknown name is never silently grouped.
_SECTOR = {
    "HDFCBANK": "bank", "ICICIBANK": "bank", "SBIN": "bank", "AXISBANK": "bank",
    "KOTAKBANK": "bank",
    "TCS": "it", "INFY": "it", "WIPRO": "it", "HCLTECH": "it",
    "RELIANCE": "energy", "BPCL": "energy", "IOC": "energy", "ONGC": "energy",
    "NTPC": "power", "POWERGRID": "power",
    "MARUTI": "auto", "TATAMOTORS": "auto",
    "TATASTEEL": "metals", "JSWSTEEL": "metals", "HINDALCO": "metals",
    "SUNPHARMA": "pharma", "CIPLA": "pharma",
    "ITC": "fmcg", "TITAN": "consumer", "LT": "infra",
}

# How much of the ranking conviction is allowed to carry. Alignment keeps a real
# but minority say, which is proportionate to how well it is evidenced.
_CONFIDENCE_WEIGHT = 0.7
_ALIGNMENT_WEIGHT = 0.3


def _rank_score(c: dict, floor: float | None = None) -> float:
    """Blend conviction with global alignment on a common scale.

    Both inputs are normalised to 0..1 before weighting, and that matters more
    than it looks:

      * **Confidence is rescaled over its USABLE range, not 0-1.** Nothing below
        `confidence_threshold` (0.65) is ever emitted, so raw confidence only
        ever spans 0.65-1.0. Weighting the raw value gave conviction an
        effective spread of ~0.25 against alignment's full 1.0 — so alignment
        could still outvote it at the extremes, which is the exact failure the
        blend exists to prevent. Rescaling makes the 70/30 split real.
      * **Alignment is squashed into -1..1.** It is an expected-move percentage
        of unbounded magnitude; without this, one violent overnight move in a
        single driver dominates the ranking exactly as the lexicographic sort
        did.
    """
    if floor is None:
        floor = get_settings().confidence_threshold
    raw = float(c.get("confidence") or 0.0)
    span = max(1.0 - floor, 1e-6)
    confidence = max(0.0, min((raw - floor) / span, 1.0))

    alignment = float(c["global_context"].get("alignment_score") or 0.0)
    squashed = max(-1.0, min(alignment / 2.0, 1.0))
    return _CONFIDENCE_WEIGHT * confidence + _ALIGNMENT_WEIGHT * squashed


def sector_of(symbol: str) -> str:
    return _SECTOR.get(symbol.upper(), f"other:{symbol.upper()}")


def _cap_per_sector(candidates: list[dict], max_per_sector: int) -> list[dict]:
    """Keep at most `max_per_sector` candidates from any one sector.

    With 5 banks and 4 IT names in the universe, a single NASDAQ move gives
    every IT stock a similar alignment score and they surface together — four
    apparently independent ideas that are one bet in disguise. If the read is
    wrong they all lose at once, which is precisely the correlated-position
    problem the risk limits exist to prevent, delivered by the product's own
    headline feature.

    Assumes `candidates` is already sorted best-first, so the survivor of each
    sector is its strongest candidate.
    """
    if max_per_sector <= 0:
        return candidates
    kept: list[dict] = []
    seen: dict[str, int] = {}
    for c in candidates:
        sector = sector_of(c["symbol"])
        if seen.get(sector, 0) >= max_per_sector:
            c["dropped_reason"] = f"sector cap: already showing {max_per_sector} from {sector}"
            continue
        seen[sector] = seen.get(sector, 0) + 1
        c["sector"] = sector
        kept.append(c)
    return kept


def _rr(signal) -> float | None:
    reward = abs(signal.target_price - signal.entry_price)
    risk = abs(signal.entry_price - signal.stop_loss)
    return round(reward / risk, 2) if risk else None


def _dominant_drivers(cues: list[dict], threshold: float = 1.0) -> dict[str, float]:
    """Overnight drivers that moved enough to plausibly matter today."""
    out = {}
    for c in cues:
        if c["symbol"] in _DRIVER_LABEL and abs(c["change_pct"]) >= threshold:
            out[c["symbol"]] = c["change_pct"]
    return out


def _global_context(sens: dict, movers: dict[str, float], direction: str,
                    min_expected_move: float = 0.15) -> dict:
    """Explain how today's overnight moves plausibly hit this stock, and whether
    that supports or contradicts the proposed trade direction.

    Only relationships strong enough to be real (see `meaningful` in the
    sensitivity engine) are used — a weak correlation dressed up as a reason is
    worse than saying nothing.
    """
    drivers = sens.get("drivers", {}) if isinstance(sens, dict) else {}
    reasons: list[str] = []
    expected_total = 0.0

    for drv, move_pct in movers.items():
        info = drivers.get(drv) or {}
        beta = info.get("beta")
        if beta is None or not info.get("meaningful"):
            continue
        expected = beta * move_pct
        if abs(expected) < min_expected_move:
            continue
        expected_total += expected
        label = _DRIVER_LABEL.get(drv, drv)
        reasons.append(
            f"{label} moved {move_pct:+.2f}% overnight; this stock's beta to {label} "
            f"is {beta:+.2f} over {info.get('n_days', '?')} days "
            f"(correlation {info.get('correlation')}), implying roughly {expected:+.2f}%."
        )

    if not reasons:
        return {"alignment_score": 0.0, "expected_move_pct": None,
                "agrees_with_direction": None, "reasons": []}

    # Does the overnight picture point the same way as the trade?
    wants_up = direction == "BUY"
    agrees = (expected_total > 0) == wants_up
    # Supporting context ranks up; contradicting context ranks down.
    score = abs(expected_total) if agrees else -abs(expected_total)

    if not agrees:
        reasons.append(
            f"CONFLICT: overnight drivers imply {expected_total:+.2f}%, which works "
            f"against this {direction} call. Treat with caution."
        )

    return {
        "alignment_score": round(score, 3),
        "expected_move_pct": round(expected_total, 2),
        "agrees_with_direction": agrees,
        "reasons": reasons,
    }


def _shared_drivers(candidates: list[dict]) -> str:
    """Name candidates that rest on the SAME overnight driver, if any.

    One NASDAQ move gives every IT name a similar alignment score, so they
    surface together and read as several independent ideas. They are one bet: if
    the read is wrong they all lose at once. The sector cap limits how many can
    appear, but two DIFFERENT sectors can still be driven by one cue, so the
    narrative has to be able to say so.
    """
    by_driver: dict[str, list[str]] = {}
    for c in candidates:
        for reason in c["global_context"].get("reasons", []):
            for label in _DRIVER_LABEL.values():
                if reason.startswith(label):
                    by_driver.setdefault(label, []).append(c["symbol"])
    return "; ".join(
        f"{', '.join(sorted(set(syms)))} all move on {driver}"
        for driver, syms in by_driver.items()
        if len(set(syms)) > 1
    )


def _events_block(events: dict) -> str:
    """Real material for the WHY behind the overnight cues -- absent
    entirely (not "none found") when a source couldn't be checked at all
    (no key, vendor down), so the prompt never implies a quiet macro window
    that was actually just an unreachable one."""
    releases = events.get("fred_releases")
    headlines = events.get("headlines") or []
    parts: list[str] = []

    if releases is None:
        parts.append("US MACRO RELEASES: not available this run (FRED unreachable or unconfigured).")
    elif releases:
        lines = "\n".join(
            f"  {r['name']}: {r['actual']} (prior {r['prior']}), as of {r['date']}" for r in releases
        )
        parts.append(f"NEW US MACRO RELEASES SINCE THE LAST BRIEF:\n{lines}")
    else:
        parts.append("US MACRO RELEASES: none newly published since the last brief.")

    if headlines:
        lines = "\n".join(f"  {h['title']} ({h.get('publisher') or 'unknown source'})" for h in headlines[:8])
        parts.append(f"RECENT MACRO/GOLD/USD HEADLINES:\n{lines}")

    return "\n\n".join(parts)


async def _narrative(llm: LlmClient, cues: dict, candidates: list[dict], events: dict | None = None) -> str:
    """LLM writes the connecting prose. It is given the computed numbers and
    explicitly told not to invent any."""
    s = cues["summary"]
    cue_lines = "\n".join(
        f"  {c['name']}: {c['value']} ({c['change_pct']:+.2f}%)" for c in cues["cues"]
    )
    cand_lines = "\n".join(
        f"  {c['symbol']} ({c.get('sector', 'unclassified')}) {c['direction']} "
        f"entry {c['entry']} target {c['target']} stop {c['stop']} | "
        f"{'; '.join(c['global_context']['reasons']) or 'no strong global driver'}"
        for c in candidates
    ) or "  (no candidates cleared the filters today)"

    shared = _shared_drivers(candidates)
    events_block = _events_block(events) if events else ""

    prompt = (
        "Write the opening narrative for a pre-market trading brief for Indian markets.\n\n"
        f"OVERNIGHT CUES:\n{cue_lines}\n\n"
        f"COMPUTED READ: bias={s['bias']} ({s['label']}), cue strength={s['signal_strength']} "
        f"(uncalibrated — describes how strongly the cues agree, NOT a hit rate), "
        f"US avg {s['us_avg_pct']}%, Asia avg {s['asia_avg_pct']}%\n"
        f"NOTES: {'; '.join(s['notes']) or 'none'}\n\n"
        + (f"{events_block}\n\n" if events_block else "")
        + f"CANDIDATES:\n{cand_lines}\n\n"
        + (f"CORRELATION WARNING: {shared}\n\n" if shared else "")
        + "Write 3-4 sentences a trader reads in under 20 seconds: what happened overnight, "
        "what it implies for the Indian open, and where the opportunity or risk sits. "
        + ("If a macro release above plausibly explains one of the overnight cue moves, say so "
           "explicitly (e.g. \"CPI printed above the prior reading, which likely pushed the dollar "
           "and weighed on...\") — connect the release to the cue, don't just list both separately. " if events_block else "")
        + ("State plainly that the correlated candidates above are ONE bet, not several — "
           "a user who takes all of them is concentrated, not diversified. " if shared else "")
        + "Use ONLY the numbers above — invent nothing. Do not promise returns or certainty. "
        "Plain prose, no bullet points, no headings."
    )

    try:
        resp = await asyncio.to_thread(
            llm.chat,
            temperature=0, max_tokens=350,
            messages=[
                {"role": "system", "content": prompts.BRIEF_SYSTEM},
                {"role": "user", "content": prompt},
            ],
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        logger.warning("Brief narrative generation failed: %s", e)
        # Deterministic fallback — the brief still ships without the model.
        return (
            f"{s['label']}. US markets averaged {s['us_avg_pct']}% and Asia {s['asia_avg_pct']}% overnight. "
            + (" ".join(s["notes"]) if s["notes"] else "")
        ).strip()


async def publish(brief: dict) -> bool:
    """Store the brief via the NestJS internal endpoint."""
    settings = get_settings()
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.post(
                f"{settings.api_service_url}/api/internal/brief",
                headers={"x-internal-key": settings.internal_api_key},
                json=brief,
            )
            r.raise_for_status()
        logger.info("Morning brief published for %s", brief.get("date"))
        return True
    except httpx.HTTPError as e:
        logger.warning("Brief publish failed: %s", e)
        return False
