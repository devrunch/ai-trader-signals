"""
Morning brief ranking and sector capping (checklist 08).

The brief is the client's headline deliverable, so how it orders its shortlist
is the most visible judgement the product makes.
"""
from __future__ import annotations

from app.signals import brief


def candidate(symbol: str, confidence: float, alignment: float) -> dict:
    return {
        "symbol": symbol,
        "confidence": confidence,
        "global_context": {"alignment_score": alignment},
    }


def rank(cands: list[dict]) -> list[str]:
    ordered = sorted(cands, key=lambda c: -brief._rank_score(c))
    return [c["symbol"] for c in ordered]


# -- ranking ---------------------------------------------------------------

def test_conviction_outranks_a_thin_global_story():
    """The bug: a lexicographic sort on (alignment, confidence) means alignment
    decides outright, because two floats are never exactly equal. A 0.66
    signal with a slightly better beta story beat a 0.95 one, every time —
    and alignment is the least evidenced input in the system."""
    thin_story_low_conviction = candidate("A", confidence=0.66, alignment=0.40)
    strong_conviction = candidate("B", confidence=0.95, alignment=0.30)
    assert rank([thin_story_low_conviction, strong_conviction])[0] == "B"


def test_alignment_still_breaks_ties_between_equal_conviction():
    assert rank([
        candidate("A", confidence=0.80, alignment=0.10),
        candidate("B", confidence=0.80, alignment=0.90),
    ])[0] == "B"


def test_a_conflicting_global_read_pushes_a_candidate_down():
    """A negative alignment score means the overnight drivers work AGAINST the
    trade. It must cost the candidate rank, not merely fail to help."""
    conflicted = candidate("A", confidence=0.85, alignment=-1.5)
    supported = candidate("B", confidence=0.80, alignment=1.5)
    assert rank([conflicted, supported])[0] == "B"


def test_one_violent_driver_cannot_dominate_the_ranking():
    """Alignment is squashed before weighting, so an extreme overnight move
    cannot reproduce the old lexicographic behaviour by magnitude alone."""
    extreme_but_unconvinced = candidate("A", confidence=0.66, alignment=50.0)
    convinced = candidate("B", confidence=0.95, alignment=0.0)
    assert rank([extreme_but_unconvinced, convinced])[0] == "B"


def test_a_missing_alignment_score_does_not_crash_the_ranking():
    c = {"symbol": "A", "confidence": 0.8, "global_context": {}}
    assert brief._rank_score(c) > 0


def test_confidence_is_rescaled_over_its_usable_range_not_zero_to_one():
    """Nothing below the 0.65 threshold is ever emitted, so raw confidence only
    spans 0.65-1.0. Weighting the raw value left conviction with an effective
    spread of ~0.25 against alignment's full 1.0 — so alignment could still
    outvote it, which is the failure the blend exists to prevent."""
    at_floor = {"symbol": "A", "confidence": 0.65, "global_context": {"alignment_score": 0.0}}
    at_ceiling = {"symbol": "B", "confidence": 1.0, "global_context": {"alignment_score": 0.0}}
    assert brief._rank_score(at_floor, floor=0.65) == 0.0
    assert brief._rank_score(at_ceiling, floor=0.65) == brief._CONFIDENCE_WEIGHT


# -- sector cap ------------------------------------------------------------

def test_four_it_names_collapse_to_one():
    """One NASDAQ move gives every IT stock a similar alignment score, so they
    surface together — four ideas that are one bet."""
    cands = [candidate(s, 0.9, 1.0) for s in ("TCS", "INFY", "WIPRO", "HCLTECH")]
    kept = brief._cap_per_sector(cands, 1)
    assert len(kept) == 1
    assert kept[0]["sector"] == "it"


def test_the_cap_keeps_the_strongest_of_each_sector():
    """`_cap_per_sector` assumes best-first ordering, so the survivor is the
    highest-ranked member of its sector."""
    cands = sorted(
        [candidate("TCS", 0.70, 1.0), candidate("INFY", 0.95, 1.0), candidate("SBIN", 0.80, 1.0)],
        key=lambda c: -brief._rank_score(c),
    )
    kept = brief._cap_per_sector(cands, 1)
    assert {c["symbol"] for c in kept} == {"INFY", "SBIN"}


def test_different_sectors_all_survive():
    cands = [candidate(s, 0.9, 1.0) for s in ("TCS", "SBIN", "RELIANCE", "MARUTI")]
    assert len(brief._cap_per_sector(cands, 1)) == 4


def test_a_dropped_candidate_says_why():
    cands = [candidate("TCS", 0.9, 1.0), candidate("INFY", 0.8, 1.0)]
    brief._cap_per_sector(cands, 1)
    assert "sector cap" in cands[1]["dropped_reason"]


def test_an_unknown_symbol_is_never_silently_grouped():
    """A symbol missing from the map gets its own bucket rather than landing in
    a shared 'other' that would cap unrelated stocks against each other."""
    assert brief.sector_of("SOMENEWCO") != brief.sector_of("ANOTHERNEWCO")
    cands = [candidate("SOMENEWCO", 0.9, 1.0), candidate("ANOTHERNEWCO", 0.9, 1.0)]
    assert len(brief._cap_per_sector(cands, 1)) == 2


def test_a_cap_of_zero_or_less_disables_capping():
    cands = [candidate(s, 0.9, 1.0) for s in ("TCS", "INFY", "WIPRO")]
    assert len(brief._cap_per_sector(cands, 0)) == 3


# -- correlation warning ---------------------------------------------------

def with_reasons(symbol: str, *reasons: str) -> dict:
    return {"symbol": symbol, "confidence": 0.9,
            "global_context": {"alignment_score": 1.0, "reasons": list(reasons)}}


def test_candidates_sharing_a_driver_are_named_as_one_bet():
    """Two names moving on the same overnight cue are not diversification."""
    cands = [
        with_reasons("TCS", "NASDAQ moved +1.80% overnight; this stock's beta ..."),
        with_reasons("SBIN", "NASDAQ moved +1.80% overnight; this stock's beta ..."),
    ]
    shared = brief._shared_drivers(cands)
    assert "NASDAQ" in shared
    assert "SBIN" in shared and "TCS" in shared


def test_candidates_on_different_drivers_are_not_flagged():
    cands = [
        with_reasons("TCS", "NASDAQ moved +1.80% overnight; ..."),
        with_reasons("BPCL", "crude moved -2.40% overnight; ..."),
    ]
    assert brief._shared_drivers(cands) == ""


def test_a_single_candidate_is_never_flagged_as_correlated():
    assert brief._shared_drivers([with_reasons("TCS", "NASDAQ moved +1.80% ...")]) == ""


def test_candidates_with_no_global_reason_are_not_flagged():
    assert brief._shared_drivers([with_reasons("TCS"), with_reasons("SBIN")]) == ""
