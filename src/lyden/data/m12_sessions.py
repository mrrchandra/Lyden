"""M12 sessions: long-distance fact sessions + 2-hop CoT traces.

Extends :mod:`lyden.data.clm_sessions` for the 32K context-warming run:

- :func:`long_fact_session` stretches the fact -> question distance with
  filler text so a training window must carry the fact across (up to) the
  whole context to reach the answer.
- :func:`cot_session` chains two facts and writes the bridging step out
  ("... living in the barn which stands on Green Farm ...") between the
  question and the answer, as visible reasoning.
- :func:`filler_text` / :func:`n_filler_for_distance` build and calibrate
  that filler with any tokenizer so distances are token-true, not guessed.

Sessions remain plain documents (bos ... eos) - the LM loss covers the
whole transcript - and the WRITE/REPLACE payloads keep the M11 escape
rules, so the runtime CLM loop can replay them unchanged.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import TYPE_CHECKING

from lyden.data.clm_sessions import _COLORS, _ITEMS, _NOUNS, _PLACES, clm_session

if TYPE_CHECKING:
    from lyden.data.tokenizer import LydenTokenizer

#: words per filler sentence below
_SENTWORDS = 9

_DISTANCE_ERROR = 0.1


def _pick(rnd: random.Random, choices: Sequence[str], *, exclude: str | None = None) -> str:
    while True:
        pick = rnd.choice(choices)
        if pick != exclude:
            return pick


def _filler_sentence(rnd: random.Random) -> str:
    noun = _pick(rnd, _NOUNS)
    place = _pick(rnd, _PLACES)
    item = _pick(rnd, _ITEMS)
    color = _pick(rnd, _COLORS)
    verb = _pick(rnd, ("found", "saw", "carried", "sold"))
    return f"The {noun} {verb} a {color} {item} near {place}."


def filler_text(rnd: random.Random, *, n_sentences: int) -> str:
    """Fresh filler prose: neutral sentences sharing the session vocabulary."""
    return " ".join(_filler_sentence(rnd) for _ in range(n_sentences))


def n_filler_for_distance(
    distance: int,
    tokenizer: LydenTokenizer,
    rnd: random.Random,
    *,
    low: int = 1,
    high: int = 200_000,
) -> int:
    """Filler sentence count whose real token length ~= ``distance`` (±10%)."""
    if distance <= 0:
        raise ValueError(f"distance must be positive, got {distance}")

    def length(n: int) -> int:
        return len(tokenizer.encode(filler_text(rnd, n_sentences=n)))

    lo, hi = low, high
    if length(hi) < distance:
        raise ValueError(
            f"distance {distance} exceeds the probe range ({length(hi)} tokens at n={hi})"
        )
    while lo < hi:
        mid = (lo + hi) // 2
        if length(mid) < distance:
            lo = mid + 1
        else:
            hi = mid
    for n in (lo, lo - 1):
        if n >= low and abs(length(n) - distance) <= distance * _DISTANCE_ERROR:
            return n
    raise ValueError(f"cannot calibrate {distance} tokens within {_DISTANCE_ERROR:.0%} error")


def long_fact_session(rnd: random.Random, *, n_filler: int) -> str:
    """A fact-fix session whose consequence lands ``n_filler`` sentences later.

    Same WRITE/REPLACE/question contract as :func:`synthetic_fact_session`;
    neutral filler text pushes the correction and question far from the fact
    so the model must retain the buffer note across the distance.
    """
    noun = _pick(rnd, _NOUNS)
    name = "Ori" if noun == "owl" else "Rex" if noun == "fox" else "Mirren"
    place_a = _pick(rnd, _PLACES)
    item = _pick(rnd, _ITEMS)
    color = _pick(rnd, _COLORS)
    place_b = _pick(rnd, _PLACES, exclude=place_a)

    story = (
        f"{name} the {noun} lives in {place_a}. "
        f"Every evening {name} keeps a {color} {item} on the shelf. "
        f"The {item} never leaves {place_a}."
    )
    note = f"Notes: {name} ({noun}) home {place_a}; {item} {color} kept nearby."
    correction_entities = filler_text(rnd, n_sentences=n_filler)
    correction = (
        f"{name} the {noun} moved to {place_b} after the rain. "
        f"The {color} {item} was carried along. "
        + correction_entities
        + f" {name} settled into {place_b} for good."
    )
    corrected_note = f"Notes: {name} ({noun}) home {place_b}; {item} {color} kept nearby."
    question = f"Where does {name} live now?"
    answer = f"{name} now lives in {place_b}."

    return clm_session(
        story,
        note=note,
        correction=correction,
        corrected_note=corrected_note,
        question=question,
        answer=answer,
    )


def cot_session(rnd: random.Random) -> str:
    """A two-hop fact session with the bridge written out before the answer.

    Facts: the character lives in place_a, and place_a stands on the grounds
    of larger place_b. The question asks about place_b, so the safe answer
    path passes through the primary fact. The generation 该 leads with the
    primary fact, then the bridge, then the final answer; the trace makes
    the intermediate step visible to the LM loss.
    """

    class _Buffer:
        pass

    noun = _pick(rnd, _NOUNS)
    name = "Ori" if noun == "owl" else "Rex" if noun == "fox" else "Mirren"
    places = list(_PLACES)
    rnd.shuffle(places)
    farm_town_a, farm_town_b = places[0], places[1]
    _pick(rnd, _ITEMS)
    _pick(rnd, _COLORS)

    story = (
        f"{name} the {noun} lives in {farm_town_a}. "
        f"{farm_town_a.capitalize()} stands on the grounds of {farm_town_b}. "
        f"The {farm_town_b} is a wide green field with a windmill in the middle."
    )
    note = f"Notes: {name} home {farm_town_a}; {farm_town_a} part of {farm_town_b}."
    correction = "Ground rules: the buffer says the home place and its larger region."
    corrected_note = f"Notes: {name} home {farm_town_a}; {farm_town_a} part of {farm_town_b}."
    question = f"Where is {name} living, through the place it lives in?"
    trace = (
        f"{name} lives in {farm_town_a}, and {farm_town_a} is part of {farm_town_b}, "
        f"so through the place, {name} is at {farm_town_b}."
    )
    answer = f"{name} is at {farm_town_b}, via {farm_town_a}."

    transcript = clm_session(
        story,
        note=note,
        correction=correction,
        corrected_note=corrected_note,
        question=question,
        answer=answer,
    )
    # splice the trace between question and answer
    head, sep, tail = transcript.rpartition(f"{answer}\n")
    assert sep, "clm_session layout changed"
    return head + trace + "\n" + answer + "\n" + tail

