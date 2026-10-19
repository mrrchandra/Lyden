"""CLM "ideal expert session" transcripts (M11).

A session is a plain-text document (bos-framed like any other document) in
which the model's DESIRED behavior is written out: it summarizes a context
page into the RAM buffer with a WRITE, applies a contradiction with a
grounded REPLACE (line number computed through the real :class:`ContextBuffer`
machinery), and answers a question using the corrected fact. Training on
these makes :func:`lyden.memory.loop.generate_with_clm` behave the same way
at inference.

Grammar compatibility: payloads are escaped the way
:func:`lyden.memory.commands._unescape` expects (backslash and double-quote).
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterator, Sequence

from lyden.memory.context_buffer import ContextBuffer


def _escape_payload(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def clm_session(
    story: str,
    *,
    note: str,
    correction: str,
    corrected_note: str,
    question: str,
    answer: str,
) -> str:
    """Build one fact-fix session transcript.

    Args:
        story: the context page content (wrapped in <|context|> markers).
        note: the buffer line the model writes after reading the story.
        correction: plain-text page that contradicts the note.
        corrected_note: the replacement buffer line (same slot as ``note``).
        question: asked AFTER the fix; must be answerable from the buffer.
        answer: the fact-correct answer text.

    REPLACE's line number is grounded: the WRITE goes through a real
    :class:`ContextBuffer`, so what lands on line 1 when the buffer starts
    empty is addressed as line 1 — exactly as the live loop would behave.
    """
    buffer = ContextBuffer()
    write_line = buffer.write(note)
    assert write_line >= 1

    lines = [
        "<|bos|>",
        "<|context|>",
        story,
        "<|/context|>",
        f'<|edit|>WRITE with "{_escape_payload(note)}"<|/edit|>',
        correction,
        f'<|edit|>REPLACE line {write_line} with "{_escape_payload(corrected_note)}"<|/edit|>',
        question,
        answer,
        "<|eos|>",
    ]
    return "\n".join(lines)


# ------------------------------------------------------- synthetic factories
_NOUNS = ("fox", "owl", "cat", "rabbit", "bear", "mouse", "duck", "deer")
_PLACES = (
    "a red barn",
    "the mill house",
    "a hollow oak",
    "the boathouse",
    "a stone cottage",
    "the lighthouse",
    "a root cellar",
    "the greenhouse",
)
_ITEMS = ("lantern", "umbrella", "teapot", "gloves", "harp", "mirror", "basket", "kite")
_COLORS = ("amber", "cobalt", "ivory", "jade", "rust-red", "pearl-grey", "gold", "ink-blue")


def _pick(rnd: random.Random, choices: Sequence[str], *, exclude: str | None = None) -> str:
    while True:
        pick = rnd.choice(choices)
        if pick != exclude:
            return pick


def synthetic_fact_session(rnd: random.Random) -> str:
    """One fully synthetic fact-fix session (self-check by construction).

    Story: two grounded facts (a home and a colored item). Correction: the
    character MOVES, contradicting the home only. The note, corrected note,
    question and answer are consistent by construction, so each held-out
    session double-serves as its own regression oracle (the M11 eval script
    generates fresh ones from unseen seeds and scores the same contract).
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
    correction = f"{name} the {noun} moved to {place_b}; the {color} {item} was carried along."
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


def mixed_documents(
    stories: Iterator[str],
    session_factory: Callable[[], str],
    ratio: int = 3,
) -> Iterator[str]:
    """Interleave one synthetic session every ``ratio`` plain documents.

    Pattern: [session, story x ratio] repeating; when the story stream ends,
    trailing sessions are no longer emitted. Keeps packing windows locally
    productive while sessions hold a steady ~1/(ratio+1) share of documents.
    """
    emitted = 0
    while True:
        yield session_factory()
        emitted += 1
        for _ in range(ratio):
            story = next(stories, None)
            if story is None:
                return
            yield story


def heldout_sessions(prefix_seed: int, n: int) -> list[tuple[str, str]]:
    """(eval_prompt, expected_answer) pairs from held-out synthetic seeds.

    The prompt is everything through the question (after <|bos|>, before the
    answer and <|eos|>); the target is the corrected-fact answer line. Fresh
    seeds never seen in training make the probe a true generalization test.
    """
    pairs: list[tuple[str, str]] = []
    for seed in range(prefix_seed, prefix_seed + n):
        session = synthetic_fact_session(random.Random(seed))
        lines = [line for line in session.split("\n") if line.strip()]
        target = lines[-2]  # answer line; <|eos|> is last
        prompt = "\n".join(lines[1:-2]) + "\n"
        pairs.append((prompt, target))
    return pairs
