"""CLM session transcript generator tests (M11).

The generator bakes "ideal expert sessions" into text documents using the
REAL buffer machinery (ContextBuffer + apply_edit), so REPLACE line numbers
are grounded in the actual buffer state — the same flow generate_with_clm
produces at inference.
"""

from __future__ import annotations

import random

from lyden.memory.commands import apply_edit, consume_edit_commands
from lyden.memory.context_buffer import ContextBuffer


def build() -> str:
    from lyden.data.clm_sessions import clm_session

    return clm_session(
        story="Luna the cat lives in a red barn. Every morning she chases sparrows.",
        note="Characters: Luna the cat (red barn, chases sparrows).",
        correction="A storm blew the barn roof off; Luna moved to the mill house.",
        corrected_note="Characters: Luna the cat (mill house after the storm, chases sparrows).",
        question="Where does Luna live now?",
        answer="Luna lives in the mill house after the storm blew the barn roof off.",
    )


def test_session_transcript_replays_commands_to_corrected_buffer() -> None:
    session = build()
    _rest, commands = consume_edit_commands(session)
    assert len(commands) == 2
    buffer = ContextBuffer()
    for cmd in commands:
        apply_edit(buffer, cmd)
    assert "mill house after the storm" in buffer.render()
    assert "red barn" not in buffer.render(), "original note must be replaced, not kept"


def test_session_replace_targets_the_written_line() -> None:
    """Buffer starts empty, so WRITE lands on line 1 and the later REPLACE
    must address that same line."""
    session = build()
    assert "REPLACE line 1 with" in session


def test_session_transcript_shape() -> None:
    session = build()
    assert session.startswith("<|bos|>")
    assert session.endswith("<|eos|>")
    assert (
        "<|context|>\nLuna the cat lives in a red barn."
        " Every morning she chases sparrows.\n<|/context|>"
    ) in session
    lines = [line for line in session.split("\n") if line.strip()]
    assert any(line.startswith("A storm blew the barn") for line in lines), (
        "correction sentence appears as plain page text"
    )
    assert "Where does Luna live now?" in session
    assert "Luna lives in the mill house" in session


def test_session_note_with_quotes_survives_replay() -> None:
    from lyden.data.clm_sessions import clm_session

    session = clm_session(
        story="Sun, rain, wind.",
        note='Rule: the "wind" period ends at dusk.',
        correction="The wind stopped at noon instead.",
        corrected_note='Rule: the "wind" period ends at noon.',
        question="When does the wind period end?",
        answer="The wind period ends at noon.",
    )
    _rest, commands = consume_edit_commands(session)
    buffer = ContextBuffer()
    for cmd in commands:
        apply_edit(buffer, cmd)
    assert buffer.read(1) == 'Rule: the "wind" period ends at noon.'


# ---------------------------------------------- M11 session factory + mixing
def test_synthetic_fact_story_roundtrips() -> None:
    """A synthetic fact page + correction must produce a session whose replay
    reflects the corrected fact, and the answer must name a home that is NOT
    the story's original one."""

    from lyden.data.clm_sessions import _PLACES, synthetic_fact_session

    session = synthetic_fact_session(random.Random(3))
    rest, commands = consume_edit_commands(session)
    assert len(commands) == 2
    buffer = ContextBuffer()
    for cmd in commands:
        apply_edit(buffer, cmd)
    assert "moved" not in buffer.render(), "correction page is not buffer content"
    story = session.split("<|/context|>")[0]
    answer = [line for line in session.split("\n") if line.strip()][-2]
    assert any(place in answer and place not in story for place in _PLACES), (
        "answer must name the corrected home"
    )


def test_sample_sessions_are_seeded_and_valid() -> None:
    from lyden.data.clm_sessions import synthetic_fact_session

    rnd = random.Random(11)
    sessions = [synthetic_fact_session(rnd) for _ in range(5)]
    assert len({s for s in sessions}) == 5, "seeded randomness must vary across sessions"
    assert all(s.startswith("<|bos|>") and s.endswith("<|eos|>") for s in sessions)


def test_mixed_documents_interleave_sessions_every_ratio() -> None:
    """Pattern: one session, ratio plain stories, repeat — so every packing
    window sees session ops at a ~1/(ratio+1) doc frequency."""
    from lyden.data.clm_sessions import mixed_documents

    stories = [f"story-{i}" for i in range(6)]
    sessions_added: list[str] = []

    def factory() -> str:
        sessions_added.append(f"session-{len(sessions_added)}")
        return sessions_added[-1]

    out = list(mixed_documents(iter(stories), factory, ratio=3))
    assert out[0] == "session-0"
    assert out[1:4] == ["story-0", "story-1", "story-2"]
    assert out[4] == "session-1"
    assert out[5:8] == ["story-3", "story-4", "story-5"]
    assert all(d.startswith("story-") or d.startswith("session-") for d in out)


def test_heldout_sessions_derive_prompt_and_target() -> None:
    from lyden.data.clm_sessions import heldout_sessions

    pairs = heldout_sessions(prefix_seed=50_000, n=4)
    assert len(pairs) == 4
    seen_targets = set()
    for prompt, target in pairs:
        assert target != "" and "<|bos|>" not in prompt and "<|eos|>" not in prompt
        assert target not in prompt, "the answer must never leak into the eval prompt"
        assert prompt.rstrip().split("\n")[-1].startswith("Where does"), (
            "prompt must end on the question"
        )
        assert "REPLACE line" in prompt, "the edit must be inside the context"
        seen_targets.add(target)
    assert len(seen_targets) > 2, "targets vary across seeds"
