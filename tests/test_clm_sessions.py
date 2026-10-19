"""CLM session transcript generator tests (M11).

The generator bakes "ideal expert sessions" into text documents using the
REAL buffer machinery (ContextBuffer + apply_edit), so REPLACE line numbers
are grounded in the actual buffer state — the same flow generate_with_clm
produces at inference.
"""

from __future__ import annotations

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
