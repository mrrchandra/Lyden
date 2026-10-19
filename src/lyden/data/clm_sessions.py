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
