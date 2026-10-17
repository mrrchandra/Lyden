"""RAM-only editable context buffer (CLM memory).

The buffer is a list of human-readable LINES; token-level views are derived
from ``render()`` via the tokenizer on demand. All addressing in edit
commands is 1-based line numbers (per spec example "REPLACE line 42 ...").
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lyden.memory.commands import EditCommand


class ContextBuffer:
    """List of text lines plus derived views. Never touches disk."""

    def __init__(self, lines: list[str] | None = None) -> None:
        self.lines: list[str] = list(lines) if lines else []

    # ------------------------------------------------------------ views
    def render(self) -> str:
        return "\n".join(self.lines)

    @property
    def n_lines(self) -> int:
        return len(self.lines)

    # ------------------------------------------------------------ ops
    def read(self, line: int, end_line: int | None = None) -> str:
        """1-based inclusive slice."""
        end = line if end_line is None else end_line
        return "\n".join(self.lines[line - 1 : end])

    def write(self, text: str, after_line: int | None = None) -> int:
        """Append (or insert after a line); returns the 1-based line number
        where the FIRST written line landed."""
        new_lines = text.split("\n")
        if after_line is None or after_line >= len(self.lines):
            self.lines.extend(new_lines)
            first = len(self.lines) - len(new_lines) + 1
        else:
            self.lines[after_line:after_line] = new_lines
            first = after_line + 1
        return first

    def replace(self, line: int, text: str) -> None:
        self.lines[line - 1 : line] = text.split("\n")

    def delete(self, line: int, end_line: int | None = None) -> None:
        end = line if end_line is None else end_line
        del self.lines[line - 1 : end]

    def summarize(
        self, line: int, summarizer: Callable[[str], str], end_line: int | None = None
    ) -> str:
        """Compress a span via ``summarizer`` (the model itself) in place."""
        end = line if end_line is None else end_line
        span = "\n".join(self.lines[line - 1 : end])
        short = summarizer(span)
        del self.lines[line - 1 : end]
        self.lines[line - 1 : line - 1] = short.split("\n")
        return short

    # ------------------------------------------------------------ snapshot
    @classmethod
    def from_text_with_edits(
        cls, text: str, summarizer: Callable[[str], str] | None = None
    ) -> tuple[ContextBuffer, list[EditCommand]]:
        """Build a buffer from rendered text, consuming any edit commands
        found in it (user-driven edits). Returns (buffer, commands_applied)."""
        from lyden.memory.commands import apply_edit, consume_edit_commands

        rest, cmds = consume_edit_commands(text)
        buf = cls(rest.split("\n"))
        for cmd in cmds:
            apply_edit(buf, cmd, summarizer=summarizer)
        return buf, cmds

    def state_dict(self) -> dict[str, object]:
        """RAM-only snapshot (for serialization into checkpoints/tests)."""
        return {"lines": list(self.lines)}

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> ContextBuffer:
        lines = state.get("lines", [])
        return cls(list(lines) if isinstance(lines, list) else [])
