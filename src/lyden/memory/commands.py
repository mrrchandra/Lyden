"""Edit-command grammar: parse + execute ("Commands should be textual and
human-readable", spec 4.6).

Supported forms (all wrapped in <|edit|> ... <|/edit|>):
  READ [ line N | lines A to B ]
  WRITE [ after line N ] with "text"
  REPLACE line N with "text"
  DELETE line N
  SUMMARIZE line N
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from lyden.memory.context_buffer import ContextBuffer

OPEN, CLOSE = "<|edit|>", "<|/edit|>"

_PAYLOAD = r'"((?:[^"\\]|\\.)*)"'


@dataclass
class EditCommand:
    verb: str
    lines: tuple[int, int] | None = None  # 1-based (start, end)
    after_line: int | None = None
    text: str | None = None


def _unescape(s: str) -> str:
    return s.replace("\\\\", "\x00").replace('\\"', '"').replace("\\n", "\n").replace("\x00", "\\")


def _parse_inner(inner: str) -> EditCommand:
    inner = inner.strip()
    patterns: tuple[tuple[str, re.Pattern[str]], ...] = (
        ("DELETE", re.compile(r"DELETE line (\d+)")),
        ("SUMMARIZE", re.compile(r"SUMMARIZE line (\d+)")),
        ("REPLACE", re.compile(rf"REPLACE line (\d+) with {_PAYLOAD}")),
        ("READ_R", re.compile(r"READ lines (\d+) to (\d+)")),
        ("READ_L", re.compile(r"READ line (\d+)")),
        ("READ_B", re.compile(r"READ")),
        ("WRITE_A", re.compile(rf"WRITE after line (\d+) with {_PAYLOAD}")),
        ("WRITE_B", re.compile(rf"WRITE with {_PAYLOAD}")),
    )
    for verb, pattern in patterns:
        m = pattern.fullmatch(inner)
        if not m:
            continue
        g = m.groups()
        if verb == "DELETE":
            n = int(g[0])
            return EditCommand("DELETE", lines=(n, n))
        if verb == "SUMMARIZE":
            n = int(g[0])
            return EditCommand("SUMMARIZE", lines=(n, n))
        if verb == "REPLACE":
            n = int(g[0])
            return EditCommand("REPLACE", lines=(n, n), text=_unescape(g[1]))
        if verb == "READ_R":
            return EditCommand("READ", lines=(int(g[0]), int(g[1])))
        if verb == "READ_L":
            n = int(g[0])
            return EditCommand("READ", lines=(n, n))
        if verb == "READ_B":
            return EditCommand("READ")
        if verb == "WRITE_A":
            return EditCommand("WRITE", after_line=int(g[0]), text=_unescape(g[1]))
        return EditCommand("WRITE", text=_unescape(g[0]))
    raise ValueError(f"unknown edit command: {inner!r}")


def parse_edit_command(text: str) -> EditCommand | None:
    """Extract the *first* complete <|edit|>...<|/edit|> command, or None."""
    pattern = re.compile(rf"{re.escape(OPEN)}(.*?){re.escape(CLOSE)}", re.DOTALL)
    m = pattern.search(text)
    if not m:
        return None
    return _parse_inner(m.group(1))


def _target_span(cmd: EditCommand, n_lines: int) -> tuple[int, int]:
    if cmd.lines is None:
        raise ValueError("command needs a line target")
    a, b = cmd.lines
    a = max(1, min(a, n_lines))
    b = max(1, min(b, n_lines))
    return a, b


def apply_edit(
    buffer: ContextBuffer,
    cmd: EditCommand,
    summarizer: Callable[[str], str] | None = None,
) -> str:
    """Execute one command against the buffer; returns the echo/result text."""
    if cmd.verb == "READ":
        if cmd.lines is None:
            return buffer.render()
        a, b = _target_span(cmd, buffer.n_lines)
        return buffer.read(a, b)
    if cmd.verb == "WRITE":
        if cmd.text is None:
            raise ValueError("WRITE needs a payload")
        line = buffer.write(cmd.text, after_line=cmd.after_line)
        return f"written to line {line}"
    if cmd.verb == "REPLACE":
        if cmd.text is None:
            raise ValueError("REPLACE needs a payload")
        a, _ = _target_span(cmd, buffer.n_lines)
        buffer.replace(a, cmd.text)
        return f"replaced line {a}"
    if cmd.verb == "DELETE":
        a, b = _target_span(cmd, buffer.n_lines)
        buffer.delete(a, b)
        return f"deleted lines {a}-{b}"
    if cmd.verb == "SUMMARIZE":
        if summarizer is None:
            raise ValueError("SUMMARIZE needs a summarizer (model-backed)")
        a, b = _target_span(cmd, buffer.n_lines)
        return buffer.summarize(a, summarizer=summarizer, end_line=b)
    raise ValueError(f"unhandled verb: {cmd.verb}")


def consume_edit_commands(text: str) -> tuple[str, list[EditCommand]]:
    """Remove all complete commands from text in order; return the remaining
    text and the parsed commands. Unclosed <\\|edit\\|> openers: the fragment
    AFTER the interceptor point is kept verbatim (still raw text)."""
    remaining: list[str] = []
    cmds: list[EditCommand] = []
    pos = 0
    while True:
        nxt = text.find(OPEN, pos)
        if nxt == -1:
            remaining.append(text[pos:])
            break
        close_at = text.find(CLOSE, nxt)
        if close_at == -1:
            remaining.append(text[pos:nxt])
            remaining.append(text[nxt:])
            break
        cmds.append(_parse_inner(text[nxt + len(OPEN) : close_at]))
        remaining.append(text[pos:nxt])
        pos = close_at + len(CLOSE)
    return "".join(remaining), cmds
