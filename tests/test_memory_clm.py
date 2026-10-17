"""M8: CLM editable context memory — buffer, command parser, executor, loop."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.cuda

requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from lyden.memory import (  # noqa: E402
    ContextBuffer,
    apply_edit,
    generate_with_clm,
    parse_edit_command,
)


# ------------------------------------------------------------ buffer
def test_buffer_round_trips() -> None:
    b = ContextBuffer(["alpha", "beta", "gamma"])
    assert b.read(2) == "beta"
    assert b.read(1, 3) == "alpha\nbeta\ngamma"
    b.write("delta")
    assert b.read(4) == "delta"
    b.write("first", after_line=0)
    assert b.read(1) == "first"
    assert b.read(4) == "gamma"
    b.replace(2, "ALPHA")
    assert b.read(2) == "ALPHA"
    b.delete(2)
    assert b.read(1, 2) == "first\nbeta"
    assert b.summarize(2, summarizer=lambda s: s[:3]) == "bet"
    assert b.read(2) == "bet"
    assert b.render() == "first\nbet\ngamma\ndelta"


def test_buffer_renders_render_join() -> None:
    b = ContextBuffer(["one", "two"])
    assert b.render() == "one\ntwo"
    b.write("three")
    assert b.render() == "one\ntwo\nthree"


# ------------------------------------------------------------ parser
def test_parse_read_variants() -> None:
    cmd = parse_edit_command('<|edit|>READ line 7<|/edit|>')
    assert cmd is not None and cmd.verb == "READ" and cmd.lines == (7, 7)
    cmd = parse_edit_command('<|edit|>READ lines 3 to 9<|/edit|>')
    assert cmd is not None and cmd.verb == "READ" and cmd.lines == (3, 9)
    cmd = parse_edit_command('<|edit|>READ<|/edit|>')
    assert cmd is not None and cmd.verb == "READ" and cmd.lines is None


def test_parse_write_replace_delete_summarize() -> None:
    cmd = parse_edit_command('<|edit|>WRITE with "hello world"<|/edit|>')
    assert cmd is not None and cmd.verb == "WRITE" and cmd.text == "hello world"
    cmd = parse_edit_command('<|edit|>WRITE after line 4 with "two\\nwords"<|/edit|>')
    assert cmd is not None and cmd.after_line == 4 and cmd.text == "two\nwords"
    cmd = parse_edit_command('<|edit|>REPLACE line 42 with "User prefers Python"<|/edit|>')
    assert cmd is not None and cmd.verb == "REPLACE" and cmd.lines == (42, 42)
    assert cmd.text == "User prefers Python"
    cmd = parse_edit_command('<|edit|>DELETE line 3<|/edit|>')
    assert cmd is not None and cmd.verb == "DELETE" and cmd.lines == (3, 3)
    cmd = parse_edit_command('<|edit|>SUMMARIZE line 12<|/edit|>')
    assert cmd is not None and cmd.verb == "SUMMARIZE" and cmd.lines == (12, 12)


def test_parse_ignores_incomplete_and_unknown() -> None:
    assert parse_edit_command('no commands here') is None
    assert parse_edit_command('<|edit|>REPLACE line 1 with "unclosed') is None
    with pytest.raises(ValueError):
        cmd = parse_edit_command('<|edit|>TELEPORT line 1<|/edit|>')
        assert cmd is not None
        apply_edit(ContextBuffer(), cmd)


# ------------------------------------------------------------ executor
def test_apply_edit_operations() -> None:
    b = ContextBuffer(["a", "b", "c"])
    out = apply_edit(b, parse_edit_command('<|edit|>READ line 2<|/edit|>'))  # type: ignore[arg-type]
    assert out == "b"
    out = apply_edit(b, parse_edit_command('<|edit|>READ lines 1 to 3<|/edit|>'))  # type: ignore[arg-type]
    assert out == "a\nb\nc"
    out = apply_edit(b, parse_edit_command('<|edit|>WRITE with "hi mom"<|/edit|>'))  # type: ignore[arg-type]
    assert b.read(4) == "hi mom"
    out = apply_edit(b, parse_edit_command('<|edit|>WRITE after line 1 with "x"<|/edit|>'))  # type: ignore[arg-type]
    assert b.read(2) == "x" and b.n_lines == 5
    apply_edit(b, parse_edit_command('<|edit|>REPLACE line 2 with "y"<|/edit|>'))  # type: ignore[arg-type]
    assert b.read(2) == "y"
    apply_edit(b, parse_edit_command('<|edit|>DELETE line 1<|/edit|>'))  # type: ignore[arg-type]
    assert b.read(1) == "y"
    out = apply_edit(
        b,
        parse_edit_command('<|edit|>SUMMARIZE line 1<|/edit|>'),  # type: ignore[arg-type]
        summarizer=lambda s: s.upper(),
    )
    assert out == "Y" and b.read(1) == "Y"
    with pytest.raises(ValueError):
        apply_edit(b, parse_edit_command('<|edit|>SUMMARIZE line 1<|/edit|>'))  # type: ignore[arg-type]


def test_prompt_edits_consumed() -> None:
    cmd_a = '<|edit|>REPLACE line 1 with "NEW ONE"<|/edit|>'
    cmd_b = '<|edit|>WRITE with "tail"<|/edit|>'
    prompt = f"line one\n{cmd_a}\n{cmd_b}"
    buffer, rest_text = ContextBuffer.from_text_with_edits(prompt)
    assert "NEW ONE" in buffer.lines
    assert "tail" in buffer.lines
    assert "<|edit|>" not in buffer.render()


# ------------------------------------------------------------ generation loop
def _tiny_model_and_tokenizer(vocab: int = 8192):
    from lyden.config import MambaConfig, ModelConfig, MoEConfig
    from lyden.data.tokenizer import LydenTokenizer
    from lyden.model.model import LydenModel

    torch.manual_seed(3)
    # dim 256 + d_state 64 is the smallest d_in_proj-aligned geometry
    # (see mamba_block alignment checks).
    cfg = ModelConfig(
        vocab_size=vocab,
        dim=256,
        n_layers=2,
        layer_pattern="mm",
        mamba=MambaConfig(d_state=64),
        moe=MoEConfig(n_routed=2, n_shared=1, top_k=1, expert_hidden=16),
    )
    model = LydenModel(cfg).cuda().eval()
    tok = LydenTokenizer.load(Path("~/lyden-data-nano/tokenizer").expanduser())
    return model, tok


@requires_gpu
def test_generate_with_clm_prompt_edits() -> None:
    model, tok = _tiny_model_and_tokenizer()
    prompt = 'L1\n<|edit|>REPLACE line 1 with "REDACTED"<|/edit|>'
    text, buffer = generate_with_clm(model, tok, prompt, max_new_tokens=24, max_edits=2)
    assert buffer.read(1) == "REDACTED"
    assert "<|edit|>" not in text


@requires_gpu
def test_generate_with_clm_injected_edit() -> None:
    model, tok = _tiny_model_and_tokenizer()

    class ScriptedModel(torch.nn.Module):
        """Emits a fixed token stream one token per step (greedy)."""

        def __init__(self, ids: list[int], vocab: int) -> None:
            super().__init__()
            self.ids = list(ids)
            self.vocab = vocab
            self.dummy = torch.nn.Parameter(torch.zeros(1))
            self.step = 0

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            logits = torch.full(
                (x.shape[0], x.shape[1], self.vocab), -1e9, device=x.device
            )
            if len(self.ids) > 0 and self.step < len(self.ids):
                logits[:, -1, self.ids[self.step]] = 1e9
            self.step += 1
            return logits

    cmd_ids = tok.encode('<|edit|>WRITE with "scripted hello"<|/edit|>')
    stub = ScriptedModel(cmd_ids, tok.vocab_size)
    text, buffer = generate_with_clm(
        stub, tok, "seed text", max_new_tokens=64, max_edits=1, temperature=0.0
    )
    assert "scripted hello" in buffer.lines
    assert "<|edit|>" not in text
