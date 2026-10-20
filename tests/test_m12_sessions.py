"""M12 data sessions: long-distance fact sessions, filler, 2-hop CoT traces."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, "src")

from lyden.data.m12_sessions import (
    cot_session,
    filler_text,
    long_fact_session,
    n_filler_for_distance,
)
from lyden.data.tokenizer import LydenTokenizer

ROOT = Path(__file__).resolve().parents[1]
REAL_TOK = ROOT / "lyden-data" / "tokenizer"


@pytest.fixture(scope="module")
def tiny_tokenizer(tmp_path_factory: pytest.TempPathFactory) -> LydenTokenizer:
    sample = (
        "Rex the fox lives in a red barn. Every evening Rex keeps a "
        "amber lantern on the shelf. The barn is quiet at night. "
        "Ori lived in the mill house, and the mill house stands on Green Farm. "
    )
    sample = sample * 200
    out = tmp_path_factory.mktemp("tok") / "tokenizer-tiny"
    return LydenTokenizer.train(sample, vocab_size=512, out_dir=out)


def test_filler_text_is_fresh_and_counted() -> None:
    text = filler_text(random.Random(7), n_sentences=8)
    assert text.count(".") == 8
    assert 8 * 7 <= len(text.split()) <= 8 * 11


def test_n_filler_for_distance_hits_target(tiny_tokenizer: LydenTokenizer) -> None:
    rnd = random.Random(1)
    n = n_filler_for_distance(16_000, tiny_tokenizer, rnd)
    filler = filler_text(rnd, n_sentences=n)
    filler_ids = len(tiny_tokenizer.encode(filler))
    assert abs(filler_ids - 16_000) <= 16_000 * 0.1


def test_long_fact_session_places_question_at_depth(tiny_tokenizer: LydenTokenizer) -> None:
    rnd = random.Random(3)
    n = n_filler_for_distance(4_000, tiny_tokenizer, rnd)
    session = long_fact_session(rnd, n_filler=n)
    ids = tiny_tokenizer.encode(session)
    eos_pos = ids.index(2)
    question = [ln for ln in session.splitlines() if ln.startswith("Where does ")][0]
    assert "live now?" in question
    name = question.removeprefix("Where does ").removesuffix(" live now?")
    assert eos_pos >= 3_400
    assert "<|edit|>WRITE" in session
    assert session.count("<|edit|>") == 2
    # fact-before-filler-before-question-before-answer ordering
    assert session.index(f"{name} the") < session.index("moved to")
    assert session.index("moved to") < session.index(question)
    answer_line = session.split(question, 1)[1].strip().splitlines()[0]
    assert answer_line.startswith(f"{name} now lives in ")


def test_cot_session_is_two_hops_with_trace() -> None:
    session = cot_session(random.Random(5))
    # answer references the BRIDGED fact, not the primary story fact
    assert "via" in session
    assert "<|edit|>WRITE" in session and "<|/edit|>" in session
    assert "<|bos|>" in session and "<|eos|>" in session
    # story states the bridge fact; trace repeats it after the question
    assert "stands on the grounds of" in session
    assert session.index("stands on the grounds of") < session.index("Where is")
    trace_start = session.index("so through the place")
    question_start = session.index("Where is")
    answer_start = session.index("via")
    assert question_start < trace_start < answer_start
