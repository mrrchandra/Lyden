"""Tests for tokenizer training/loading and encode/decode round-trips."""

from __future__ import annotations

from pathlib import Path

import pytest

from lyden.data.tokenizer import LydenTokenizer

SAMPLE = [
    "Once upon a time there was a little girl named Lily. She loved to play outside.",
    "The dog ran fast through the park and found a big stick under the tree.",
    "Tim and his mom went to the store to buy apples and bananas for breakfast.",
    "It was a sunny day, so Tom rode his bike all the way to school with his friends.",
    "Lily saw a rainbow after the rain and thought it was the most beautiful thing.",
    '<|edit|>REPLACE line 3 with "User likes Python"<|/edit|>',
    "Zażółć gęślą jaźń — naïve café test of unicode.",
]


@pytest.fixture(scope="module")
def tiny_tokenizer(tmp_path_factory: pytest.TempPathFactory) -> LydenTokenizer:
    out = tmp_path_factory.mktemp("tok") / "tokenizer-tiny"
    sample = SAMPLE * 200  # enough text for a tiny BPE
    return LydenTokenizer.train(sample, vocab_size=512, out_dir=out)


def test_special_token_ids_reserved_and_stable(tiny_tokenizer: LydenTokenizer) -> None:
    ids = tiny_tokenizer.special_token_ids()
    assert ids == {
        "<|pad|>": 0,
        "<|bos|>": 1,
        "<|eos|>": 2,
        "<|edit|>": 3,
        "<|/edit|>": 4,
        "<|context|>": 5,
        "<|/context|>": 6,
    }
    assert tiny_tokenizer.vocab_size <= 512  # BPE target; tiny sample yields fewer


def test_encode_decode_roundtrip(tiny_tokenizer: LydenTokenizer) -> None:
    for text in SAMPLE:
        ids = tiny_tokenizer.encode(text)
        assert tiny_tokenizer.decode(ids) == text


def test_bos_eos_framing(tiny_tokenizer: LydenTokenizer) -> None:
    ids = tiny_tokenizer.encode("hello", add_bos=True, add_eos=True)
    assert ids[0] == tiny_tokenizer.bos_id == 1
    assert ids[-1] == tiny_tokenizer.eos_id == 2
    plain = tiny_tokenizer.encode("hello")
    assert len(ids) == len(plain) + 2


def test_edit_markers_tokenize_as_specials(tiny_tokenizer: LydenTokenizer) -> None:
    ids = tiny_tokenizer.encode(SAMPLE[5])
    special = {tiny_tokenizer.special_token_ids()[k] for k in ("<|edit|>", "<|/edit|>")}
    assert special.issubset(set(ids))


def test_load_persisted(tmp_path: Path) -> None:
    out = tmp_path / "tok"
    LydenTokenizer.train(SAMPLE * 50, vocab_size=512, out_dir=out)
    loaded = LydenTokenizer.load(out)
    assert loaded.encode("hello", add_eos=True) == LydenTokenizer.load(out).encode(
        "hello", add_eos=True
    )


def test_load_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="prepare_data"):
        LydenTokenizer.load(tmp_path / "nowhere")
