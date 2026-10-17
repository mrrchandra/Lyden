"""Tokenizer training and loading.

A byte-level BPE tokenizer built with HuggingFace ``tokenizers``, trained from
scratch on a sample of the training corpus. Special tokens cover bos/eos/pad
and the CLM edit/context markers so the model can emit them as native tokens.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers


class LydenTokenizer:
    """Wrapper around a HF ``Tokenizer`` with Lyden's special-token registry."""

    #: Special tokens added during training, in stable id order (ids 0..6).
    SPECIAL_TOKENS: tuple[str, ...] = (
        "<|pad|>",
        "<|bos|>",
        "<|eos|>",
        "<|edit|>",
        "<|/edit|>",
        "<|context|>",
        "<|/context|>",
    )

    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer
        self._special_ids = {tok: i for i, tok in enumerate(self.SPECIAL_TOKENS)}

    # ------------------------------------------------------------------ train/load
    @classmethod
    def train(
        cls,
        sample: Iterable[str],
        vocab_size: int,
        out_dir: str | Path,
    ) -> "LydenTokenizer":
        """Train byte-level BPE on a text sample and persist it to ``out_dir``.

        ``sample`` should be large enough that BPE learns reasonable merges
        (tens of thousands of documents work well at 32K vocab).
        """
        tok = Tokenizer(models.BPE(unk_token=None))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        tok.post_processor = processors.ByteLevel(trim_offsets=False)

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            show_progress=False,
            special_tokens=list(cls.SPECIAL_TOKENS),
        )
        tok.train_from_iterator(sample, trainer)
        out = cls(tok)
        out.save(out_dir)
        return out

    @classmethod
    def load(cls, tokenizer_dir: str | Path) -> "LydenTokenizer":
        """Load a previously trained tokenizer directory."""
        tokenizer_dir = Path(tokenizer_dir).expanduser()
        path = tokenizer_dir / "tokenizer.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Trained tokenizer not found at {path}; run scripts/prepare_data.py first"
            )
        return cls(Tokenizer.from_file(str(path)))

    # ------------------------------------------------------------------ properties
    @property
    def vocab_size(self) -> int:
        """Total vocab size including the seven special tokens."""
        return self.tokenizer.get_vocab_size(with_added_tokens=True)

    @property
    def pad_id(self) -> int:
        return self._special_ids["<|pad|>"]

    @property
    def bos_id(self) -> int:
        return self._special_ids["<|bos|>"]

    @property
    def eos_id(self) -> int:
        return self._special_ids["<|eos|>"]

    def special_token_ids(self) -> dict[str, int]:
        """Stable id mapping for the special-token registry."""
        return dict(self._special_ids)

    # ------------------------------------------------------------------ encode/decode
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """Encode text; optionally frame with the bos/eos token ids.

        Special markers in the text (e.g. ``<|edit|>``) encode to the dedicated
        special token because they were registered during training.
        """
        ids = self.tokenizer.encode(text).ids
        if add_bos:
            ids = [self.bos_id, *ids]
        if add_eos:
            ids = [*ids, self.eos_id]
        return ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        """Batched encode (parallelized in Rust; used by the pretokenizer)."""
        return [e.ids for e in self.tokenizer.encode_batch(texts)]

    def decode(self, ids: list[int], skip_special: bool = False) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special)

    # ------------------------------------------------------------------ persistence
    def save(self, out_dir: str | Path) -> None:
        out_dir = Path(out_dir).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(out_dir / "tokenizer.json"), pretty=False)
        meta = {
            "vocab_size": self.vocab_size,
            "special_tokens": list(self.SPECIAL_TOKENS),
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
