from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


@dataclass
class CorpusMetadata:
    dataset: str
    subset: str | None
    revision: str | None
    tokenizer: str
    eos_token_id: int
    train_tokens: int
    validation_tokens: int
    sequence_length: int | None = None
    tokenizer_revision: str | None = None
    dtype: str = "uint32"
    train_sha256: str | None = None
    validation_sha256: str | None = None


class TokenCorpus:
    def __init__(self, path: str | Path, sequence_length: int) -> None:
        self.path = Path(path)
        self.sequence_length = sequence_length
        self.tokens = np.memmap(self.path, dtype=np.uint32, mode="r")
        self.blocks = len(self.tokens) // sequence_length
        if self.blocks == 0:
            raise ValueError(f"{self.path} has no complete token blocks")

    def __len__(self) -> int:
        return self.blocks

    def block(self, index: int) -> Tensor:
        start = index * self.sequence_length
        array = np.array(self.tokens[start : start + self.sequence_length], dtype=np.int64)
        return torch.from_numpy(array)


class TokenBatcher:
    def __init__(
        self,
        corpus: TokenCorpus,
        batch_size: int,
        *,
        seed: int,
        repeat: bool,
        epoch: int = 0,
        cursor: int = 0,
    ) -> None:
        self.corpus = corpus
        self.batch_size = batch_size
        self.seed = seed
        self.repeat = repeat
        self.epoch = epoch
        self.cursor = cursor
        self._order = self._make_order(epoch)

    def _make_order(self, epoch: int) -> np.ndarray:
        order = np.arange(len(self.corpus))
        np.random.default_rng(self.seed + epoch).shuffle(order)
        return order

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self.epoch, "cursor": self.cursor}

    def _advance_epoch(self) -> None:
        if not self.repeat:
            raise StopIteration
        self.epoch += 1
        self.cursor = 0
        self._order = self._make_order(self.epoch)

    def next(self) -> Tensor:
        if self.cursor + self.batch_size > len(self._order):
            self._advance_epoch()
        indices = self._order[self.cursor : self.cursor + self.batch_size]
        self.cursor += self.batch_size
        return torch.stack([self.corpus.block(int(index)) for index in indices])


def sha256_file(path: str | Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_metadata(cache_dir: str | Path) -> CorpusMetadata:
    values = json.loads((Path(cache_dir) / "metadata.json").read_text(encoding="utf-8"))
    return CorpusMetadata(**values)
