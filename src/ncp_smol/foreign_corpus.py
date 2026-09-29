"""Pack a foreign-corpus token pool with the same tokenizer and EOS packing as prepare.py."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .data import sha256_file

FOREIGN_POOLS: dict[str, dict[str, str]] = {
    "wikipedia": {
        "dataset": "wikimedia/wikipedia",
        "subset": "20231101.en",
        "split": "train",
        "text_column": "text",
    },
    "code": {
        # codeparrot/github-code still ships only a legacy dataset script (dead under
        # datasets>=3); code_search_net is parquet-native, ungated, and its
        # func_code_string column is plain source code.
        "dataset": "code_search_net",
        "subset": "python",
        "split": "train",
        "text_column": "func_code_string",
    },
}


def _token_stream(
    rows: Any,
    tokenizer: PreTrainedTokenizerBase,
    text_column: str,
    batch_size: int = 256,
) -> Any:
    batch: list[str] = []
    for row in rows:
        text = row.get(text_column)
        if not text:
            continue
        batch.append(text)
        if len(batch) < batch_size:
            continue
        for ids in tokenizer(batch, add_special_tokens=False)["input_ids"]:
            yield from ids
            yield tokenizer.eos_token_id
        batch.clear()
    if batch:
        for ids in tokenizer(batch, add_special_tokens=False)["input_ids"]:
            yield from ids
            yield tokenizer.eos_token_id


def _write_tokens(path: Path, stream: Any, count: int) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    written = 0
    buffer = np.empty(min(count, 1_000_000), dtype=np.uint32)
    with temporary.open("wb") as handle:
        while written < count:
            size = 0
            while size < len(buffer) and written + size < count:
                try:
                    buffer[size] = next(stream)
                except StopIteration:
                    handle.write(buffer[:size].tobytes())
                    temporary.replace(path)
                    return written + size
                size += 1
            handle.write(buffer[:size].tobytes())
            written += size
    temporary.replace(path)
    return written


def prepare_foreign_pool(
    name: str,
    *,
    tokenizer_name: str,
    tokenizer_revision: str | None,
    output_dir: str | Path,
    tokens: int = 10_000_000,
    seed: int = 17,
    shuffle_buffer: int = 10_000,
) -> dict[str, Any]:
    """Build one foreign token pool, mirroring ncp_smol.prepare packing rules."""
    from datasets import load_dataset

    if name not in FOREIGN_POOLS:
        raise ValueError(f"unknown foreign pool: {name}")
    spec = FOREIGN_POOLS[name]
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        revision=tokenizer_revision,
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer must define eos_token_id")
    rows = load_dataset(
        spec["dataset"],
        spec["subset"],
        split=spec["split"],
        streaming=True,
    ).shuffle(seed=seed, buffer_size=shuffle_buffer)
    output = Path(output_dir) / name
    written = _write_tokens(
        output / "pool.bin",
        _token_stream(rows, tokenizer, spec["text_column"]),
        tokens,
    )
    payload: dict[str, Any] = {
        "pool": name,
        "dataset": spec["dataset"],
        "subset": spec["subset"],
        "tokenizer": tokenizer_name,
        "tokenizer_revision": tokenizer_revision,
        "tokens": written,
        "sha256": sha256_file(output / "pool.bin"),
    }
    (output / "metadata.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return payload
