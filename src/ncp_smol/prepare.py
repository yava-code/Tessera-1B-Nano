from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .data import CorpusMetadata, sha256_file
from .experiment import DataConfig, load_experiment


def _documents(config: DataConfig, seed: int) -> tuple[Iterable[dict], Iterable[dict] | None]:
    from datasets import load_dataset

    common = {
        "path": config.dataset,
        "name": config.subset,
        "revision": config.revision,
        "streaming": True,
    }
    train = load_dataset(split=config.train_split, **common)
    validation = (
        load_dataset(split=config.validation_split, **common) if config.validation_split else None
    )
    train = train.shuffle(seed=seed, buffer_size=config.shuffle_buffer)
    if validation is not None:
        validation = validation.shuffle(seed=seed + 1, buffer_size=config.shuffle_buffer)
    return train, validation


def _token_stream(
    rows: Iterable[dict],
    tokenizer: PreTrainedTokenizerBase,
    text_column: str,
    batch_size: int = 256,
) -> Iterator[int]:
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


def _write_tokens(path: Path, stream: Iterator[int], count: int) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    buffer = np.empty(min(count, 1_000_000), dtype=np.uint32)
    with path.open("wb") as handle:
        while written < count:
            size = 0
            while size < len(buffer) and written + size < count:
                try:
                    buffer[size] = next(stream)
                except StopIteration:
                    handle.write(buffer[:size].tobytes())
                    return written + size
                size += 1
            handle.write(buffer[:size].tobytes())
            written += size
    return written


def prepare(config_path: str | Path) -> dict[str, object]:
    experiment = load_experiment(config_path)
    config = experiment.data
    output = Path(config.cache_dir)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        experiment.model.base_model,
        revision=experiment.model.revision,
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer must define eos_token_id")

    train_rows, validation_rows = _documents(config, experiment.run.seed)
    cache_train_tokens = config.cache_train_tokens or config.train_tokens
    if validation_rows is None:
        shared = _token_stream(train_rows, tokenizer, config.text_column)
        validation_written = _write_tokens(
            output / "validation.bin", shared, config.validation_tokens
        )
        train_written = _write_tokens(output / "train.bin", shared, cache_train_tokens)
    else:
        validation_written = _write_tokens(
            output / "validation.bin",
            _token_stream(validation_rows, tokenizer, config.text_column),
            config.validation_tokens,
        )
        train_written = _write_tokens(
            output / "train.bin",
            _token_stream(train_rows, tokenizer, config.text_column),
            cache_train_tokens,
        )

    metadata = CorpusMetadata(
        dataset=config.dataset,
        subset=config.subset,
        revision=config.revision,
        tokenizer=experiment.model.base_model,
        eos_token_id=tokenizer.eos_token_id,
        train_tokens=train_written,
        validation_tokens=validation_written,
        sequence_length=config.sequence_length,
        tokenizer_revision=experiment.model.revision,
    )
    metadata.train_sha256 = sha256_file(output / "train.bin")
    metadata.validation_sha256 = sha256_file(output / "validation.bin")
    payload = asdict(metadata)
    (output / "metadata.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic packed-token corpora")
    parser.add_argument("config")
    args = parser.parse_args()
    print(json.dumps(prepare(args.config), indent=2))


if __name__ == "__main__":
    main()
