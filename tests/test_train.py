import json
from pathlib import Path

import numpy as np
from test_modeling import tiny_model

from ncp_smol import train


class DummyTokenizer:
    def save_pretrained(self, path: Path) -> None:
        (Path(path) / "tokenizer_config.json").write_text("{}", encoding="utf-8")


def test_trainer_writes_resumable_checkpoint(tmp_path: Path, monkeypatch) -> None:
    cache = tmp_path / "data"
    cache.mkdir()
    np.arange(64, dtype=np.uint32).tofile(cache / "train.bin")
    np.arange(32, dtype=np.uint32).tofile(cache / "validation.bin")
    (cache / "metadata.json").write_text(
        json.dumps(
            {
                "dataset": "synthetic",
                "subset": None,
                "revision": "test",
                "tokenizer": "test",
                "eos_token_id": 2,
                "train_tokens": 64,
                "validation_tokens": 32,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "run"
    config = tmp_path / "train.yaml"
    config.write_text(
        f"""
run:
  name: integration
  mode: ncp
  output_dir: '{output.as_posix()}'
  seed: 17
  resume: true
model:
  base_model: test
  chunk_size: 4
  segments: 4
  codebook_size: 8
  concept_layers: 1
  insert_layer: 1
  dtype: float32
data:
  dataset: synthetic
  subset: null
  revision: test
  train_split: train
  validation_split: validation
  text_column: text
  cache_dir: '{cache.as_posix()}'
  sequence_length: 16
  train_tokens: 64
  validation_tokens: 32
  overfit: false
optim:
  learning_rate: 0.001
  micro_batch_size: 1
  grad_accum_steps: 1
train:
  log_every_steps: 1
  eval_every_tokens: 32
  save_every_tokens: 32
  eval_batches: 1
  max_wall_time_minutes: 5
  run_budget_usd: 1
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(train, "make_model", lambda *_: tiny_model())
    monkeypatch.setattr(
        train.AutoTokenizer, "from_pretrained", lambda *_args, **_kwargs: DummyTokenizer()
    )

    result = train.run(config)
    checkpoint = Path(result["checkpoint"])
    state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))

    assert result["step"] == 4
    assert result["tokens"] == 64
    assert state["batcher"] == {"epoch": 0, "cursor": 4}
    assert (checkpoint / "model.safetensors").exists()
    assert (checkpoint / "optimizer.pt").exists()
    assert (checkpoint / "tokenizer_config.json").exists()
