import json
from pathlib import Path

from ncp_smol.publish import build_card


def test_model_card_uses_recorded_metrics(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        """
run:
  name: test
  mode: ncp
  output_dir: runs/test
model:
  base_model: example/base
  chunk_size: 4
  segments: 4
  codebook_size: 8
  concept_layers: 1
  insert_layer: 1
  dtype: float32
data:
  dataset: example/data
  subset: null
  revision: abc
  train_split: train
  validation_split: validation
  text_column: text
  cache_dir: data
  sequence_length: 16
  train_tokens: 64
  validation_tokens: 32
optim:
  micro_batch_size: 1
  grad_accum_steps: 1
""".strip(),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"tokens_seen": 64, "billable_seconds": 3600}),
        encoding="utf-8",
    )
    evaluation = tmp_path / "eval.json"
    evaluation.write_text(
        json.dumps(
            {
                "predicted": {"ntp_loss": 2.5, "perplexity": 12.1825},
                "intervention": {
                    "zero_minus_predicted": 0.1,
                    "shuffle_minus_predicted": 0.2,
                },
            }
        ),
        encoding="utf-8",
    )

    card = build_card(config, checkpoint, evaluation)

    assert "Held-out NTP loss | 2.5000" in card
    assert "Training tokens | 64" in card
    assert "Tracked compute estimate | $2.74" in card
    assert "# Tessera-1B-Nano" in card
    assert "Paragon Intelligence Labs" in card
    assert 'AutoModelForCausalLM.from_pretrained("REPO_ID", trust_remote_code=True)' in card
    assert "costs +0.1000 nats" in card


def test_model_card_includes_codebook_row(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        """
run:
  name: test
  mode: ncp
  output_dir: runs/test
model:
  base_model: example/base
  chunk_size: 4
  segments: 4
  codebook_size: 8
  concept_layers: 1
  insert_layer: 1
  dtype: float32
data:
  dataset: example/data
  subset: null
  revision: abc
  train_split: train
  validation_split: validation
  text_column: text
  cache_dir: data
  sequence_length: 16
  train_tokens: 64
  validation_tokens: 32
optim:
  micro_batch_size: 1
  grad_accum_steps: 1
""".strip(),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"tokens_seen": 64, "billable_seconds": 3600}),
        encoding="utf-8",
    )
    evaluation = tmp_path / "eval.json"
    evaluation.write_text(
        json.dumps(
            {
                "predicted": {
                    "ntp_loss": 2.5,
                    "perplexity": 12.1825,
                    "codebook_perplexity": 7.5476508140563965,
                    "codebook_usage": 0.856249988079071,
                }
            }
        ),
        encoding="utf-8",
    )

    card = build_card(config, checkpoint, evaluation)

    assert "Codebook perplexity / usage | 7.5477 / 85.6%" in card


def test_model_card_gate_branch(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        """
run:
  name: tinystories-overfit
  mode: ncp
  output_dir: runs/test
model:
  base_model: example/base
  chunk_size: 4
  segments: 4
  codebook_size: 8
  concept_layers: 1
  insert_layer: 1
  dtype: float32
data:
  dataset: roneneldan/TinyStories
  subset: null
  revision: abc
  train_split: train
  validation_split: validation
  text_column: text
  cache_dir: data
  sequence_length: 16
  train_tokens: 64
  validation_tokens: 32
optim:
  micro_batch_size: 1
  grad_accum_steps: 1
""".strip(),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"tokens_seen": 64, "billable_seconds": 3600}),
        encoding="utf-8",
    )
    evaluation = tmp_path / "eval.json"
    evaluation.write_text(
        json.dumps(
            {
                "predicted": {
                    "ntp_loss": 2.5,
                    "perplexity": 12.1825,
                    "codebook_perplexity": 2.43,
                    "codebook_usage": 0.406,
                },
                "intervention": {
                    "zero_minus_predicted": 0.028,
                    "shuffle_minus_predicted": 0.0,
                },
            }
        ),
        encoding="utf-8",
    )

    card = build_card(config, checkpoint, evaluation)

    assert "# Tessera-135M-Gate" in card
    assert "architecture-gate checkpoint" in card
    assert "overfit" in card
    assert "costs +0.0280 nats" in card
    assert "Codebook perplexity / usage | 2.4300 / 40.6%" in card


def test_model_card_baseline_branch(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        """
run:
  name: test
  mode: ntp
  output_dir: runs/test
model:
  base_model: example/base
  chunk_size: 4
  segments: 4
  codebook_size: 8
  concept_layers: 1
  insert_layer: 1
  dtype: float32
data:
  dataset: example/data
  subset: null
  revision: abc
  train_split: train
  validation_split: validation
  text_column: text
  cache_dir: data
  sequence_length: 16
  train_tokens: 64
  validation_tokens: 32
optim:
  micro_batch_size: 1
  grad_accum_steps: 1
""".strip(),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"tokens_seen": 64, "billable_seconds": 3600}),
        encoding="utf-8",
    )
    evaluation = tmp_path / "eval.json"
    evaluation.write_text(
        json.dumps({"predicted": {"ntp_loss": 2.5, "perplexity": 12.1825}}),
        encoding="utf-8",
    )

    card = build_card(config, checkpoint, evaluation)

    assert "# Tessera-1B-Nano-Base" in card
    assert "matched NTP-only baseline" in card
    assert "Tessera-1B-Nano" in card
    assert "Zero feedback delta" not in card
    assert "AutoModelForCausalLM.from_pretrained(\"REPO_ID\")" in card
    assert "trust_remote_code" not in card
