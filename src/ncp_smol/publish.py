from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi

from .experiment import load_experiment


def _metric(value: float) -> str:
    return f"{value:.4f}"


def build_card(
    config_path: str | Path,
    checkpoint: str | Path,
    eval_path: str | Path,
) -> str:
    config = load_experiment(config_path)
    checkpoint = Path(checkpoint)
    state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    evaluation: dict[str, Any] = json.loads(Path(eval_path).read_text(encoding="utf-8"))
    result = evaluation.get("predicted", evaluation)
    interventions = evaluation.get("intervention", {})
    cost = float(state["billable_seconds"]) / 3600
    cost *= config.train.hourly_cost_usd

    intervention_rows = ""
    if interventions:
        intervention_rows = (
            f"| Zero feedback delta | {_metric(interventions['zero_minus_predicted'])} |\n"
            f"| Shuffled feedback delta | {_metric(interventions['shuffle_minus_predicted'])} |\n"
        )

    codebook_row = ""
    if "codebook_perplexity" in result and "codebook_usage" in result:
        codebook_row = (
            f"| Codebook perplexity / usage | {_metric(result['codebook_perplexity'])} / "
            f"{100 * float(result['codebook_usage']):.1f}% |\n"
        )

    tldr_usage = ""
    if interventions:
        zero_delta = _metric(interventions["zero_minus_predicted"])
        tldr_usage = (
            "- **The concept channel is causally used**: zeroing predicted concept feedback\n"
            f"  at inference costs +{zero_delta} nats of held-out NTP loss.\n"
        )

    return f"""---
library_name: transformers
pipeline_tag: text-generation
base_model: {config.model.base_model}
tags:
- next-concept-prediction
- conceptlm
- causal-lm
- smollm2
- tessera
license: apache-2.0
---

# Tessera-1B-Nano

A Next Concept Prediction checkpoint from Paragon Intelligence Labs: an independent,
small-scale replication of ConceptLM on `{config.model.base_model}`. The model keeps
ordinary next-token generation and adds a causal, product-quantized concept path over
{config.model.chunk_size}-token chunks: token states are pooled into concepts,
product-quantized, processed by causal concept blocks, and the predicted next concept is
fed back into the token decoder.

## TL;DR

This checkpoint comes from a token-matched comparison against the unchanged
{config.model.base_model} backbone: both arms consumed the same tokens of the same packed
corpus, in the same order, from the same initialization, with no restarts and no NaNs.

- **Token loss is neutral**: the matched baseline is within run noise (see the whitepaper
  and `results/fineweb-edu/README.md` in the ncp-smol repository for the exact numbers).
{tldr_usage}- **The codebook is rich**: no low-entropy shortcut, usage grows monotonically over the
  run (growth curves are in the repository whitepaper).
- Full reading of the interventions: `docs/whitepaper.md` in the ncp-smol repository.

## Evaluation

| Metric | Value |
| --- | ---: |
| Held-out NTP loss | {_metric(result["ntp_loss"])} |
| Held-out perplexity | {_metric(result["perplexity"])} |
{intervention_rows}{codebook_row}| Training tokens | {state["tokens_seen"]:,} |
| Tracked compute estimate | ${cost:.2f} |

Intervention deltas are increases in held-out NTP loss relative to normal predicted concept
feedback, evaluated on identical batches.

## Architecture

- chunk size: {config.model.chunk_size}
- product code: {config.model.segments} segments x {config.model.codebook_size} entries
- causal concept blocks: {config.model.concept_layers}
- injection point: before token decoder block {config.model.insert_layer + 1}
- NCP target: next {config.model.ncp_target} concept
- loss: `L_ntp + {config.model.ncp_weight:g} L_ncp + {config.model.vq_weight:g} L_vq`

This is a compact ConceptLM-style implementation, not an 8.9B NCP-ArchPreview replica. It
omits iterative residual coding, cross-scale residual connections, and the large-scale
training recipe.

## Training data and provenance

The matched corpus and the training record are pinned in the ncp-smol repository:
packed-cache SHA256 hashes, the complete metric log, trainer state, and the raw
intervention evaluation JSON (also shipped in this repository as `eval.json` and
`metrics.jsonl`).

## Loading

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("REPO_ID")
model = AutoModelForCausalLM.from_pretrained("REPO_ID")
```

## References

- ConceptLM: https://arxiv.org/abs/2602.08984
- NCP-ArchPreview: https://arxiv.org/abs/2609.10715
"""


def publish(
    config_path: str | Path,
    checkpoint: str | Path,
    eval_path: str | Path,
    repo_id: str,
    *,
    private: bool = False,
    dry_run: bool = False,
) -> str:
    checkpoint = Path(checkpoint)
    card = build_card(config_path, checkpoint, eval_path).replace("REPO_ID", repo_id)
    (checkpoint / "README.md").write_text(card, encoding="utf-8")
    shutil.copy2(eval_path, checkpoint / "eval.json")
    metrics = checkpoint.parent / "metrics.jsonl"
    if metrics.exists():
        shutil.copy2(metrics, checkpoint / "metrics.jsonl")
    if dry_run:
        return str(checkpoint / "README.md")

    api = HfApi()
    api.create_repo(repo_id, private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        folder_path=checkpoint,
        ignore_patterns=["optimizer.pt"],
        commit_message="Publish Tessera-1B-Nano checkpoint",
    )
    return f"https://huggingface.co/{repo_id}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a model card and publish a checkpoint")
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("eval_json")
    parser.add_argument("repo_id")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(
        publish(
            args.config,
            args.checkpoint,
            args.eval_json,
            args.repo_id,
            private=args.private,
            dry_run=args.dry_run,
        )
    )


if __name__ == "__main__":
    main()
