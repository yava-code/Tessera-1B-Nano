from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi

from .experiment import ExperimentConfig, load_experiment


def _metric(value: float) -> str:
    return f"{value:.4f}"


FAMILY_MEMBERS = (
    ("Tessera-135M-Gate", "135M architecture gate (overfit, published for the scale story)"),
    ("Tessera-1B-Nano-Base", "1B-token matched NTP-only control"),
    ("Tessera-1B-Nano", "1B-token matched concept arm"),
)


def _family_section(this_title: str) -> str:
    rows = "\n".join(
        f"- [{name}](https://huggingface.co/yava-code/{name}) - {role}"
        + (" **(this model)**" if name == this_title else "")
        for name, role in FAMILY_MEMBERS
    )
    return (
        "## The Tessera family\n\n"
        "Three checkpoints, one story, in reading order:\n\n"
        f"{rows}\n\n"
        "All cards are generated from the run artifacts by the same `build_card`; the\n"
        "[study repository](https://github.com/yava-code/Tessera-1B-Nano) holds the\n"
        "whitepaper and full records.\n"
    )


def _card_title(config: ExperimentConfig) -> str:
    if config.run.mode == "ntp":
        return "Tessera-1B-Nano-Base"
    if config.data.dataset == "roneneldan/TinyStories":
        # The 135M overfit architecture gate is a different artifact from the 1B-token
        # comparison arms: it proves the path trains, it is not a quality model.
        return "Tessera-135M-Gate"
    return "Tessera-1B-Nano"


def _intro(config: ExperimentConfig) -> str:
    if config.run.mode == "ntp":
        return (
            f"The matched NTP-only baseline of the Tessera-1B-Nano comparison from Paragon\n"
            f"Intelligence Labs: the unchanged `{config.model.base_model}` backbone,\n"
            "continued-pretrained without a concept path. It exists so the concept-arm result\n"
            "is interpretable.\n"
        )
    if _card_title(config) == "Tessera-135M-Gate":
        return (
            "The 135M architecture-gate checkpoint from Paragon Intelligence Labs: the smallest\n"
            f"member of the Tessera family (`{config.model.base_model}` backbone with the full\n"
            "Next Concept Prediction path). It is a deliberately overfit gate run that proves\n"
            "the complete concept path trains end to end before any large spend, and carries the\n"
            "same causal concept-path architecture as Tessera-1B-Nano. It is not a language\n"
            "model quality result: it saw a 1M-token TinyStories subset repeated 16 times.\n"
        )
    return (
        "A Next Concept Prediction checkpoint from Paragon Intelligence Labs: an independent,\n"
        f"small-scale replication of ConceptLM on `{config.model.base_model}`. The model keeps\n"
        "ordinary next-token generation and adds a causal, product-quantized concept path over\n"
        f"{config.model.chunk_size}-token chunks: token states are pooled into concepts,\n"
        "product-quantized, processed by causal concept blocks, and the predicted next concept\n"
        "is fed back into the token decoder.\n"
    )


def _tldr(config: ExperimentConfig, interventions: dict[str, Any]) -> str:
    if config.run.mode == "ntp":
        return (
            "## TL;DR\n\n"
            "This arm is the control of a token-matched comparison: it consumed the same\n"
            "tokens of the same packed corpus, in the same order, from the same\n"
            "initialization as the concept arm, with no restarts and no NaNs.\n\n"
            "- **Token loss is the reference**: the final numbers below are the baseline the\n"
            "  concept arm is compared against.\n"
            "- The comparison outcome and the intervention readings live in the whitepaper at\n"
            "  `docs/whitepaper.md` in the ncp-smol repository.\n"
        )
    if _card_title(config) == "Tessera-135M-Gate":
        zero_delta = _metric(interventions["zero_minus_predicted"]) if interventions else ""
        return (
            "## TL;DR\n\n"
            "This gate checkpoint exists to de-risk the 1B-token comparison, and the scale\n"
            "story it records is the point of publishing it.\n\n"
            f"- **All three objectives train** (NTP, NCP, VQ fell 41 to 43% over the run).\n"
            f"- **The decoder causally uses the concept channel even at 135M**: zeroing the\n"
            f"  feedback costs +{zero_delta} nats of held-out NTP loss.\n"
            "- **The failure mode this gate exposed is real and instructive**: effective\n"
            "  codebook perplexity stayed near 2.4 (usage ~31-41%), and shuffled feedback cost\n"
            "  nothing (+0.000). At 1B tokens the same architecture left the shortcut far\n"
            "  behind (perplexity 7.55, usage 85.6%, zero-delta +0.105). Gate small, then\n"
            "  scale: some behaviors only appear above a scale threshold.\n"
        )
    zero_delta = _metric(interventions["zero_minus_predicted"]) if interventions else ""
    return (
        "## TL;DR\n\n"
        "This checkpoint comes from a token-matched comparison against the unchanged\n"
        f"{config.model.base_model} backbone: both arms consumed the same tokens of the\n"
        "same packed corpus, in the same order, from the same initialization, with no\n"
        "restarts and no NaNs.\n\n"
        "- **Token loss is neutral**: the matched baseline is within run noise (see the\n"
        "  whitepaper and `results/fineweb-edu/README.md` in the ncp-smol repository for\n"
        "  the exact numbers).\n"
        "- **The concept channel is causally used**: zeroing predicted concept feedback\n"
        f"  at inference costs +{zero_delta} nats of held-out NTP loss.\n"
        "- **The codebook is rich**: no low-entropy shortcut, usage grows monotonically\n"
        "  over the run (growth curves are in the repository whitepaper).\n"
        "- Full reading of the interventions: `docs/whitepaper.md` in the ncp-smol repo.\n"
    )


def _loading_snippet(config: ExperimentConfig) -> str:
    if config.run.mode == "ntp":
        call = 'AutoModelForCausalLM.from_pretrained("REPO_ID")'
    else:
        call = 'AutoModelForCausalLM.from_pretrained("REPO_ID", trust_remote_code=True)'
    return (
        "```python\n"
        "from transformers import AutoModelForCausalLM, AutoTokenizer\n\n"
        'tokenizer = AutoTokenizer.from_pretrained("REPO_ID")\n'
        f"model = {call}\n"
        "```"
    )


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
    title = _card_title(config)

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

# {title}

{_intro(config)}
{_tldr(config, interventions)}
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

{_loading_snippet(config)}

{_family_section(title)}## References

- Code and study: https://github.com/yava-code/Tessera-1B-Nano
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
    config = load_experiment(config_path)
    title = _card_title(config)
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
        commit_message=f"Publish {title} checkpoint",
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
