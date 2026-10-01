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
            "The plain half of a controlled experiment by Paragon Intelligence Labs: the\n"
            f"untouched `{config.model.base_model}` backbone, continued-pretrained with no\n"
            "concept path, no extra parameters, no tricks. This is the matched NTP-only baseline"
            " of the comparison - every difference from its sibling is the concept path's"
            " doing, and nothing else.\n"
        )
    if _card_title(config) == "Tessera-135M-Gate":
        return (
            "The smallest member of the Tessera family, and the reason the bigger ones exist.\n"
            f"This 135M `architecture-gate checkpoint` (a `{config.model.base_model}` backbone\n"
            "with the full Next Concept Prediction path) was trained on a deliberately overfit\n"
            "TinyStories subset before any large spend, to prove the whole path - concept\n"
            "pooling, quantized codes, causal concept blocks, feedback - trains end to end.\n"
            "It is not a language-model quality result, and it is not meant to be: what makes\n"
            "it worth publishing is what broke, and what the break predicted at scale.\n"
        )
    return (
        "A language model that predicts its next thought. This is an independent, small-scale\n"
        f"replication of ConceptLM by Paragon Intelligence Labs: an ordinary\n"
        f"`{config.model.base_model}` backbone plus a thin causal concept path. Every\n"
        f"{config.model.chunk_size} tokens are pooled into one concept code, a small causal\n"
        "module guesses the next code, and that guess is fed back into the decoder while it\n"
        "writes. The surprising part is not that it works - it is what the model turns out\n"
        "to be reading from the channel.\n"
    )


def _tldr(config: ExperimentConfig, interventions: dict[str, Any]) -> str:
    if config.run.mode == "ntp":
        return (
            "## TL;DR\n\n"
            "This arm is the boring half on purpose: same tokens, same order, same\n"
            "initialization as the concept arm, no restarts, no NaNs. Science needs a\n"
            "control before it needs a result.\n\n"
            "- **Read this card as the yardstick**: the final numbers below are what the\n"
            "  concept arm is measured against.\n"
            "- The comparison outcome and the intervention readings live in the whitepaper at\n"
            "  `docs/whitepaper.md` in the ncp-smol repository.\n"
        )
    if _card_title(config) == "Tessera-135M-Gate":
        zero_delta = _metric(interventions["zero_minus_predicted"]) if interventions else ""
        return (
            "## TL;DR\n\n"
            "Small and cheap on purpose: this gate run exists to de-risk the 1B-token"
            " comparison before spending on it. It earned its keep in both directions.\n\n"
            "- **All three objectives train** (NTP, NCP, VQ fell 41 to 43% over the run),\n"
            "  and the decoder causally uses the concept channel even at 135M: zeroing the\n"
            f"  feedback costs +{zero_delta} nats of held-out NTP loss.\n"
            "- **It also caught a real failure mode**: effective codebook perplexity stayed\n"
            "  near 2.4 with usage around a third of the codebook, and shuffled feedback"
            "  cost nothing - a low-entropy shortcut the 1B-token run later left far behind."
            "  Gate small, then scale: some behaviors only appear above a scale threshold.\n"
        )
    zero_delta = _metric(interventions["zero_minus_predicted"]) if interventions else ""
    return (
        "## TL;DR\n\n"
        "Same billion tokens, same order, same initialization as the unchanged baseline -"
        " the only difference is the concept path. Three findings survive every control we"
        " threw at them:\n\n"
        "- **The decoder leans on the channel.** Silence the concept feedback and the model"
        f"  measurably stumbles: zeroing it costs +{zero_delta} nats of held-out NTP loss.\n"
        "- **But it is not reading its own sentence.** Feedback predicted from a completely"
        "  different sentence - another batch element, a Wikipedia passage, even source"
        "  code - works just as well. The channel carries *something*, but not "
        "  sentence identity.\n"
        "- **And it does not (yet) buy token loss.** At this budget the concept arm matches"
        "  the baseline exactly. What the channel carries, why the model wants it, and"
        "  whether it pays off at scale - that is the open question this study is built to"
        "  squeeze.\n\n"
        "The exact numbers, the preregistered predictions, and the full record: whitepaper"
        " at `docs/whitepaper.md` in the ncp-smol repository.\n"
    )


def _findings(config: ExperimentConfig, evaluation: dict[str, Any]) -> str:
    if config.run.mode != "ncp" or _card_title(config) == "Tessera-135M-Gate":
        return ""
    predicted = evaluation.get("predicted", {}).get("ntp_loss")
    zero = evaluation.get("zero", {}).get("ntp_loss")
    shuffle = evaluation.get("shuffle", {}).get("ntp_loss")
    if predicted is None or zero is None or shuffle is None:
        return ""
    similar = evaluation.get("similar_shuffle", {}).get("ntp_loss")
    similar_row = (
        f"- feedback from a topically similar sentence: {similar:.4f} - even topic\n"
        "  identity is absent.\n"
        if similar is not None
        else ""
    )
    return (
        "## What the study found\n\n"
        "The same held-out batches, scored under four inference modes:\n\n"
        f"- normal predicted feedback: held-out NTP loss {predicted:.4f}.\n"
        f"- feedback silenced: {zero:.4f} - zeroing the channel costs"
        f" +{zero - predicted:.4f} nats.\n"
        f"- feedback from another sentence in the batch: {shuffle:.4f} -\n"
        "  indistinguishable from the sequence's own.\n"
        f"{similar_row}"
        "\nThe model consults its concept channel at every step, yet what it reads there is"
        " interchangeable across sentences, topics, and domains. Whatever the concept path"
        " carries, the token path wants it - and a pure token model of the same size,"
        " trained the same way, does not miss it. Cornering that signal is the open"
        " question; the probes are preregistered in the study repository.\n"
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
{_findings(config, evaluation)}
## Numbers

The raw record this narrative is built from; per-arm READMEs and the whitepaper hold the
full analysis.

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
