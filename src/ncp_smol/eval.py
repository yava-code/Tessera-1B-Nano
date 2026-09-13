from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from .data import TokenBatcher, TokenCorpus
from .experiment import load_experiment
from .modeling import NcpSmolForCausalLM
from .runtime import make_model, seed_everything, token_cross_entropy, write_json


@torch.no_grad()
def evaluate_checkpoint(
    config_path: str | Path,
    checkpoint: str | Path,
    *,
    batches: int | None = None,
) -> dict[str, Any]:
    config = load_experiment(config_path)
    seed_everything(config.run.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(config, Path(checkpoint)).to(device).eval()
    corpus = TokenCorpus(
        Path(config.data.cache_dir) / "validation.bin",
        config.data.sequence_length,
    )
    count = batches or config.train.eval_batches
    modes = ["predicted", "zero", "shuffle"] if config.run.mode == "ncp" else ["predicted"]
    results: dict[str, Any] = {}

    for mode in modes:
        batcher = TokenBatcher(
            corpus,
            config.optim.micro_batch_size,
            seed=config.run.seed + 20_000,
            repeat=True,
        )
        loss_sum = 0.0
        offset_sum = torch.zeros(config.model.chunk_size, device=device)
        offset_count = torch.zeros(config.model.chunk_size, device=device)
        indices = []
        for _ in range(count):
            input_ids = batcher.next().to(device, non_blocking=True)
            kwargs = {"concept_mode": mode} if config.run.mode == "ncp" else {}
            outputs = model(input_ids=input_ids, labels=input_ids, **kwargs)
            token_loss = token_cross_entropy(outputs.logits, input_ids)
            loss_sum += token_loss.mean().item()
            positions = torch.arange(1, input_ids.shape[1], device=device) % config.model.chunk_size
            for offset in range(config.model.chunk_size):
                selected = token_loss[:, positions == offset]
                offset_sum[offset] += selected.sum()
                offset_count[offset] += selected.numel()
            if isinstance(model, NcpSmolForCausalLM):
                indices.append(outputs.code_indices.cpu())

        mean_loss = loss_sum / count
        entry: dict[str, Any] = {
            "ntp_loss": mean_loss,
            "perplexity": math.exp(min(mean_loss, 20)),
            "loss_by_chunk_offset": {
                str(index): float(offset_sum[index] / offset_count[index].clamp_min(1))
                for index in range(config.model.chunk_size)
            },
        }
        if isinstance(model, NcpSmolForCausalLM):
            entry.update(model.quantizer.usage(torch.cat(indices)))
        results[mode] = entry

    if config.run.mode == "ncp":
        predicted = results["predicted"]["ntp_loss"]
        results["intervention"] = {
            "zero_minus_predicted": results["zero"]["ntp_loss"] - predicted,
            "shuffle_minus_predicted": results["shuffle"]["ntp_loss"] - predicted,
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate loss and concept interventions")
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("--batches", type=int)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = evaluate_checkpoint(args.config, args.checkpoint, batches=args.batches)
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
