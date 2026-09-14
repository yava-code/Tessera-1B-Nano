from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .data import TokenBatcher, TokenCorpus, read_metadata
from .experiment import ExperimentConfig, load_experiment
from .modeling import NcpSmolForCausalLM
from .runtime import (
    cosine_schedule,
    latest_checkpoint,
    make_model,
    seed_everything,
    torch_dtype,
    write_json,
)


def _checkpoint_state(path: Path) -> dict[str, Any]:
    return json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))


def _save(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    output_dir: Path,
    step: int,
    tokens_seen: int,
    batcher: TokenBatcher,
    billable_seconds: float,
    tokenizer: PreTrainedTokenizerBase,
) -> Path:
    target = output_dir / f"step-{step:08d}"
    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(target, safe_serialization=True)
    tokenizer.save_pretrained(target)
    for name in ("experiment.json", "data_metadata.json"):
        shutil.copy2(output_dir / name, target / name)
    torch.save(
        {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
        target / "optimizer.pt",
    )
    write_json(
        target / "trainer_state.json",
        {
            "step": step,
            "tokens_seen": tokens_seen,
            "batcher": batcher.state_dict(),
            "billable_seconds": billable_seconds,
        },
    )
    write_json(output_dir / "latest.json", {"checkpoint": target.name})
    return target


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    config: ExperimentConfig,
    device: torch.device,
) -> dict[str, float]:
    corpus = TokenCorpus(
        Path(config.data.cache_dir) / "validation.bin",
        config.data.sequence_length,
    )
    batcher = TokenBatcher(
        corpus,
        config.optim.micro_batch_size,
        seed=config.run.seed + 10_000,
        repeat=True,
    )
    model.eval()
    totals: dict[str, float] = {}
    code_indices = []
    for _ in range(config.train.eval_batches):
        input_ids = batcher.next().to(device, non_blocking=True)
        outputs = model(input_ids=input_ids, labels=input_ids)
        metrics = {"loss": outputs.loss}
        if config.run.mode == "ncp":
            metrics.update(
                ntp_loss=outputs.ntp_loss,
                ncp_loss=outputs.ncp_loss,
                vq_loss=outputs.vq_loss,
            )
            code_indices.append(outputs.code_indices.cpu())
        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach())

    result = {name: value / config.train.eval_batches for name, value in totals.items()}
    if isinstance(model, NcpSmolForCausalLM):
        result.update(model.quantizer.usage(torch.cat(code_indices)))
    model.train()
    return result


def run(config_path: str | Path) -> dict[str, Any]:
    config = load_experiment(config_path)
    seed_everything(config.run.seed)
    output_dir = Path(config.run.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "experiment.json", asdict(config))
    metadata = read_metadata(config.data.cache_dir)
    expected_metadata = {
        "dataset": config.data.dataset,
        "subset": config.data.subset,
        "revision": config.data.revision,
        "tokenizer": config.model.base_model,
        "tokenizer_revision": config.model.revision,
        "sequence_length": config.data.sequence_length,
    }
    mismatches = {
        name: (getattr(metadata, name), expected)
        for name, expected in expected_metadata.items()
        if getattr(metadata, name) not in {None, expected}
    }
    if mismatches:
        raise ValueError(f"token cache metadata does not match config: {mismatches}")
    write_json(output_dir / "data_metadata.json", asdict(metadata))
    required_cache_tokens = config.data.cache_train_tokens or config.data.train_tokens
    if metadata.train_tokens < required_cache_tokens:
        raise ValueError("token cache is smaller than the requested training budget")
    cache_root = Path(config.data.cache_dir)
    expected_sizes = {
        "train.bin": metadata.train_tokens * 4,
        "validation.bin": metadata.validation_tokens * 4,
    }
    incomplete = {}
    for name, size in expected_sizes.items():
        path = cache_root / name
        actual = path.stat().st_size if path.exists() else 0
        if actual < size:
            incomplete[name] = actual
    if incomplete:
        raise ValueError(f"token cache files are incomplete: {incomplete}")

    resume_from = latest_checkpoint(output_dir) if config.run.resume else None
    state = _checkpoint_state(resume_from) if resume_from else {}
    model = make_model(config, resume_from)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.base_model,
        revision=config.model.revision,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if config.train.compile:
        if config.run.mode == "ncp":
            raise ValueError("torch.compile is not enabled for the hook-based NCP path")
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optim.learning_rate,
        betas=(config.optim.beta1, config.optim.beta2),
        eps=config.optim.eps,
        weight_decay=config.optim.weight_decay,
        fused=device.type == "cuda",
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_schedule(
            step,
            config.max_steps,
            config.optim.warmup_ratio,
            config.optim.min_lr_ratio,
        ),
    )
    if resume_from:
        saved = torch.load(resume_from / "optimizer.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])

    train_corpus = TokenCorpus(
        Path(config.data.cache_dir) / "train.bin",
        config.data.sequence_length,
    )
    batch_state = state.get("batcher", {})
    batcher = TokenBatcher(
        train_corpus,
        config.optim.micro_batch_size,
        seed=config.run.seed,
        repeat=config.data.overfit,
        epoch=batch_state.get("epoch", 0),
        cursor=batch_state.get("cursor", 0),
    )
    step = int(state.get("step", 0))
    tokens_seen = int(state.get("tokens_seen", 0))
    starting_tokens = tokens_seen
    prior_seconds = float(state.get("billable_seconds", 0.0))
    next_eval = (
        (tokens_seen // config.train.eval_every_tokens) + 1
    ) * config.train.eval_every_tokens
    next_save = (
        (tokens_seen // config.train.save_every_tokens) + 1
    ) * config.train.save_every_tokens
    started = time.monotonic()
    log_path = output_dir / "metrics.jsonl"
    model.train()
    compute_dtype = torch_dtype(config.model.dtype)

    while step < config.max_steps:
        optimizer.zero_grad(set_to_none=True)
        sums: dict[str, float] = {}
        for _ in range(config.optim.grad_accum_steps):
            input_ids = batcher.next().to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=compute_dtype,
                enabled=device.type == "cuda" and compute_dtype != torch.float32,
            ):
                outputs = model(input_ids=input_ids, labels=input_ids)
                loss = outputs.loss / config.optim.grad_accum_steps
            loss.backward()
            metrics = {"loss": outputs.loss}
            if config.run.mode == "ncp":
                metrics.update(
                    ntp_loss=outputs.ntp_loss,
                    ncp_loss=outputs.ncp_loss,
                    vq_loss=outputs.vq_loss,
                )
            for name, value in metrics.items():
                sums[name] = sums.get(name, 0.0) + float(value.detach())
            tokens_seen += input_ids.numel()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.optim.max_grad_norm)
        optimizer.step()
        scheduler.step()
        step += 1

        if step % config.train.log_every_steps == 0 or step == 1:
            elapsed = time.monotonic() - started
            payload = {
                "event": "train",
                "step": step,
                "tokens": tokens_seen,
                "tokens_per_second": (tokens_seen - starting_tokens) / max(elapsed, 1e-6),
                "learning_rate": scheduler.get_last_lr()[0],
                "grad_norm": float(grad_norm),
                "estimated_cost_usd": (
                    (prior_seconds + elapsed)
                    / 3600
                    * config.train.hourly_cost_usd
                ),
                **{name: value / config.optim.grad_accum_steps for name, value in sums.items()},
            }
            line = json.dumps(payload, sort_keys=True)
            print(line, flush=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

        if tokens_seen >= next_eval:
            metrics = evaluate(model, config, device)
            payload = {"event": "eval", "step": step, "tokens": tokens_seen, **metrics}
            line = json.dumps(payload, sort_keys=True)
            print(line, flush=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
            next_eval += config.train.eval_every_tokens

        if tokens_seen >= next_save:
            _save(
                model,
                optimizer,
                scheduler,
                output_dir,
                step,
                tokens_seen,
                batcher,
                prior_seconds + time.monotonic() - started,
                tokenizer,
            )
            next_save += config.train.save_every_tokens

        elapsed_minutes = (time.monotonic() - started) / 60
        estimated_cost = (
            (prior_seconds + time.monotonic() - started)
            / 3600
            * config.train.hourly_cost_usd
        )
        if (
            elapsed_minutes >= config.train.max_wall_time_minutes
            or estimated_cost >= config.train.run_budget_usd
        ):
            break

    billable_seconds = prior_seconds + time.monotonic() - started
    checkpoint = _save(
        model,
        optimizer,
        scheduler,
        output_dir,
        step,
        tokens_seen,
        batcher,
        billable_seconds,
        tokenizer,
    )
    return {
        "checkpoint": str(checkpoint),
        "step": step,
        "tokens": tokens_seen,
        "estimated_cost_usd": (
            billable_seconds / 3600 * config.train.hourly_cost_usd
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an NTP or NTP+NCP checkpoint")
    parser.add_argument("config")
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2))


if __name__ == "__main__":
    main()
