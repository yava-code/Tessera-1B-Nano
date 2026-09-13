from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from transformers import AutoModelForCausalLM

from .experiment import ExperimentConfig
from .modeling import NcpSmolForCausalLM


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def torch_dtype(name: str) -> torch.dtype:
    values = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    try:
        return values[name]
    except KeyError as error:
        raise ValueError(f"unsupported dtype: {name}") from error


def make_model(config: ExperimentConfig, checkpoint: Path | None = None) -> torch.nn.Module:
    dtype = torch_dtype(config.model.dtype)
    if checkpoint is not None:
        if config.run.mode == "ncp":
            return NcpSmolForCausalLM.from_pretrained(checkpoint, dtype=dtype)
        return AutoModelForCausalLM.from_pretrained(checkpoint, dtype=dtype)

    if config.run.mode == "ncp":
        return NcpSmolForCausalLM.from_backbone(
            config.model.base_model,
            revision=config.model.revision,
            chunk_size=config.model.chunk_size,
            segments=config.model.segments,
            codebook_size=config.model.codebook_size,
            concept_layers=config.model.concept_layers,
            insert_layer=config.model.insert_layer,
            ncp_target=config.model.ncp_target,
            ncp_weight=config.model.ncp_weight,
            vq_weight=config.model.vq_weight,
            dtype=dtype,
        )
    return AutoModelForCausalLM.from_pretrained(
        config.model.base_model,
        revision=config.model.revision,
        dtype=dtype,
    )


def latest_checkpoint(output_dir: str | Path) -> Path | None:
    root = Path(output_dir)
    marker = root / "latest.json"
    if not marker.exists():
        return None
    value = json.loads(marker.read_text(encoding="utf-8"))["checkpoint"]
    path = root / value
    return path if path.is_dir() else None


def cosine_schedule(step: int, total: int, warmup_ratio: float, min_ratio: float) -> float:
    warmup = max(1, round(total * warmup_ratio))
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_ratio + 0.5 * (1 - min_ratio) * (1 + math.cos(math.pi * progress))


def token_cross_entropy(logits: Tensor, labels: Tensor) -> Tensor:
    token_loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].float().transpose(1, 2),
        labels[:, 1:],
        reduction="none",
        ignore_index=-100,
    )
    return token_loss


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
