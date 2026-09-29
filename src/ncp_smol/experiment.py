from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RunConfig:
    name: str
    mode: str
    output_dir: str
    seed: int = 17
    resume: bool = True


@dataclass
class ModelConfig:
    base_model: str = "HuggingFaceTB/SmolLM2-360M"
    revision: str | None = None
    chunk_size: int = 4
    segments: int | None = None
    codebook_size: int = 64
    concept_layers: int = 2
    insert_layer: int = 1
    ncp_target: str = "continuous"
    ncp_weight: float = 1.0
    vq_weight: float = 1.0
    codebook_normalization: str = "none"
    dtype: str = "bfloat16"


@dataclass
class DataConfig:
    dataset: str
    subset: str | None
    revision: str | None
    train_split: str
    validation_split: str | None
    text_column: str
    cache_dir: str
    sequence_length: int
    train_tokens: int
    validation_tokens: int
    cache_train_tokens: int | None = None
    shuffle_buffer: int = 10_000
    overfit: bool = False


@dataclass
class OptimConfig:
    learning_rate: float = 3e-5
    min_lr_ratio: float = 0.1
    warmup_ratio: float = 0.01
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    micro_batch_size: int = 8
    grad_accum_steps: int = 16
    max_grad_norm: float = 1.0


@dataclass
class TrainConfig:
    log_every_steps: int = 10
    eval_every_tokens: int = 100_000_000
    save_every_tokens: int = 250_000_000
    eval_batches: int = 32
    max_wall_time_minutes: int = 1_320
    hourly_cost_usd: float = 2.74
    run_budget_usd: float = 60.0
    compile: bool = False


@dataclass
class ExperimentConfig:
    run: RunConfig
    model: ModelConfig
    data: DataConfig
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @property
    def tokens_per_step(self) -> int:
        return self.data.sequence_length * self.optim.micro_batch_size * self.optim.grad_accum_steps

    @property
    def max_steps(self) -> int:
        return max(1, self.data.train_tokens // self.tokens_per_step)

    def validate(self) -> None:
        if self.run.mode not in {"ntp", "ncp"}:
            raise ValueError("run.mode must be 'ntp' or 'ncp'")
        if self.data.sequence_length % self.model.chunk_size:
            raise ValueError("sequence_length must be divisible by chunk_size")
        if self.data.train_tokens < self.tokens_per_step:
            raise ValueError("train_tokens must cover at least one optimizer step")
        if self.optim.micro_batch_size < 1 or self.optim.grad_accum_steps < 1:
            raise ValueError("batch sizes must be positive")


def _build(cls: type[Any], values: dict[str, Any] | None) -> Any:
    return cls(**(values or {}))


def load_experiment(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    root = os.environ.get("NCP_ROOT", str(config_path.resolve().parent.parent))
    text = config_path.read_text(encoding="utf-8").replace("${NCP_ROOT}", root)
    values = yaml.safe_load(text)
    config = ExperimentConfig(
        run=_build(RunConfig, values.get("run")),
        model=_build(ModelConfig, values.get("model")),
        data=_build(DataConfig, values.get("data")),
        optim=_build(OptimConfig, values.get("optim")),
        train=_build(TrainConfig, values.get("train")),
    )
    config.validate()
    return config
