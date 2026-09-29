from __future__ import annotations

import sys
from pathlib import Path

import modal

from modal_requirements import REQUIREMENTS

ROOT = Path(__file__).parent
REMOTE_ROOT = "/root/ncp-smol"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(*REQUIREMENTS)
    .add_local_file(ROOT / "modal_requirements.py", "/root/modal_requirements.py", copy=True)
    .add_local_dir(ROOT / "src", f"{REMOTE_ROOT}/src", copy=True)
    .add_local_dir(ROOT / "configs", f"{REMOTE_ROOT}/configs", copy=True)
    .env(
        {
            "PYTHONPATH": f"{REMOTE_ROOT}/src",
            "NCP_ROOT": "/vol",
            "HF_HOME": "/vol/.cache/huggingface",
            "HF_DATASETS_CACHE": "/vol/.cache/huggingface/datasets",
            "TORCH_HOME": "/vol/.cache/torch",
        }
    )
)

app = modal.App("ncp-smol", image=image)
volume = modal.Volume.from_name("ncp-smol", create_if_missing=True)


def _config(name: str) -> str:
    return f"{REMOTE_ROOT}/configs/{name}"


@app.function(cpu=8, memory=16384, volumes={"/vol": volume}, timeout=86_400)
def prepare_data(config: str) -> dict[str, object]:
    sys.path.insert(0, f"{REMOTE_ROOT}/src")
    from ncp_smol.prepare import prepare

    return prepare(_config(config))


@app.function(cpu=8, memory=16384, volumes={"/vol": volume}, timeout=86_400)
def prepare_foreign(name: str) -> dict[str, object]:
    """Pack a foreign-corpus pool (wikipedia | code) with the project tokenizer."""
    sys.path.insert(0, f"{REMOTE_ROOT}/src")
    from ncp_smol.experiment import load_experiment
    from ncp_smol.foreign_corpus import prepare_foreign_pool

    experiment = load_experiment(_config("fineweb-edu-ncp.yaml"))
    return prepare_foreign_pool(
        name,
        tokenizer_name=experiment.model.base_model,
        tokenizer_revision=experiment.model.revision,
        output_dir="/vol/.cache/data/foreign",
        tokens=10_000_000,
        seed=experiment.run.seed,
    )


@app.function(
    gpu="A100-40GB",
    cpu=8,
    memory=32768,
    max_containers=1,
    volumes={"/vol": volume},
    timeout=82_800,
)
def train(config: str) -> dict[str, object]:
    sys.path.insert(0, f"{REMOTE_ROOT}/src")
    from ncp_smol.train import run

    return run(_config(config))


@app.function(
    gpu="A100-40GB",
    cpu=4,
    memory=16384,
    volumes={"/vol": volume},
    timeout=3_600,
)
def verify_hf(
    base_repo: str = "yava-code/Tessera-1B-Nano-Base",
    concept_repo: str = "yava-code/Tessera-1B-Nano",
    blocks: int = 256,
) -> dict[str, object]:
    """Re-measure the committed eval from the published HF weights on /vol held-out data."""
    import hashlib
    import json

    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM

    sys.path.insert(0, f"{REMOTE_ROOT}/src")
    from ncp_smol.runtime import token_cross_entropy

    validation_path = "/vol/.cache/data/fineweb-edu-1b/validation.bin"
    sha = hashlib.sha256(open(validation_path, "rb").read()).hexdigest()
    with open("/vol/runs/fineweb-edu-ntp/data_metadata.json", encoding="utf-8") as handle:
        metadata = json.load(handle)
    sha_matches = sha == metadata["validation_sha256"]

    sequence_length = 1024
    tokens = np.memmap(validation_path, dtype=np.uint32, mode="r")
    order = np.arange(len(tokens) // sequence_length)
    np.random.default_rng(17 + 20_000).shuffle(order)
    order = order[:blocks]
    device = torch.device("cuda")

    results: dict[str, object] = {"validation_sha256_matches": sha_matches, "blocks": blocks}
    for name, repo, trust in (
        ("base", base_repo, False),
        ("concept", concept_repo, True),
    ):
        _ = hf_hub_download(repo, "README.md")  # fail fast if the repo moved
        model = AutoModelForCausalLM.from_pretrained(
            repo,
            trust_remote_code=trust,
            dtype=torch.bfloat16,
        ).to(device)
        model.eval()
        predicted_total = 0.0
        zero_total = 0.0
        with torch.no_grad():
            for start in range(0, len(order), 8):
                chunk = order[start : start + 8]
                batch = torch.stack(
                    [
                        torch.from_numpy(
                            np.array(
                                tokens[index * sequence_length : (index + 1) * sequence_length],
                                dtype=np.int64,
                            )
                        )
                        for index in chunk
                    ]
                ).to(device)
                outputs = model(input_ids=batch)
                predicted_total += float(
                    token_cross_entropy(outputs.logits, batch).mean().item()
                ) * len(chunk)
                if trust:
                    zero_outputs = model(input_ids=batch, concept_mode="zero")
                    zero_total += float(
                        token_cross_entropy(zero_outputs.logits, batch).mean().item()
                    ) * len(chunk)
        mean_ntp = predicted_total / len(order)
        entry: dict[str, float] = {"held_out_ntp": round(mean_ntp, 4)}
        if trust:
            entry["zero_feedback_ntp"] = round(zero_total / len(order), 4)
            entry["zero_feedback_delta"] = round(zero_total / len(order) - mean_ntp, 4)
        results[name] = entry
    return results


@app.function(
    gpu="A100-40GB",
    cpu=4,
    memory=16384,
    volumes={"/vol": volume},
    timeout=3_600,
)
def cross_domain_probe(
    checkpoint: str,
    pools: str = "wikipedia,code",
    batches: int = 32,
) -> dict[str, object]:
    """Run the cross-domain feedback partner probe on the final NCP checkpoint."""
    sys.path.insert(0, f"{REMOTE_ROOT}/src")
    import torch

    from ncp_smol.data import TokenBatcher, TokenCorpus
    from ncp_smol.experiment import load_experiment
    from ncp_smol.modeling import NcpSmolForCausalLM
    from ncp_smol.probes import cross_domain_delta, foreign_feedback
    from ncp_smol.runtime import latest_checkpoint, seed_everything

    config = load_experiment(_config("fineweb-edu-ncp.yaml"))
    seed_everything(config.run.seed)
    device = torch.device("cuda")
    resolved = checkpoint
    if checkpoint == "latest":
        found = latest_checkpoint(config.run.output_dir)
        if found is None:
            raise ValueError("no checkpoint found")
        resolved = str(found)
    model = NcpSmolForCausalLM.from_pretrained(resolved).to(device).eval()

    corpus = TokenCorpus(
        Path(config.data.cache_dir) / "validation.bin",
        config.data.sequence_length,
    )
    batch_size = config.optim.micro_batch_size
    results: dict[str, object] = {"checkpoint": resolved, "batches": batches}
    for pool_name in pools.split(","):
        pool_name = pool_name.strip()
        pool_corpus = TokenCorpus(
            Path("/vol/.cache/data/foreign") / pool_name / "pool.bin",
            config.data.sequence_length,
        )
        pool_batcher = TokenBatcher(pool_corpus, batch_size, seed=config.run.seed, repeat=True)
        pool = foreign_feedback(model, pool_batcher, batches=batches, device=device)
        delta = cross_domain_delta(
            model,
            corpus,
            pool,
            batches=batches,
            batch_size=batch_size,
            seed=config.run.seed + 20_000,
            device=device,
        )
        entry = dict(delta)
        entry["pool_vectors"] = int(pool.shape[0])
        results[pool_name] = entry
    return results


@app.function(
    gpu="A100-40GB",
    cpu=4,
    memory=16384,
    volumes={"/vol": volume},
    timeout=14_400,
)
def evaluate(
    config: str,
    checkpoint: str,
    batches: int = 32,
    output: str | None = None,
) -> dict[str, object]:
    sys.path.insert(0, f"{REMOTE_ROOT}/src")
    from ncp_smol.eval import evaluate_checkpoint
    from ncp_smol.experiment import load_experiment
    from ncp_smol.runtime import latest_checkpoint, write_json

    config_path = _config(config)
    if checkpoint == "latest":
        experiment = load_experiment(config_path)
        resolved = latest_checkpoint(experiment.run.output_dir)
        if resolved is None:
            raise ValueError(f"no checkpoint found for {config}")
        checkpoint = str(resolved)
    result = evaluate_checkpoint(config_path, checkpoint, batches=batches)
    output = output or f"/vol/artifacts/{Path(config).stem}-{Path(checkpoint).name}-eval.json"
    write_json(output, result)
    return {"output": output, "metrics": result}


@app.function(cpu=2, memory=4096, volumes={"/vol": volume}, timeout=600)
def read_progress(config: str) -> dict[str, object]:
    sys.path.insert(0, f"{REMOTE_ROOT}/src")
    from ncp_smol.experiment import load_experiment
    from ncp_smol.progress import progress_from_files

    experiment = load_experiment(_config(config))
    return progress_from_files(
        Path(experiment.run.output_dir),
        max_steps=experiment.max_steps,
    )


@app.local_entrypoint()
def progress(config: str) -> None:
    print(read_progress.remote(config))


@app.local_entrypoint()
def prepare_foreign_entry(name: str) -> None:
    print(prepare_foreign.remote(name))


@app.local_entrypoint()
def cross_domain(
    checkpoint: str = "latest",
    pools: str = "wikipedia,code",
    batches: int = 32,
) -> None:
    print(cross_domain_probe.remote(checkpoint, pools, batches))


@app.local_entrypoint()
def eval(
    config: str,
    checkpoint: str = "latest",
    batches: int = 32,
    output: str | None = None,
) -> None:
    print(evaluate.remote(config, checkpoint, batches, output))
