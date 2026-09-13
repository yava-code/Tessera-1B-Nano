from __future__ import annotations

import sys
from pathlib import Path

import modal

ROOT = Path(__file__).parent
REMOTE_ROOT = "/root/ncp-smol"
REQUIREMENTS = (ROOT / "requirements-modal.txt").read_text(encoding="utf-8").splitlines()

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(*REQUIREMENTS)
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


@app.function(
    gpu="A100-40GB",
    cpu=8,
    memory=32768,
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
    output = output or f"/vol/artifacts/{Path(config).stem}-eval.json"
    write_json(output, result)
    return {"output": output, "metrics": result}


@app.local_entrypoint()
def prepare(config: str = "tinystories-overfit.yaml") -> None:
    print(prepare_data.remote(config))


@app.local_entrypoint()
def fit(config: str = "tinystories-overfit.yaml") -> None:
    print(train.remote(config))


@app.local_entrypoint()
def eval(
    config: str,
    checkpoint: str = "latest",
    batches: int = 32,
    output: str | None = None,
) -> None:
    print(evaluate.remote(config, checkpoint, batches, output))
