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

app = modal.App("ncp-smol-tiny", image=image)
volume = modal.Volume.from_name("ncp-smol", create_if_missing=True)


@app.function(gpu="L4", cpu=8, memory=32768, volumes={"/vol": volume}, timeout=10_800)
def train(config: str) -> dict[str, object]:
    sys.path.insert(0, f"{REMOTE_ROOT}/src")
    from ncp_smol.train import run

    return run(f"{REMOTE_ROOT}/configs/{config}")


@app.function(gpu="L4", cpu=4, memory=16384, volumes={"/vol": volume}, timeout=3_600)
def evaluate(config: str, checkpoint: str, batches: int = 32) -> dict[str, object]:
    sys.path.insert(0, f"{REMOTE_ROOT}/src")
    from ncp_smol.eval import evaluate_checkpoint
    from ncp_smol.experiment import load_experiment
    from ncp_smol.runtime import latest_checkpoint, write_json

    config_path = f"{REMOTE_ROOT}/configs/{config}"
    if checkpoint == "latest":
        experiment = load_experiment(config_path)
        resolved = latest_checkpoint(experiment.run.output_dir)
        if resolved is None:
            raise ValueError(f"no checkpoint found for {config}")
        checkpoint = str(resolved)
    result = evaluate_checkpoint(config_path, checkpoint, batches=batches)
    output = f"/vol/artifacts/{Path(config).stem}-{Path(checkpoint).name}-eval.json"
    write_json(output, result)
    return {"output": output, "metrics": result}


@app.local_entrypoint()
def fit(config: str = "tinystories-overfit.yaml") -> None:
    print(train.remote(config))


@app.local_entrypoint()
def eval(
    config: str = "tinystories-overfit.yaml",
    checkpoint: str = "latest",
    batches: int = 32,
) -> None:
    print(evaluate.remote(config, checkpoint, batches))
