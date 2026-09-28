from __future__ import annotations

import sys
from pathlib import Path

import modal

from modal_requirements import REQUIREMENTS

# Self-contained on purpose: Modal ships only this module into the container, so it must
# not import from modal_app.py (that cross-import crash-looped the function on start).
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

app = modal.App("ncp-smol-publish", image=image)
volume = modal.Volume.from_name("ncp-smol", create_if_missing=True)


def _config(name: str) -> str:
    return f"{REMOTE_ROOT}/configs/{name}"


@app.function(
    cpu=2,
    memory=4096,
    volumes={"/vol": volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=14_400,
)
def publish_checkpoint(
    config: str,
    checkpoint: str,
    eval_json: str,
    repo_id: str,
    private: bool = False,
) -> str:
    sys.path.insert(0, f"{REMOTE_ROOT}/src")
    from ncp_smol.publish import publish
    from ncp_smol.runtime import latest_checkpoint

    config_path = _config(config)
    if checkpoint == "latest":
        from ncp_smol.experiment import load_experiment

        experiment = load_experiment(config_path)
        resolved = latest_checkpoint(experiment.run.output_dir)
        if resolved is None:
            raise ValueError(f"no checkpoint found for {config}")
        checkpoint = str(resolved)

    return publish(
        config_path,
        checkpoint,
        eval_json,
        repo_id,
        private=private,
    )


@app.local_entrypoint()
def publish(
    config: str,
    checkpoint: str,
    eval_json: str,
    repo_id: str,
    private: bool = False,
) -> None:
    print(publish_checkpoint.remote(config, checkpoint, eval_json, repo_id, private))
