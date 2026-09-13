from __future__ import annotations

import sys

import modal

from modal_app import REMOTE_ROOT, _config, image, volume

app = modal.App("ncp-smol-publish", image=image)


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
    from ncp_smol.experiment import load_experiment
    from ncp_smol.publish import publish
    from ncp_smol.runtime import latest_checkpoint

    config_path = _config(config)
    if checkpoint == "latest":
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
