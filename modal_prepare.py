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

app = modal.App("ncp-smol-prepare", image=image)
volume = modal.Volume.from_name("ncp-smol", create_if_missing=True)


@app.function(cpu=8, memory=16384, volumes={"/vol": volume}, timeout=86_400)
def prepare_data(config: str) -> dict[str, object]:
    sys.path.insert(0, f"{REMOTE_ROOT}/src")
    from ncp_smol.prepare import prepare

    return prepare(f"{REMOTE_ROOT}/configs/{config}")


@app.local_entrypoint()
def prepare(config: str = "tinystories-overfit.yaml") -> None:
    print(prepare_data.remote(config))
