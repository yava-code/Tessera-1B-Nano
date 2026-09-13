from __future__ import annotations

import argparse
import json

import torch

from ncp_smol.modeling import NcpSmolForCausalLM

REVISIONS = {
    "HuggingFaceTB/SmolLM2-135M": "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
    "HuggingFaceTB/SmolLM2-360M": "f8027fd0eaeea54caa13c31d31b9fdc459c38b49",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the NCP path on a real SmolLM2 checkpoint")
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    args = parser.parse_args()

    model = NcpSmolForCausalLM.from_backbone(
        args.model,
        revision=REVISIONS.get(args.model),
        chunk_size=4,
        codebook_size=64,
        concept_layers=2,
        insert_layer=1,
        dtype=torch.float32,
    ).eval()
    total = sum(parameter.numel() for parameter in model.parameters())
    backbone = sum(parameter.numel() for parameter in model.backbone.parameters())
    input_ids = torch.tensor([[1, 504, 969, 198, 268, 314, 260, 357]])
    with torch.no_grad():
        output = model(input_ids=input_ids, labels=input_ids)
    print(
        json.dumps(
            {
                "shape": list(output.logits.shape),
                "parameters": total,
                "concept_parameters": total - backbone,
                "ntp_loss": float(output.ntp_loss),
                "ncp_loss": float(output.ncp_loss),
                "vq_loss": float(output.vq_loss),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
