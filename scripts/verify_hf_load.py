"""Verify the published Hugging Face checkpoints load and generate.

Checks, per repository:
- the checkpoint loads through `AutoModelForCausalLM` (the baseline as a plain llama
  model, the concept arm with `trust_remote_code=True` against the modules shipped in
  the repo);
- greedy generation returns text;
- a teacher-forced forward produces a finite loss (and, for the concept arm, the
  zero-feedback loss is reported next to the predicted-feedback loss).

Prints one JSON verdict per repo and exits non-zero on failure. CPU is enough:

    .venv/Scripts/python.exe -X utf8 scripts/verify_hf_load.py
"""

from __future__ import annotations

import argparse
import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPOS = {
    "baseline": "yava-code/Tessera-1B-Nano-Base",
    "concept": "yava-code/Tessera-1B-Nano",
}
PROMPT = "The theory of relativity states"


def verify(repo: str, *, trust_remote_code: bool, max_new_tokens: int) -> dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(
        repo,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch.float32,
    ).to(device)
    model.eval()

    inputs = tokenizer(PROMPT, return_tensors="pt").to(device)
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        forward = model(**inputs, labels=inputs["input_ids"])
    text = tokenizer.decode(generated[0], skip_special_tokens=True)

    verdict: dict[str, object] = {
        "repo": repo,
        "trust_remote_code": trust_remote_code,
        "generated_prefix": text[:96],
        "forward_loss": round(float(forward.loss), 4),
    }
    if trust_remote_code:
        with torch.no_grad():
            zero = model(**inputs, labels=inputs["input_ids"], concept_mode="zero")
        verdict["zero_feedback_loss"] = round(float(zero.loss), 4)
    if not text.strip():
        raise AssertionError(f"{repo}: generation produced no text")
    if not torch.isfinite(forward.loss):
        raise AssertionError(f"{repo}: forward loss is not finite")
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    args = parser.parse_args()

    failures: list[str] = []
    for key, repo in REPOS.items():
        try:
            verdict = verify(
                repo,
                trust_remote_code=key == "concept",
                max_new_tokens=args.max_new_tokens,
            )
        except Exception as error:  # noqa: BLE001 - the script reports and exits non-zero
            failures.append(f"{repo}: {type(error).__name__}: {error}")
            print(json.dumps({"repo": repo, "status": "FAILED", "error": str(error)[:200]}))
            continue
        verdict["status"] = "OK"
        print(json.dumps(verdict, indent=2))

    if failures:
        sys.exit(1)
    print("both published checkpoints load and generate")


if __name__ == "__main__":
    main()
