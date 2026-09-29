from __future__ import annotations

import functools
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASELINE_REPO = "yava-code/Tessera-1B-Nano-Base"
CONCEPT_REPO = "yava-code/Tessera-1B-Nano"
EXAMPLE_PROMPTS = [
    "The most common metals are",
    "Photosynthesis is the process used by plants",
    "In the nineteenth century, transport changed because",
    "A simple recipe for bread needs flour, water and",
]
DEFAULT_MAX_NEW_TOKENS = 48
MAX_NEW_TOKENS_LIMIT = 160
INTRO_NOTE = (
    "The comparison is token-matched: both arms consumed the same billion tokens in the"
    " same order. On held-out data the concept arm matches the baseline's token loss"
    " (+0.03%), its decoder demonstrably consumes the concept channel (zeroing the"
    " feedback costs +0.1050 nats), yet shuffled feedback is as good as its own"
    " (+0.0008): the use is sequence-generic."
)

MODELS: dict[str, tuple[AutoTokenizer, Any]] = {}


def _torch_dtype() -> torch.dtype:
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


def load_models() -> None:
    if MODELS:
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for key, repo, trust in (
        ("base", BASELINE_REPO, False),
        ("concept", CONCEPT_REPO, True),
    ):
        tokenizer = AutoTokenizer.from_pretrained(repo)
        model = AutoModelForCausalLM.from_pretrained(
            repo,
            trust_remote_code=trust,
            dtype=_torch_dtype(),
        ).to(device)
        model.eval()
        MODELS[key] = (tokenizer, model)


def prompt_ntp_loss(
    tokenizer: AutoTokenizer,
    model: Any,
    prompt: str,
    *,
    zero_feedback: bool = False,
) -> float:
    """Teacher-forced next-token loss of the prompt (the metric the study reports)."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = (
            model(**inputs, concept_mode="zero")
            if zero_feedback
            else model(**inputs)
        )
    logits = outputs.logits[:, :-1, :]
    targets = inputs["input_ids"][:, 1:]
    return float(
        torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="mean",
        )
    )


def generate(
    tokenizer: AutoTokenizer,
    model: Any,
    prompt: str,
    max_new_tokens: int,
    *,
    zero_feedback: bool = False,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    if zero_feedback:
        # generate() rejects unknown kwargs, so route concept_mode through forward.
        model.forward = functools.partial(model.forward, concept_mode="zero")
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
            )
    finally:
        if zero_feedback:
            del model.forward
    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    )


def run_comparison(prompt: str, max_new_tokens: int) -> tuple[str, str, str, str, str]:
    prompt = prompt.strip() or EXAMPLE_PROMPTS[0]
    max_new_tokens = max(16, min(int(max_new_tokens), MAX_NEW_TOKENS_LIMIT))
    load_models()
    base_tok, base_model = MODELS["base"]
    concept_tok, concept_model = MODELS["concept"]

    base_text = generate(base_tok, base_model, prompt, max_new_tokens)
    concept_text = generate(concept_tok, concept_model, prompt, max_new_tokens)
    zero_text = generate(
        concept_tok,
        concept_model,
        prompt,
        max_new_tokens,
        zero_feedback=True,
    )

    base_loss = prompt_ntp_loss(base_tok, base_model, prompt)
    concept_loss = prompt_ntp_loss(concept_tok, concept_model, prompt)
    zero_loss = prompt_ntp_loss(concept_tok, concept_model, prompt, zero_feedback=True)
    delta = zero_loss - concept_loss

    base_note = f"teacher-forced NTP loss on the prompt: {base_loss:.3f}"
    concept_note = (
        f"teacher-forced NTP loss on the prompt: {concept_loss:.3f}; zeroing the concept"
        f" feedback costs {delta:+.3f} nats here (held-out average: +0.105)"
    )
    return base_text, base_note, concept_text, zero_text, concept_note
