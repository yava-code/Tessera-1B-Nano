from __future__ import annotations

import functools
from typing import Any

try:  # must precede torch/transformers imports: it patches CUDA allocation
    import spaces
except ImportError:  # pragma: no cover - local run
    spaces = None

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
DEFAULT_SEED = 7
INTRO_NOTE = (
    "The comparison is token-matched: both arms consumed the same billion tokens in the"
    " same order. On held-out data the concept arm matches the baseline's token loss"
    " (+0.03%), its decoder demonstrably consumes the concept channel (zeroing the"
    " feedback costs +0.1050 nats), yet shuffled feedback is as good as its own"
    " (+0.0008): the use is sequence-generic."
)

MODELS: dict[str, tuple[AutoTokenizer, Any]] = {}


def load_models(*, gpu_dtype: bool = False) -> None:
    """Load both checkpoints on CPU; gpu_dtype selects bf16 for the ZeroGPU move."""
    if MODELS:
        return
    dtype = torch.bfloat16 if gpu_dtype else torch.float32
    for key, repo, trust in (
        ("base", BASELINE_REPO, False),
        ("concept", CONCEPT_REPO, True),
    ):
        tokenizer = AutoTokenizer.from_pretrained(repo)
        model = AutoModelForCausalLM.from_pretrained(
            repo,
            trust_remote_code=trust,
            dtype=dtype,
        )
        model.eval()
        MODELS[key] = (tokenizer, model)


def _device(model: Any) -> torch.device:
    return next(model.parameters()).device


def _generate(
    tokenizer: AutoTokenizer,
    model: Any,
    prompt: str,
    max_new_tokens: int,
    *,
    seed: int,
    zero_feedback: bool = False,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(_device(model))
    if zero_feedback:
        # generate() rejects unknown kwargs, so route concept_mode through forward.
        model.forward = functools.partial(model.forward, concept_mode="zero")
    # transformers 4.x multinomial sampling draws from the global RNG (no generator
    # argument), so reproducibility needs a device-global seed reset per call.
    device = _device(model)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
    else:
        torch.manual_seed(seed)
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


def _prompt_loss(
    tokenizer: AutoTokenizer,
    model: Any,
    prompt: str,
    *,
    zero_feedback: bool = False,
) -> float:
    inputs = tokenizer(prompt, return_tensors="pt").to(_device(model))
    with torch.no_grad():
        outputs = model(**inputs, concept_mode="zero") if zero_feedback else model(**inputs)
    logits = outputs.logits[:, :-1, :]
    targets = inputs["input_ids"][:, 1:]
    return float(
        torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="mean",
        )
    )


def _run(prompt: str, max_new_tokens: int, seed: int) -> tuple[str, str, str, str, str]:
    if set(MODELS) != {"base", "concept"}:
        raise RuntimeError("Models are still loading; retry in a minute.")
    prompt = prompt.strip() or EXAMPLE_PROMPTS[0]
    max_new_tokens = max(16, min(int(max_new_tokens), MAX_NEW_TOKENS_LIMIT))
    seed = max(0, min(int(seed), 2**31 - 1))
    base_tok, base_model = MODELS["base"]
    concept_tok, concept_model = MODELS["concept"]

    # Identical seeds across the arms make repeats bit-reproducible; per-arm offsets
    # only decorrelate the columns, they change nothing on a repeated click.
    base_text = _generate(base_tok, base_model, prompt, max_new_tokens, seed=seed)
    concept_text = _generate(concept_tok, concept_model, prompt, max_new_tokens, seed=seed + 1)
    zero_text = _generate(
        concept_tok,
        concept_model,
        prompt,
        max_new_tokens,
        seed=seed + 2,
        zero_feedback=True,
    )

    base_loss = _prompt_loss(base_tok, base_model, prompt)
    concept_loss = _prompt_loss(concept_tok, concept_model, prompt)
    zero_loss = _prompt_loss(concept_tok, concept_model, prompt, zero_feedback=True)
    delta = zero_loss - concept_loss

    base_note = f"teacher-forced NTP loss on the prompt: {base_loss:.3f}"
    concept_note = (
        f"teacher-forced NTP loss on the prompt: {concept_loss:.3f}; zeroing the concept"
        f" feedback costs {delta:+.3f} nats here (held-out average: +0.105)"
    )
    return base_text, base_note, concept_text, zero_text, concept_note


if spaces is not None:  # ZeroGPU: models live on CPU, compute moves to a shared A100

    @spaces.GPU(duration=45)
    def _gpu_run(prompt: str, max_new_tokens: int, seed: int) -> tuple[str, str, str, str, str]:
        # Only the device move and the math run inside the GPU window: downloads and
        # CPU loading stay in the main process, or ZeroGPU kills the call on timeout.
        for _, model in MODELS.values():
            if next(model.parameters()).device.type != "cuda":
                model.to("cuda")
        return _run(prompt, max_new_tokens, seed)

    def run_comparison(
        prompt: str, max_new_tokens: int, seed: int
    ) -> tuple[str, str, str, str, str]:
        load_models(gpu_dtype=True)
        return _gpu_run(prompt, max_new_tokens, seed)

else:  # local: plain cpu/cuda execution

    def run_comparison(
        prompt: str, max_new_tokens: int, seed: int
    ) -> tuple[str, str, str, str, str]:
        load_models()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for _, model in MODELS.values():
            model.to(device)
        return _run(prompt, max_new_tokens, seed)
