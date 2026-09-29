"""Cross-domain feedback partner probe (docs/probe-cross-domain.md)."""

from __future__ import annotations

import torch
from torch import Tensor

from .data import TokenBatcher, TokenCorpus
from .modeling import ConceptMode, NcpSmolForCausalLM


@torch.no_grad()
def foreign_feedback(
    model: NcpSmolForCausalLM,
    pool_batcher: TokenBatcher,
    *,
    batches: int,
    device: torch.device,
) -> Tensor:
    """Collect predicted per-chunk feedback vectors on foreign-corpus batches."""
    pool: list[Tensor] = []
    for _ in range(batches):
        input_ids = pool_batcher.next().to(device, non_blocking=True)
        model(input_ids=input_ids, concept_mode="predicted")
        last = model._hook_state.get("last_feedback")
        if last is None:
            raise RuntimeError("model did not expose last_feedback")
        pool.append(last.detach())
    return torch.cat(pool, dim=0)


@torch.no_grad()
def cross_domain_delta(
    model: NcpSmolForCausalLM,
    corpus: TokenCorpus,
    pool: Tensor,
    *,
    batches: int,
    batch_size: int,
    seed: int,
    device: torch.device,
    mode: ConceptMode = "predicted",
) -> dict[str, float]:
    """Mean NTP loss on held-out batches with feedback replaced by foreign partners.

    Returns both the overridden loss and the reference `predicted` loss on the exact
    same batches (same seed, same order), so the delta needs no cross-run comparison.
    """

    def run(use_override: bool) -> float:
        batcher = TokenBatcher(corpus, batch_size, seed=seed, repeat=True)
        total = 0.0
        count = 0
        for _ in range(batches):
            input_ids = batcher.next().to(device, non_blocking=True)
            model._hook_state["override_feedback"] = pool if use_override else None
            try:
                outputs = model(input_ids=input_ids, labels=input_ids, concept_mode=mode)
                loss = outputs.ntp_loss
            finally:
                model._hook_state["override_feedback"] = None
            total += float(loss.item())
            count += 1
        return total / max(count, 1)

    overridden = run(True)
    reference = run(False)
    return {
        "overridden_ntp_loss": overridden,
        "reference_ntp_loss": reference,
        "delta": overridden - reference,
    }
