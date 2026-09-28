# Model card (Hugging Face)

The card below is what `src/ncp_smol/publish.py::build_card` generates for the final
checkpoint and eval artifact; this document mirrors it with the concrete numbers so the
uploaded card and the docs agree. The repo id flows in through the `REPO_ID` placeholder
at publish time, so no manual replacement is needed. Upload stays gated on the Modal
`huggingface` secret (`HF_TOKEN`); the procedure is
[docs/publishing-checklist.md](docs/publishing-checklist.md).

---

````
---
library_name: transformers
pipeline_tag: text-generation
base_model: HuggingFaceTB/SmolLM2-360M
tags:
- next-concept-prediction
- conceptlm
- causal-lm
- smollm2
- tessera
license: apache-2.0
---

# Tessera-1B-Nano

A Next Concept Prediction checkpoint from Paragon Intelligence Labs: an independent,
small-scale replication of ConceptLM on `HuggingFaceTB/SmolLM2-360M`. The model keeps
ordinary next-token generation and adds a causal, product-quantized concept path over
4-token chunks: token states are pooled into concepts, product-quantized, processed by
causal concept blocks, and the predicted next concept is fed back into the token decoder.

## TL;DR

This checkpoint comes from a token-matched comparison against the unchanged
HuggingFaceTB/SmolLM2-360M backbone: both arms consumed the same tokens of the same packed
corpus, in the same order, from the same initialization, with no restarts and no NaNs.

- **Token loss is neutral**: the matched baseline is within run noise (see the whitepaper
  and `results/fineweb-edu/README.md` in the ncp-smol repository for the exact numbers).
- **The concept channel is causally used**: zeroing predicted concept feedback
  at inference costs +0.1050 nats of held-out NTP loss.
- **The codebook is rich**: no low-entropy shortcut, usage grows monotonically over the
  run (growth curves are in the repository whitepaper).
- Full reading of the interventions: `docs/whitepaper.md` in the ncp-smol repository.

## Evaluation

| Metric | Value |
| --- | ---: |
| Held-out NTP loss | 2.5142 |
| Held-out perplexity | 12.3568 |
| Zero feedback delta | 0.1050 |
| Shuffled feedback delta | 0.0008 |
| Codebook perplexity / usage | 7.5477 / 85.6% |
| Training tokens | 999,948,288 |
| Tracked compute estimate | $30.27 |

Intervention deltas are increases in held-out NTP loss relative to normal predicted concept
feedback, evaluated on identical batches.

## Architecture

- chunk size: 4
- product code: 15 segments x 64 entries
- causal concept blocks: 2
- injection point: before token decoder block 2
- NCP target: next continuous concept
- loss: `L_ntp + 1 L_ncp + 1 L_vq`

This is a compact ConceptLM-style implementation, not an 8.9B NCP-ArchPreview replica. It
omits iterative residual coding, cross-scale residual connections, and the large-scale
training recipe.

## Training data and provenance

The matched corpus and the training record are pinned in the ncp-smol repository:
packed-cache SHA256 hashes, the complete metric log, trainer state, and the raw
intervention evaluation JSON (also shipped in this repository as `eval.json` and
`metrics.jsonl`).

## Loading

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("REPO_ID")
model = AutoModelForCausalLM.from_pretrained("REPO_ID")
```

## References

- ConceptLM: https://arxiv.org/abs/2602.08984
- NCP-ArchPreview: https://arxiv.org/abs/2609.10715
````
