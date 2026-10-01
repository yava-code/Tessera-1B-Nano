# Tessera: a language model that predicts its next thought

**Tessera-1B-Nano** ([HF weights](https://huggingface.co/yava-code/Tessera-1B-Nano),
[live demo](https://huggingface.co/spaces/yava-code/tessera-comparison)) is an independent,
small-scale replication of **ConceptLM** by Paragon Intelligence Labs. An ordinary
SmolLM2-360M backbone gains a thin causal concept path: every four tokens are pooled into
one concept code, a small module guesses the *next* code, and the guess is fed back into
the decoder while it writes.

The matched experiment then produced a result strange enough to build a study around:

- **The decoder leans on the channel.** Silence the concept feedback and held-out loss
  gets worse - the model reads this thing constantly.
- **But it is not reading its own sentence.** Feed it concepts predicted from a different
  sentence in the batch - or from a Wikipedia passage, or from source code - and nothing
  changes. Whatever it consumes there, it is not sentence identity, and not even topic.
- **And at this budget it buys nothing a pure token model misses.** A token-matched
  baseline without any concept path lands exactly even.

So: a model that constantly consumes a signal that carries ... what? That question - with
preregistered predictions, full metric logs, and honest cost records - is the study. Read
the [whitepaper](docs/whitepaper.md), or [play with both models side by
side](https://huggingface.co/spaces/yava-code/tessera-comparison) before reading anything.

`ncp-smol` is the internal project id. The family: the 135M architecture gate
([Tessera-135M-Gate](https://huggingface.co/yava-code/Tessera-135M-Gate)), the NTP-only
control ([Tessera-1B-Nano-Base](https://huggingface.co/yava-code/Tessera-1B-Nano-Base)) and
the concept arm ([Tessera-1B-Nano](https://huggingface.co/yava-code/Tessera-1B-Nano)), all
in [one collection](https://huggingface.co/collections/yava-code/tessera-next-concept-prediction-gated-and-scaled-6abc3b7ad9177be38bcb3ff4).

## The record, in one screen

- **TinyStories architecture gate (135M, overfit)**: all three objectives train (losses
  fell 41-43%), and the gate exposed a real failure mode - a low-entropy codebook shortcut
  that disappears at scale. Record:
  [results/tinystories-overfit](results/tinystories-overfit/README.md).
- **Quantized-target pilot**: a second failure mode caught honestly - near-zero initial
  NCP loss from tightly clustered codewords - fixed with variance-matched codewords, after
  which discretization is token-neutral. Record: [results/h5-quantized](results/h5-quantized/README.md).
- **The 1B-token matched comparison** (this is the headline study): both arms consumed
  999,948,288 tokens of the same packed corpus in the same order; the concept arm matches
  the baseline on token loss, uses its channel heavily, and that use is sequence-generic.
  Verdict: [results/fineweb-edu](results/fineweb-edu/README.md).
- **Sequence length 4096** (4x context, same token budget): the neutrality transfers, the
  zero-feedback penalty *grows* (+0.1373), and the channel's cost shows up as compute:
  1.8x the baseline's wall time per token. Verdict:
  [results/fineweb-edu-4096](results/fineweb-edu-4096/README.md).

Publication materials built from that record: the whitepaper
([docs/whitepaper.md](docs/whitepaper.md)), post drafts for X, Reddit/HN, and a Russian-language
short post ([docs/posts.md](docs/posts.md)), and the Hugging Face model card draft
([docs/model-card.md](docs/model-card.md)). The step-by-step publication procedure,
creating the Modal `huggingface` secret and running the remote publish, is
[docs/publishing-checklist.md](docs/publishing-checklist.md).

This repository is built for one controlled question: does the ConceptLM objective produce
a useful signal when continued pretraining is reduced to a 360M backbone and roughly one
billion tokens?

## Model

| Component | ncp-smol setting |
| --- | --- |
| Token backbone | `HuggingFaceTB/SmolLM2-360M` |
| Total parameters | 382.7M: 361.8M backbone + 20.8M concept path |
| Chunk size | 4 tokens |
| Concept code | 15 product segments, 64 entries each |
| Concept module | 2 causal Llama decoder blocks |
| Injection point | before token decoder block 2 |
| Training loss | `L_ntp + L_ncp + L_vq` |
| Generation API | ordinary next-token generation |

The TinyStories overfit config uses SmolLM2-135M to make architecture checks cheap. The
FineWeb-Edu comparison uses the 360M backbone for both arms. It is not a reproduction of the
8.9B NCP-ArchPreview scale or training recipe; the omitted pieces are listed in
[docs/design.md](docs/design.md).

## Causality

The concept decoder is causal. A concept formed from token chunk `i` predicts concept
`i + 1`; only that prediction is exposed to the token path. Ground-truth future concepts
are detached loss targets and never enter the decoder. The feedback alignment leaves the
first `k - 1` positions at zero, then introduces each prediction at the boundary where its
source chunk is fully in the past.

`tests/test_modeling.py` changes a sequence suffix and asserts that every earlier logit stays
unchanged. This is the regression test for future leakage.

## Reproduce

Keep model and dataset caches on `D:`:

```powershell
$env:HF_HOME = "D:\code\NCP\.cache\huggingface"
$env:HF_DATASETS_CACHE = "D:\code\NCP\.cache\huggingface\datasets"
$env:TORCH_HOME = "D:\code\NCP\.cache\torch"
$env:NCP_ROOT = "D:\code\NCP"
$env:UV_CACHE_DIR = "D:\code\NCP\.cache\uv"
$env:TMP = "D:\code\NCP\.cache\tmp"
$env:TEMP = "D:\code\NCP\.cache\tmp"

uv sync --python 3.11 --all-extras --locked
.venv\Scripts\pytest
```

TinyStories architecture overfit:

```powershell
.venv\Scripts\ncp-smol-prepare configs\tinystories-overfit.yaml
.venv\Scripts\ncp-smol-train configs\tinystories-overfit.yaml
$checkpoint = (Get-Content runs\tinystories-overfit\latest.json | ConvertFrom-Json).checkpoint
.venv\Scripts\ncp-smol-eval configs\tinystories-overfit.yaml `
  "runs\tinystories-overfit\$checkpoint"
```

Modal runs use a persistent `/vol` volume, so prepared tokens and resumable checkpoints do
not disappear with a container:

```powershell
.venv\Scripts\modal.exe run modal_prepare.py::prepare --config tinystories-overfit.yaml
.venv\Scripts\modal.exe run modal_tiny.py::fit --config tinystories-overfit.yaml

.venv\Scripts\modal.exe run modal_prepare.py::prepare --config fineweb-edu-ncp.yaml
.venv\Scripts\modal.exe deploy modal_app.py
.venv\Scripts\python.exe modal_submit.py submit fineweb-edu-ntp.yaml
.venv\Scripts\python.exe modal_submit.py submit fineweb-edu-ncp.yaml
```

Main training uses a deployed function and a durable spawned call. The returned
`function_call_id` can be checked from another machine or client session:

```powershell
.venv\Scripts\python.exe modal_submit.py status fc-...
```

The two FineWeb-Edu arms read the same packed token cache. They share the backbone,
initialization source, sequence length, token order, optimizer, and token budget. The only
training difference is the concept path and its losses.

## Evaluation contract

A comparison is reportable only after all three checks pass:

1. TinyStories overfit lowers NTP, NCP, and VQ loss.
2. NTP-only and NTP+NCP consume the same number of FineWeb-Edu tokens.
3. Held-out NTP loss is measured on the same batches. For NCP, zeroing or shuffling concept
   feedback is also measured on those exact batches.

Loss by token offset within each four-token chunk is reported separately. This distinguishes
a concept path that changes parameters from one that supplies useful boundary information.
The hypotheses and falsification rules are in [docs/experiments.md](docs/experiments.md).

## Scale and cost

The main configs request 1.0B tokens per arm, within the intended 0.5–1.5B range. Each A100
run has a `$60` software cap and a 22-hour wall-clock limit. The tracked all-in allocation
rate is `$2.74/hour`: one A100-40GB, eight CPU cores, and 32 GiB of memory at the rates used
when the run was configured. The two training arms are therefore capped at about `$120`
combined, excluding data preparation, evaluation, and storage. The 135M TinyStories gate
runs separately on an L4 with a `$6` cap. Actual tokens, wall time, and the same cost estimate
are written into every checkpoint rather than inferred later.

The main-run numbers are experiment settings, not claimed spend. Completed TinyStories costs
and metrics are taken directly from `trainer_state.json`, `metrics.jsonl`, and the eval JSON.

After evaluation, build the checkpoint card locally before uploading:

```powershell
$checkpoint = (Get-Content runs\fineweb-edu-ncp\latest.json | ConvertFrom-Json).checkpoint
.venv\Scripts\ncp-smol-publish configs\fineweb-edu-ncp.yaml `
  "runs\fineweb-edu-ncp\$checkpoint" artifacts\fineweb-edu-ncp-eval.json `
  yava-code/Tessera-1B-Nano --dry-run
```

Remove `--dry-run` only after reviewing the generated card. Optimizer state is kept locally
for resume and excluded from the Hugging Face upload.

For checkpoints that remain on the Modal volume, evaluation and publication can stay remote:

```powershell
.venv\Scripts\modal.exe run modal_app.py::eval --config fineweb-edu-ncp.yaml `
  --checkpoint latest `
  --output /vol/artifacts/fineweb-edu-ncp-eval.json
.venv\Scripts\modal.exe run modal_publish.py::publish --config fineweb-edu-ncp.yaml `
  --checkpoint latest `
  --eval-json /vol/artifacts/fineweb-edu-ncp-eval.json `
  --repo-id yava-code/Tessera-1B-Nano
```

Remote publication expects a Modal secret named `huggingface` with the `HF_TOKEN` key. Create
and review that secret in Modal before invoking the publish entrypoint. The full
step-by-step checklist is in [docs/publishing-checklist.md](docs/publishing-checklist.md).

## References

- [ConceptLM: Predicting Concepts, Not Just Tokens](https://arxiv.org/abs/2602.08984)
- [NCP-ArchPreview](https://arxiv.org/abs/2609.10715)
- [LUMIA-Group/ConceptLM](https://github.com/LUMIA-Group/ConceptLM)

Apache-2.0 licensed.
