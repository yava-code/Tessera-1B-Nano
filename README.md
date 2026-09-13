# ncp-smol

Small-scale, independent reproduction of Next Concept Prediction on SmolLM2.

`ncp-smol` keeps token-level autoregressive generation, but adds a thin latent path. Every
four token states are pooled into a concept, product-quantized, passed through a causal
concept module, and used to predict the next concept. The predicted concept is then added
back to the token backbone before the second decoder layer.

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
.venv\Scripts\ncp-smol-eval configs\tinystories-overfit.yaml `
  runs\tinystories-overfit\step-XXXXXXXX
```

Modal runs use a persistent `/vol` volume, so prepared tokens and resumable checkpoints do
not disappear with a container:

```powershell
.venv\Scripts\modal.exe run modal_app.py::prepare --config tinystories-overfit.yaml
.venv\Scripts\modal.exe run modal_app.py::fit --config tinystories-overfit.yaml

.venv\Scripts\modal.exe run modal_app.py::prepare --config fineweb-edu-ncp.yaml
.venv\Scripts\modal.exe run modal_app.py::fit --config fineweb-edu-ntp.yaml
.venv\Scripts\modal.exe run modal_app.py::fit --config fineweb-edu-ncp.yaml
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

The main configs request 1.0B tokens per arm, within the intended 0.5–1.5B range. Each GPU
job has a `$110` hard software cap. The current 22-hour A100-40GB timeout corresponds to
about `$53` in GPU time at `$2.10/hour`, plus the configured 15% allowance for CPU, memory,
and storage. TinyStories has a `$6` cap. Actual tokens, wall time, and the same cost estimate
are written into every checkpoint rather than inferred later.

These numbers are experiment settings, not claimed spend or completed results. Published
results should come directly from `trainer_state.json`, `metrics.jsonl`, and the eval JSON.

After evaluation, build the checkpoint card locally before uploading:

```powershell
.venv\Scripts\ncp-smol-publish configs\fineweb-edu-ncp.yaml `
  runs\fineweb-edu-ncp\step-XXXXXXXX artifacts\fineweb-edu-ncp-eval.json `
  USERNAME/ncp-smol-360m --dry-run
```

Remove `--dry-run` only after reviewing the generated card. Optimizer state is kept locally
for resume and excluded from the Hugging Face upload.

For checkpoints that remain on the Modal volume, evaluation and publication can stay remote:

```powershell
.venv\Scripts\modal.exe run modal_app.py::eval --config fineweb-edu-ncp.yaml `
  --checkpoint /vol/runs/fineweb-edu-ncp/step-XXXXXXXX
.venv\Scripts\modal.exe run modal_app.py::publish --config fineweb-edu-ncp.yaml `
  --checkpoint /vol/runs/fineweb-edu-ncp/step-XXXXXXXX `
  --eval-json /vol/artifacts/fineweb-edu-ncp-eval.json `
  --repo-id USERNAME/ncp-smol-360m
```

Remote publication expects a Modal secret named `huggingface` with the `HF_TOKEN` key. Create
and review that secret in Modal before invoking the publish entrypoint.

## References

- [ConceptLM: Predicting Concepts, Not Just Tokens](https://arxiv.org/abs/2602.08984)
- [NCP-ArchPreview](https://arxiv.org/abs/2609.10715)
- [LUMIA-Group/ConceptLM](https://github.com/LUMIA-Group/ConceptLM)

Apache-2.0 licensed.
