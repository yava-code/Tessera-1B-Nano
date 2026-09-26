# FineWeb-Edu NTP-only arm

This is the control arm of the matched continued-pretraining comparison. It trains the
unmodified SmolLM2-360M backbone with standard next-token prediction on the shared packed
FineWeb-Edu cache. The NTP+NCP arm reads the same files, the same token order, and the same
optimizer settings; the only training difference is the concept path and its losses.

## Run

| Item | Value |
| --- | --- |
| Backbone | `HuggingFaceTB/SmolLM2-360M` (revision `f8027fd0`) |
| Hardware | Modal A100-40GB, 8 CPU cores, 32 GiB |
| Packed train corpus | 1,000,000,000 tokens (SHA256 in `data_metadata.json`) |
| Packed validation set | 10,000,000 tokens |
| Tokens seen | 999,948,288 |
| Optimizer steps | 7,629 (131,072 tokens per step) |
| Sequence length | 1,024 |
| Batch | 8 x 16 accumulated at LR 3e-05, cosine to 10% |
| Wall time | 32,694 s (9.08 h) |
| Configured cost estimate | $24.85 |
| Resume events | none; single uninterrupted container |

The cost estimate uses the configured $2.74/h all-in allocation rate. It is not a Modal
invoice. No NaNs, no gradient spikes (max grad norm 0.407), and no mid-run configuration
change occurred.

## Train loss

Means compare the first and last ten logged train windows.

| Loss | First 10 | Last 10 | Change |
| --- | ---: | ---: | ---: |
| NTP | 2.5437 | 2.5130 | -1.2% |

Mean throughput was 30,504 tokens/s after container warmup.

## Held-out dynamics

Evaluation runs every 100M tokens on fixed held-out batches (32 blocks of the 10M-token
validation split, batcher seed 37).

| Tokens | Held-out NTP loss |
| ---: | ---: |
| 100M | 2.5195 |
| 200M | 2.5164 |
| 300M | 2.5146 |
| 400M | 2.5130 |
| 500M | 2.5122 |
| 600M | 2.5113 |
| 700M | 2.5103 |
| 800M | 2.5096 |
| 900M | 2.5092 |

Held-out loss decreases monotonically across the full token budget. The curve is shallow
because continued pretraining starts from an already-trained 360M checkpoint and the LR is
small; the comparison against the NCP arm reads this same curve at the final checkpoint.

## Final held-out evaluation

The final checkpoint was evaluated on the volume with the standard eval entrypoint (32
batches, identical batcher seed as the intervention evaluations used by the NCP arm).

| Metric | Value |
| --- | ---: |
| Held-out NTP loss | 2.5135 |
| Held-out perplexity | 12.35 |
| Loss at chunk offset 0 | 2.5105 |
| Loss at chunk offset 1 | 2.5199 |
| Loss at chunk offset 2 | 2.5086 |
| Loss at chunk offset 3 | 2.5152 |

Chunk offsets are reported only for comparability with the NCP arm's boundary profile; a
pure token model has no concept boundary, so the offsets are noise-level differences.

## Comparison contract

This arm is the baseline for the two reportable checks in docs/experiments.md:

1. **H2 token matching**: 999,948,288 tokens consumed, identical packed corpus (SHA256
   `c42f4bff...`), identical token order and seed. The NCP arm must report the same token
   count for the comparison to be valid.
2. **Final held-out NTP loss**: 2.5135 measured on the same batches the NCP intervention
   evaluation uses. The verdict is written in `results/fineweb-edu/README.md` once the NCP
   arm finishes.

## Artifacts

- `metrics.jsonl`: complete train and periodic evaluation log (772 records);
- `step-00007629-eval.json`: final held-out evaluation with loss by chunk offset;
- `step-00007629-trainer-state.json`: step, token, cursor, and wall-time record;
- `experiment.json` and `data_metadata.json`: pinned run and data provenance.

The checkpoint itself (`step-00007629`) stays on the Modal `ncp-smol` volume with optimizer
state for resume and is not duplicated into the repository.
