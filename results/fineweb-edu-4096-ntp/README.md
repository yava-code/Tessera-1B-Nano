# FineWeb-Edu NTP-only arm, sequence length 4096

This is the control arm of the seq-4096 matched probe: the unmodified SmolLM2-360M backbone
trained with standard next-token prediction at `sequence_length: 4096` on a fresh packed
FineWeb-Edu cache, at the same 1B-token budget and seed as the 1024 arms. The NCP arm reads
the same cache, the same token order, and the same optimizer settings; the only training
difference is the concept path and its losses.

## Run

| Item | Value |
| --- | --- |
| Backbone | `HuggingFaceTB/SmolLM2-360M` (revision `f8027fd0`) |
| Hardware | Modal A100-40GB, 8 CPU cores, 32 GiB |
| Packed train corpus | 1,000,000,000 tokens, seq 4096 (SHA256 in `data_metadata.json`) |
| Packed validation set | 10,000,000 tokens |
| Tokens seen | 999,948,288 |
| Optimizer steps | 7,629 (131,072 tokens per step) |
| Sequence length | 4,096 |
| Batch | 2 x 16 accumulated at LR 3e-05, cosine to 10% |
| Wall time | 37,096 s (10.30 h) |
| Configured cost estimate | $28.23 |
| Resume events | none; single uninterrupted container |

The cost estimate uses the configured $2.74/h all-in allocation rate. It is not a Modal
invoice. No NaNs, no gradient spikes (max grad norm 0.320), and no mid-run configuration
change occurred.

## Train loss

Means compare the first and last ten logged train windows.

| Loss | First 10 | Last 10 | Change |
| --- | ---: | ---: | ---: |
| NTP | 2.4843 | 2.4649 | -0.8% |

Mean throughput was 26,930 tokens/s (versus 30,504 tokens/s for the 1024 arm: longer
sequences pay more attention overhead per token at this model size).

## Held-out dynamics

Evaluation runs every 100M tokens on fixed held-out batches (32 blocks of the 10M-token
validation split, batcher seed 37), at the arm's own sequence length.

| Tokens | Held-out NTP loss |
| ---: | ---: |
| 100M | 2.4744 |
| 200M | 2.4711 |
| 300M | 2.4695 |
| 400M | 2.4694 |
| 500M | 2.4688 |
| 600M | 2.4683 |
| 700M | 2.4680 |
| 800M | 2.4679 |
| 900M | 2.4678 |

Held-out loss decreases monotonically across the full token budget. At the 900M mark the
4096 arm sits at 2.4678 versus 2.5092 for the 1024 arm; this within-arm gap is the
long-context boost shared by the evaluation sequence length and the arm's training length,
and is exactly why the preregistered headline for this probe is the between-arm gap
measured at 4096.

## Final held-out evaluation

The final checkpoint was evaluated on the volume with the standard eval entrypoint (32
batches, identical batcher seed as the intervention evaluations used by the NCP arm).

| Metric | Value |
| --- | ---: |
| Held-out NTP loss | 2.5367 |
| Held-out perplexity | 12.64 |
| Loss at chunk offset 0 | 2.5355 |
| Loss at chunk offset 1 | 2.5527 |
| Loss at chunk offset 2 | 2.5254 |
| Loss at chunk offset 3 | 2.5331 |

Chunk offsets are reported only for comparability with the NCP arm's boundary profile; a
pure token model has no concept boundary, so the offsets are noise-level differences.

## Comparison contract

This arm is the baseline for the preregistered seq-4096 checks in docs/experiments.md
(probe 4, predictions (a)-(d)):

1. **Token matching**: 999,948,288 tokens consumed, fresh packed corpus at seq 4096, same
   seed 17 and corpus revisions as the 1024 arms. The NCP arm must report the same token
   count for the comparison to be valid.
2. **Headline**: the between-arm NTP gap at 4096 (NCP minus this arm's 2.5367), preregistered
   as neutral within +/-0.005.
3. The verdict is written in `results/fineweb-edu-4096/` once the NCP arm finishes.

## Artifacts

- `metrics.jsonl`: complete train and periodic evaluation log (772 records);
- `step-00007629-eval.json`: final held-out evaluation with loss by chunk offset;
- `step-00007629-trainer-state.json`: step, token, cursor, and wall-time record;
- `experiment.json` and `data_metadata.json`: pinned run and data provenance.

The checkpoint itself (`step-00007629`) stays on the Modal `ncp-smol` volume with optimizer
state for resume and is not duplicated into the repository.
