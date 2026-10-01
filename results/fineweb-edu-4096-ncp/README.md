# FineWeb-Edu NTP+NCP arm, sequence length 4096

This is the concept arm of the seq-4096 matched probe: the SmolLM2-360M backbone with the
concept-feedback path trained at `sequence_length: 4096` on the same packed FineWeb-Edu
cache, token budget, and seed as the NTP control arm (`results/fineweb-edu-4096-ntp/`).
The two arms read the same cache, the same token order, and the same optimizer settings;
the only training difference is the concept path and its losses.

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
| Wall time | 66,888 s (18.58 h) across two containers |
| Configured cost estimate | $50.91 ($40.01 + $10.90 resume) |
| Resume events | 1; see below |

The cost estimate uses the configured $2.74/h all-in allocation rate. It is not a Modal
invoice. No NaNs, no gradient spikes (max grad norm 0.603), and no mid-run configuration
change to model, data, or optimizer settings.

### Budget stop and resume

The preregistered config carried `run_budget_usd: 40.0`, calibrated on the 1024 arms'
throughput. At sequence 4096 the concept path pays a quadratic cross-attention cost the
1024 calibration did not reflect: throughput was 15,013 tokens/s in segment 1 (versus
30,504 tokens/s for the NTP-1024 arm). The budget guard therefore stopped the run at
step 6,020 (789,053,440 tokens, $40.01) instead of the planned 1B. The run resumed from
`step-00006020` with a config identical to the preregistered one except
`run_budget_usd: 52.0` (and the run name); the resume event reloaded the optimizer and
scheduler state and the batcher cursor, discarded 0 metric lines, and completed the
remaining 210.9M tokens at 14,624 tokens/s. Both segments are in the single
`metrics.jsonl`; the stop-time snapshot is preserved as `metrics-preresume.jsonl` and
`experiment-preresume.json`.

## Train loss

Means compare the first and last ten logged train windows.

| Loss | First 10 | Last 10 | Change |
| --- | ---: | ---: | ---: |
| Total (NTP+NCP+VQ) | 7.3414 | 6.0110 | -18.1% |
| NTP component | 2.4842 | 2.4655 | -0.8% |
| NCP concept-prediction | 2.4278 | 1.7916 | -26.2% |
| VQ codebook | 2.4293 | 1.7539 | -27.8% |

The NTP component tracks the control arm (first/last ten means 2.4843/2.4649) throughout,
as required for token matching.

## Held-out dynamics

Evaluation runs every 100M tokens on fixed held-out batches (32 blocks of the 10M-token
validation split, batcher seed 27), reporting the model's total training loss
(NTP+NCP+VQ), not the NTP-only headline metric.

| Tokens | Held-out total loss |
| ---: | ---: |
| 100M | 7.0449 |
| 200M | 6.8588 |
| 300M | 6.6239 |
| 400M | 6.4617 |
| 500M | 6.3195 |
| 600M | 6.2085 |
| 700M | 6.1348 |
| 800M | 6.0705 |
| 900M | 6.0348 |

Held-out loss decreases monotonically across the full token budget, across both
containers.

## Final held-out evaluation

The final checkpoint was evaluated on the volume with the standard eval entrypoint (32
batches, identical batcher seed and batches as the NTP control arm), measuring NTP-only
token cross-entropy under concept interventions.

| Metric | Value |
| --- | ---: |
| Predicted NTP loss | 2.5372 |
| Held-out perplexity | 12.64 |
| Zero-feedback NTP loss | 2.6746 |
| Shuffle NTP loss | 2.5383 |
| Similar-shuffle NTP loss | 2.5383 |
| Codebook perplexity / usage | 7.22 / 84.3% |

Per-chunk-offset profile (predicted): 2.5360 / 2.5529 / 2.5260 / 2.5340 — the mild
offset-1 elevation matches the control arm's profile shape.

## Preregistered verdict (probe 4, predictions (a)-(d))

Predictions are recorded in docs/experiments.md before execution.

| # | Prediction | Result | Verdict |
| --- | --- | --- | --- |
| a | Between-arm NTP gap at 4096 neutral within +/-0.005 | **+0.0006** (2.5372 vs 2.5367) | **confirmed** |
| b | Zero-feedback delta persists at 4096 | **+0.1373**, larger than the 1024 value +0.1050 | **confirmed** |
| c | Shuffle/similar deltas near zero in 2-seq microbatches | **+0.0011** / **+0.0011** | **confirmed** |
| d | Codebook reaches the 1B-1024 endpoint (7.55 ppl) | 7.22 ppl, 84.3% usage vs 85.6% | **missed, narrowly** |

The headline: the token-loss neutrality of the concept path survives a 4x sequence length
(a), and the model's reliance on the feedback channel is not an artifact of the short
context — removing it costs more at 4096 than at 1024 (b), while partner-identity
controls stay at noise level (c). The codebook lands just under the 1024 endpoint in
both perplexity and usage (d); given the identical token budget and the higher codebook
demand of 32 chunks per sequence, this reads as a within-noise shortfall rather than a
length effect, but it is recorded as a miss.

## Comparison contract

1. **Token matching**: both arms consumed 999,948,288 tokens of the same packed corpus
   (SHA256 in each `data_metadata.json`), same seed 17, same corpus revisions.
2. **Headline**: between-arm NTP gap +0.0006, inside the preregistered +/-0.005 band.
3. The run-level verdict is written in `results/fineweb-edu-4096/README.md` and
   docs/experiments.md (probe 4).

## Artifacts

- `metrics.jsonl`: complete train and periodic evaluation log (772 records), both
  containers;
- `step-00007629-eval.json`: final held-out evaluation with interventions, codebook
  usage, and loss by chunk offset;
- `step-00007629-trainer-state.json`: step, token, cursor, and billable-time record;
- `experiment.json` and `data_metadata.json`: pinned run and data provenance;
- `metrics-preresume.jsonl` and `experiment-preresume.json`: stop-time snapshot of the
  budget-interrupted first container.

The checkpoint itself (`step-00007629`) stays on the Modal `ncp-smol` volume with
optimizer state for resume and is not duplicated into the repository.
