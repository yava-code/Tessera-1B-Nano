# FineWeb-Edu NCP arm

This is the treatment arm of the matched continued-pretraining comparison: the same
SmolLM2-360M backbone as the NTP-only arm plus the causal concept path (product VQ, two
concept decoder blocks, injection before token decoder block 2), trained with
`L_ntp + L_ncp + L_vq`. Both arms read the same packed FineWeb-Edu cache in the same order.

## Run

| Item | Value |
| --- | --- |
| Backbone | `HuggingFaceTB/SmolLM2-360M` (revision `f8027fd0`) |
| Concept path | 15 segments x 64 codewords, 2 causal blocks, `ncp_target: continuous` |
| Hardware | Modal A100-40GB, 8 CPU cores, 32 GiB |
| Packed train corpus | 1,000,000,000 tokens, SHA256 `c42f4bff...` |
| Tokens seen | 999,948,288 |
| Optimizer steps | 7,629 (131,072 tokens per step) |
| Wall time | 39,772 s (11.05 h) |
| Configured cost estimate | $30.27 |
| Resume events | none; single uninterrupted container |

Token count matches the NTP arm exactly (999,948,288), so the comparison contract in
docs/experiments.md is satisfied. No NaNs, no hard gradient spikes (max grad norm 2.08
during the first step, 0.28–0.40 thereafter), no mid-run configuration change. Mean
throughput was 24,932 tokens/s; the concept path costs about 18% throughput relative to
the NTP arm's 30,504.

## Train loss

Means compare the first and last ten logged train windows.

| Loss | First 10 | Last 10 | Change |
| --- | ---: | ---: | ---: |
| NTP | 2.5438 | 2.5136 | -1.2% |
| NCP | 2.4505 | 1.8220 | -25.6% |
| VQ | 2.4572 | 1.7890 | -27.2% |
| Total | 7.4514 | 6.1246 | -17.8% |

All three objectives fall without degenerate collapse, unlike the quantized-target pilot
whose NCP loss stayed near 1e-6.

## Held-out dynamics

Evaluation runs every 100M tokens on fixed held-out batches (32 blocks, batcher seed 37),
the same schedule and batches as the NTP arm.

| Tokens | NTP | NCP | VQ | Code usage | Code perplexity |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100M | 2.5196 | 2.3136 | 2.3197 | 74.8% | 2.93 |
| 200M | 2.5166 | 2.2134 | 2.2130 | 75.6% | 3.69 |
| 300M | 2.5147 | 2.1170 | 2.1094 | 73.3% | 4.15 |
| 400M | 2.5132 | 2.0312 | 2.0168 | 77.5% | 5.23 |
| 500M | 2.5124 | 1.9733 | 1.9535 | 80.5% | 5.91 |
| 600M | 2.5115 | 1.9155 | 1.8913 | 82.1% | 6.46 |
| 700M | 2.5106 | 1.8700 | 1.8424 | 83.4% | 6.89 |
| 800M | 2.5099 | 1.8485 | 1.8185 | 84.1% | 7.19 |
| 900M | 2.5096 | 1.8274 | 1.7959 | 84.3% | 7.41 |

The codebook keeps enriching throughout training: effective perplexity grows 2.93 -> 7.41
and usage climbs to 84%, far beyond the low-entropy shortcut the TinyStories overfit gate
fell into (ppl ~2.4, usage ~31-41%).

## Token-path comparison with the NTP arm

Mean train NTP loss per 763-step window, NCP arm minus NTP-only arm on identical token
ranges: +0.0000, +0.0002, +0.0002, +0.0003, +0.0003, +0.0004, +0.0005, +0.0005, +0.0006,
+0.0006. The concept path tracks the pure token model to within half a thousandth of a nat
everywhere, drifting marginally above it late in training.

## Final held-out evaluation and interventions

The final checkpoint was evaluated on identical batches in three modes (H3), with codebook
statistics (H6) and loss by chunk offset (H4).

| Mode | Held-out NTP | Perplexity |
| --- | ---: | ---: |
| Predicted feedback | 2.5142 | 12.36 |
| Zero feedback | 2.6192 | 13.73 |
| Shuffled feedback | 2.5150 | 12.37 |

| Check | Value | Reading |
| --- | ---: | --- |
| Zero-feedback delta (H3) | +0.1050 | removing concepts costs 4.2% higher NTP loss: the decoder uses the channel |
| Shuffle-feedback delta (H3) | +0.0008 | sequence-identical feedback is no better than another sequence's: use is generic, not sequence-specific |
| Codebook perplexity / usage | 7.55 / 85.6% | passes the non-trivial-vocabulary check of H6 |
| H6 shuffle check | delta ~ 0 | does not pass: the vocabulary is rich but its use is sequence-generic |
| Offset-0 vs offset-3 zero-delta | +0.1066 vs +0.1039 | concept help decays slightly across the chunk: direction matches H4, magnitude is small |

Zero-feedback loss by chunk offset: 2.6177, 2.6254, 2.6142, 2.6197 (offsets 0-3);
predicted: 2.5111, 2.5204, 2.5095, 2.5158.

## Verdict at this scale

H2 is neutral at 1B tokens: final held-out NTP loss 2.5142 versus 2.5135 for the NTP-only
arm (+0.0007, about +0.03%). The concept path neither helps nor hurts the token objective
under this recipe, while learning a rich (H6 perplexity 7.55) concept vocabulary whose
predicted feedback the decoder demonstrably consumes (zero-delta +0.105) in a
sequence-generic way (shuffle-delta ~0). A neutral H2 with a strongly used, generic latent
channel is a legitimate result; the follow-up candidates are scale, schedule, or target
geometry, and they are listed in docs/experiments.md rather than claimed here.

## Artifacts

- `metrics.jsonl`: complete train and periodic evaluation log (772 records);
- `step-00007629-eval.json`: final intervention evaluation on the volume's held-out batches;
- `step-00007629-trainer-state.json`: step, token, cursor, and wall-time record;
- `experiment.json` and `data_metadata.json`: pinned run and data provenance.

The checkpoint (`step-00007629`) stays on the Modal `ncp-smol` volume with optimizer state;
the eval JSON is also at `artifacts/fineweb-edu-ncp-eval.json` on the volume.
