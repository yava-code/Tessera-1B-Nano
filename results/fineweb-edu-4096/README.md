# FineWeb-Edu comparison at sequence length 4096

Both matched arms of preregistered probe 4 (docs/experiments.md) are complete. Each
consumed 999,948,288 tokens of the same packed corpus, in the same order, from the same
initialization, under the same optimizer and schedule. The H2 contract transfers to
sequence length 4096.

| Split | Tokens | SHA256 |
| --- | ---: | --- |
| Train | 1,000,000,000 | `c42f4bffe9e225eca05ee5fafa9bfc42875d014125f18f47919be611f775a413` |
| Validation | 10,000,000 | `61ecc77c5d586daa10c5938a6719463380e8e64347c0194e09635651561d4330` |

These hashes are identical to the 1024 arms' packed cache: the packer's token stream is
independent of the window length, so the 4096 arms read the very same token sequence as
the 1024 arms, regrouped into 4x longer windows. `data_metadata.json` differs from the
1024 cache only in `sequence_length`.

## Matched result

| | NTP-only | NTP+NCP |
| --- | ---: | ---: |
| Tokens | 999,948,288 | 999,948,288 |
| Wall time | 10.30 h | 18.58 h (one budget stop + resume) |
| Tracked cost estimate | $28.23 | $50.91 |
| Final held-out NTP loss | **2.5367** | **2.5372** |
| Final held-out perplexity | 12.64 | 12.64 |
| Throughput | 26,930 tok/s | 15,013 tok/s |

Final held-out NTP loss differs by **+0.0006 (+0.02%) in favor of the NTP-only arm** —
neutral, inside the preregistered +/-0.005 band. Full records:
`../fineweb-edu-4096-ntp/README.md` and `../fineweb-edu-4096-ncp/README.md`.

The cost asymmetry is real and worth naming: at 4096 the concept path pays a quadratic
cross-attention cost the decoder side does not, so the NCP arm runs at 1.8x the NTP arm's
wall time per token (at 1024 the ratio was 1.2x). Sequence length is not free for the
channel; whether that cost can be amortized (chunk-parallel encoding, cached feedback) is
an engineering question the science result does not depend on.

## Preregistered predictions (a)-(d)

| # | Prediction | Result | Verdict |
| --- | --- | --- | --- |
| a | Between-arm NTP gap at 4096 neutral within +/-0.005 | **+0.0006** | confirmed |
| b | Zero-feedback delta persists at 4096 | **+0.1373** (vs +0.1050 at 1024) | confirmed |
| c | Shuffle/similar deltas near zero in 2-seq microbatches | **+0.0011** / **+0.0011** | confirmed |
| d | Codebook reaches the 1B-1024 endpoint (7.55 ppl) | **7.22 ppl, 84.3% usage** | missed, narrowly |

## What this probe establishes

1. **The neutrality result is not a short-context artifact.** At 4x the sequence length,
   with 16x more chunks per microbatch and the same token budget, the concept path still
   neither helps nor hurts token loss (a). The matched-scale conclusion of the published
   comparison transfers to long contexts.
2. **The channel's use is chunk-offset-based, not context-window-based.** Zeroing the
   feedback costs *more* at 4096 (+0.1373) than at 1024 (+0.1050): the decoder leans on
   the concept signal at least as much when each sequence carries 32 chunk boundaries
   instead of 2 (b). The intervention signature scales with the number of in-window
   decisions, not with context length itself.
3. **Partner-identity controls stay at noise under 2-sequence microbatches** (c): with
   only one alternative sequence to shuffle, the control is weaker than at 1024, exactly
   as preregistered; it still shows nothing sequence-specific.
4. **The codebook lands just under the 1024 endpoint** (d): 7.22 ppl / 84.3% usage vs
   7.55 / 85.6%. With the same token budget spread over 8x more per-sequence chunks, the
   shortfall is small and within what codebook growth noise at this scale suggests, but
   it is recorded as a preregistered miss, not waved away.

## Artifacts

- `../fineweb-edu-4096-ntp/` and `../fineweb-edu-4096-ncp/`: per-arm mirrors with full
  metric logs, final evals, trainer states, and provenance;
- the NCP arm's budget stop at 789M tokens and its resume to the full 1B budget are
  documented in `../fineweb-edu-4096-ncp/README.md`.
