# H5 — quantized versus continuous NCP target (both with variance-matched codewords)

The preregistered repeat of the aborted direct target swap. Both arms are token-matched:
same SmolLM2-135M base (revision `93efa2f0`), same TinyStories cache (16,777,216 tokens
each, identical order, seed 17), same optimizer, `overfit` regime on the 1M-token cache,
L4 on Modal. The only differences are `ncp_target: continuous|quantized` and the run
name. Both use the new `codebook_normalization: variance`, the stated H5 prerequisite:
transformed codewords are rescaled per segment to unit spread before quantization and
prediction.

## Prerequisite check (the reason the first swap was aborted)

Initial quantized-target NCP loss at the exact pilot geometry:

- aborted pilot (`ncp_target: quantized`, no normalization): **2.80e-7** at step 1;
- replication in this pass (`none`): **3.11e-7**;
- this run's arm (`variance`): **0.88** at step 1, in the same order as the continuous
  target's ~1.0.

The variance fix restores a meaningful error signal; the ablation is now token-matched
in gradient-scale terms, not just in data terms.

## Results at the final checkpoint (step 4096, 16,777,216 tokens)

| metric | continuous | quantized |
|---|---|---|
| held-out NTP loss (intervention eval, 16 batches) | **1.8493** | **1.8495** |
| NCP train loss (last eval window) | 0.5688 | **0.1409** |
| codebook perplexity | 9.80 | **10.21** |
| codebook usage | 18.1% | 17.9% |
| zero-feedback delta | +0.1720 | **+0.1883** |
| shuffle delta | +0.0083 | +0.0042 |
| similar-shuffle delta | +0.0087 | +0.0028 |
| tracked cost | $1.39 | $1.39 |

Both arms learn, both codebooks enrich, both decoders causally use the channel, and in
both arms the use is again sequence-generic (shuffle and similar-shuffle near zero while
zeroing hurts). The quantized arm even shows a slightly larger zero-delta.

## Verdict on H5

**Neutral on the headline metric, positive on the mechanism.** With the prerequisite
satisfied, the choice of target no longer determines the initial NTP trajectory: both
arms land within +0.0002 NTP of each other (run noise), against the aborted pilot where
the quantized arm could not learn at all. The quantized target compresses the NCP loss
~4x (predicting one of 64 codes per segment is easier than regressing a continuous
state) without any token-loss penalty or codebook collapse at this scale.

This does not overturn the paper's released-code choice, but it removes "quantized
target degenerates at small scale" from the list of open explanations. The remaining H5
question (does the target choice matter at 4-8B tokens where the paper's gains appear)
is deferred with the rest of the scale-up probes.

## Files

- `step-00004096-eval-continuous.json` / `step-00004096-eval-quantized.json`:
  intervention evals (predicted/zero/shuffle/similar_shuffle, offsets, codebook stats);
- `metrics-continuous.jsonl` / `metrics-quantized.jsonl`: complete train/eval logs;
- configs: `configs/h5-quantized-continuous.yaml` / `configs/h5-quantized-quantized.yaml`
  (identical except run name, output dir, and `ncp_target`);
- implementation: `codebook_normalization` in `src/ncp_smol/quantizer.py`,
  `configuration.py`, `experiment.py`, `runtime.py` (default `none` = back-compatible;
  published checkpoints are unaffected).
