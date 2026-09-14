# Experiment protocol

The primary comparison is held-out next-token loss after matched continued pretraining.
Generation samples are useful for inspection, but not the headline metric at this scale.

## H1 — optimization

On the fixed TinyStories subset, NTP, NCP, and VQ losses should all fall. Failure of either
auxiliary loss means the concept path is not ready for the main run, even if token loss moves.

## H2 — token-matched comparison

NTP-only and NTP+NCP start from the same SmolLM2 checkpoint and consume the same packed
FineWeb-Edu blocks in the same order. Report final held-out NTP loss and perplexity together
with tokens and estimated cost. A neutral or negative result remains valid; unequal token
counts do not.

## H3 — concept use, not merely extra parameters

Evaluate the trained NCP checkpoint in three modes on identical batches:

- `predicted`: normal concept feedback;
- `zero`: feedback removed;
- `shuffle`: feedback assigned to another sequence in the same batch at the same chunk offset.

If both interventions leave NTP loss unchanged within run noise, the decoder is not making
material use of the learned concept signal. This is stronger evidence than auxiliary loss
reduction alone.

## H4 — boundary profile

Report token cross entropy separately for offsets 0–3 inside each chunk. A genuinely useful
predicted concept should have its clearest effect near the start of the chunk it conditions.
This is an exploratory signature, not a guaranteed result.

## H5 — what should NCP predict?

There is a useful discrepancy between the paper and the released Llama path. The paper
regresses the predicted code mixture toward the next continuous pooled state; the code uses
the selected next codeword. A short token-matched `continuous` versus `quantized` ablation can
measure three things before spending on a full second run: early NCP gradient scale, codebook
utilization, and the zero-feedback intervention delta. This tests whether discretization is
serving as the target itself or as a constrained prediction basis.

The first direct swap failed this scale check: quantized-target NCP loss stayed around
`1e-6`, versus order `1` for the continuous target. The transformed codewords were too tightly
clustered for selected-code prediction to define a meaningful initial error. A valid repeat
therefore needs normalized or variance-matched codewords before it can be called token-matched.

## H6 — the global-code shortcut

Codebook usage alone can hide a low-entropy solution. The TinyStories gate used more than
40% of code indices during intervention evaluation, yet its effective codebook perplexity was
only about 2.4 and shuffling feedback across sequences did not change NTP loss. A useful
concept vocabulary should pass all three checks: non-trivial effective perplexity, a positive
zero-feedback delta, and a positive sequence-shuffle delta.

For the next pilot, record code perplexity and both intervention deltas throughout training.
Introduce a minimal diversity term only if a scale-corrected target still takes the same
shortcut; otherwise target geometry is the cleaner explanation.

## Minimum publishable record

Archive the config, model revision, dataset revision, token-cache hashes, metrics log,
trainer state, and eval JSON. Claims in the README or model card should be generated from
those artifacts, with no extrapolation from partial runs.
