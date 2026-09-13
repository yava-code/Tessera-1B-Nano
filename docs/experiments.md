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
- `shuffle`: feedback assigned to the wrong chunk.

If both interventions leave NTP loss unchanged within run noise, the decoder is not making
material use of the learned concept signal. This is stronger evidence than auxiliary loss
reduction alone.

## H4 — boundary profile

Report token cross entropy separately for offsets 0–3 inside each chunk. A genuinely useful
predicted concept should have its clearest effect near the start of the chunk it conditions.
This is an exploratory signature, not a guaranteed result.

## Minimum publishable record

Archive the config, model revision, dataset revision, token-cache hashes, metrics log,
trainer state, and eval JSON. Claims in the README or model card should be generated from
those artifacts, with no extrapolation from partial runs.
