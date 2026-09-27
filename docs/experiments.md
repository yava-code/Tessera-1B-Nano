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

## Outcomes — FineWeb-Edu 1B comparison (September 2026)

Both arms are complete and matched: 999,948,288 tokens each, no restarts, no NaNs
(results/fineweb-edu). Status per hypothesis:

- **H1 — pass at both scales.** TinyStories gate: NTP, NCP, and VQ train losses fell
  41–43% over 16.8M tokens. FineWeb-Edu arms: NTP −1.2%, NCP −25.6%, VQ −27.2% (first
  versus last ten train windows), no degenerate collapse.
- **H2 — neutral.** Final held-out NTP loss 2.5135 (NTP-only) versus 2.5142 (NTP+NCP):
  +0.0007 (+0.03%) in favor of the pure token arm; perplexity 12.35 versus 12.36.
  Tracked cost $24.85 versus $30.27 at $2.74/h.
- **H3 — split verdict, replicated at both scales.** Zeroing concept feedback costs
  +0.1050 NTP loss: the decoder uses the channel. Shuffled feedback from another sequence
  in the same batch costs only +0.0008: the use is sequence-generic. The same split —
  zero hurts, shuffle does not — appeared already in the TinyStories gate.
- **H4 — direction only.** The zero-delta decays across the chunk (+0.1066 at offset 0,
  +0.1039 at offset 3): the predicted concept helps most where it is freshest, but the
  gradient of the effect is small.
- **H5 — open, with a stated prerequisite.** The direct quantized-target swap failed the
  scale check (NCP loss ~1e-6 because transformed codewords are too tightly clustered).
  A valid repeat needs normalized or variance-matched codewords before the ablation is
  token-matched.
- **H6 — two of three checks.** Codebook perplexity 7.55 and usage 85.6% at the final
  checkpoint, grown monotonically from 2.93/74.8% at 100M tokens — no low-entropy
  shortcut. The shuffle check fails (delta ~ 0), consistent with H3.

## What the original paper reports

Calibration against ConceptLM (arXiv 2602.08984), so the neutral H2 is read in context
rather than as a failure:

- Table 4: `L_ntp` alone reaches ppl 69.28; `L_ntp + L_ncp` gives 76.30 and
  `L_ntp + L_vq` gives 75.64 — both auxiliary pairs alone are *worse* than pure NTP.
  Only the full `L_ntp + L_ncp + L_vq` triple reaches 68.13. Our neutral H2 with
  strongly falling auxiliary losses reproduces this structure at 1B tokens.
- The paper's reported gains come from the full triple at larger scale (Pythia-410M on
  300B tokens, GPT-2 1.5B on 8B, Llama-3.1-8B continual on 9.6B; +0.4 average accuracy
  over the PM baseline). Nothing there contradicts a neutral result at 1B tokens.
- Mechanistically, concept layers run on T/k positions. With sequence length 1024 and
  chunk size 4 this is 256 concept positions per sequence — a coarse, sequence-level
  summary channel. That is consistent with the sequence-generic use measured by the
  shuffle control.

## Minimum publishable record

Archive the config, model revision, dataset revision, token-cache hashes, metrics log,
trainer state, and eval JSON. Claims in the README or model card should be generated from
those artifacts, with no extrapolation from partial runs.

## Next probes

Ordered by cost and information gained:

1. **Similar-shuffle probe (CPU).** Shuffle feedback to the most similar sequence in the
   batch instead of a random one. If similar-shuffle hurts while exact shuffle does not,
   the channel carries topic-level rather than sequence-identity signal.
2. **H5 quantized-target with normalized codewords.** The stated prerequisite above;
   a short token-matched ablation before any second full run.
3. **Sequence length 4096 at the same token budget.** More concept positions per sequence
   tests whether channel granularity limits usefulness.
4. **4–8B tokens.** The same matched design at several times the current budget,
   approaching the scales where the original paper reports gains (8–10B).
