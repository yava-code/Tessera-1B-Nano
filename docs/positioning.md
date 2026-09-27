# Project positioning

Use a claim only after the public repository, checkpoint, and matched comparison exist.
The FineWeb-Edu matched comparison is complete; these claims are written against its
recorded artifacts (results/fineweb-edu and per-arm READMEs), not extrapolations.

## Calm

1. **Ran a token-matched FineWeb-Edu comparison of NTP-only versus NTP+NCP on
   SmolLM2-360M: exactly 999,948,288 tokens per arm, same data order, same optimizer,
   no restarts, no NaNs, for about $55 of tracked compute.**

2. **Reproduced the ConceptLM objective end to end at 360M and measured not just token
   loss but causal concept interventions: zero-feedback, sequence-shuffle, codebook
   perplexity, usage, and loss by chunk offset.**

## Bold

3. **Showed the learned concept channel is real but generic: the decoder demonstrably
   consumes predicted concept feedback (+0.105 NTP loss when zeroed) through a rich,
   non-collapsed codebook (perplexity 7.55, 85.6% usage), while shuffled feedback from
   another sequence works as well as the sequence's own.**

4. **Provided the first independent matched-scale check of Next Concept Prediction:
   a neutral token-loss result at 1B tokens that mirrors the original paper's own
   structure, where auxiliary pairs alone underperform pure NTP and only the full
   triple wins at much larger scale.**

## Short

5. **Reproduced NCP at 360M: a used, rich, but sequence-generic concept channel — and an
   honest neutral verdict at 1B tokens, with the next probes stated in advance.**
