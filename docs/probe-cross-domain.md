# Cross-domain feedback partner probe

Follow-up to the similar-shuffle result. That probe replaced each sequence's concept
feedback with the most similar sequence's feedback *from the same FineWeb-Edu batch* and
measured a delta of +0.0008 — indistinguishable from random shuffle. The stated caveat was
that partners inside a FineWeb-Edu batch are already same-corpus, so the probe could only
test topic identity within one distribution. This probe removes that caveat: partners now
come from a **different corpus entirely**, so their concept feedback cannot accidentally
match the query's topic.

## Design

For each held-out evaluation batch the trained concept arm runs the normal `predicted`
pass. Its predicted per-chunk feedback vectors `F` (batch, chunks, hidden) are captured,
then re-injected from a **foreign pool**: predicted-feedback vectors computed by the same
model on sequences from another corpus. Two foreign pools are used, ordered by expected
distance from FineWeb-Edu prose:

1. **Wikipedia (20231101.en sample)** — different register and topic mix, but still
   natural-language prose. Expectation written before the run: near-zero delta, because
   FineWeb-Edu already covers encyclopedic content and the measured use is
   sequence-generic.
2. **Code (code_search_net, python split)** — source code token distribution is far
   from web prose. (The originally planned codeparrot/github-code still ships only a
   legacy dataset script, dead under datasets>=3.) This is the strongest form of the
   probe: if even code-derived feedback does not hurt, the channel is insensitive to
   partner domain at the observed effect size (+0.105 baseline for zeroing).

## Mechanism (inference-time only)

- `NcpSmolForCausalLM` gains two pieces of state on `_hook_state`:
  - `capture_feedback`: after computing `predicted` feedback, store it on
    `self._hook_state["last_feedback"]` (mode `predicted` only).
  - `override_feedback`: a (batch, chunks, hidden) tensor, sliced/padded to the current
    batch size, injected in place of the predicted feedback. `mode` stays `predicted`
    so the rest of the path (losses, code indices) is untouched.
- `ncp_smol/probes.py`:
  - `foreign_feedback(model, pool_batcher, batches, device)` — runs the model over
    foreign-corpus batches, captures feedback, returns a tensor pool.
  - `cross_domain_delta(model, corpus, pool, batches, seed, device)` — for each eval
    batch, passes the model in `predicted` mode with `override_feedback` drawn from the
    pool, measures `token_cross_entropy` on identical held-out FineWeb-Edu batches, and
    returns the mean loss delta versus the true `predicted` pass.
- New `TokenCorpus` files are built on Modal by `modal_app.py::prepare_foreign` from
  streaming corpora with the same tokenizer, EOS packing, and 10M-token validation-style
  size as the existing held-out set. SHA256 is recorded in the probe artifact.

## Preregistered predictions (before the run)

Recorded before executing, following the same convention as H3/H6:

- `zero` (committed): +0.1050 — the reference intervention.
- `shuffle` (committed): +0.0008 — within-batch random partners.
- `similar_shuffle` (committed): +0.0008 — within-batch most-similar partners.
- **Wikipedia pool: +0.001 to +0.01** — partners are topic-distant, but the channel was
  measured as sequence-generic, so domain mismatch alone should not change much.
- **Code pool: +0.01 to +0.03** — the strongest mismatch; if the channel carries any
  register-level prior, this is where a small cost appears. If the code-pool delta lands
  near +0.10, the channel encodes "is my input prose" — a surprising and publishable
  signal in itself.

## Outcome criteria

- All foreign deltas ≈ 0 (within ±2 SE ≈ ±0.02): the channel's usefulness is
  robust to partner domain at this scale; the sequence-generic reading stands, now with
  an out-of-distribution control.
- A graded penalty (code > wiki > own): the channel carries register/topic-level
  information beyond sequence identity; quantifies *how much* of the +0.105 is
  topic-specific.
- Code-pool delta ≈ zero-delta: the channel gates on input domain; new follow-up —
  mixed-domain training.

## Result (2026-09-29, A100, 32 batches, seed 17/37)

- wikipedia partners: delta **+0.0016** (2.51588 vs 2.51433);
- code partners: delta **+0.0011** (2.51539 vs 2.51433);
- artifact: `results/fineweb-edu-ncp/step-00007629-eval-cross-domain.json`.

First criterion hit: both deltas are inside the preregistered wiki band at its low edge
and far below the +0.105 zero-delta. Partner domain does not matter at this effect size;
together with similar-shuffle (+0.0008) the sequence-generic reading now holds across
partner identity, similarity, and domain. The prose-gate hypothesis is ruled out.
