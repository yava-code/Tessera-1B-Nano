# Finishing the FineWeb-Edu matched run

The NCP arm was submitted as a durable Modal call during the previous session. This file
records how to monitor it and what to do when it finishes.

## Submitted run

| Item | Value |
| --- | --- |
| Config | `configs/fineweb-edu-ncp.yaml` |
| App | `ncp-smol` (deployed from `modal_app.py`) |
| Function call id | `fc-01M3ER9EDMVHHW7647WGMA80R8` |
| Submitted | 2026-09-26, durable spawned call |
| Budget cap | `$60` tracked estimate, 22 h wall-clock (trainer stops itself) |

The call id is also saved locally in `state/submissions.json` (git-ignored), so
`--config` lookups work without remembering the id.

## Monitoring

All of these are read-only and safe to run while training:

```powershell
.venv\Scripts\python.exe -X utf8 modal_submit.py status --config fineweb-edu-ncp.yaml
.venv\Scripts\python.exe -X utf8 modal_submit.py logs --config fineweb-edu-ncp.yaml --tail 20
.venv\Scripts\modal.exe run modal_app.py::progress --config fineweb-edu-ncp.yaml
```

Healthy startup evidence recorded at submission time: step 100 reached 13.1M tokens at
22.3k tokens/s, grad norm about 0.28, `ncp_loss` about 2.43 (order 1, unlike the degenerate
quantized pilot at 1e-6). Expected finish: roughly 12.5 h and about $34, at step 7629 with
999,948,288 tokens — the same token count as the NTP arm.

## When the run finishes

1. Confirm the final checkpoint exists on the volume:
   `runs/fineweb-edu-ncp/step-00007629/trainer_state.json` with
   `tokens_seen = 999948288` (equal to the NTP arm's count — this is the H2 requirement).
2. Run the intervention evaluation on the volume:

   ```powershell
   .venv\Scripts\modal.exe run modal_app.py::eval --config fineweb-edu-ncp.yaml `
     --checkpoint latest `
     --output /vol/artifacts/fineweb-edu-ncp-eval.json
   ```

   This produces held-out NTP loss in `predicted`, `zero`, and `shuffle` modes on identical
   batches (H3), plus loss by chunk offset (H4) and codebook usage/perplexity (H6 checks).
3. Copy artifacts from the volume into `results/fineweb-edu-ncp/`: `metrics.jsonl`,
   the eval JSON, the final `trainer_state.json`, `experiment.json`, `data_metadata.json`.
   Write a README in the style of `results/tinystories-overfit/README.md`: run table,
   train-loss means for NTP/NCP/VQ, held-out dynamics, intervention deltas, artifact list.
4. Update `results/fineweb-edu/README.md`: replace the "neither arm has started" note with
   the H2 verdict — final held-out NTP loss for both arms, tokens, and tracked cost.
   The NTP baseline is already evaluated: NTP loss 2.5135, perplexity 12.35
   (`artifacts/fineweb-edu-ntp-eval.json` on the volume).
5. Optional publication stays gated as before: build the card with `--dry-run` locally,
   review it, and only then create the `huggingface` Modal secret (`HF_TOKEN`) and run
   `modal_publish.py`. Publication was intentionally not performed in this session.

## Failure handling

- If the call dies (OOM, preemption, error), `status` reports it; `runs/` on the volume
  keeps the last checkpoint and `resume: true` restarts from it. Resubmit with
  `modal_submit.py submit fineweb-edu-ncp.yaml` — a fresh call id replaces the saved one.
- The trainer stops itself at the `$60` cap or 22 h; the resulting checkpoint is still
  valid, but the comparison is only reportable at equal token counts, so prefer letting it
  reach step 7629 unless the budget stops it first.
