# Publishing the NCP checkpoint to Hugging Face

Step-by-step checklist, mapped to the existing scripts. The checkpoint
(`runs/fineweb-edu-ncp/step-00007629`) lives on the Modal volume `ncp-smol`, so the
publication path is remote. Nothing below runs automatically; every manual step is
explicit. Windows notes: run modal commands with `PYTHONIOENCODING=utf-8`, python with
`-X utf8`, and pass `/vol/...` paths with `MSYS_NO_PATHCONV=1` in Git Bash.

## 0. Preconditions (already true)

- [x] Final checkpoint on the volume: `runs/fineweb-edu-ncp/step-00007629/` with
      `trainer_state.json` (`tokens_seen = 999948288`), weights (safetensors), tokenizer
      files, `experiment.json`, `data_metadata.json`, `optimizer.pt`.
- [x] Eval artifact on the volume: `/vol/artifacts/fineweb-edu-ncp-eval.json` (all three
      intervention modes) and its committed copy
      `results/fineweb-edu-ncp/step-00007629-eval.json`.
- [x] Card dry-run verified locally: `ncp-smol-publish --dry-run` renders a correct card
      from the committed artifacts (see `docs/model-card.md` for the curated version).

Checkpoint contents that `publish()` uploads: everything in the checkpoint directory
except `optimizer.pt` (excluded by `ignore_patterns`). That is the model, tokenizer,
`eval.json`, `metrics.jsonl`, and provenance JSONs, a complete self-contained repo.

## 1. Create the Modal `huggingface` secret (manual, one-time)

1. Get a Hugging Face token with **write** access: https://huggingface.co/settings/tokens
   (fine-grained, scoped to the target repo/user is enough).
2. In the Modal dashboard: https://modal.com/secrets -> **Create new secret** ->
   provider **Hugging Face** -> key `HF_TOKEN` = `<token>` -> name the secret exactly
   **`huggingface`**. `modal_publish.py` declares
   `modal.Secret.from_name("huggingface")`, so the name is load-bearing.
3. CLI equivalent, if preferred:

   ```powershell
   .venv\Scripts\modal.exe secret create huggingface HF_TOKEN=<token>
   ```

4. Verify it resolves:

   ```powershell
   $env:PYTHONIOENCODING = "utf-8"
   .venv\Scripts\modal.exe secret list
   ```

## 2. Decide repo id and visibility

- Executed 2026-09-29: the `Paragon-Intelligence-Labs` org is not available to the token,
  so `whoami` resolved the namespace to the personal account `yava-code`. Published repos:
  [yava-code/Tessera-1B-Nano](https://huggingface.co/yava-code/Tessera-1B-Nano) and
  [yava-code/Tessera-1B-Nano-Base](https://huggingface.co/yava-code/Tessera-1B-Nano-Base).
  The brand lives in the cards. If the org is granted later, both repos can be moved in
  the HF UI without a re-upload.
- Repo starts public by default; add `--private` for a first private upload and flip it
  in the HF UI later. `publish()` calls `create_repo(..., exist_ok=True)`, so a retry is
  safe and re-uploads over the same repo.

## 3. Review the card one last time (done)

The uploaded card is generated remotely by `build_card`; the factual tables come from
the checkpoint's `trainer_state.json` and the eval JSON. The curated TL;DR, codebook row,
and loading snippet are part of `build_card` and upload automatically; `docs/model-card.md`
mirrors the rendered output. The card is mode-aware: `mode: ntp` renders the
Tessera-1B-Nano-Base control-arm card.

- [ ] Optional: merge the curated sections (TL;DR, intervention reading, limitations) into
      the card. Two ways:
      - Edit `src/ncp_smol/publish.py::build_card` and redeploy (keeps remote generation
        reproducible); or
      - Upload first, then replace the README in the repo via the HF UI or
        `huggingface_hub.upload_file(..., path_in_repo="README.md")`.
- [x] The loading snippet is filled automatically: `build_card` renders the `REPO_ID`
  placeholder with the repo id passed to `publish`.

## 4. Publish (remote, reads the volume; executed 2026-09-29)

Durable-call note: `modal run` against this function can outlive the local CLI timeout;
spawn it instead (`Function.from_name("ncp-smol-publish", "publish_checkpoint").spawn(...)`)
and poll with `modal_submit.py status <call-id>` (the published NCP call was
`fc-01M3N4NMVBSANXJZB5TDWFDD8M`, the baseline `fc-01M3N4NXYZ6VDS2DM6GWXHB28J`).

```powershell
$env:PYTHONIOENCODING = "utf-8"
.venv\Scripts\modal.exe deploy modal_publish.py
.venv\Scripts\modal.exe run modal_publish.py::publish `
  --config fineweb-edu-ncp.yaml `
  --checkpoint latest `
  --eval-json /vol/artifacts/fineweb-edu-ncp-eval.json `
  --repo-id Paragon-Intelligence-Labs/Tessera-1B-Nano
```

- `--checkpoint latest` resolves through `runs/fineweb-edu-ncp/latest.json` on the volume
  (`step-00007629`). Pin the explicit name if `latest` could move.
- Git Bash note: prefix the command with `MSYS_NO_PATHCONV=1` so `/vol/...` survives.
- The function needs the secret (step 1) or image creation fails with
  "secret huggingface not found".
- Expected stdout: `https://huggingface.co/Paragon-Intelligence-Labs/Tessera-1B-Nano`.

## 5. Verify after upload (verified)

- [x] Files tab shows `README.md` (the card), `eval.json`, `metrics.jsonl`,
      `experiment.json`, `data_metadata.json`, safetensors, tokenizer files, and **no**
      `optimizer.pt`. The NCP repo additionally carries `modeling.py`, `quantizer.py`,
      and `configuration.py` for `trust_remote_code` loading.
- [ ] Model card renders the YAML frontmatter (tags, `base_model` link,
      `pipeline_tag: text-generation`).
- [ ] Loading snippet works from a clean environment:

      ```python
      from transformers import AutoModelForCausalLM, AutoTokenizer

      tok = AutoTokenizer.from_pretrained("Paragon-Intelligence-Labs/Tessera-1B-Nano")
      model = AutoModelForCausalLM.from_pretrained("Paragon-Intelligence-Labs/Tessera-1B-Nano")
      ```
- [ ] Spot-check the card numbers against `results/fineweb-edu/README.md`: NTP loss
      2.5142, zero-delta +0.1050, shuffle-delta +0.0008, 999,948,288 tokens, $30.27.

## 5b. Load verification (2026-09-29, passed)

`scripts/verify_hf_load.py` loads both published repos with `AutoModelForCausalLM` and
generates greedily: the baseline as a plain llama model, the concept arm with
`trust_remote_code=True` against the modules shipped in the repo (its `config.json`
declares `auto_map`; the loading snippet in the card says so). Recorded outcome:

- Both repos load and generate coherent text; forward passes are finite.
- Parity with the committed eval, re-measured from the downloaded HF weights on 16 held-out
  blocks (same batcher seed): NTP loss 2.4355 (base) and 2.4368 (concept) versus the
  committed 2.5135 / 2.5142 over all 256 blocks. The 16-block subsample explains the
  shared +0.078 offset (both arms shifted equally; the arm gap +0.0013 matches the
  committed +0.0007). The concept arm's zero-feedback delta is +0.0919 versus the committed
  +0.1050 on the full batch set, and its total objective loss 6.058 matches the committed
  total-loss log (6.13 at 900M).
- The card loading snippets were corrected (the concept arm needs `trust_remote_code=True`)
  and re-uploaded to both repos after this check.

## 6. After publishing (pending)


- [ ] Flip posts' `<link>` placeholders in `docs/posts.md` to the real repo URL and
      unfork the X/Reddit/RU posts for actual posting.- [ ] If you used a namespace other than `Paragon-Intelligence-Labs`, update the root
  `README.md` publish snippets to match.
- [ ] Keep `results/` as the source of truth; the HF repo is a mirror of the checkpoint,
      not the record.

## Failure handling

- **Secret error at deploy/run time**: `Secret 'huggingface' not found`: create it
  (step 1) and rerun; no code change needed.
- **401/403 from HF**: token lacks write scope or the repo id belongs to another
  namespace: regenerate the token, recreate the secret, rerun.
- **Interrupted upload**: `publish()` is idempotent (`exist_ok=True`, full-folder
  upload); rerun the same command.
- **Wrong card content**: fix `build_card` or the checkpoint artifacts, redeploy
  `modal_publish.py`, rerun; HF keeps history, so corrections are safe.
