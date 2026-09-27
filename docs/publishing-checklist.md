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
`eval.json`, `metrics.jsonl`, and provenance JSONs — a complete, self-contained repo.

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

- Default assumed in the docs: `USERNAME/ncp-smol-360m`. Replace `USERNAME` with the HF
  account that owns the token.
- Repo starts public by default; add `--private` for a first private upload and flip it
  in the HF UI later. `publish()` calls `create_repo(..., exist_ok=True)`, so a retry is
  safe and re-uploads over the same repo.

## 3. Review the card one last time

The uploaded card is generated remotely by `build_card` — the factual tables come from
the checkpoint's `trainer_state.json` and the eval JSON, not from the docs drafts. The
curated narrative in `docs/model-card.md` is *not* uploaded automatically.

- [ ] Optional: merge the curated sections (TL;DR, intervention reading, limitations) into
      the card. Two ways:
      - Edit `src/ncp_smol/publish.py::build_card` and redeploy (keeps remote generation
        reproducible); or
      - Upload first, then replace the README in the repo via the HF UI or
        `huggingface_hub.upload_file(..., path_in_repo="README.md")`.
- [ ] Replace `USERNAME` in the loading snippet either way (`build_card` renders the
      literal repo id passed to `publish`, so step 4 fixes this automatically if the
      template keeps `REPO_ID`).

## 4. Publish (remote, reads the volume)

```powershell
$env:PYTHONIOENCODING = "utf-8"
.venv\Scripts\modal.exe deploy modal_publish.py
.venv\Scripts\modal.exe run modal_publish.py::publish `
  --config fineweb-edu-ncp.yaml `
  --checkpoint latest `
  --eval-json /vol/artifacts/fineweb-edu-ncp-eval.json `
  --repo-id USERNAME/ncp-smol-360m
```

- `--checkpoint latest` resolves through `runs/fineweb-edu-ncp/latest.json` on the volume
  (`step-00007629`). Pin the explicit name if `latest` could move.
- Git Bash note: prefix the command with `MSYS_NO_PATHCONV=1` so `/vol/...` survives.
- The function needs the secret (step 1) or image creation fails with
  "secret huggingface not found".
- Expected stdout: `https://huggingface.co/USERNAME/ncp-smol-360m`.

## 5. Verify after upload

- [ ] Files tab shows `README.md` (the card), `eval.json`, `metrics.jsonl`,
      `experiment.json`, `data_metadata.json`, safetensors, tokenizer files — and **no**
      `optimizer.pt`.
- [ ] Model card renders the YAML frontmatter (tags, `base_model` link,
      `pipeline_tag: text-generation`).
- [ ] Loading snippet works from a clean environment:

      ```python
      from transformers import AutoModelForCausalLM, AutoTokenizer

      tok = AutoTokenizer.from_pretrained("USERNAME/ncp-smol-360m")
      model = AutoModelForCausalLM.from_pretrained("USERNAME/ncp-smol-360m")
      ```

      (`trust_remote_code` in the generated snippet is harmless but unnecessary — the
      checkpoint is plain transformers + safetensors.)
- [ ] Spot-check the card numbers against `results/fineweb-edu/README.md`: NTP loss
      2.5142, zero-delta +0.1050, shuffle-delta +0.0008, 999,948,288 tokens, $30.27.

## 6. After publishing

- [ ] Flip posts' `<link>` placeholders in `docs/posts.md` to the real repo URL and
      unfork the X/Reddit/RU posts for actual posting.
- [ ] Update the root `README.md` publish snippets if the repo id changed from
      `USERNAME/ncp-smol-360m`.
- [ ] Keep `results/` as the source of truth; the HF repo is a mirror of the checkpoint,
      not the record.

## Failure handling

- **Secret error at deploy/run time**: `Secret 'huggingface' not found` — create it
  (step 1) and rerun; no code change needed.
- **401/403 from HF**: token lacks write scope or the repo id belongs to another
  namespace — regenerate the token, recreate the secret, rerun.
- **Interrupted upload**: `publish()` is idempotent (`exist_ok=True`, full-folder
  upload); rerun the same command.
- **Wrong card content**: fix `build_card` or the checkpoint artifacts, redeploy
  `modal_publish.py`, rerun; HF keeps history, so corrections are safe.
