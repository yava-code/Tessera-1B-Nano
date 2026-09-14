# Quantized-target pilot

This run isolates the target discrepancy between the ConceptLM paper and its released Llama
path. It matches the continuous-target TinyStories configuration in model, data order,
optimizer, seed, and intended token budget; only `ncp_target` changes to `quantized`.

The pilot was stopped after 614,400 tokens and 150 steps. NCP loss started at
`2.80e-7` and remained between `2.79e-7` and `2.07e-6`, compared with approximately `1.33`
for the continuous target. The configured cost estimate at stop was $0.037.

This is a degenerate objective at the current initialization scale. The transformed codewords
are tightly clustered, so both a softmax-weighted code mixture and the selected codeword are
nearly identical before the concept predictor has learned anything. VQ loss remains around
`1.3`, but quantized-target NCP receives almost no useful error signal.

The result rules out a direct target swap as a fair ablation. A follow-up would first normalize
or variance-match transformed codewords, then verify that initial NCP gradient scale is
comparable to the continuous objective. No checkpoint was published from this intentionally
terminated run; `metrics.jsonl` contains the complete record.
