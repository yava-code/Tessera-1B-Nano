# Contributing

Keep changes close to the experiment: causal correctness, deterministic data order, matched
controls, and checkpoint reproducibility take priority over framework abstractions.

Before opening a pull request, run:

```powershell
.venv\Scripts\ruff.exe check .
.venv\Scripts\pytest.exe
```

Architecture changes should include a leakage test or a short explanation of why the
existing prefix-invariance test still covers them. Do not add reported results without the
config and raw eval artifact that produced them.
