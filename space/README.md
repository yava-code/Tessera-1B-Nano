---
title: Tessera-1B-Nano comparison
emoji: 🧩
colorFrom: gray
colorTo: blue
sdk: gradio
app_file: app.py
suggested_hardware: zero-a10g
license: apache-2.0
short_description: Side-by-side generation with a live concept readout
---

# Tessera-1B-Nano vs Tessera-1B-Nano-Base

Side-by-side demo for the token-matched comparison described in the
[Tessera-1B-Nano repository](https://github.com/yava-code/Tessera-1B-Nano). Three generations per run:
the NTP-only baseline, the concept arm with predicted feedback, andthe concept arm with its concept feedback zeroed at inference. A readout panel reports the teacher-forced NTP
loss of each model on the prompt and the zero-feedback penalty of the concept arm.

Runs on ZeroGPU: the checkpoints load on CPU at startup and each request moves compute to
a shared A100 through the `spaces.GPU` decorator.

The checkpoints: [Tessera-1B-Nano](https://huggingface.co/yava-code/Tessera-1B-Nano) and
[Tessera-1B-Nano-Base](https://huggingface.co/yava-code/Tessera-1B-Nano-Base). The full
study, including the committed intervention numbers (zeroing costs +0.1050 nats on held-out
data, shuffling costs +0.0008), is in the repository whitepaper.
