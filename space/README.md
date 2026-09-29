---
title: Tessera-1B-Nano comparison
emoji: 🧩
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
license: apache-2.0
short_description: Side-by-side generation and a live concept-intervention readout
---

# Tessera-1B-Nano vs Tessera-1B-Nano-Base

Side-by-side demo for the token-matched comparison described in the
[Tessera-1B-Nano repository](https://github.com/yava-code/Tessera-1B-Nano). Three generations per run:
the NTP-only baseline, the concept arm with predicted feedback, and the concept arm with
its concept feedback zeroed at inference. A readout panel reports the teacher-forced NTP
loss of each model on the prompt and the zero-feedback penalty of the concept arm.

The checkpoints: [Tessera-1B-Nano](https://huggingface.co/yava-code/Tessera-1B-Nano) and
[Tessera-1B-Nano-Base](https://huggingface.co/yava-code/Tessera-1B-Nano-Base). The full
study, including the committed intervention numbers (zeroing costs +0.1050 nats on held-out
data, shuffling costs +0.0008), is in the repository whitepaper.
