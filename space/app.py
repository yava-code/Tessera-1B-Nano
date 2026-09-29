from __future__ import annotations

import gradio as gr
from comparison import (
    BASELINE_REPO,
    CONCEPT_REPO,
    DEFAULT_MAX_NEW_TOKENS,
    EXAMPLE_PROMPTS,
    INTRO_NOTE,
    MAX_NEW_TOKENS_LIMIT,
    run_comparison,
)

with gr.Blocks(title="Tessera-1B-Nano comparison") as demo:
    gr.Markdown(
        "# Tessera-1B-Nano vs Tessera-1B-Nano-Base\n\n" + INTRO_NOTE
    )
    prompt_box = gr.Textbox(
        label="Prompt",
        placeholder="Type a prompt, or pick an example below",
        lines=2,
    )
    max_new_tokens_slider = gr.Slider(
        minimum=16,
        maximum=MAX_NEW_TOKENS_LIMIT,
        value=DEFAULT_MAX_NEW_TOKENS,
        step=8,
        label="New tokens per generation",
    )
    run_button = gr.Button("Generate side by side", variant="primary")
    gr.Examples(examples=EXAMPLE_PROMPTS, inputs=[prompt_box])

    gr.Markdown("## Generations")
    base_name = BASELINE_REPO.split("/")[-1]
    concept_name = CONCEPT_REPO.split("/")[-1]
    with gr.Row():
        with gr.Column():
            base_link = f"[{base_name}](https://huggingface.co/{BASELINE_REPO})"
            gr.Markdown(f"**{base_link}** (NTP-only)")
            base_output = gr.Textbox(label="Baseline generation", lines=6)
            base_loss_note = gr.Markdown()
        with gr.Column():
            concept_link = f"[{concept_name}](https://huggingface.co/{CONCEPT_REPO})"
            gr.Markdown(f"**{concept_link}** (NTP + concepts)")
            concept_output = gr.Textbox(label="Concept arm, predicted feedback", lines=6)
            zero_output = gr.Textbox(label="Concept arm, zeroed feedback", lines=6)
            concept_loss_note = gr.Markdown()

    gr.Markdown(
        "### Reading the outputs\n\n"
        "- The two arms write differently because sampling seeds differ, not because one\n"
        "  is better: their held-out losses differ by +0.03%.\n"
        "- The zeroed-feedback column shows what the concept arm writes when its latent\n"
        "  channel is silenced at inference.\n"
        "- The loss readout is the honest signal: it is computed teacher-forced on your\n"
        "  prompt, so it shows the intervention penalty directly instead of relying on\n"
        "  sampled text to diverge.\n\n"
        f"Checkpoints: [{CONCEPT_REPO}](https://huggingface.co/{CONCEPT_REPO}),\n"
        f"[{BASELINE_REPO}](https://huggingface.co/{BASELINE_REPO}). Study and whitepaper:\n"
        "the [ncp-smol repository](https://github.com/yava-code/ncp-smol)."
    )

    run_button.click(
        run_comparison,
        inputs=[prompt_box, max_new_tokens_slider],
        outputs=[base_output, base_loss_note, concept_output, zero_output, concept_loss_note],
    )
    prompt_box.submit(
        run_comparison,
        inputs=[prompt_box, max_new_tokens_slider],
        outputs=[base_output, base_loss_note, concept_output, zero_output, concept_loss_note],
    )

if __name__ == "__main__":
    demo.launch()
