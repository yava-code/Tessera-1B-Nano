from __future__ import annotations

import comparison
import gradio as gr
from comparison import (
    BASELINE_REPO,
    CONCEPT_REPO,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_SEED,
    EXAMPLE_PROMPTS,
    INTRO_NOTE,
    MAX_NEW_TOKENS_LIMIT,
    run_comparison,
)

# Load both checkpoints at startup, outside any ZeroGPU window: the GPU worker forks
# from this process per request and must always see fully loaded models.
comparison.load_models(gpu_dtype=True)


def handler(prompt: str, max_new_tokens: int, seed: int) -> tuple[str, str, str, str, str]:
    try:
        return run_comparison(prompt, max_new_tokens, seed)
    except Exception as error:  # surface the real failure inside the UI
        import traceback

        message = f"{type(error).__name__}: {error}\n{traceback.format_exc()[-800:]}"
        return "", f"ERROR: {message}", "", "", message


with gr.Blocks(title="Tessera-1B-Nano comparison") as demo:
    gr.Markdown(
        "# We taught a small LLM to predict its next thought\n\n" + INTRO_NOTE
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
    seed_number = gr.Number(
        value=DEFAULT_SEED,
        label="Seed (same seed = same generations)",
        precision=0,
        minimum=0,
        maximum=2**31 - 1,
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
        "## Reading the outputs\n\n"
        "- **Left vs middle:** two models that cost the same and trained the same. Watch"
        "  how differently they write.\n"
        "- **Middle vs right:** the concept model with its concept feedback silenced. If"
        "  the concept channel carries anything the model uses, these two should diverge."
        "  They do.\n"
        "- The teacher-forced loss line is the honest signal: it is computed on your"
        "  prompt, so it shows the intervention penalty directly instead of relying on"
        "  sampled text to diverge.\n\n"
        f"Checkpoints: [{CONCEPT_REPO}](https://huggingface.co/{CONCEPT_REPO}) and\n"
        f"[{BASELINE_REPO}](https://huggingface.co/{BASELINE_REPO}). The whole family (the\n"
        "135M gate, both 1B arms) lives in [this\n"
        "collection](https://huggingface.co/collections/yava-code/tessera-next-concept-prediction-gated-and-scaled-6abc3b7ad9177be38bcb3ff4).\n"
        "Study and whitepaper: the\n"
        "[Tessera-1B-Nano repository](https://github.com/yava-code/Tessera-1B-Nano)."
    )

    with gr.Accordion("Under the hood: the numbers", open=False):
        gr.Markdown(
            "**Setup.** Both arms are the `HuggingFaceTB/SmolLM2-360M` backbone continued"
            " on the same packed FineWeb-Edu corpus: 999,948,288 tokens each, same order,"
            " same seed, same schedule. The only training difference is the concept path"
            " and its auxiliary losses.\n\n"
            "**Headline intervention numbers (held-out, 32 batches):**\n\n"
            "- zeroing concept feedback costs **+0.1050 nats** of NTP loss - the decoder"
            " uses the channel;\n"
            "- feeding feedback predicted from a different sentence in the batch costs"
            " **+0.0008** - the use is sequence-generic, not sequence-specific;\n"
            "- topic-similar foreign feedback costs **+0.0008**, feedback predicted from"
            " Wikipedia **+0.0016**, from source code **+0.0011** - the channel does not"
            " even carry topic identity;\n"
            "- at sequence length 4096 (4x context) the same pattern holds: gap to the"
            " baseline **+0.0006**, zero-cost **+0.1373**, shuffled **+0.0011**;\n"
            "- codebook: 7.55 effective perplexity, 85.6% usage (135M gate: 2.41 / 33%).\n\n"
            "The token-loss bottom line at 1B tokens is **neutral** (2.5135 vs 2.5142)."
            " What the concept path buys at larger budgets is the open question the"
            " study is built to answer.\n\n"
            "Full records and the preregistered protocol: "
            "[whitepaper](https://github.com/yava-code/Tessera-1B-Nano/blob/main/docs/whitepaper.md)"
            " and [results](https://github.com/yava-code/Tessera-1B-Nano/tree/main/results)."
        )

    run_button.click(
        handler,
        inputs=[prompt_box, max_new_tokens_slider, seed_number],
        outputs=[base_output, base_loss_note, concept_output, zero_output, concept_loss_note],
    )
    prompt_box.submit(
        handler,
        inputs=[prompt_box, max_new_tokens_slider, seed_number],
        outputs=[base_output, base_loss_note, concept_output, zero_output, concept_loss_note],
    )

if __name__ == "__main__":
    demo.launch()
