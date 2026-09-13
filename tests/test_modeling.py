from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, LlamaConfig

from ncp_smol.configuration import NcpSmolConfig
from ncp_smol.modeling import NcpSmolForCausalLM


def tiny_model() -> NcpSmolForCausalLM:
    backbone = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        attention_dropout=0.0,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    config = NcpSmolConfig(
        backbone_config=backbone.to_dict(),
        chunk_size=4,
        segments=4,
        codebook_size=8,
        concept_layers=1,
        insert_layer=1,
    )
    return NcpSmolForCausalLM(config)


def test_forward_exposes_all_objectives() -> None:
    model = tiny_model()
    input_ids = torch.randint(3, 64, (2, 16))

    output = model(input_ids=input_ids, labels=input_ids)
    output.loss.backward()

    assert output.logits.shape == (2, 16, 64)
    assert output.code_indices.shape == (2, 4, 4)
    assert all(
        torch.isfinite(value)
        for value in (output.loss, output.ntp_loss, output.ncp_loss, output.vq_loss)
    )
    assert model.concept_head.weight.grad is not None


def test_future_tokens_do_not_change_prefix_logits() -> None:
    torch.manual_seed(7)
    model = tiny_model().eval()
    first = torch.randint(3, 64, (1, 16))
    second = first.clone()
    second[:, 10:] = torch.randint(3, 64, (1, 6))

    with torch.no_grad():
        first_logits = model(input_ids=first).logits
        second_logits = model(input_ids=second).logits

    torch.testing.assert_close(first_logits[:, :10], second_logits[:, :10], atol=1e-6, rtol=1e-5)


def test_interventions_change_the_concept_conditioning() -> None:
    torch.manual_seed(11)
    model = tiny_model().eval()
    input_ids = torch.randint(3, 64, (1, 16))

    with torch.no_grad():
        predicted = model(input_ids=input_ids, concept_mode="predicted").logits
        zero = model(input_ids=input_ids, concept_mode="zero").logits
        shuffled = model(input_ids=input_ids, concept_mode="shuffle").logits

    assert not torch.equal(predicted[:, 3:], zero[:, 3:])
    assert not torch.equal(predicted[:, 7:], shuffled[:, 7:])


def test_feedback_is_shifted_by_k_minus_one_positions() -> None:
    model = tiny_model()
    predicted = torch.tensor([[[1.0], [2.0], [3.0]]]).expand(1, 3, 32)

    feedback = model._align_feedback(predicted, length=12)

    expected = torch.tensor([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3], dtype=torch.float32)
    torch.testing.assert_close(feedback[0, :, 0], expected)


def test_generation_remains_next_token_autoregressive() -> None:
    model = tiny_model().eval()
    prompt = torch.tensor([[1, 7, 9, 11]])

    output = model.generate(prompt, max_new_tokens=3, do_sample=False)

    assert output.shape == (1, 7)


def test_save_load_roundtrip(tmp_path: Path) -> None:
    model = tiny_model().eval()
    input_ids = torch.randint(3, 64, (1, 12))
    with torch.no_grad():
        expected = model(input_ids=input_ids).logits

    model.save_pretrained(tmp_path, safe_serialization=True)
    loaded = NcpSmolForCausalLM.from_pretrained(tmp_path).eval()

    with torch.no_grad():
        actual = loaded(input_ids=input_ids).logits
    torch.testing.assert_close(actual, expected)


def test_hugging_face_auto_class_roundtrip(tmp_path: Path) -> None:
    model = tiny_model().eval()
    model.save_pretrained(tmp_path, safe_serialization=True)

    loaded = AutoModelForCausalLM.from_pretrained(tmp_path, trust_remote_code=True)

    assert loaded.__class__.__name__ == "NcpSmolForCausalLM"
