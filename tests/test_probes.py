import numpy as np
import pytest
import torch

from ncp_smol.data import TokenBatcher, TokenCorpus
from ncp_smol.foreign_corpus import FOREIGN_POOLS, prepare_foreign_pool
from ncp_smol.probes import cross_domain_delta, foreign_feedback
from tests.test_modeling import tiny_model


def test_predicted_mode_captures_feedback_pool() -> None:
    torch.manual_seed(5)
    model = tiny_model().eval()
    input_ids = torch.randint(3, 64, (2, 16))

    with torch.no_grad():
        model(input_ids=input_ids)

    last = model._hook_state["last_feedback"]
    assert last is not None
    assert last.shape == (2, 4, 32)
    assert last.dtype == torch.float32


def test_override_with_captured_pool_reproduces_predicted_logits() -> None:
    torch.manual_seed(6)
    model = tiny_model().eval()
    input_ids = torch.randint(3, 64, (2, 16))

    with torch.no_grad():
        model(input_ids=input_ids)
        pool = model._hook_state["last_feedback"].clone()
        predicted = model(input_ids=input_ids, concept_mode="predicted").logits
        model._hook_state["override_feedback"] = pool
        try:
            overridden = model(input_ids=input_ids, concept_mode="predicted").logits
        finally:
            model._hook_state["override_feedback"] = None

    torch.testing.assert_close(overridden, predicted)


def test_override_with_foreign_feedback_changes_logits_but_not_aux_losses() -> None:
    torch.manual_seed(7)
    model = tiny_model().eval()
    first = torch.randint(3, 64, (2, 16))
    second = torch.randint(3, 64, (2, 16))

    with torch.no_grad():
        model(input_ids=first)
        first_pool = model._hook_state["last_feedback"].clone()
        clean = model(input_ids=second, labels=second, concept_mode="predicted")
        model._hook_state["override_feedback"] = first_pool
        try:
            foreign = model(input_ids=second, labels=second, concept_mode="predicted")
        finally:
            model._hook_state["override_feedback"] = None

    assert not torch.equal(foreign.logits[:, 4:], clean.logits[:, 4:])
    torch.testing.assert_close(foreign.ncp_loss, clean.ncp_loss)
    torch.testing.assert_close(foreign.vq_loss, clean.vq_loss)


def test_override_rejects_mismatched_chunks_and_empty_pool() -> None:
    model = tiny_model()
    predicted = torch.randn(2, 4, 32)

    with pytest.raises(ValueError, match="chunks"):
        model._fit_override(predicted, torch.randn(4, 3, 32))
    with pytest.raises(ValueError, match="empty"):
        model._fit_override(predicted, torch.empty(0, 4, 32))


def test_cross_domain_delta_measures_reference_and_override(tmp_path) -> None:
    torch.manual_seed(9)
    model = tiny_model().eval()
    rng = np.random.default_rng(11)
    tokens = rng.integers(3, 64, size=(8 * 1024,), dtype=np.uint32)
    corpus_path = tmp_path / "pool.bin"
    corpus_path.write_bytes(tokens.tobytes())
    corpus = TokenCorpus(corpus_path, 1024)

    pool = foreign_feedback(
        model,
        TokenBatcher(corpus, 2, seed=3, repeat=True),
        batches=2,
        device=torch.device("cpu"),
    )
    result = cross_domain_delta(
        model,
        corpus,
        pool,
        batches=2,
        batch_size=2,
        seed=5,
        device=torch.device("cpu"),
    )

    assert set(result) == {"overridden_ntp_loss", "reference_ntp_loss", "delta"}
    assert result["reference_ntp_loss"] > 0.0
    assert torch.isfinite(torch.tensor(result["delta"]))
    repeat = cross_domain_delta(
        model,
        corpus,
        pool,
        batches=2,
        batch_size=2,
        seed=5,
        device=torch.device("cpu"),
    )
    assert result == repeat


def test_foreign_pool_registry_and_unknown_name(tmp_path) -> None:
    assert set(FOREIGN_POOLS) == {"wikipedia", "code"}
    with pytest.raises(ValueError, match="unknown foreign pool"):
        prepare_foreign_pool(
            "fictional", tokenizer_name="x", tokenizer_revision=None, output_dir=tmp_path
        )
