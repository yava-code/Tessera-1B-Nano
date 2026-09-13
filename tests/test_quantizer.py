import torch

from ncp_smol.quantizer import ProductVectorQuantizer


def test_product_quantizer_shapes_and_detached_target() -> None:
    quantizer = ProductVectorQuantizer(hidden_size=24, segments=3, codebook_size=8)
    concepts = torch.randn(2, 5, 24, requires_grad=True)

    output = quantizer(concepts)

    assert output.codes.shape == concepts.shape
    assert output.indices.shape == (2, 5, 3)
    output.loss.backward()
    assert concepts.grad is None
    assert quantizer.codebook.grad is None
    assert quantizer.transform.layers[0][0].weight.grad is not None


def test_expected_code_is_differentiable() -> None:
    quantizer = ProductVectorQuantizer(hidden_size=24, segments=3, codebook_size=8)
    logits = torch.randn(2, 5, 3, 8, requires_grad=True)

    expected = quantizer.expected(logits)
    expected.square().mean().backward()

    assert expected.shape == (2, 5, 24)
    assert logits.grad is not None
