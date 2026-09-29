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


def test_default_normalization_is_backward_compatible() -> None:
    quantizer = ProductVectorQuantizer(hidden_size=24, segments=3, codebook_size=8)
    assert quantizer.normalization == "none"
    assert quantizer.transform.normalization == "none"
    with torch.no_grad():
        codes = quantizer.transformed_codes()
    assert codes.std() < 0.5  # unnormalized initial cloud stays tight


def test_variance_normalization_rescales_codeword_cloud() -> None:
    quantizer = ProductVectorQuantizer(
        hidden_size=24, segments=3, codebook_size=8, normalization="variance"
    )
    assert quantizer.normalization == "variance"
    with torch.no_grad():
        codes = quantizer.transformed_codes()
    stds = codes.float().std(dim=1)
    torch.testing.assert_close(stds, torch.ones_like(stds), atol=0.05, rtol=0.05)


def test_variance_normalization_restores_initial_quantized_target_error() -> None:
    """Reproduce the preregistered H5 failure and its fix.

    With quantized targets both sides of the NCP loss live in the transformed-codeword
    cloud. At the default init that cloud is tiny (std ~5e-4 on the real pilot
    geometry), so the selected-code error starts at ~3e-7 (measured 2.8e-7 in the
    aborted pilot, 3.1e-7 in replication) and the arm receives no gradient signal.
    Variance matching rescales the cloud to order 1 (measured NCP loss 1.23, matching
    the continuous target's ~1.33), making the target prediction meaningful.
    """
    torch.manual_seed(17)
    plain = ProductVectorQuantizer(hidden_size=24, segments=3, codebook_size=8)
    normalized = ProductVectorQuantizer(
        hidden_size=24, segments=3, codebook_size=8, normalization="variance"
    )
    with torch.no_grad():
        plain_codes = plain.transformed_codes()
        normalized_codes = normalized.transformed_codes()
    # Codeword spread across the codebook dimension: the scale the selected-code NCP
    # error lives at (measured 5e-4 on the real 135M pilot geometry, exactly 1.0
    # after variance matching).
    plain_spread = plain_codes.float().std(dim=1).mean()
    normalized_spread = normalized_codes.float().std(dim=1).mean()
    assert plain_spread < 0.5, (
        f"expected the degenerate tight target cloud, got spread {plain_spread:.2e}"
    )
    assert 0.9 < normalized_spread < 1.1, (
        f"expected unit codeword spread, got {normalized_spread:.2e}"
    )
