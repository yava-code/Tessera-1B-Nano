from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class QuantizerOutput:
    codes: Tensor
    indices: Tensor
    loss: Tensor


class CodebookTransform(nn.Module):
    def __init__(self, segments: int, dim: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, 2 * dim),
                    nn.ReLU(),
                    nn.Linear(2 * dim, dim),
                )
                for _ in range(segments)
            ]
        )

    def forward(self, codebook: Tensor) -> Tensor:
        return torch.stack(
            [layer(codes) for layer, codes in zip(self.layers, codebook, strict=True)]
        )


class ProductVectorQuantizer(nn.Module):
    def __init__(self, hidden_size: int, segments: int, codebook_size: int) -> None:
        super().__init__()
        if hidden_size % segments:
            raise ValueError("hidden_size must be divisible by segments")

        self.hidden_size = hidden_size
        self.segments = segments
        self.codebook_size = codebook_size
        self.segment_dim = hidden_size // segments
        self.register_buffer(
            "codebook",
            torch.empty(segments, codebook_size, self.segment_dim),
        )
        self.transform = CodebookTransform(segments, self.segment_dim)
        nn.init.normal_(self.codebook, std=0.02)

    def transformed_codes(self) -> Tensor:
        return self.transform(self.codebook)

    def forward(self, concepts: Tensor) -> QuantizerOutput:
        if concepts.shape[-1] != self.hidden_size:
            raise ValueError(f"expected hidden size {self.hidden_size}, got {concepts.shape[-1]}")

        shape = concepts.shape[:-1]
        x = concepts.reshape(*shape, self.segments, self.segment_dim)
        target = x.detach()
        codes = self.transformed_codes()
        distances = (
            target.square().sum(dim=-1, keepdim=True)
            + codes.square()
            .sum(dim=-1)
            .view(*([1] * len(shape)), self.segments, self.codebook_size)
            - 2 * torch.einsum("...sd,snd->...sn", target, codes)
        )
        indices = distances.argmin(dim=-1)
        gather_index = indices.unsqueeze(-1).expand(*indices.shape, self.segment_dim)
        expanded = codes.view(*([1] * len(shape)), *codes.shape).expand(*shape, *codes.shape)
        quantized = expanded.gather(-2, gather_index.unsqueeze(-2)).squeeze(-2)
        loss = F.mse_loss(quantized, target)
        return QuantizerOutput(quantized.reshape(*shape, self.hidden_size), indices, loss)

    def expected(self, logits: Tensor) -> Tensor:
        if logits.shape[-2:] != (self.segments, self.codebook_size):
            raise ValueError("logits do not match the product codebook")
        probs = logits.float().softmax(dim=-1).to(logits.dtype)
        predicted = torch.einsum("...sn,snd->...sd", probs, self.transformed_codes())
        return predicted.flatten(-2)

    @torch.no_grad()
    def usage(self, indices: Tensor) -> dict[str, float]:
        flat = indices.reshape(-1, self.segments)
        perplexities = []
        active = []
        for segment in range(self.segments):
            counts = torch.bincount(flat[:, segment], minlength=self.codebook_size).float()
            probs = counts / counts.sum().clamp_min(1)
            entropy = -(probs * probs.clamp_min(1e-12).log()).sum()
            perplexities.append(entropy.exp())
            active.append((counts > 0).float().mean())
        return {
            "codebook_perplexity": torch.stack(perplexities).mean().item(),
            "codebook_usage": torch.stack(active).mean().item(),
        }
