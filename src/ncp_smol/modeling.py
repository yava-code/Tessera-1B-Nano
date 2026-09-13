from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, GenerationMixin, LlamaConfig, PreTrainedModel
from transformers.modeling_outputs import ModelOutput
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaRotaryEmbedding,
)

from .configuration import NcpSmolConfig
from .quantizer import ProductVectorQuantizer

ConceptMode = Literal["predicted", "zero", "shuffle"]


@dataclass
class NcpCausalLMOutput(ModelOutput):
    loss: Tensor | None = None
    logits: Tensor | None = None
    ntp_loss: Tensor | None = None
    ncp_loss: Tensor | None = None
    vq_loss: Tensor | None = None
    code_indices: Tensor | None = None


class NcpSmolForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = NcpSmolConfig
    base_model_prefix = "backbone"
    main_input_name = "input_ids"
    _supports_cache_class = False
    _no_split_modules = ["LlamaDecoderLayer"]
    _tied_weights_keys = ["backbone.lm_head.weight"]

    def __init__(self, config: NcpSmolConfig) -> None:
        super().__init__(config)
        backbone_config = LlamaConfig.from_dict(config.backbone_config)
        self.backbone = LlamaForCausalLM(backbone_config)

        concept_config = LlamaConfig.from_dict(config.backbone_config)
        concept_config._attn_implementation = "eager"
        self.concept_layers = nn.ModuleList(
            [LlamaDecoderLayer(concept_config, i) for i in range(config.concept_layers)]
        )
        self.concept_rotary = LlamaRotaryEmbedding(concept_config)
        self.quantizer = ProductVectorQuantizer(
            hidden_size=config.hidden_size,
            segments=config.segments,
            codebook_size=config.codebook_size,
        )
        self.concept_head = nn.Linear(
            config.hidden_size,
            config.segments * config.codebook_size,
        )
        self._hook_state: dict[str, Any] = {}
        self._hook = self.backbone.model.layers[config.insert_layer].register_forward_pre_hook(
            self._inject_concepts,
            with_kwargs=True,
        )
        self.post_init()
        self.init_concept_from_backbone()
        self.tie_weights()

    @classmethod
    def from_backbone(
        cls,
        model_name_or_path: str,
        *,
        revision: str | None = None,
        chunk_size: int = 4,
        segments: int | None = None,
        codebook_size: int = 64,
        concept_layers: int = 2,
        insert_layer: int = 1,
        ncp_target: str = "continuous",
        ncp_weight: float = 1.0,
        vq_weight: float = 1.0,
        **load_kwargs: Any,
    ) -> NcpSmolForCausalLM:
        base = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            revision=revision,
            **load_kwargs,
        )
        if not isinstance(base, LlamaForCausalLM):
            raise TypeError("ncp-smol currently requires a Llama-compatible backbone")
        config = NcpSmolConfig(
            backbone_config=base.config.to_dict(),
            base_model_name_or_path=model_name_or_path,
            base_model_revision=revision,
            chunk_size=chunk_size,
            segments=segments,
            codebook_size=codebook_size,
            concept_layers=concept_layers,
            insert_layer=insert_layer,
            ncp_target=ncp_target,
            ncp_weight=ncp_weight,
            vq_weight=vq_weight,
        )
        model = cls(config)
        model.backbone.load_state_dict(base.state_dict())
        model.init_concept_from_backbone()
        model.tie_weights()
        return model

    def init_concept_from_backbone(self) -> None:
        source = self.backbone.model.layers
        for index, layer in enumerate(self.concept_layers):
            layer.load_state_dict(source[index % len(source)].state_dict())

    def get_input_embeddings(self) -> nn.Module:
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.backbone.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        return self.backbone.get_output_embeddings()

    def set_output_embeddings(self, value: nn.Module) -> None:
        self.backbone.set_output_embeddings(value)

    def tie_weights(self) -> None:
        self.backbone.tie_weights()

    def _concept_mask(self, valid: Tensor, dtype: torch.dtype) -> Tensor:
        batch, length = valid.shape
        mask = torch.full(
            (length, length),
            torch.finfo(dtype).min,
            dtype=dtype,
            device=valid.device,
        ).triu(diagonal=1)
        mask = mask.view(1, 1, length, length).expand(batch, 1, length, length).clone()
        return mask.masked_fill(~valid[:, None, None, :], torch.finfo(dtype).min)

    def _concept_path(
        self,
        hidden: Tensor,
        attention_mask: Tensor | None,
        mode: ConceptMode,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        batch, length, width = hidden.shape
        chunks = length // self.config.chunk_size
        zero = hidden.sum() * 0
        if chunks == 0:
            return torch.zeros_like(hidden), {
                "ncp_loss": zero,
                "vq_loss": zero,
                "code_indices": torch.empty(
                    batch, 0, self.config.segments, dtype=torch.long, device=hidden.device
                ),
            }

        usable = chunks * self.config.chunk_size
        concepts = (
            hidden[:, :usable]
            .reshape(
                batch,
                chunks,
                self.config.chunk_size,
                width,
            )
            .mean(dim=2)
        )
        if attention_mask is None:
            valid = torch.ones(batch, chunks, dtype=torch.bool, device=hidden.device)
        else:
            valid = (
                attention_mask[:, :usable]
                .bool()
                .reshape(batch, chunks, self.config.chunk_size)
                .all(dim=2)
            )

        quantized = self.quantizer(concepts)
        positions = torch.arange(chunks, device=hidden.device)
        position_ids = (positions * self.config.chunk_size).unsqueeze(0).expand(batch, chunks)
        concept_hidden = concepts
        rotary = self.concept_rotary(concept_hidden, position_ids)
        causal_mask = self._concept_mask(valid, concept_hidden.dtype)
        for layer in self.concept_layers:
            concept_hidden = layer(
                concept_hidden,
                attention_mask=causal_mask,
                position_ids=position_ids,
                cache_position=positions,
                position_embeddings=rotary,
                use_cache=False,
            )

        logits = self.concept_head(concept_hidden).view(
            batch,
            chunks,
            self.config.segments,
            self.config.codebook_size,
        )
        predicted = self.quantizer.expected(logits)

        if chunks > 1 and valid[:, 1:].any():
            target_mask = valid[:, 1:].unsqueeze(-1)
            target = quantized.codes if self.config.ncp_target == "quantized" else concepts
            squared = (predicted[:, :-1] - target[:, 1:].detach()).square()
            ncp_loss = (squared * target_mask).sum() / (target_mask.sum().clamp_min(1) * width)
        else:
            ncp_loss = zero

        if mode == "zero":
            predicted = torch.zeros_like(predicted)
        elif mode == "shuffle":
            predicted = predicted.roll(shifts=1, dims=0)

        feedback = self._align_feedback(predicted.detach(), length)
        return feedback, {
            "ncp_loss": ncp_loss,
            "vq_loss": quantized.loss,
            "code_indices": quantized.indices,
        }

    def _align_feedback(self, predicted: Tensor, length: int) -> Tensor:
        batch, _, width = predicted.shape
        prefix = torch.zeros(batch, 1, width, dtype=predicted.dtype, device=predicted.device)
        repeated = torch.cat((prefix, predicted), dim=1).repeat_interleave(
            self.config.chunk_size,
            dim=1,
        )
        return repeated[:, 1 : length + 1]

    def _inject_concepts(
        self,
        _module: nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        if not self._hook_state.get("active", False):
            return args, kwargs

        hidden = args[0] if args else kwargs["hidden_states"]
        feedback, aux = self._concept_path(
            hidden,
            self._hook_state.get("attention_mask"),
            self._hook_state["mode"],
        )
        self._hook_state["aux"] = aux
        if args:
            args = (hidden + feedback, *args[1:])
        else:
            kwargs["hidden_states"] = hidden + feedback
        return args, kwargs

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
        concept_mode: ConceptMode = "predicted",
        use_cache: bool | None = None,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> NcpCausalLMOutput | tuple[Tensor, ...]:
        if concept_mode not in {"predicted", "zero", "shuffle"}:
            raise ValueError(f"unknown concept_mode: {concept_mode}")

        self._hook_state = {
            "active": True,
            "attention_mask": attention_mask,
            "mode": concept_mode,
            "aux": None,
        }
        try:
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                labels=None,
                use_cache=False,
                return_dict=True,
                **kwargs,
            )
            aux = self._hook_state["aux"]
        finally:
            self._hook_state["active"] = False
        if aux is None:
            raise RuntimeError("concept injection hook did not run")

        ntp_loss = None
        total_loss = None
        if labels is not None:
            shift_logits = outputs.logits[:, :-1].float().contiguous()
            shift_labels = labels[:, 1:].contiguous()
            ntp_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            total_loss = (
                ntp_loss
                + self.config.ncp_weight * aux["ncp_loss"]
                + self.config.vq_weight * aux["vq_loss"]
            )

        result = NcpCausalLMOutput(
            loss=total_loss,
            logits=outputs.logits,
            ntp_loss=ntp_loss,
            ncp_loss=aux["ncp_loss"],
            vq_loss=aux["vq_loss"],
            code_indices=aux["code_indices"],
        )
        if return_dict:
            return result
        return tuple(value for value in result.values() if value is not None)

    def prepare_inputs_for_generation(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": False,
        }


NcpSmolForCausalLM.register_for_auto_class("AutoModelForCausalLM")
