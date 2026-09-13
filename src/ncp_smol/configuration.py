from __future__ import annotations

from typing import Any

from transformers import LlamaConfig, PretrainedConfig


class NcpSmolConfig(PretrainedConfig):
    model_type = "ncp_smol"

    def __init__(
        self,
        backbone_config: dict[str, Any] | None = None,
        base_model_name_or_path: str = "HuggingFaceTB/SmolLM2-360M",
        base_model_revision: str | None = None,
        chunk_size: int = 4,
        segments: int | None = None,
        codebook_size: int = 64,
        concept_layers: int = 2,
        insert_layer: int = 1,
        ncp_target: str = "continuous",
        ncp_weight: float = 1.0,
        vq_weight: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        backbone_config = backbone_config or LlamaConfig().to_dict()
        hidden_size = int(backbone_config["hidden_size"])
        heads = int(backbone_config["num_attention_heads"])
        segments = heads if segments is None else segments

        if chunk_size < 2:
            raise ValueError("chunk_size must be at least 2")
        if hidden_size % segments:
            raise ValueError("hidden_size must be divisible by segments")
        if not 0 < insert_layer < int(backbone_config["num_hidden_layers"]):
            raise ValueError("insert_layer must split the token backbone")
        if concept_layers < 1:
            raise ValueError("concept_layers must be positive")
        if codebook_size < 2:
            raise ValueError("codebook_size must be at least 2")
        if ncp_target not in {"quantized", "continuous"}:
            raise ValueError("ncp_target must be 'quantized' or 'continuous'")

        self.backbone_config = backbone_config
        self.hidden_size = hidden_size
        self.vocab_size = int(backbone_config["vocab_size"])
        self.num_hidden_layers = int(backbone_config["num_hidden_layers"])
        self.num_attention_heads = heads
        self.num_key_value_heads = int(backbone_config["num_key_value_heads"])
        self.use_cache = False
        self.base_model_name_or_path = base_model_name_or_path
        self.base_model_revision = base_model_revision
        self.chunk_size = chunk_size
        self.segments = segments
        self.codebook_size = codebook_size
        self.concept_layers = concept_layers
        self.insert_layer = insert_layer
        self.ncp_target = ncp_target
        self.ncp_weight = ncp_weight
        self.vq_weight = vq_weight
        self.architectures = ["NcpSmolForCausalLM"]
        self.auto_map = {
            "AutoConfig": "configuration.NcpSmolConfig",
            "AutoModelForCausalLM": "modeling.NcpSmolForCausalLM",
        }


NcpSmolConfig.register_for_auto_class()
