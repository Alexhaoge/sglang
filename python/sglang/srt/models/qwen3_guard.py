# Adapted from qwen3.py
"""Qwen3Guard-Stream: a Qwen3 backbone plus four classification heads.

The model does not generate text.  One forward pass over the prompt emits, for
every token, a risk level and a content category -- one pair for the assistant
response and one pair for the user query.  There is no decode loop: a request
is a single prefill and its answer is the head logits (``StreamGuardResult``).

Multi-turn streaming is a *session* concern, not a model concern.  The caller
opens a streaming session (``Engine.open_session(streaming=True)``) and sends
each turn as a request in that session; the session keeps the KV of the
previous turns, so a turn only prefills the tokens appended since the last one
and the returned logits cover exactly those tokens.
"""

import logging
from typing import Iterable, Optional, Tuple

import torch
from torch import nn

from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.utils import get_layer_id
from sglang.srt.managers.utils import StreamGuardResult
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from sglang.srt.models.qwen3 import Qwen3Model
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)


class Qwen3ForGuardModel(nn.Module):
    # BitandBytes specific attributes
    default_bitsandbytes_target_modules = [
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]
    bitsandbytes_stacked_params_mapping = {
        # shard_name, weight_name, index
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_parallel().pp_group
        self.config = config
        self.quant_config = quant_config
        self.model = Qwen3Model(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )

        # Two independent head pairs: (risk level, category) for the response
        # and the same for the query.  Each pair shares one pre-projection and
        # one norm, then splits into its two classification heads.
        inner = config.guard_inner_size
        self.risk_level_category_pre = nn.Linear(config.hidden_size, inner, bias=False)
        self.risk_level_category_layernorm = RMSNorm(inner, eps=config.rms_norm_eps)
        self.risk_level_head = nn.Linear(inner, config.num_risk_level, bias=False)
        self.category_head = nn.Linear(inner, config.num_category, bias=False)

        self.query_risk_level_category_pre = nn.Linear(
            config.hidden_size, inner, bias=False
        )
        self.query_risk_level_category_layernorm = RMSNorm(
            inner, eps=config.rms_norm_eps
        )
        self.query_risk_level_head = nn.Linear(
            inner, config.num_query_risk_level, bias=False
        )
        self.query_category_head = nn.Linear(
            inner, config.num_query_category, bias=False
        )

        # For EAGLE3 support, mirroring the causal-LM wrapper.
        self.capture_aux_hidden_states = False

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> StreamGuardResult:
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        if self.capture_aux_hidden_states:
            hidden_states, _ = hidden_states

        if not self.pp_group.is_last_rank:
            return hidden_states

        # Every token of this extend needs a label -- the caller asked for
        # exactly these tokens and the prefix match was capped to leave them
        # out of the cache -- so the heads run over the full hidden state, not
        # over a logits_to_keep tail.
        x = self.risk_level_category_layernorm(
            self.risk_level_category_pre(hidden_states)
        )
        query_x = self.query_risk_level_category_layernorm(
            self.query_risk_level_category_pre(hidden_states)
        )
        return StreamGuardResult(
            risk_level_logits=self.risk_level_head(x),
            category_logits=self.category_head(x),
            query_risk_level_logits=self.query_risk_level_head(query_x),
            query_category_logits=self.query_category_head(query_x),
            hidden_states=hidden_states,
            bid=-1,
        )

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # The guard heads sit at the top level of the checkpoint, outside
        # "model.", and load through the plain default loader.
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "Embedding" in self.config.name_or_path:
                name = add_prefix(name, "model")
            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self.model, "start_layer")
                and (
                    layer_id < self.model.start_layer
                    or layer_id >= self.model.end_layer
                )
            ):
                continue

            if "rotary_emb.inv_freq" in name or "projector" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                # A guard model has no lm_head; a checkpoint that ships one is
                # tying it to the embedding, which the backbone already loaded.
                continue
            if name.startswith("model.vision_tower") and name not in params_dict:
                continue
            if "scale" in name:
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if name in params_dict.keys():
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                else:
                    logger.warning(f"Parameter {name} not found in params_dict")

    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        self.model.load_kv_cache_scales(quantization_param_path)


EntryClass = Qwen3ForGuardModel
