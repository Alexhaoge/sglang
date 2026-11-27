# Copyright 2023-2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""NPU hooks for MiMo-V2.5/V2.6 that the model files do not provide.

The Ascend contribution guide asks new hardware support to arrive as new files
rather than as edits to existing code, so both hooks below are installed here
and keep ``srt/models`` untouched.

1. **Fused-qkv re-sharding.** The checkpoint ships
   ``attention_projection_layout: fused_qkv``: ``qkv_proj`` is interleaved over
   ``num_key_value_heads`` (4) TP shards. The stock loader can only *merge*
   shards, so an attention-TP width above the interleave is rejected -- that is
   exactly the shape a 16-die node needs, where ``attn_tp == tp == 16`` (no
   attention-DP) or ``attn_tp == 8`` (``--attn-dp-size 2``). The wrapper splits a
   shard's query heads and replicates its KV heads, which is the layout
   ``QKVParallelLinear.num_kv_head_replicas`` and
   ``ModelConfig.get_num_kv_heads()`` already describe ("with fewer heads than
   groups, each GPU keeps one").

2. **Audio-tokenizer device.** ``AudioEncoderMixin.build_audio_encoder`` builds
   ``torch.device(f"cuda:{torch.cuda.current_device()}")``, which every
   multimodal MiMo checkpoint reaches at construction time -- even for a
   text-only request, because the encoders are built whenever the config carries
   both ``vision_config`` and ``audio_config``. The wrapper loads the tokenizer on
   the process's real accelerator instead.
"""

import torch

# Rows of the whole checkpoint tensor for one fused-qkv layer.
def ckpt_fused_qkv_rows(num_q_heads, num_kv_heads, head_dim, v_head_dim, ckpt_tp):
    q_rows_per_shard = (num_q_heads // ckpt_tp) * head_dim
    kv_heads_per_shard = max(1, num_kv_heads // ckpt_tp)
    return ckpt_tp * (
        q_rows_per_shard
        + kv_heads_per_shard * head_dim
        + kv_heads_per_shard * v_head_dim
    )


def fused_qkv_rows_for_rank(
    *,
    num_q_heads,
    num_kv_heads,
    head_dim,
    v_head_dim,
    tp_rank,
    tp_size,
    ckpt_tp,
):
    """Rows this attention-TP rank takes when ``tp_size > ckpt_tp``.

    Query heads of the owning checkpoint shard are split further; that shard's
    KV heads are replicated onto every worker of the shard, so a worker's KV
    slice is one head.
    """
    if tp_size % ckpt_tp:
        raise ValueError(
            f"fused qkv replication needs attention tp_size ({tp_size}) to be a "
            f"multiple of the checkpoint interleave ({ckpt_tp})."
        )
    repetition = tp_size // ckpt_tp
    if num_q_heads % ckpt_tp or num_q_heads % tp_size:
        raise ValueError(
            f"query head count {num_q_heads} must divide by both the checkpoint "
            f"interleave ({ckpt_tp}) and the attention TP width ({tp_size})."
        )
    if num_kv_heads % ckpt_tp:
        raise ValueError(
            f"KV head count {num_kv_heads} must divide by the checkpoint "
            f"interleave ({ckpt_tp})."
        )
    if tp_size % num_kv_heads:
        raise ValueError(
            f"attention TP width {tp_size} must be a multiple of the KV head "
            f"count ({num_kv_heads}) so that KV heads can be replicated; got "
            f"remainder {tp_size % num_kv_heads}."
        )

    kv_heads_per_shard = num_kv_heads // ckpt_tp
    q_rows_per_shard = (num_q_heads // ckpt_tp) * head_dim
    q_rows_per_rank = q_rows_per_shard // repetition
    if q_rows_per_rank * repetition != q_rows_per_shard:
        raise ValueError(
            f"a checkpoint shard's {q_rows_per_shard} query rows cannot be split "
            f"across {repetition} workers."
        )
    k_rows_per_shard = kv_heads_per_shard * head_dim
    v_rows_per_shard = kv_heads_per_shard * v_head_dim
    shard_rows = q_rows_per_shard + k_rows_per_shard + v_rows_per_shard

    shard_index = tp_rank // repetition
    within_shard = tp_rank % repetition
    shard_base = shard_index * shard_rows

    q_start = shard_base + within_shard * q_rows_per_rank
    q_rows = torch.arange(q_start, q_start + q_rows_per_rank, dtype=torch.int64)

    kv_head = tp_rank // (tp_size // num_kv_heads)
    if kv_head // kv_heads_per_shard != shard_index:
        raise ValueError(
            f"attention TP rank {tp_rank} replicates KV head {kv_head} but takes "
            f"its query heads from checkpoint shard {shard_index}."
        )
    head_in_shard = kv_head % kv_heads_per_shard
    k_start = shard_base + q_rows_per_shard + head_in_shard * head_dim
    v_start = (
        shard_base + q_rows_per_shard + k_rows_per_shard + head_in_shard * v_head_dim
    )
    k_rows = torch.arange(k_start, k_start + head_dim, dtype=torch.int64)
    v_rows = torch.arange(v_start, v_start + v_head_dim, dtype=torch.int64)
    return torch.cat([q_rows, k_rows, v_rows])


def _replicating_qkv_loader(original):
    """Wrap the model's fused-qkv loader with the replication direction."""

    def load(
        name,
        param,
        loaded_weight,
        expected_fused_tp_size=None,
        deferred_scale_inv=None,
        *,
        qkv_proj,
    ):
        from sglang.srt.distributed.utils import get_group_rank_size
        from sglang.srt.lora.layers import unwrap_lora_layer
        from sglang.srt.model_loader.weight_utils import default_weight_loader

        module = unwrap_lora_layer(qkv_proj)
        tp_rank, tp_size = get_group_rank_size(module.tp_group)
        ckpt_tp = (
            expected_fused_tp_size if expected_fused_tp_size is not None else tp_size
        )
        if ckpt_tp % tp_size == 0:
            # Narrower than (or equal to) the interleave: the model's own
            # merging loader is the right one.
            return original(
                name,
                param,
                loaded_weight,
                expected_fused_tp_size,
                deferred_scale_inv,
                qkv_proj=qkv_proj,
            )
        if tp_size % ckpt_tp:
            raise ValueError(
                f"MiMoV2 fused qkv_proj checkpoint is TP={ckpt_tp}-interleaved, "
                f"which neither splits nor replicates over attention "
                f"tp_size={tp_size} while loading {name}."
            )
        if loaded_weight.dim() == 0:
            # Per-tensor companions are identical on every rank.
            default_weight_loader(param, loaded_weight)
            return
        if "weight_scale_inv" in name:
            raise ValueError(
                f"MiMoV2 fused qkv_proj {name}: a block-quantized scale_inv cannot "
                f"be re-sharded onto attention tp_size={tp_size} > checkpoint "
                f"interleave={ckpt_tp}, because its 128-row block grid is anchored "
                "to the checkpoint layout. Either launch with attention TP "
                f"{ckpt_tp} (--attn-dp-size {tp_size // ckpt_tp} at "
                f"--tp-size {tp_size}), or convert the checkpoint to per-channel "
                "quantization (e.g. modelslim), where the scale follows the same "
                "row indices as the weight."
            )

        rows = fused_qkv_rows_for_rank(
            num_q_heads=module.total_num_heads,
            num_kv_heads=module.total_num_kv_heads,
            head_dim=module.head_size,
            v_head_dim=module.v_head_size,
            tp_rank=tp_rank,
            tp_size=tp_size,
            ckpt_tp=ckpt_tp,
        )
        if rows.numel() != param.shape[0]:
            raise ValueError(
                f"MiMoV2 fused qkv_proj {name}: selecting {rows.numel()} rows but "
                f"the local parameter has {param.shape[0]}; attention "
                f"tp_size={tp_size} does not match this layer's head layout."
            )
        default_weight_loader(
            param, loaded_weight.index_select(0, rows.to(loaded_weight.device))
        )

    return load


def _install_qkv_replication():
    from sglang.srt.models import mimo_v2

    wrapped = _replicating_qkv_loader(mimo_v2.load_mimo_v2_qkv_proj_weight)
    mimo_v2.load_mimo_v2_qkv_proj_weight = wrapped
    # The MTP/draft module imported the loader by name, so it holds its own
    # binding and needs the wrapper too.
    from sglang.srt.models import mimo_v2_nextn

    mimo_v2_nextn.load_mimo_v2_qkv_proj_weight = wrapped


def _install_audio_tokenizer_device():
    import torch_npu

    from sglang.srt.models.mimo_audio import AudioEncoderMixin
    from sglang.srt.utils import get_device

    # The model file builds its device with
    # `torch.device(f"cuda:{torch.cuda.current_device()}")` *before* calling the
    # loader, so that read has to resolve on an NPU host too. torch_npu's
    # transfer_to_npu normally provides this alias; pin it explicitly so the
    # audio path does not depend on that import having happened.
    torch.cuda.current_device = torch_npu.npu.current_device

    original = AudioEncoderMixin._load_mimo_audio_tokenizer

    def load(path, device):
        # Load the tokenizer on whatever accelerator this process runs on.
        return original(path, torch.device(get_device()))

    AudioEncoderMixin._load_mimo_audio_tokenizer = staticmethod(load)


def _is_mimo_v2(model_config) -> bool:
    from sglang.srt.configs.model_config import MIMO_V2_MODEL_ARCHS

    architectures = model_config.hf_config.architectures or ()
    return any(arch in MIMO_V2_MODEL_ARCHS for arch in architectures)


def install_mimo_v2_npu_support():
    """Install both hooks at the first model instantiation.

    ``init_npu_backend`` runs while ``model_runner`` is still being imported, and
    several attention modules import ``model_runner`` at module scope, so
    importing the MiMo model files there risks a partially-initialized import.
    The loader's ``_initialize_model`` runs once the import graph is settled and
    before any weight is read, so the hooks are applied from there instead -- and
    only for a MiMo checkpoint.
    """
    from sglang.srt.model_loader import loader as model_loader

    original = model_loader._initialize_model

    def initialize_model(model_config, *args, **kwargs):
        model_loader._initialize_model = original  # one-shot
        if _is_mimo_v2(model_config):
            _install_qkv_replication()
            _install_audio_tokenizer_device()
        return original(model_config, *args, **kwargs)

    model_loader._initialize_model = initialize_model
