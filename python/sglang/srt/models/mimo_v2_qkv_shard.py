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
"""Row layout of MiMo-V2's TP-interleaved fused ``qkv_proj`` checkpoint.

MiMo-V2.5/V2.6 ship ``attention_projection_layout: fused_qkv``: ``qkv_proj`` is
interleaved over ``num_key_value_heads`` (4) TP shards and each shard holds
``[q heads of the shard | its k heads | its v heads]``.

A worker whose attention-TP width differs from that interleave has to re-cut the
tensor, and the two directions are not symmetric:

* **narrower** (``attn_tp <= ckpt_tp``) - a worker merges several checkpoint
  shards, i.e. every worker holds *fewer* heads than a shard contains;
* **wider** (``attn_tp > ckpt_tp``) - a worker splits one shard's *query* heads
  further while that shard's *KV* heads are **replicated**, because there are
  fewer KV heads than workers. This is the layout
  :meth:`ModelConfig.get_num_kv_heads` describes as "with fewer heads than
  groups, each GPU keeps one", and the one ``MiMoV2Attention`` asserts when it
  picks ``num_kv_heads = max(1, total_num_kv_heads // attn_tp_size)``.

Only the wider direction is computed here (it is the one that needs no
knowledge of the quantizer). The functions return row indices into the
checkpoint tensor, so callers can apply them to the weight, and to any
per-output-channel companion tensor (``weight_scale``), with the same layout.

Kept free of model imports so the CPU test suite can pin the arithmetic without
torch_npu or a checkpoint.
"""

import re

import torch

# ``layers.<id>.`` appears in every fused-qkv parameter name.
_LAYER_ID_RE = re.compile(r"layers\.(\d+)\.")


def get_ckpt_qkv_geometry(config, name, *, is_swa=None):
    """``(num_q_heads, num_kv_heads, head_dim, v_head_dim)`` for ``name``'s layer.

    SWA layers use the ``swa_*`` fields because the checkpoint interleaves those
    with their own head counts. ``is_swa`` overrides the classification for
    callers whose parameter names carry no ``layers.<id>.`` (the MTP layer is
    always windowed). Returns ``None`` when the layer cannot be identified.
    """
    if is_swa is None:
        match = _LAYER_ID_RE.search(name)
        if match is None:
            return None
        layer_id = int(match.group(1))
        pattern = getattr(config, "hybrid_layer_pattern", None)
        is_swa = (
            pattern is not None
            and 0 <= layer_id < len(pattern)
            and pattern[layer_id] == 1
        )
    if is_swa:
        return (
            config.swa_num_attention_heads,
            config.swa_num_key_value_heads,
            config.swa_head_dim,
            getattr(config, "swa_v_head_dim", config.swa_head_dim),
        )
    return (
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
        getattr(config, "v_head_dim", config.head_dim),
    )


def replicated_fused_qkv_row_indices(
    *,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    v_head_dim: int,
    tp_rank: int,
    tp_size: int,
    ckpt_tp: int,
) -> torch.Tensor:
    """Rows of the checkpoint tensor that belong to ``tp_rank`` when it replicates.

    Args:
        num_q_heads: query heads in the layer (all ranks).
        num_kv_heads: KV heads in the layer (all ranks).
        head_dim: K (and Q) head width.
        v_head_dim: V head width; MiMo is asymmetric (192 vs 128).
        tp_rank: this worker's attention-TP rank.
        tp_size: attention-TP width (``tp // (attn_dp * attn_cp)``).
        ckpt_tp: the checkpoint's interleave, i.e. ``num_key_value_heads`` for a
            ``fused_qkv`` checkpoint.

    Returns:
        A 1-D ``int64`` tensor of length
        ``(num_q_heads // tp_size) * head_dim + head_dim + v_head_dim``.
    """
    if ckpt_tp <= 0 or tp_size % ckpt_tp:
        raise ValueError(
            f"fused qkv replication needs attention tp_size ({tp_size}) to be a "
            f"multiple of the checkpoint interleave ({ckpt_tp}); use the sharding "
            "path for the other direction."
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
            f"attention TP width {tp_size} must be a multiple of the KV head count "
            f"({num_kv_heads}) for KV replication; got remainder "
            f"{tp_size % num_kv_heads}."
        )
    kv_heads_per_shard = num_kv_heads // ckpt_tp
    q_heads_per_shard = num_q_heads // ckpt_tp
    q_rows_per_shard = q_heads_per_shard * head_dim
    q_rows_per_rank = q_rows_per_shard // repetition
    if q_rows_per_rank * repetition != q_rows_per_shard:
        raise ValueError(
            f"a checkpoint shard's {q_heads_per_shard} query heads cannot be split "
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

    # Which KV head this worker holds, in the model's own numbering, and where
    # that head sits inside the checkpoint shard.
    kv_head = tp_rank // (tp_size // num_kv_heads)
    owning_shard = kv_head // kv_heads_per_shard
    if owning_shard != shard_index:
        raise ValueError(
            f"internal layout mismatch: attention TP rank {tp_rank} replicates "
            f"KV head {kv_head} (checkpoint shard {owning_shard}) but its query "
            f"heads come from shard {shard_index}."
        )
    head_in_shard = kv_head % kv_heads_per_shard
    k_start = shard_base + q_rows_per_shard + head_in_shard * head_dim
    v_start = (
        shard_base
        + q_rows_per_shard
        + k_rows_per_shard
        + head_in_shard * v_head_dim
    )
    k_rows = torch.arange(k_start, k_start + head_dim, dtype=torch.int64)
    v_rows = torch.arange(v_start, v_start + v_head_dim, dtype=torch.int64)
    return torch.cat([q_rows, k_rows, v_rows])


def ckpt_fused_qkv_rows(
    num_q_heads: int, num_kv_heads: int, head_dim: int, v_head_dim: int, ckpt_tp: int
) -> int:
    """Rows in the whole checkpoint tensor for one fused-qkv layer."""
    q_rows_per_shard = (num_q_heads // ckpt_tp) * head_dim
    kv_heads_per_shard = max(1, num_kv_heads // ckpt_tp)
    return ckpt_tp * (
        q_rows_per_shard + kv_heads_per_shard * head_dim + kv_heads_per_shard * v_head_dim
    )
