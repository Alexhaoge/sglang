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
"""Windowed + attention-sink vision attention for Ascend NPU.

MiMo-V2.5/V2.6's vision tower windows 24 of its 28 layers and gives each of
those layers a learnable attention sink (``vision_config.use_sink: true``).
`VisionAttention` forwards both the window and the sink to the backend, but the
Ascend backend pins its FIA call to ``sparse_mode=0`` with neither an attention
mask nor ``learnable_sink``, so those 24 layers would silently run as global,
sink-free attention -- wrong output, no error.

FIA can express both (``sparse_mode=4`` with ``pre_tokens``/``next_tokens`` is a
band, and ``learnable_sink`` takes the per-head logit), but that combination is
not verified for the shape a vision tower uses: varlen (``TND``), no page table,
and ``s1 > 1``. Until it is, ``WindowedVisionAscendAttention`` reproduces the
semantics with ``scaled_dot_product_attention`` so the path is correct before it
is fast. The follow-up is an sgl-kernel-npu operator for varlen window + sink
vision attention; this module then becomes its caller.

Semantics, deliberately exact rather than approximate:

* **window** - additive ``-inf`` outside ``|i - j| <= w``. The band is
  bidirectional, matching the HF reference
  (``modeling_mimo_v2.py::MiMoVisionAttention._build_window_mask``), not causal.
  The tower's "sliding window" is a token permutation *plus* this band; the
  permutation is model code and already runs unchanged on NPU, so only the band
  is expressible here.
* **sink** - the head's learned logit joins the softmax denominator without
  contributing a value. ``SINK_VIRTUAL_TOKEN`` (the default) appends one zero
  K/V row and puts the logit in that row's score, which is what flash-attn's
  ``sinks=`` computes -- so NPU matches what the CUDA path in this repo emits.
  ``SINK_REFERENCE_KEY0`` instead adds the logit to key 0's score, which is what
  the released HF modeling file does; it is kept because the two encodings
  disagree, and a GPU-vs-reference comparison is the only way to settle which
  one the checkpoint was trained under.

Queries are processed in row blocks, and each block only keeps the keys its band
can see, so the mask never materializes as ``[T, T]``.

Installation: ``install_vision_window_attention()`` (called once per worker from
``init_npu_backend``) swaps the ``ascend_attn`` backend for
``WindowedVisionAscendAttention``. That subclass delegates to the stock backend
whenever it sees neither a window nor a sink, so every other vision model on NPU
keeps the existing FIA path unchanged.
"""

import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.vision import VisionAscendAttention

# How the learnable sink logit enters the softmax. See the module docstring.
SINK_VIRTUAL_TOKEN = "virtual_token"
SINK_REFERENCE_KEY0 = "reference_key0"
DEFAULT_SINK_ENCODING = SINK_VIRTUAL_TOKEN

# Element budget for one materialized mask block (query rows times visible keys,
# per attention head). 8M bf16 elements ~= 16 MB, which keeps the temporary well
# below the ViT activation footprint while still giving tensor-core-friendly
# shapes.
_MASK_ELEMENT_BUDGET = 8 * 1024 * 1024

# Smallest useful query block; below this the SDPA launch overhead dominates.
_MIN_QUERY_BLOCK = 32


def _plain_window(window_size) -> int:
    """Collapse ``VisionAttention``'s ``(left, right)`` window to one radius.

    The tower always passes ``(w, w)``. A half-open band (``left != right``) has
    no faithful single-radius equivalent, so reject it loudly instead of
    silently using the larger side.
    """
    if isinstance(window_size, (tuple, list)):
        left, right = (list(window_size) + [-1, -1])[:2]
    else:
        left = right = window_size
    left = -1 if left is None else int(left)
    right = -1 if right is None else int(right)
    if left <= 0 and right <= 0:
        return -1
    if left != right:
        raise ValueError(
            "ViT windowed attention needs a symmetric window, got "
            f"window_size={tuple(window_size)}; splitting the band is not supported."
        )
    if left <= 0:
        raise ValueError(f"ViT window size must be positive, got {left}.")
    return left


def _segment_lengths(cu_seqlens) -> list:
    if cu_seqlens is None:
        raise ValueError("ViT windowed attention needs cu_seqlens.")
    if isinstance(cu_seqlens, torch.Tensor):
        cu_seqlens = cu_seqlens.detach().to("cpu").to(torch.int64).tolist()
    else:
        cu_seqlens = [int(v) for v in cu_seqlens]
    if len(cu_seqlens) < 2:
        raise ValueError(f"cu_seqlens needs at least two entries, got {cu_seqlens}.")
    lengths = [cu_seqlens[i + 1] - cu_seqlens[i] for i in range(len(cu_seqlens) - 1)]
    if any(length < 0 for length in lengths):
        raise ValueError(f"cu_seqlens must be non-decreasing, got {cu_seqlens}.")
    if cu_seqlens[0] != 0:
        raise ValueError(f"cu_seqlens must start at 0, got {cu_seqlens}.")
    return lengths


def _query_block_rows(num_heads: int, keys_per_row: int) -> int:
    rows = _MASK_ELEMENT_BUDGET // max(1, num_heads * keys_per_row)
    return max(_MIN_QUERY_BLOCK, min(rows, 1024))


def _segment_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
    window: int,
    sinks: torch.Tensor,
    sink_encoding: str,
) -> torch.Tensor:
    """Attention for one image/video segment.

    ``q`` is ``[S, H, D]``; ``k``/``v`` are ``[S, Hkv, D]`` (GQA when Hkv < H).
    """
    seq_len, num_heads, _ = q.shape
    num_kv_heads = k.shape[1]
    if num_heads % num_kv_heads:
        raise ValueError(
            f"Query heads ({num_heads}) must be a multiple of KV heads ({num_kv_heads})."
        )
    if num_heads != num_kv_heads:
        groups = num_heads // num_kv_heads
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)

    # [S, H, D] -> [1, H, S, D] for SDPA.
    qh = q.transpose(0, 1).unsqueeze(0)
    kh = k.transpose(0, 1).unsqueeze(0)
    vh = v.transpose(0, 1).unsqueeze(0)

    sink_logits = None
    if sinks is not None:
        sink_logits = sinks.to(dtype=q.dtype, device=q.device).reshape(-1)
        if sink_logits.numel() != num_heads:
            raise ValueError(
                f"sinks must hold one logit per query head ({num_heads}), got "
                f"{sink_logits.numel()}."
            )
        sink_logits = sink_logits.view(1, num_heads, 1)

    keys_per_row = seq_len if window < 0 else min(seq_len, 2 * window + 1)
    rows = _query_block_rows(num_heads, keys_per_row)

    # The sink logit differs per head, so a mask carrying it has to be
    # materialized per head; a pure band mask does not. Only the virtual-token
    # encoding adds a mask column (for the appended key).
    if sink_encoding not in (SINK_VIRTUAL_TOKEN, SINK_REFERENCE_KEY0):
        raise ValueError(
            f"Unknown sink encoding {sink_encoding!r}; expected "
            f"{SINK_VIRTUAL_TOKEN!r} or {SINK_REFERENCE_KEY0!r}."
        )
    use_virtual_sink = sink_logits is not None and sink_encoding == SINK_VIRTUAL_TOKEN
    mask_heads = num_heads if sink_logits is not None else 1

    out = torch.empty_like(q)
    for start in range(0, seq_len, rows):
        end = min(start + rows, seq_len)
        if window < 0:
            key_start, key_end = 0, seq_len
        else:
            key_start = max(0, start - window)
            key_end = min(seq_len, end + window)

        q_block = qh[:, :, start:end, :].contiguous()
        k_block = kh[:, :, key_start:key_end, :].contiguous()
        v_block = vh[:, :, key_start:key_end, :].contiguous()
        block_rows = end - start
        block_keys = key_end - key_start

        mask_cols = block_keys + (1 if use_virtual_sink else 0)
        mask = None
        if window >= 0 or sink_logits is not None:
            mask = torch.zeros(
                (1, mask_heads, block_rows, mask_cols),
                dtype=q.dtype,
                device=q.device,
            )
            if window >= 0:
                row_pos = torch.arange(start, end, device=q.device)
                col_pos = torch.arange(key_start, key_end, device=q.device)
                visible = (row_pos[:, None] - col_pos[None, :]).abs() <= window
                mask[:, :, :, :block_keys] = torch.zeros(
                    (block_rows, block_keys), dtype=q.dtype, device=q.device
                ).masked_fill_(~visible, float("-inf"))

        if sink_logits is not None:
            if use_virtual_sink:
                # The sink becomes key `block_keys`: a zero key/value row whose
                # score is the logit, so it enters the softmax denominator and
                # contributes nothing to the output.
                k_block = torch.cat(
                    (k_block, q.new_zeros((1, num_heads, 1, k_block.shape[-1]))), dim=2
                )
                v_block = torch.cat(
                    (v_block, q.new_zeros((1, num_heads, 1, v_block.shape[-1]))), dim=2
                )
                mask[:, :, :, block_keys] = sink_logits
            elif key_start == 0:
                # HF reference encoding: bias the score of the sequence's first
                # key. Only the block that actually contains key 0 applies it --
                # further blocks start past it, so in the reference math the key
                # is masked out and the bias has no effect there.
                mask[:, :, :, 0] = mask[:, :, :, 0] + sink_logits

        attn = F.scaled_dot_product_attention(
            q_block, k_block, v_block, attn_mask=mask, scale=scale
        )
        out[start:end] = attn.squeeze(0).transpose(0, 1)

    return out


def vision_window_sink_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens,
    num_heads: int,
    scale: float,
    window_size=(-1, -1),
    sinks: torch.Tensor = None,
    sink_encoding: str = DEFAULT_SINK_ENCODING,
) -> torch.Tensor:
    """ViT attention with a bidirectional window and an optional attention sink.

    Args:
        q: ``[T, H, D]`` queries, flattened over all images/videos in the batch.
        k: ``[T, Hkv, D]`` keys.
        v: ``[T, Hkv, D]`` values.
        cu_seqlens: cumulative per-image token counts, ``[B + 1]``, starting at 0.
        num_heads: number of query heads (used to size the mask blocks).
        scale: softmax scale; ``head_dim ** -0.5`` when the caller passes ``None``.
        window_size: ``(w, w)`` or ``w``; ``-1`` (or both non-positive) disables
            the window. ``(left, right)`` with ``left != right`` is rejected.
        sinks: ``[H]`` learnable sink logits, one per query head, or ``None``.
        sink_encoding: see the module docstring.

    Returns:
        ``[T, H, D]`` attention output in ``q``'s dtype and device.
    """
    if q.dim() != 3:
        raise ValueError(f"q must be [T, H, D], got shape {tuple(q.shape)}.")
    if q.shape[0] == 0:
        return q

    window = _plain_window(window_size)
    if scale is None:
        scale = q.shape[-1] ** -0.5
    scale = float(scale)

    lengths = _segment_lengths(cu_seqlens)
    total = sum(lengths)
    if total != q.shape[0]:
        raise ValueError(
            f"cu_seqlens covers {total} tokens but q has {q.shape[0]}; the vision "
            "encoder flattens every image into one sequence, so the two must agree."
        )

    out = q.new_empty(q.shape)
    offset = 0
    for length in lengths:
        if length == 0:
            continue
        out[offset : offset + length] = _segment_attention(
            q[offset : offset + length],
            k[offset : offset + length],
            v[offset : offset + length],
            scale=scale,
            window=window,
            sinks=sinks,
            sink_encoding=sink_encoding,
        )
        offset += length
    return out


class WindowedVisionAscendAttention(VisionAscendAttention):
    """Ascend vision attention that honours a window and a learnable sink.

    Delegates to :class:`VisionAscendAttention` unless the caller supplied one of
    the two features, so models that use neither keep the stock FIA path
    byte-for-byte.
    """

    def forward(
        self,
        q,
        k,
        v,
        cu_seqlens,
        bsz,
        seq_len,
        softmax_scale=None,
        forward_metadata=None,
        attention_mask=None,
        **kwargs,
    ):
        window = kwargs.get("window_size", (-1, -1))
        if not isinstance(window, (tuple, list)):
            window = (window, window)
        window = tuple(window)
        sinks = kwargs.get("s_aux", None)
        if sinks is None and not any(int(w) >= 0 for w in window):
            return super().forward(
                q=q,
                k=k,
                v=v,
                cu_seqlens=cu_seqlens,
                bsz=bsz,
                seq_len=seq_len,
                softmax_scale=softmax_scale,
                forward_metadata=forward_metadata,
                attention_mask=attention_mask,
                **kwargs,
            )

        if forward_metadata is not None:
            meta_cu_seqlens = forward_metadata.cu_seqlens
        elif isinstance(cu_seqlens, torch.Tensor):
            meta_cu_seqlens = cu_seqlens
        else:
            raise RuntimeError(
                "Windowed vision attention needs the per-image cumulative "
                "sequence lengths; pass forward_metadata (see "
                "prepare_vision_attention_metadata) or a cu_seqlens tensor."
            )

        _, num_heads, head_size = q.shape
        return vision_window_sink_attention(
            q,
            k,
            v,
            cu_seqlens=meta_cu_seqlens,
            num_heads=num_heads,
            scale=softmax_scale if softmax_scale is not None else head_size**-0.5,
            window_size=window,
            sinks=sinks,
            sink_encoding=DEFAULT_SINK_ENCODING,
        )


def install_vision_window_attention() -> None:
    """Route ``ascend_attn`` through :class:`WindowedVisionAscendAttention`.

    Called from ``init_npu_backend`` (once per worker). Registering the subclass
    under the existing name keeps the change inside the NPU backend: the shared
    vision attention code is not modified, and every model that never passes a
    window or a sink is unaffected.
    """
    from sglang.srt.layers.attention import vision

    vision.QKV_BACKEND_IMPL["ascend_attn"] = WindowedVisionAscendAttention
