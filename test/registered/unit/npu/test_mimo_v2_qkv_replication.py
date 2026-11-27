"""CPU checks for the Ascend-side MiMo-V2 fused-qkv re-sharding hook.

The checkpoint ships `attention_projection_layout: fused_qkv`, interleaved over
`num_key_value_heads` (4) TP shards. The stock loader can only merge shards, so
an attention-TP width above the interleave -- TP16 with no attention-DP
(attn_tp 16) or with `--attn-dp-size 2` (attn_tp 8) -- is rejected. The NPU hook
in `hardware_backend/npu/mimo_v2.py` splits a shard's query heads further and
replicates its KV heads instead: the layout `QKVParallelLinear` and
`ModelConfig.get_num_kv_heads()` describe as "with fewer heads than groups, each
GPU keeps one".

These tests need neither NPU nor a checkpoint: they rebuild a fused-qkv tensor
whose rows are numbered, tag each row with what it holds, and check per rank that
the hook keeps exactly that rank's query heads plus one replicated KV head.
"""

import sys
import unittest
from types import ModuleType
from unittest.mock import patch

import torch

from sglang.srt.hardware_backend.npu.mimo_v2 import (
    _replicating_qkv_loader,
    ckpt_fused_qkv_rows,
    fused_qkv_rows_for_rank,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

# Row kinds, so a selected row can be decoded back into what it holds.
_Q, _K, _V = 0, 1, 2


def _build_ckpt(*, num_q_heads, num_kv_heads, head_dim, v_head_dim, ckpt_tp):
    """A fused-qkv checkpoint whose row index equals its row number."""
    q_heads_per_shard = num_q_heads // ckpt_tp
    kv_heads_per_shard = max(1, num_kv_heads // ckpt_tp)
    tags = []
    for shard in range(ckpt_tp):
        q_head_base = shard * q_heads_per_shard
        for i in range(q_heads_per_shard * head_dim):
            tags.append((_Q, q_head_base + i // head_dim, i % head_dim))
        kv_head_base = shard * kv_heads_per_shard
        for i in range(kv_heads_per_shard * head_dim):
            tags.append((_K, kv_head_base + i // head_dim, i % head_dim))
        for i in range(kv_heads_per_shard * v_head_dim):
            tags.append((_V, kv_head_base + i // v_head_dim, i % v_head_dim))
    row_ids = torch.arange(len(tags), dtype=torch.int64).unsqueeze(1)
    return row_ids, tags


def _expected_rows(tags, *, num_q_heads, num_kv_heads, head_dim, v_head_dim, tp_rank, tp_size):
    """What the model expects this rank to hold, in load order."""
    q_heads_per_rank = num_q_heads // tp_size
    expected = []
    for head in range(tp_rank * q_heads_per_rank, (tp_rank + 1) * q_heads_per_rank):
        expected += [(_Q, head, d) for d in range(head_dim)]
    kv_head = tp_rank // (tp_size // num_kv_heads)
    expected += [(_K, kv_head, d) for d in range(head_dim)]
    expected += [(_V, kv_head, d) for d in range(v_head_dim)]
    del tags  # the layout above is derived from the model, not from the checkpoint
    return expected


class TestFusedQkvRows(CustomTestCase):
    def check_geometry(self, geometry, *, tp_size, ckpt_tp):
        num_q_heads, num_kv_heads, head_dim, v_head_dim = geometry
        _, tags = _build_ckpt(
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            v_head_dim=v_head_dim,
            ckpt_tp=ckpt_tp,
        )
        self.assertEqual(
            len(tags),
            ckpt_fused_qkv_rows(
                num_q_heads, num_kv_heads, head_dim, v_head_dim, ckpt_tp
            ),
        )
        rows_per_rank = (
            (num_q_heads // tp_size) * head_dim + head_dim + v_head_dim
        )
        for tp_rank in range(tp_size):
            rows = fused_qkv_rows_for_rank(
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                v_head_dim=v_head_dim,
                tp_rank=tp_rank,
                tp_size=tp_size,
                ckpt_tp=ckpt_tp,
            )
            self.assertEqual(rows.numel(), rows_per_rank)
            self.assertEqual(
                [tags[i] for i in rows.tolist()],
                _expected_rows(
                    tags,
                    num_q_heads=num_q_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    v_head_dim=v_head_dim,
                    tp_rank=tp_rank,
                    tp_size=tp_size,
                ),
                f"tp_rank={tp_rank}",
            )

    def test_global_layer_replicating_twice(self):
        # MiMo-V2.5 global layer, scaled down: 8 q / 2 kv / head_dim 4 / v 2.
        self.check_geometry((8, 2, 4, 2), tp_size=4, ckpt_tp=2)

    def test_global_layer_replicating_four_times(self):
        self.check_geometry((8, 2, 4, 2), tp_size=8, ckpt_tp=2)

    def test_swa_layer_with_two_kv_heads_per_shard(self):
        # SWA layers carry more KV heads (8 vs 4 in the real checkpoint), so one
        # shard holds two of them and a rank has to pick one: the offset path.
        self.check_geometry((8, 4, 4, 2), tp_size=4, ckpt_tp=2)

    def test_swa_layer_replicating_four_times(self):
        self.check_geometry((8, 4, 4, 2), tp_size=8, ckpt_tp=2)

    def test_real_mimo_shapes_attn_tp_8_and_16(self):
        """The 16-die widths: attn_tp 8 (--attn-dp-size 2) and 16 (no DP)."""
        for tp_size in (8, 16):
            with self.subTest(tp_size=tp_size):
                self.check_geometry((64, 4, 192, 128), tp_size=tp_size, ckpt_tp=4)

    def test_every_query_head_is_covered_exactly_once(self):
        _, tags = _build_ckpt(
            num_q_heads=8, num_kv_heads=2, head_dim=4, v_head_dim=2, ckpt_tp=2
        )
        q_rows = [i for i, tag in enumerate(tags) if tag[0] == _Q]
        seen = []
        for tp_rank in range(8):
            rows = fused_qkv_rows_for_rank(
                num_q_heads=8,
                num_kv_heads=2,
                head_dim=4,
                v_head_dim=2,
                tp_rank=tp_rank,
                tp_size=8,
                ckpt_tp=2,
            )
            seen.extend(i for i in rows.tolist() if tags[i][0] == _Q)
        self.assertEqual(sorted(seen), q_rows)

    def test_kv_head_is_replicated_the_expected_number_of_times(self):
        _, tags = _build_ckpt(
            num_q_heads=8, num_kv_heads=2, head_dim=4, v_head_dim=2, ckpt_tp=2
        )
        holders = {}
        for tp_rank in range(8):
            rows = fused_qkv_rows_for_rank(
                num_q_heads=8,
                num_kv_heads=2,
                head_dim=4,
                v_head_dim=2,
                tp_rank=tp_rank,
                tp_size=8,
                ckpt_tp=2,
            )
            k_heads = {tags[i][1] for i in rows.tolist() if tags[i][0] == _K}
            self.assertEqual(len(k_heads), 1, f"tp_rank={tp_rank}")
            for head in k_heads:
                holders[head] = holders.get(head, 0) + 1
        # Two KV heads, eight workers -> four workers per head.
        self.assertEqual(holders, {0: 4, 1: 4})


class TestRejection(CustomTestCase):
    def test_width_that_neither_splits_nor_replicates(self):
        with self.assertRaises(ValueError):
            fused_qkv_rows_for_rank(
                num_q_heads=12,
                num_kv_heads=3,
                head_dim=4,
                v_head_dim=2,
                tp_rank=0,
                tp_size=5,
                ckpt_tp=2,
            )

    def test_tp_not_a_multiple_of_kv_heads(self):
        with self.assertRaises(ValueError):
            fused_qkv_rows_for_rank(
                num_q_heads=8,
                num_kv_heads=3,
                head_dim=4,
                v_head_dim=2,
                tp_rank=0,
                tp_size=4,
                ckpt_tp=1,
            )

    def test_q_heads_must_divide_the_attention_tp_width(self):
        with self.assertRaises(ValueError):
            fused_qkv_rows_for_rank(
                num_q_heads=10,
                num_kv_heads=2,
                head_dim=4,
                v_head_dim=2,
                tp_rank=0,
                tp_size=4,
                ckpt_tp=2,
            )

    def test_kv_heads_must_divide_the_interleave(self):
        with self.assertRaises(ValueError):
            fused_qkv_rows_for_rank(
                num_q_heads=8,
                num_kv_heads=3,
                head_dim=4,
                v_head_dim=2,
                tp_rank=0,
                tp_size=4,
                ckpt_tp=2,
            )


class _FakeQKVLinear:
    """Stands in for QKVParallelLinear; only the attributes the hook reads."""

    def __init__(self, *, size, num_q_heads, num_kv_heads, head_dim, v_head_dim):
        # tp_group carries (rank, size); the fake get_group_rank_size returns it.
        self.tp_group = None
        self.tp_size = size
        self.total_num_heads = num_q_heads
        self.total_num_kv_heads = num_kv_heads
        self.head_size = head_dim
        self.v_head_size = v_head_dim


class TestLoaderWrapper(CustomTestCase):
    """The wrapper must delegate the merging direction and replicate the other."""

    def run_wrapper(
        self,
        *,
        geometry,
        rank,
        size,
        ckpt_tp,
        name="model.layers.0.self_attn.qkv_proj.weight",
    ):
        num_q_heads, num_kv_heads, head_dim, v_head_dim = geometry
        ckpt, _ = _build_ckpt(
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            v_head_dim=v_head_dim,
            ckpt_tp=ckpt_tp,
        )
        rows_per_rank = (num_q_heads // size) * head_dim + head_dim + v_head_dim
        param = torch.empty(rows_per_rank, 1, dtype=torch.int64)
        qkv_proj = _FakeQKVLinear(
            size=size,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            v_head_dim=v_head_dim,
        )

        delegated = []
        loaded = []

        def original(*args, **kwargs):
            delegated.append(args)
            return "delegated"

        weight_utils = ModuleType("sglang.srt.model_loader.weight_utils")
        weight_utils.default_weight_loader = lambda target, value: loaded.append(
            value.clone()
        )
        distributed_utils = ModuleType("sglang.srt.distributed.utils")
        distributed_utils.get_group_rank_size = lambda group: (rank, size)
        lora_layers = ModuleType("sglang.srt.lora.layers")
        lora_layers.unwrap_lora_layer = lambda module: module

        loader = _replicating_qkv_loader(original)
        with patch.dict(
            sys.modules,
            {
                "sglang.srt.model_loader.weight_utils": weight_utils,
                "sglang.srt.distributed.utils": distributed_utils,
                "sglang.srt.lora.layers": lora_layers,
            },
        ):
            outcome = loader(name, param, ckpt, ckpt_tp, None, qkv_proj=qkv_proj)
        return outcome, delegated, loaded, ckpt

    def test_merging_direction_delegates(self):
        outcome, delegated, loaded, _ = self.run_wrapper(
            geometry=(8, 2, 4, 2), rank=1, size=4, ckpt_tp=4
        )
        self.assertEqual(outcome, "delegated")
        self.assertEqual(len(delegated), 1)
        self.assertFalse(loaded)

    def test_replicating_direction_loads_this_ranks_rows(self):
        outcome, delegated, loaded, ckpt = self.run_wrapper(
            geometry=(8, 2, 4, 2), rank=5, size=8, ckpt_tp=2
        )
        self.assertIsNone(outcome)
        self.assertFalse(delegated)
        self.assertEqual(len(loaded), 1)
        _, tags = _build_ckpt(
            num_q_heads=8, num_kv_heads=2, head_dim=4, v_head_dim=2, ckpt_tp=2
        )
        got = [tags[int(row)] for row in loaded[0].flatten().tolist()]
        expected = [(_Q, 5, d) for d in range(4)]  # one q head at tp_size=8
        expected += [(_K, 1, d) for d in range(4)]  # kv head 5 // (8 // 2)
        expected += [(_V, 1, d) for d in range(2)]
        self.assertEqual(got, expected)
        self.assertEqual(ckpt.shape[1], 1)

    def test_block_quantized_scale_is_rejected_with_guidance(self):
        with self.assertRaises(ValueError) as caught:
            self.run_wrapper(
                geometry=(8, 2, 4, 2),
                rank=0,
                size=8,
                ckpt_tp=2,
                name="model.layers.0.self_attn.qkv_proj.weight_scale_inv",
            )
        message = str(caught.exception)
        self.assertIn("block", message)
        self.assertIn("modelslim", message)

    def test_width_that_fits_neither_direction_is_rejected(self):
        with self.assertRaises(ValueError):
            self.run_wrapper(geometry=(8, 2, 4, 2), rank=0, size=6, ckpt_tp=4)


if __name__ == "__main__":
    unittest.main()
