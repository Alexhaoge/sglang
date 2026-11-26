"""CPU checks for MiMo-V2's fused-qkv re-sharding when attention TP is wider.

`attention_projection_layout: fused_qkv` interleaves `qkv_proj` over
`num_key_value_heads` (4) TP shards. A worker whose attention-TP width is a
multiple of that has to split one shard's query heads further and replicate that
shard's KV heads -- the layout `ModelConfig.get_num_kv_heads()` documents as
"with fewer heads than groups, each GPU keeps one". That is what makes TP16 with
no attention-DP (attn_tp 16) or `--attn-dp-size 2` (attn_tp 8) loadable at all,
instead of only attn_tp <= 4.

The arithmetic lives in `mimo_v2_qkv_shard`, which imports torch only, so it can
be pinned here without torch_npu or a checkpoint.
"""

import unittest

import torch

from sglang.srt.models.mimo_v2_qkv_shard import (
    ckpt_fused_qkv_rows,
    get_ckpt_qkv_geometry,
    replicated_fused_qkv_row_indices,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

# Row kinds, so a selected row can be decoded back into what it holds.
_Q, _K, _V = 0, 1, 2


def _build_ckpt(*, num_q_heads, num_kv_heads, head_dim, v_head_dim, ckpt_tp):
    """A fused-qkv checkpoint whose rows are tagged with their (kind, head, dim)."""
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
    rows = torch.arange(len(tags), dtype=torch.int64).unsqueeze(1)
    return rows, tags


class TestReplicatedFusedQkvRows(CustomTestCase):
    def check_geometry(self, geometry, *, tp_size, ckpt_tp):
        """Every rank must get its own query heads and one replicated KV head."""
        num_q_heads, num_kv_heads, head_dim, v_head_dim = geometry
        weight, tags = _build_ckpt(
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            v_head_dim=v_head_dim,
            ckpt_tp=ckpt_tp,
        )
        self.assertEqual(
            weight.shape[0],
            ckpt_fused_qkv_rows(
                num_q_heads, num_kv_heads, head_dim, v_head_dim, ckpt_tp
            ),
        )
        q_heads_per_rank = num_q_heads // tp_size
        rows_per_rank = q_heads_per_rank * head_dim + head_dim + v_head_dim

        for tp_rank in range(tp_size):
            rows = replicated_fused_qkv_row_indices(
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                v_head_dim=v_head_dim,
                tp_rank=tp_rank,
                tp_size=tp_size,
                ckpt_tp=ckpt_tp,
            )
            self.assertEqual(rows.numel(), rows_per_rank)

            # The layout the model itself expects: q heads are partitioned over
            # attention TP; a KV head is replicated onto `tp_size // nkv` ranks.
            expected = []
            for head in range(
                tp_rank * q_heads_per_rank, (tp_rank + 1) * q_heads_per_rank
            ):
                expected += [(_Q, head, d) for d in range(head_dim)]
            kv_head = tp_rank // (tp_size // num_kv_heads)
            expected += [(_K, kv_head, d) for d in range(head_dim)]
            expected += [(_V, kv_head, d) for d in range(v_head_dim)]

            self.assertEqual(
                [tags[i] for i in rows.tolist()], expected, f"tp_rank={tp_rank}"
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
        """The A3-lite widths: attn_tp 8 (--attn-dp-size 2) and 16 (no DP)."""
        for tp_size in (8, 16):
            with self.subTest(tp_size=tp_size):
                self.check_geometry((64, 4, 192, 128), tp_size=tp_size, ckpt_tp=4)

    def test_every_query_head_is_covered_exactly_once(self):
        weight, tags = _build_ckpt(
            num_q_heads=8, num_kv_heads=2, head_dim=4, v_head_dim=2, ckpt_tp=2
        )
        q_rows = [i for i, tag in enumerate(tags) if tag[0] == _Q]
        seen = []
        for tp_rank in range(8):
            rows = replicated_fused_qkv_row_indices(
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
        per_rank_k_rows = []
        for tp_rank in range(8):
            rows = replicated_fused_qkv_row_indices(
                num_q_heads=8,
                num_kv_heads=2,
                head_dim=4,
                v_head_dim=2,
                tp_rank=tp_rank,
                tp_size=8,
                ckpt_tp=2,
            )
            per_rank_k_rows.append(sum(1 for i in rows.tolist() if tags[i][0] == _K))
        # Each worker takes exactly one K head worth of rows (head_dim = 4).
        self.assertEqual(per_rank_k_rows, [4] * 8)


class TestRejection(CustomTestCase):
    def test_width_that_neither_splits_nor_replicates(self):
        with self.assertRaises(ValueError):
            replicated_fused_qkv_row_indices(
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
            replicated_fused_qkv_row_indices(
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
            replicated_fused_qkv_row_indices(
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
            replicated_fused_qkv_row_indices(
                num_q_heads=8,
                num_kv_heads=3,
                head_dim=4,
                v_head_dim=2,
                tp_rank=0,
                tp_size=4,
                ckpt_tp=2,
            )


class TestGeometry(CustomTestCase):
    def config(self):
        return type(
            "Cfg",
            (),
            dict(
                num_attention_heads=64,
                num_key_value_heads=4,
                head_dim=192,
                v_head_dim=128,
                swa_num_attention_heads=64,
                swa_num_key_value_heads=8,
                swa_head_dim=192,
                swa_v_head_dim=128,
                hybrid_layer_pattern=[0, 1, 1, 1, 1],
            ),
        )()

    def test_global_layer_uses_the_full_attention_fields(self):
        self.assertEqual(
            get_ckpt_qkv_geometry(
                self.config(), "model.layers.0.self_attn.qkv_proj.weight"
            ),
            (64, 4, 192, 128),
        )

    def test_swa_layer_uses_the_swa_fields(self):
        self.assertEqual(
            get_ckpt_qkv_geometry(
                self.config(), "model.layers.2.self_attn.qkv_proj.weight"
            ),
            (64, 8, 192, 128),
        )

    def test_layer_index_outside_the_pattern_is_global(self):
        self.assertEqual(
            get_ckpt_qkv_geometry(
                self.config(), "model.layers.9.self_attn.qkv_proj.weight"
            ),
            (64, 4, 192, 128),
        )

    def test_unrelated_name_has_no_geometry(self):
        self.assertIsNone(
            get_ckpt_qkv_geometry(self.config(), "model.embed_tokens.weight")
        )


if __name__ == "__main__":
    unittest.main()
