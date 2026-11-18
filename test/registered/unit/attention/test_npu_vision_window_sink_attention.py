"""CPU checks for the Ascend ViT windowed + attention-sink path.

The Ascend vision backend cannot express a windowed attention band or a learnable
sink through its plain FIA call, so ``vision_window_attn`` rebuilds both on top of
SDPA and installs itself under the ``ascend_attn`` name. These tests pin the
semantics against an independent full-softmax reference and pin the dispatch, so
they run in the CPU suite -- no NPU and no vendor operator involved.

What they protect:
  * the band is bidirectional (``|i - j| <= w``), not causal;
  * a sink joins the softmax denominator and contributes no value;
  * the blocked/row-limited implementation equals the whole-sequence math,
    including when a band straddles block boundaries and when the segment list
    holds several images;
  * grouped-query attention (fewer KV than query heads) is expanded correctly;
  * a call with neither a window nor a sink still reaches the stock backend,
    i.e. installing the subclass cannot change any other vision model.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.hardware_backend.npu.attention.vision_window_attn import (
    SINK_REFERENCE_KEY0,
    SINK_VIRTUAL_TOKEN,
    WindowedVisionAscendAttention,
    install_vision_window_attention,
    vision_window_sink_attention,
)
from sglang.srt.layers.attention import vision
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


def _expand_kv(t: torch.Tensor, num_heads: int) -> torch.Tensor:
    """[S, Hkv, D] -> [S, H, D] by repeating each KV head."""
    num_kv_heads = t.shape[1]
    if num_kv_heads == num_heads:
        return t
    return t.repeat_interleave(num_heads // num_kv_heads, dim=1)


def _reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens,
    *,
    window: int,
    sinks: torch.Tensor = None,
    sink_encoding: str = SINK_VIRTUAL_TOKEN,
    scale: float = None,
) -> torch.Tensor:
    """Whole-sequence softmax reference, written the way the model means it."""
    num_heads, head_dim = q.shape[1], q.shape[2]
    scale = head_dim**-0.5 if scale is None else scale
    out = torch.empty_like(q)
    for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:]):
        length = end - start
        if length == 0:
            continue
        qc = q[start:end]
        kc = _expand_kv(k[start:end], num_heads)
        vc = _expand_kv(v[start:end], num_heads)

        # [H, S, S] scores.
        scores = torch.einsum("shd,thd->hst", qc, kc) * scale
        if window > 0:
            idx = torch.arange(length)
            band = (idx[:, None] - idx[None, :]).abs() <= window
            scores = scores.masked_fill(~band, float("-inf"))

        if sinks is None:
            values = vc
        elif sink_encoding == SINK_VIRTUAL_TOKEN:
            # One extra key whose logit is the sink and whose value is zero.
            sink_scores = sinks.reshape(num_heads, 1, 1).expand(num_heads, length, 1)
            scores = torch.cat([scores, sink_scores], dim=-1)
            values = torch.cat([vc, vc.new_zeros(1, num_heads, head_dim)], dim=0)
        elif sink_encoding == SINK_REFERENCE_KEY0:
            scores[:, :, 0] = scores[:, :, 0] + sinks.reshape(num_heads, 1)
            values = vc
        else:
            raise AssertionError(sink_encoding)

        probs = torch.softmax(scores, dim=-1)
        out[start:end] = torch.einsum("hst,thd->shd", probs, values)
    return out


class VisionWindowSinkAttentionBase(CustomTestCase):
    num_heads = 4
    num_kv_heads = 4
    head_dim = 8

    def make_inputs(self, seq_lens, seed=0):
        torch.manual_seed(seed)
        total = sum(seq_lens)
        q = torch.randn(total, self.num_heads, self.head_dim, dtype=torch.float32)
        k = torch.randn(total, self.num_kv_heads, self.head_dim, dtype=torch.float32)
        v = torch.randn(total, self.num_kv_heads, self.head_dim, dtype=torch.float32)
        cu_seqlens = [0]
        for length in seq_lens:
            cu_seqlens.append(cu_seqlens[-1] + length)
        return q, k, v, cu_seqlens

    def assert_matches_reference(
        self,
        q,
        k,
        v,
        cu_seqlens,
        *,
        window,
        sinks=None,
        sink_encoding=SINK_VIRTUAL_TOKEN,
        scale=None,
    ):
        expected = _reference_attention(
            q,
            k,
            v,
            cu_seqlens,
            window=window,
            sinks=sinks,
            sink_encoding=sink_encoding,
            scale=scale,
        )
        actual = vision_window_sink_attention(
            q,
            k,
            v,
            cu_seqlens=cu_seqlens,
            num_heads=self.num_heads,
            scale=scale,
            window_size=(window, window) if window > 0 else (-1, -1),
            sinks=sinks,
            sink_encoding=sink_encoding,
        )
        self.assertEqual(actual.shape, q.shape)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


class TestWindowOnly(VisionWindowSinkAttentionBase):
    def test_band_matches_reference(self):
        q, k, v, cu = self.make_inputs([96])
        self.assert_matches_reference(q, k, v, cu, window=8)

    def test_window_wider_than_sequence_matches_full_attention(self):
        q, k, v, cu = self.make_inputs([32])
        windowed = vision_window_sink_attention(
            q,
            k,
            v,
            cu_seqlens=cu,
            num_heads=self.num_heads,
            scale=None,
            window_size=(4096, 4096),
        )
        full = vision_window_sink_attention(
            q,
            k,
            v,
            cu_seqlens=cu,
            num_heads=self.num_heads,
            scale=None,
            window_size=(-1, -1),
        )
        torch.testing.assert_close(windowed, full, atol=1e-5, rtol=1e-5)

    def test_window_actually_clips(self):
        """A window must actually clip -- guards a dropped window kwarg."""
        q, k, v, cu = self.make_inputs([96])
        banded = vision_window_sink_attention(
            q,
            k,
            v,
            cu_seqlens=cu,
            num_heads=self.num_heads,
            scale=None,
            window_size=(8, 8),
        )
        full = vision_window_sink_attention(
            q,
            k,
            v,
            cu_seqlens=cu,
            num_heads=self.num_heads,
            scale=None,
            window_size=(-1, -1),
        )
        self.assertFalse(torch.allclose(banded, full, atol=1e-3))

    def test_window_rejects_asymmetric_band(self):
        q, k, v, cu = self.make_inputs([16])
        with self.assertRaises(ValueError):
            vision_window_sink_attention(
                q,
                k,
                v,
                cu_seqlens=cu,
                num_heads=self.num_heads,
                scale=None,
                window_size=(4, 8),
            )

    def test_window_rejects_mismatched_cu_seqlens(self):
        q, k, v, cu = self.make_inputs([16])
        with self.assertRaises(ValueError):
            vision_window_sink_attention(
                q,
                k,
                v,
                cu_seqlens=cu[:-1] + [cu[-1] - 1],
                num_heads=self.num_heads,
                scale=None,
                window_size=(4, 4),
            )


class TestSink(VisionWindowSinkAttentionBase):
    def test_window_plus_sink(self):
        q, k, v, cu = self.make_inputs([96])
        sinks = torch.randn(self.num_heads)
        self.assert_matches_reference(
            q, k, v, cu, window=8, sinks=sinks, sink_encoding=SINK_VIRTUAL_TOKEN
        )

    def test_sink_without_window(self):
        q, k, v, cu = self.make_inputs([24])
        sinks = torch.randn(self.num_heads)
        self.assert_matches_reference(
            q, k, v, cu, window=-1, sinks=sinks, sink_encoding=SINK_VIRTUAL_TOKEN
        )

    def test_reference_key0_encoding(self):
        q, k, v, cu = self.make_inputs([48])
        sinks = torch.randn(self.num_heads)
        self.assert_matches_reference(
            q, k, v, cu, window=8, sinks=sinks, sink_encoding=SINK_REFERENCE_KEY0
        )

    def test_sink_changes_the_output(self):
        """A sink must actually move the result -- guards a dropped kwarg."""
        q, k, v, cu = self.make_inputs([64])
        without = vision_window_sink_attention(
            q,
            k,
            v,
            cu_seqlens=cu,
            num_heads=self.num_heads,
            scale=None,
            window_size=(8, 8),
        )
        with_sink = vision_window_sink_attention(
            q,
            k,
            v,
            cu_seqlens=cu,
            num_heads=self.num_heads,
            scale=None,
            window_size=(8, 8),
            sinks=torch.full((self.num_heads,), 4.0),
        )
        self.assertFalse(torch.allclose(without, with_sink, atol=1e-3))

    def test_bad_sink_count_raises(self):
        q, k, v, cu = self.make_inputs([16])
        with self.assertRaises(ValueError):
            vision_window_sink_attention(
                q,
                k,
                v,
                cu_seqlens=cu,
                num_heads=self.num_heads,
                scale=None,
                window_size=(4, 4),
                sinks=torch.zeros(self.num_heads - 1),
            )

    def test_unknown_encoding_raises(self):
        q, k, v, cu = self.make_inputs([16])
        with self.assertRaises(ValueError):
            vision_window_sink_attention(
                q,
                k,
                v,
                cu_seqlens=cu,
                num_heads=self.num_heads,
                scale=None,
                window_size=(4, 4),
                sinks=torch.zeros(self.num_heads),
                sink_encoding="nonsense",
            )


class TestVarlenAndBlocking(VisionWindowSinkAttentionBase):
    num_heads = 2
    num_kv_heads = 1
    head_dim = 8

    def test_multiple_images_in_one_batch(self):
        q, k, v, cu = self.make_inputs([40, 7, 0, 63])
        self.assert_matches_reference(
            q, k, v, cu, window=8, sinks=torch.randn(self.num_heads)
        )

    def test_band_across_query_blocks(self):
        """A sequence longer than the query block must match the whole-sequence math."""
        q, k, v, cu = self.make_inputs([1500])
        self.assert_matches_reference(q, k, v, cu, window=16)

    def test_key0_sink_across_query_blocks(self):
        """The key-0 sink encoding must not leak onto later blocks' first key."""
        q, k, v, cu = self.make_inputs([1500])
        self.assert_matches_reference(
            q,
            k,
            v,
            cu,
            window=16,
            sinks=torch.randn(self.num_heads),
            sink_encoding=SINK_REFERENCE_KEY0,
        )

    def test_virtual_sink_across_query_blocks(self):
        q, k, v, cu = self.make_inputs([1500])
        self.assert_matches_reference(
            q,
            k,
            v,
            cu,
            window=16,
            sinks=torch.randn(self.num_heads),
            sink_encoding=SINK_VIRTUAL_TOKEN,
        )

    def test_grouped_query_attention(self):
        """KV heads are expanded to query heads (MiMo ViT ships 32 q / 8 kv)."""
        self.assertEqual(self.num_kv_heads, 1)
        q, k, v, cu = self.make_inputs([64])
        self.assert_matches_reference(
            q, k, v, cu, window=8, sinks=torch.randn(self.num_heads)
        )


class TestDtypes(CustomTestCase):
    def test_bfloat16_runs_and_tracks_float32(self):
        torch.manual_seed(0)
        total, num_heads, head_dim = 64, 4, 8
        q = torch.randn(total, num_heads, head_dim).bfloat16()
        k = torch.randn(total, num_heads, head_dim).bfloat16()
        v = torch.randn(total, num_heads, head_dim).bfloat16()
        sinks = torch.randn(num_heads).bfloat16()
        out = vision_window_sink_attention(
            q,
            k,
            v,
            cu_seqlens=[0, total],
            num_heads=num_heads,
            scale=None,
            window_size=(8, 8),
            sinks=sinks,
        )
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(out.float()).all())

        expected = _reference_attention(
            q.float(), k.float(), v.float(), [0, total], window=8, sinks=sinks.float()
        )
        torch.testing.assert_close(out.float(), expected, atol=3e-2, rtol=3e-2)

    def test_empty_input_is_identity(self):
        q = torch.empty(0, 4, 8)
        out = vision_window_sink_attention(
            q,
            q,
            q,
            cu_seqlens=[0, 0],
            num_heads=4,
            scale=None,
            window_size=(8, 8),
        )
        self.assertEqual(out.shape, q.shape)


class TestBackendDispatch(CustomTestCase):
    """The subclass must be a drop-in for the stock backend."""

    def setUp(self):
        self.addCleanup(
            lambda: vision.QKV_BACKEND_IMPL.__setitem__(
                "ascend_attn", vision.VisionAscendAttention
            )
        )

    def _bare_instance(self):
        # VisionAscendAttention.__init__ refuses to run off NPU; the windowed
        # path needs no instance state, so bypass the constructor.
        return object.__new__(WindowedVisionAscendAttention)

    def test_install_replaces_the_backend_entry(self):
        install_vision_window_attention()
        self.assertIs(
            vision.QKV_BACKEND_IMPL["ascend_attn"], WindowedVisionAscendAttention
        )

    def test_windowed_call_matches_the_kernel(self):
        q, k, v = (torch.randn(96, 4, 8) for _ in range(3))
        sinks = torch.randn(4)
        metadata = SimpleNamespace(cu_seqlens=torch.tensor([0, 96], dtype=torch.int32))

        got = WindowedVisionAscendAttention.forward(
            self._bare_instance(),
            q,
            k,
            v,
            cu_seqlens=None,
            bsz=1,
            seq_len=96,
            forward_metadata=metadata,
            window_size=(8, 8),
            s_aux=sinks,
        )
        expected = vision_window_sink_attention(
            q,
            k,
            v,
            cu_seqlens=[0, 96],
            num_heads=4,
            scale=8**-0.5,
            window_size=(8, 8),
            sinks=sinks,
        )
        torch.testing.assert_close(got, expected, atol=1e-6, rtol=1e-6)

    def test_plain_call_delegates_to_the_stock_backend(self):
        """No window, no sink: every other vision model must be unaffected."""
        sentinel = torch.full((2, 4, 8), 7.0)
        q = torch.zeros(2, 4, 8)
        with patch.object(
            vision.VisionAscendAttention, "forward", return_value=sentinel
        ) as stock:
            got = WindowedVisionAscendAttention.forward(
                self._bare_instance(),
                q,
                q,
                q,
                cu_seqlens=torch.tensor([0, 2], dtype=torch.int32),
                bsz=1,
                seq_len=2,
            )
        self.assertIs(got, sentinel)
        stock.assert_called_once()

    def test_windowed_call_without_metadata_fails_loudly(self):
        q = torch.zeros(4, 4, 8)
        with self.assertRaises(RuntimeError):
            WindowedVisionAscendAttention.forward(
                self._bare_instance(),
                q,
                q,
                q,
                cu_seqlens=[0, 4],
                bsz=1,
                seq_len=4,
                window_size=(8, 8),
            )


if __name__ == "__main__":
    unittest.main()
