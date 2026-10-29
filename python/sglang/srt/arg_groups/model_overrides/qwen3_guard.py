"""Config-time override declarations for qwen3_guard.

Qwen3Guard-Stream classifies a whole prefill in one pass and has no decode
loop, so everything that assumes an incremental generation request has to be
off:

* chunked prefill -- a chunk boundary would split the pass that produces the
  labels, and the model has no way to resume one;
* the overlap scheduler -- there is no decode step for it to overlap with;
* CUDA graphs -- a guard forward returns head logits, not a logits tensor on
  the static output buffer the graph replays.

The radix cache is deliberately NOT disabled.  A guard stream re-sends the
whole conversation on every call, so the prefix cache is what keeps the cost
proportional to the new tokens instead of to the whole sequence; with it off,
every call re-prefills from token 0 and the KV pool fills up until the
scheduler deadlocks.  Pass --disable-radix-cache explicitly to A/B that.
"""

import logging
from typing import Any, Dict

from sglang.srt.arg_groups.model_override_base import _register_for

logger = logging.getLogger(__name__)


@_register_for("Qwen3ForGuardModel")
def _qwen3_guard_overrides(server_args: Any, hf_config: Any) -> dict:
    overrides: Dict[str, Any] = {
        "disable_overlap_schedule": True,
        "chunked_prefill_size": -1,
        "disable_prefill_cuda_graph": True,
        "disable_decode_cuda_graph": True,
    }
    logger.info(
        "Qwen3Guard-Stream classifies a whole prefill in one pass: disabling "
        "chunked prefill, the overlap scheduler and CUDA graphs."
    )
    return overrides
