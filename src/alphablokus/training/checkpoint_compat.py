"""State-dict loading that tolerates the score head — and nothing else.

The auxiliary score head (plan ``docs/plans/score-auxiliary-target.md`` S3) has to cross
the checkpoint boundary in **both** directions:

- **Old checkpoint → score-head net.** The distillation warm-start arm loads a net trained
  before the head existed. A strict ``load_state_dict`` raises; the head's tensors must
  instead be left at their fresh initialisation, and *logged*, so the operator can see
  exactly what was not restored.
- **Score-head checkpoint → plain net.** A net trained with the head must still be
  evaluatable, exportable to ONNX and convertible for the jax bridge, all of which build
  the net from a config with ``score_head`` off. Its extra tensors are ignored.

What this deliberately does **not** do is turn every load non-strict. A mismatch that is
not the score head — an ``fc`` policy-head checkpoint loaded into a ``conv`` net, a
different filter count, a truncated file — is a genuinely wrong checkpoint and still
raises, exactly as before. Silently loading half a net is the failure mode this project
can least afford.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

    import torch.nn as nn

# Every score-head tensor is registered under this attribute name on the game net, so a
# single prefix identifies the tolerated set. Kept here (not in the game module) because
# this is framework code and must not import ``games.*``.
SCORE_HEAD_PREFIX = "score_head."


def load_state_dict_compat(module: nn.Module, state_dict: Mapping[str, Any]) -> None:
    """``load_state_dict`` that ignores score-head mismatches and raises on the rest.

    Args:
        module: The network to load into.
        state_dict: The checkpoint's ``state_dict`` payload.

    Raises:
        RuntimeError: If any **non**-score-head tensor is missing from the checkpoint or
            unexpected in it — i.e. the checkpoint does not match this architecture.
    """
    result = module.load_state_dict(state_dict, strict=False)
    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)

    score_missing = [key for key in missing if key.startswith(SCORE_HEAD_PREFIX)]
    score_unexpected = [key for key in unexpected if key.startswith(SCORE_HEAD_PREFIX)]
    other_missing = [key for key in missing if not key.startswith(SCORE_HEAD_PREFIX)]
    other_unexpected = [key for key in unexpected if not key.startswith(SCORE_HEAD_PREFIX)]

    if other_missing or other_unexpected:
        raise RuntimeError(
            "Checkpoint does not match this network architecture. "
            f"Missing tensors: {sorted(other_missing)}. Unexpected tensors: {sorted(other_unexpected)}."
        )

    if score_missing:
        logger.warning(
            "Checkpoint has no score head: {} tensor(s) left at their fresh initialisation — {}",
            len(score_missing),
            sorted(score_missing),
        )
    if score_unexpected:
        logger.info(
            "Ignored {} score-head tensor(s) from the checkpoint (this net has no score head) — {}",
            len(score_unexpected),
            sorted(score_unexpected),
        )
