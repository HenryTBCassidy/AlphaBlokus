"""State-dict loading that tolerates the auxiliary heads — and nothing else.

The auxiliary heads (the score head of ``docs/plans/score-auxiliary-target.md`` S3, the
ownership and opponent-reply heads of ``docs/plans/supervised-network-improvements.md``
N4/N5) all have to cross the checkpoint boundary in **both** directions:

- **Old checkpoint → auxiliary-head net.** The distillation warm-start arm loads a net
  trained before the head existed. A strict ``load_state_dict`` raises; the head's
  tensors must instead be left at their fresh initialisation, and *logged*, so the
  operator can see exactly what was not restored.
- **Auxiliary-head checkpoint → plain net.** A net trained with a head must still be
  evaluatable, exportable to ONNX and convertible for the jax bridge, all of which build
  the net from a config with the head off. Its extra tensors are ignored.

What this deliberately does **not** do is turn every load non-strict. A mismatch that is
not an auxiliary head — an ``fc`` policy-head checkpoint loaded into a ``conv`` net, a
different filter count, a different board size, a truncated file — is a genuinely wrong
checkpoint and still raises, exactly as before. Silently loading half a net is the
failure mode this project can least afford.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from alphablokus.aux_heads import AUX_HEAD_PREFIXES

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

    import torch.nn as nn


def _is_aux(key: str) -> bool:
    """Whether a state-dict key belongs to one of the auxiliary heads."""
    return key.startswith(AUX_HEAD_PREFIXES)


def load_state_dict_compat(module: nn.Module, state_dict: Mapping[str, Any]) -> None:
    """``load_state_dict`` that ignores auxiliary-head mismatches and raises on the rest.

    Args:
        module: The network to load into.
        state_dict: The checkpoint's ``state_dict`` payload.

    Raises:
        RuntimeError: If any **non**-auxiliary-head tensor is missing from the checkpoint
            or unexpected in it — i.e. the checkpoint does not match this architecture.
            Note that a shape mismatch on a shared tensor (a different filter count or
            board size) is reported by ``load_state_dict`` itself and raises before this
            function sees the result.
    """
    result = module.load_state_dict(state_dict, strict=False)
    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)

    aux_missing = [key for key in missing if _is_aux(key)]
    aux_unexpected = [key for key in unexpected if _is_aux(key)]
    other_missing = [key for key in missing if not _is_aux(key)]
    other_unexpected = [key for key in unexpected if not _is_aux(key)]

    if other_missing or other_unexpected:
        raise RuntimeError(
            "Checkpoint does not match this network architecture. "
            f"Missing tensors: {sorted(other_missing)}. Unexpected tensors: {sorted(other_unexpected)}."
        )

    if aux_missing:
        logger.warning(
            "Checkpoint has no auxiliary head(s): {} tensor(s) left at their fresh initialisation — {}",
            len(aux_missing),
            sorted(aux_missing),
        )
    if aux_unexpected:
        logger.info(
            "Ignored {} auxiliary-head tensor(s) from the checkpoint (this net does not build them) — {}",
            len(aux_unexpected),
            sorted(aux_unexpected),
        )
