"""Names and forward order of the network's auxiliary training heads.

One definition, imported by three places that must never disagree:

- the game net, which **builds** the heads and appends their outputs to ``forward``;
- :class:`~alphablokus.games.base_wrapper.BaseNNetWrapper`, which unpacks that
  variable-arity output and weights each head's loss;
- :mod:`alphablokus.training.checkpoint_compat`, which tolerates exactly these
  prefixes across a checkpoint boundary and raises on everything else.

Kept at the package root rather than under ``training/`` because
``alphablokus.training``'s package ``__init__`` imports the Coach, which reaches the
registry, which imports ``games/base_wrapper.py`` — importing anything from
``alphablokus.training`` at module scope from there would close that cycle (which is
why ``base_wrapper`` already imports ``score_target`` and ``checkpoint_compat``
locally, inside the functions that need them).

**The order is the contract.** ``forward`` returns ``(log_pi, value)`` followed by the
output of each *built* head in this order, so a head that is off contributes nothing —
not even a ``None`` placeholder, which would make the module untraceable and break the
web ONNX export. Two rules follow, and both are load-bearing:

1. **A new head is appended at the end.** At a fixed seed every earlier head (and the
   trunk) then initialises identically whether the new one is built or not, so a
   one-head-at-a-time A/B measures the head rather than a shifted RNG stream. Inserting
   a head in the middle would silently re-randomise every head after it.
2. **Nothing may read positionally without consulting the built set.** With score off
   and ownership on, ``outputs[2]`` is the *ownership* map;
   ``BaseNNetWrapper._split_net_outputs`` is the single place that resolves this, using
   the net's ``aux_head_names``.

No auxiliary head is ever read when choosing a move: ``predict`` / ``predict_batch`` /
``predict_encoded`` drop them all.
"""

from __future__ import annotations

from typing import Final

# Forward order. Append only — see rule 1 above.
AUX_HEAD_NAMES: Final[tuple[str, ...]] = ("score_head", "ownership_head", "reply_head")

# State-dict prefixes of the tolerated set (``checkpoint_compat``).
AUX_HEAD_PREFIXES: Final[tuple[str, ...]] = tuple(f"{name}." for name in AUX_HEAD_NAMES)


def aux_key(head_name: str) -> str:
    """``"score_head"`` → ``"score"``: the short key used in configs, losses and metrics.

    ``NetConfig`` spells the knobs ``<key>_head`` / ``<key>_loss_weight``, the loss
    methods are ``loss_<key>``, and the diagnostics dictionaries key on ``<key>``, so one
    conversion keeps all four in step instead of four hand-written mappings.
    """
    return head_name.removesuffix("_head")
