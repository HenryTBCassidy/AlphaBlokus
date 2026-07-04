"""Torch checkpoint → JAX net params (plan step G3).

Loads an ``AlphaBlokusDuo`` state dict (raw, or wrapped in the
``BaseNNetWrapper`` checkpoint format with a ``state_dict`` key) and emits the
numpy pytree :func:`games.blokusduo.jaxenv.net.forward` consumes.

Every conv+BatchNorm pair is folded into a single conv-with-bias using the
eval-mode identity ``BN(x) = x·s + t`` with ``s = γ/√(σ²+eps)``,
``t = β − μ·s``: the conv weight is scaled per output channel by ``s`` and the
bias becomes ``t`` (plus ``s``-scaled original bias if one existed — none do in
this net's conv+BN pairs). This is exact for inference, which is the only thing
the jax side ever does.

Only the conv policy head is supported — the FC head is legacy
(``NetConfig.policy_head`` default has been "conv" since 2026-06-02).

Torch is imported lazily so importing this module never requires torch at
jax-only sites (and vice versa: nothing here is needed by the python backend).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

_BN_EPS = 1e-5  # torch nn.BatchNorm2d default


def _fold_conv_bn(state: dict, conv_key: str, bn_key: str) -> dict[str, np.ndarray]:
    """Fold ``BN(conv(x))`` (conv bias=False) into one conv weight + bias."""
    weight = state[f"{conv_key}.weight"].detach().cpu().numpy().astype(np.float32)
    gamma = state[f"{bn_key}.weight"].detach().cpu().numpy().astype(np.float32)
    beta = state[f"{bn_key}.bias"].detach().cpu().numpy().astype(np.float32)
    mean = state[f"{bn_key}.running_mean"].detach().cpu().numpy().astype(np.float32)
    var = state[f"{bn_key}.running_var"].detach().cpu().numpy().astype(np.float32)
    scale = gamma / np.sqrt(var + _BN_EPS)
    return {"w": weight * scale[:, None, None, None], "b": beta - mean * scale}


def _linear(state: dict, key: str) -> dict[str, np.ndarray]:
    return {
        "w": state[f"{key}.weight"].detach().cpu().numpy().astype(np.float32),
        "b": state[f"{key}.bias"].detach().cpu().numpy().astype(np.float32),
    }


def convert_state_dict(state: dict, num_residual_blocks: int) -> dict[str, Any]:
    """Torch ``AlphaBlokusDuo`` state dict → numpy params pytree (fp32)."""
    if "policy_head.move_conv.weight" not in state:
        raise ValueError(
            "Only the conv policy head is supported by the jax net "
            "(this checkpoint looks like the legacy 'fc' head)."
        )
    params: dict[str, Any] = {
        "trunk": _fold_conv_bn(state, "conv_block.0", "conv_block.1"),
        "blocks": [
            {
                "conv1": _fold_conv_bn(state, f"residual_blocks.{i}.conv_block1.0", f"residual_blocks.{i}.conv_block1.1"),  # noqa: E501
                "conv2": _fold_conv_bn(state, f"residual_blocks.{i}.conv_block2.0", f"residual_blocks.{i}.conv_block2.1"),  # noqa: E501
            }
            for i in range(num_residual_blocks)
        ],
        "value": {
            "conv": _fold_conv_bn(state, "value_head.0", "value_head.1"),
            "fc1": _linear(state, "value_head.4"),
            "fc2": _linear(state, "value_head.6"),
        },
        "policy": {
            "move_conv": {
                "w": state["policy_head.move_conv.weight"].detach().cpu().numpy().astype(np.float32),
                "b": state["policy_head.move_conv.bias"].detach().cpu().numpy().astype(np.float32),
            },
            "pass": _linear(state, "policy_head.pass_head.2"),
        },
        "perm": state["policy_head.perm"].detach().cpu().numpy().astype(np.int32),
    }
    return params


def convert_torch_checkpoint(checkpoint_path: Path, num_residual_blocks: int) -> dict[str, Any]:
    """Load a checkpoint file (wrapper format or raw state dict) and convert."""
    import torch

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = payload.get("state_dict", payload)
    return convert_state_dict(state, num_residual_blocks)


def params_to_device(params: dict[str, Any], dtype: str = "float32") -> dict[str, Any]:
    """Numpy pytree → on-device jnp pytree in ``float32`` or ``bfloat16``.

    The permutation index stays int32 regardless of the compute dtype.
    """
    import jax
    import jax.numpy as jnp

    target = jnp.float32 if dtype == "float32" else jnp.bfloat16

    def _cast(leaf: np.ndarray) -> jnp.ndarray:
        array = jnp.asarray(leaf)
        return array if jnp.issubdtype(array.dtype, jnp.integer) else array.astype(target)

    return jax.tree.map(_cast, params)
