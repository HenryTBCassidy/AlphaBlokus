"""Inference-only JAX port of ``AlphaBlokusDuo`` (plan step G3).

Runs the exact eval-mode forward pass of the torch net
(``games/blokusduo/neuralnets/net.py``, conv policy head) on parameters
converted from a torch checkpoint by :mod:`games.blokusduo.jax.checkpoint`.
Eval-mode BatchNorm is a fixed affine transform, so the converter folds every
conv+BN pair into a single conv-with-bias — the params pytree here has no BN
notion at all. Training never happens on this side; torch remains the learner.

Parity with ``predict_encoded`` is pinned by
``tests/test_blokusduo/test_jaxenv_net.py`` (fp32 tolerance + bf16 agreement).
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

#: Torch/lax NCHW convolution dimension convention shared by every conv here.
_DIMENSION_NUMBERS = ("NCHW", "OIHW", "NCHW")

#: Params pytree: nested dicts of jnp arrays (see checkpoint.convert_torch_checkpoint).
NetParams = dict[str, Any]


def _conv(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    """3×3 'SAME' or 1×1 conv with bias, matching ``nn.Conv2d(stride=1, padding=k//2)``."""
    out = jax.lax.conv_general_dilated(
        x, weight, window_strides=(1, 1), padding="SAME", dimension_numbers=_DIMENSION_NUMBERS
    )
    return out + bias[None, :, None, None]


def forward(params: NetParams, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Eval-mode forward pass.

    Args:
        params: Converted parameter pytree (BN pre-folded into conv biases).
        x: Encoded boards, ``(B, 44, 14, 14)``, same dtype as the params.

    Returns:
        ``(log_pi, value)`` — log-softmax policy ``(B, action_size)`` in fp32
        and tanh value ``(B,)`` in fp32, mirroring the torch net's ``forward``
        (value squeezed) regardless of the compute dtype.
    """
    h = jax.nn.relu(_conv(x, params["trunk"]["w"], params["trunk"]["b"]))
    for block in params["blocks"]:
        out = jax.nn.relu(_conv(h, block["conv1"]["w"], block["conv1"]["b"]))
        out = _conv(out, block["conv2"]["w"], block["conv2"]["b"])
        h = jax.nn.relu(out + h)

    # Value head: 1×1 conv(+folded BN) → ReLU → flatten → Linear → ReLU → Linear → tanh.
    v = jax.nn.relu(_conv(h, params["value"]["conv"]["w"], params["value"]["conv"]["b"]))
    v = v.reshape(v.shape[0], -1)  # (B, 196); channel dim is 1 so layout matches torch Flatten
    v = jax.nn.relu(v @ params["value"]["fc1"]["w"].T + params["value"]["fc1"]["b"])
    value = jnp.tanh(v @ params["value"]["fc2"]["w"].T + params["value"]["fc2"]["b"])  # (B, 1)

    # Conv policy head: 1×1 conv to orientation planes → ActionCodec permutation
    # gather → pooled pass logit.
    planes = _conv(h, params["policy"]["move_conv"]["w"], params["policy"]["move_conv"]["b"])
    moves = planes.reshape(planes.shape[0], -1)[:, params["perm"]]  # (B, cells·O) in action order
    pooled = h.mean(axis=(2, 3))  # AdaptiveAvgPool2d(1) + Flatten
    pass_logit = pooled @ params["policy"]["pass"]["w"].T + params["policy"]["pass"]["b"]  # (B, 1)
    logits = jnp.concatenate([moves, pass_logit], axis=1)

    log_pi = jax.nn.log_softmax(logits.astype(jnp.float32), axis=1)
    return log_pi, value.astype(jnp.float32).reshape(-1)


forward_jit = jax.jit(forward)

#: Piece ids 1–21, used by the encoder's per-piece binary planes.
_PIECE_IDS = jnp.arange(1, 22, dtype=jnp.int8)

#: Board side; matches BlokusDuoBoard.N (kept local so this module stays torch-free).
_BOARD_SIZE = 14


def encode_states(ppb: jnp.ndarray, players: jnp.ndarray, dtype: jnp.dtype = jnp.float32) -> jnp.ndarray:
    """Batched 44-channel encoding, identical to ``board.as_multi_channel(player)``.

    Args:
        ppb: ``(B, 196)`` int8 signed placement boards (player-1 canonical).
        players: ``(B,)`` int8 current players (+1/-1).
        dtype: Output dtype (fp32 or bf16 to match the net params).

    Returns:
        ``(B, 44, 14, 14)`` planes — current player's 21 piece planes, then the
        opponent's 21, then the two aggregate planes, mirroring
        ``encode_planes_from_placement`` (games/blokusduo/board.py).
    """
    signed = (ppb * players[:, None]).reshape(-1, _BOARD_SIZE, _BOARD_SIZE)  # current player positive
    own = signed[:, None, :, :] == _PIECE_IDS[None, :, None, None]  # (B, 21, 14, 14)
    opponent = signed[:, None, :, :] == -_PIECE_IDS[None, :, None, None]
    aggregates = jnp.stack([signed > 0, signed < 0], axis=1)  # (B, 2, 14, 14)
    return jnp.concatenate([own, opponent, aggregates], axis=1).astype(dtype)
