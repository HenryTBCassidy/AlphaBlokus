"""G3: the JAX net reproduces the torch net's eval-mode forward pass.

A randomly-initialised ``AlphaBlokusDuo`` (with deliberately non-trivial BN
running stats, so the conv+BN folding is genuinely exercised) is compared
against the converted JAX net on 200 dev-cache positions:

- fp32: log-policies and values agree within tight tolerance, argmax agrees
  everywhere;
- bf16: values and policy *distributions* close (KL, log-prob bounds). Argmax
  is deliberately not asserted for bf16: a random net's 17,837 logits are
  near-uniform, so rounding legitimately reorders near-ties (measured ~83%
  agreement at KL ≈ 2e-6 — the distributions are the same, the ties are not).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from tests.test_blokusduo.conftest import DEV_CACHE_PATH

if TYPE_CHECKING:
    from alphablokus.games.blokusduo.game import BlokusDuoGame

jax = pytest.importorskip("jax")
torch = pytest.importorskip("torch")

import jax.numpy as jnp  # noqa: E402

from alphablokus.core.config import NetConfig  # noqa: E402
from alphablokus.games.blokusduo.jax.checkpoint import convert_state_dict, params_to_device  # noqa: E402
from alphablokus.games.blokusduo.jax.net import forward_jit  # noqa: E402
from alphablokus.games.blokusduo.neuralnets.net import AlphaBlokusDuo  # noqa: E402

N_POSITIONS = 200


def _random_torch_net(game: BlokusDuoGame, seed: int = 0) -> AlphaBlokusDuo:
    torch.manual_seed(seed)
    config = NetConfig(
        learning_rate=1e-3, dropout=0.0, epochs=1, batch_size=8, cuda=False,
        num_filters=32, num_residual_blocks=2, policy_head="conv",
    )
    net = AlphaBlokusDuo(
        board_rows=game.board_size, board_cols=game.board_size,
        action_size=game.get_action_size(), num_input_channels=44, config=config,
    )
    # Randomise BN running stats so eval-mode BN is a non-trivial affine map —
    # fresh nets have mean=0/var=1, which would let a broken fold pass unnoticed.
    generator = torch.Generator().manual_seed(seed + 1)
    for module in net.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.running_mean.copy_(torch.randn(module.running_mean.shape, generator=generator) * 0.5)
            module.running_var.copy_(torch.rand(module.running_var.shape, generator=generator) + 0.5)
    net.eval()
    return net


@pytest.fixture(scope="module")
def encoded_positions(blokus_game_module: BlokusDuoGame) -> np.ndarray:
    from tests.fixtures.blokus_positions import iter_cached_positions

    planes = []
    for index, (board, player, _seq) in enumerate(iter_cached_positions(blokus_game_module, DEV_CACHE_PATH)):
        if index >= N_POSITIONS:
            break
        planes.append(board.as_multi_channel(player))
    return np.stack(planes).astype(np.float32)


@pytest.mark.skipif(not DEV_CACHE_PATH.exists(), reason="dev_5000 cache not built")
def test_fp32_forward_equivalence(blokus_game_module: BlokusDuoGame, encoded_positions: np.ndarray) -> None:
    net = _random_torch_net(blokus_game_module)
    with torch.no_grad():
        torch_log_pi, torch_value = net(torch.from_numpy(encoded_positions))

    params = params_to_device(convert_state_dict(net.state_dict(), num_residual_blocks=2))
    jax_log_pi, jax_value = forward_jit(params, jnp.asarray(encoded_positions))

    np.testing.assert_allclose(np.asarray(jax_log_pi), torch_log_pi.numpy(), atol=2e-4, rtol=0)
    np.testing.assert_allclose(np.asarray(jax_value), torch_value.numpy().reshape(-1), atol=1e-5, rtol=0)
    assert np.array_equal(
        np.asarray(jax_log_pi).argmax(axis=1), torch_log_pi.numpy().argmax(axis=1)
    ), "fp32 policy argmax must agree on every position"


@pytest.mark.skipif(not DEV_CACHE_PATH.exists(), reason="dev_5000 cache not built")
def test_bf16_forward_close(blokus_game_module: BlokusDuoGame, encoded_positions: np.ndarray) -> None:
    net = _random_torch_net(blokus_game_module)
    with torch.no_grad():
        torch_log_pi, torch_value = net(torch.from_numpy(encoded_positions))

    params = params_to_device(convert_state_dict(net.state_dict(), num_residual_blocks=2), dtype="bfloat16")
    jax_log_pi, jax_value = forward_jit(params, jnp.asarray(encoded_positions, dtype=jnp.bfloat16))

    np.testing.assert_allclose(np.asarray(jax_value), torch_value.numpy().reshape(-1), atol=0.01, rtol=0)
    torch_log = torch_log_pi.numpy()
    jax_log = np.asarray(jax_log_pi)
    kl = np.sum(np.exp(torch_log) * (torch_log - jax_log), axis=1)
    assert kl.max() < 1e-4, f"bf16 policy KL too high: max {kl.max():.2e}"
    np.testing.assert_allclose(jax_log, torch_log, atol=0.05, rtol=0)


@pytest.mark.skipif(not DEV_CACHE_PATH.exists(), reason="dev_5000 cache not built")
def test_matches_wrapper_predict_encoded(
    blokus_game_module: BlokusDuoGame, encoded_positions: np.ndarray, tmp_path
) -> None:
    """End-to-end: wrapper checkpoint file → converter → same probs as predict_encoded."""
    net = _random_torch_net(blokus_game_module)
    checkpoint_path = tmp_path / "ckpt.pth.tar"
    torch.save({"state_dict": net.state_dict()}, checkpoint_path)

    from alphablokus.games.blokusduo.jax.checkpoint import convert_torch_checkpoint

    params = params_to_device(convert_torch_checkpoint(checkpoint_path, num_residual_blocks=2))
    jax_log_pi, jax_value = forward_jit(params, jnp.asarray(encoded_positions))

    with torch.no_grad():
        log_pi, value = net(torch.from_numpy(encoded_positions))
    torch_probs = torch.exp(log_pi).numpy()

    np.testing.assert_allclose(np.exp(np.asarray(jax_log_pi)), torch_probs, atol=1e-5, rtol=0)
    np.testing.assert_allclose(np.asarray(jax_value), value.numpy().reshape(-1), atol=1e-5, rtol=0)
