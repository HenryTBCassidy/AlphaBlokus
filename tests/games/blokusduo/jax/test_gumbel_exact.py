"""Small exact tests for the production Gumbel search path (bug-sweep item C6).

These pin behaviours the structural/parity suites cannot see:

- a mid-game forced-pass position (exactly one legal action, game NOT over)
  must put the entire policy target and the Sequential-Halving choice on pass;
- the root's compact window must be exactly the top-K of the LEGAL-masked
  priors — an illegal logit must never displace a legal move from a slot
  (bug-sweep item C2, pinned at the root; children share ``topk_legal``);
- identical positions searched in different batch slots under ONE shared key
  must still receive independent Gumbel noise (bug-sweep item C1 — a broadcast
  noise vector would silently collapse self-play diversity);
- short endgames solved by exhaustive negamax: the search must pick an
  exact-optimal move and must not rank a losing move above a winning one
  (bug-sweep item C3 — a completed-Q perspective/sign error at any depth
  fails this, and unlike a budget limit it does not improve with more sims).
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np
import pytest

from tests.games.blokusduo.conftest import DEV_CACHE_PATH
from tests.games.blokusduo.jax.conftest import make_search_config

if TYPE_CHECKING:
    from alphablokus.games.blokusduo.game import BlokusDuoGame

jax = pytest.importorskip("jax")
pytest.importorskip("mctx")
torch = pytest.importorskip("torch")

import jax.numpy as jnp  # noqa: E402

from alphablokus.games.blokusduo.jax.bridge import numpy_state_from_board  # noqa: E402
from alphablokus.games.blokusduo.jax.checkpoint import convert_state_dict, params_to_device  # noqa: E402
from alphablokus.games.blokusduo.jax.kernels import GameState, make_kernels  # noqa: E402
from alphablokus.games.blokusduo.jax.net import encode_states, forward  # noqa: E402
from alphablokus.games.blokusduo.jax.search import SearchConfig, make_search  # noqa: E402
from alphablokus.games.blokusduo.jax.tables import build_jax_tables  # noqa: E402


@pytest.fixture(scope="module")
def setup(tmp_path_factory, blokus_game_module: BlokusDuoGame):
    from alphablokus.games.blokusduo.nn.wrapper import NNetWrapper

    torch.manual_seed(21)
    game = blokus_game_module
    game.enable_optimised_movegen()
    nnet = NNetWrapper(game, make_search_config(tmp_path_factory.mktemp("gumbel_exact")))
    params = params_to_device(convert_state_dict(nnet.nnet.state_dict(), num_residual_blocks=1))
    kernels = make_kernels(build_jax_tables(game))
    return game, params, kernels


def _weights_by_id(topk_ids_row: np.ndarray, weights_row: np.ndarray) -> dict[int, float]:
    """Aggregate compact-slot weights per global id (padded slots share pass)."""
    out: dict[int, float] = {}
    for a, w in zip(topk_ids_row, weights_row, strict=True):
        out[int(a)] = out.get(int(a), 0.0) + float(w)
    return out


def test_forced_pass_midgame_gets_all_mass(setup) -> None:
    """Mover out of pieces, opponent still to play: pass is the ONLY legal move."""
    _game, params, kernels = setup
    state = kernels.initial_state()
    # Empty the current player's inventory only — the game is not over because
    # the opponent still has every piece; the mover's sole legal action is pass.
    slot = 0  # initial state's current player is White (+1) -> slot 0
    remaining = state.remaining.at[slot].set(False)
    state = state._replace(remaining=remaining)
    assert int(np.asarray(kernels.game_result(state, jnp.int8(1)))) == 0, "position must be ongoing"
    mask = np.asarray(kernels.legal_mask(state))
    assert mask.sum() == 1 and mask[kernels.pass_index], "test premise: pass is the only legal action"

    batch = jax.tree.map(lambda x: jnp.broadcast_to(x, (2, *x.shape)), state)
    search = make_search(kernels, SearchConfig(num_simulations=16, top_k=16, policy="gumbel", gumbel_max_considered=8))
    result = search(params, jax.random.PRNGKey(3), batch)

    chosen = np.asarray(result.chosen_global)
    assert (chosen == kernels.pass_index).all(), "search must choose the only legal action"
    for i in range(2):
        mass = _weights_by_id(np.asarray(result.topk_ids)[i], np.asarray(result.action_weights)[i])
        assert mass.get(kernels.pass_index, 0.0) == pytest.approx(1.0, abs=1e-5)


def test_terminal_root_under_gumbel_resolves_to_pass(setup) -> None:
    """A finished game (neither side can place): the gumbel path must still
    put everything on pass — the PUCT twin of this test already exists, the
    production (gumbel) path was never pinned."""
    _game, params, kernels = setup
    state = kernels.initial_state()
    state = state._replace(remaining=jnp.zeros_like(state.remaining))  # no pieces at all
    batch = jax.tree.map(lambda x: jnp.broadcast_to(x, (2, *x.shape)), state)
    search = make_search(kernels, SearchConfig(num_simulations=8, top_k=16, policy="gumbel", gumbel_max_considered=8))
    result = search(params, jax.random.PRNGKey(6), batch)
    assert (np.asarray(result.chosen_global) == kernels.pass_index).all()
    for i in range(2):
        mass = _weights_by_id(np.asarray(result.topk_ids)[i], np.asarray(result.action_weights)[i])
        assert mass.get(kernels.pass_index, 0.0) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.skipif(not DEV_CACHE_PATH.exists(), reason="dev_5000 cache not built")
def test_root_window_is_topk_of_masked_priors(setup) -> None:
    """No illegal logit may occupy a compact slot; the window is exactly the
    top-K of the masked priors (C2)."""
    from alphablokus.testing.positions import iter_cached_positions

    game, params, kernels = setup
    top_k = 16
    boards, players = [], []
    for board, player, seq in iter_cached_positions(game, DEV_CACHE_PATH):
        if len(seq) <= 4 or len(seq) >= 22:  # extremes: huge and tiny branching
            boards.append(board)
            players.append(player)
        if len(boards) >= 12:
            break
    states = GameState(
        *(
            np.stack(rows)
            for rows in zip(*(numpy_state_from_board(b, p) for b, p in zip(boards, players, strict=True)), strict=True)
        )
    )
    search = make_search(
        kernels, SearchConfig(num_simulations=24, top_k=top_k, policy="gumbel", gumbel_max_considered=top_k)
    )
    result = search(params, jax.random.PRNGKey(4), states)
    topk_ids = np.asarray(result.topk_ids)

    masks = np.asarray(kernels.legal_mask_batch(states))
    log_pi = np.asarray(forward(params, encode_states(states.ppb, states.current_player))[0])
    for i in range(len(boards)):
        masked = np.where(masks[i], log_pi[i], -np.inf)
        n_window = min(int(masks[i].sum()), top_k)
        expected = set(np.argsort(-masked, kind="stable")[:n_window].tolist())
        # Slots beyond the legal count are padded with the pass id; the real
        # window is the first n_window ids IF the code fills top-k first.
        # Padded slots (fewer legals than K) are remapped to the pass id, so
        # audit UNIQUE ids: every id in the window with a finite masked prior
        # must be legal, and their logit multiset must equal the exact top-K.
        got_ids = {a for a in topk_ids[i].tolist() if np.isfinite(masked[a])}
        assert got_ids.issubset(set(np.flatnonzero(masks[i]).tolist())), f"illegal id in window at position {i}"
        assert len(got_ids) == n_window, f"position {i}: {len(got_ids)} distinct legal ids, expected {n_window}"
        got_logits = np.sort([masked[a] for a in got_ids])
        want_logits = np.sort([masked[a] for a in expected])
        np.testing.assert_allclose(
            got_logits,
            want_logits,
            atol=1e-6,
            err_msg=f"window at position {i} is not the top-{top_k} of the masked priors",
        )


def test_gumbel_noise_is_per_slot_not_broadcast(setup) -> None:
    """One shared key, B identical states: slots must diverge (C1)."""
    _game, params, kernels = setup
    B = 32
    single = kernels.initial_state()
    states = jax.tree.map(lambda x: jnp.broadcast_to(x, (B, *x.shape)), single)
    search = make_search(kernels, SearchConfig(num_simulations=16, top_k=16, policy="gumbel", gumbel_max_considered=16))
    result = search(params, jax.random.PRNGKey(123), states)
    chosen = np.asarray(result.chosen_global)
    assert len(set(chosen.tolist())) >= 4, (
        f"only {len(set(chosen.tolist()))} distinct opening actions across {B} identical slots — "
        "Gumbel noise looks broadcast across the batch"
    )
    # a second key must produce a different per-slot pattern
    result2 = search(params, jax.random.PRNGKey(124), states)
    assert not np.array_equal(chosen, np.asarray(result2.chosen_global))


@pytest.mark.slow
def test_endgame_negamax_exact(setup) -> None:
    """Exhaustively solved short endgames: the search picks an optimal move and
    never ranks a losing move above a winning one (C3)."""
    game, params, kernels = setup
    rng = random.Random(1234)
    # Only demand optimal play on positions the search could actually solve. The
    # network here is randomly initialised, so it contributes no useful prior or
    # value — the search finds the optimal move only by enumerating the tree. If
    # the position's whole tree exceeds the simulation budget, a *correct* Gumbel
    # MCTS can still miss the optimal move, and the assertion below would fail
    # for budget reasons rather than the sign error it is meant to catch. So the
    # node cap is the simulation budget, shared across all root actions, and any
    # position that overruns it is skipped rather than asserted on.
    num_simulations = 512
    node_cap = num_simulations

    def legal_actions(board, player):
        return [int(a) for a in np.nonzero(game.valid_move_masking(board, player))[0]]

    def negamax(board, player, budget):
        budget[0] += 1
        if budget[0] > node_cap:
            raise RecursionError
        r = game.get_game_ended(board, player)
        if r != 0:
            return 0 if abs(r) < 0.5 else int(np.sign(r))
        best = -2
        for a in legal_actions(board, player):
            nb, np_ = game.get_next_state(board, player, a)
            best = max(best, -negamax(nb, np_, budget))
            if best == 1:
                break
        return best

    positions = []  # (board, player, {action: exact class})
    playouts = 0
    while len(positions) < 4 and playouts < 240:
        playouts += 1
        board, player = game.initialise_board(), 1
        history = []
        while game.get_game_ended(board, player) == 0:
            history.append((board, player))
            board, player = game.get_next_state(board, player, rng.choice(legal_actions(board, player)))
        for pos_idx in range(max(0, len(history) - 4), len(history)):
            b, p = history[pos_idx]
            depth = len(history) - pos_idx
            acts = legal_actions(b, p)
            if depth < 2 or not (2 <= len(acts) <= 16):
                continue
            try:
                classes = {}
                # One budget shared across every root action, so the counter is
                # the size of the position's whole tree. Overrunning it raises
                # RecursionError and the position is dropped.
                solve_budget = [0]
                for a in acts:
                    nb, np_ = game.get_next_state(b, p, a)
                    classes[a] = -negamax(nb, np_, solve_budget)
            except RecursionError:
                continue
            if len(set(classes.values())) >= 2 and len(positions) < 4:
                positions.append((b, p, classes))
    if len(positions) < 2:
        pytest.skip(
            f"could not collect 2 discriminating endgames solvable within {node_cap} nodes "
            f"in {playouts} playouts — no assertion is safe to make"
        )

    states = GameState(
        *(np.stack(rows) for rows in zip(*(numpy_state_from_board(b, p) for b, p, _ in positions), strict=True))
    )
    search = make_search(
        kernels,
        SearchConfig(num_simulations=num_simulations, top_k=64, policy="gumbel", gumbel_max_considered=64),
    )
    result = search(params, jax.random.PRNGKey(99), states)
    chosen = np.asarray(result.chosen_global)
    topk_ids = np.asarray(result.topk_ids)
    weights = np.asarray(result.action_weights)

    for i, (_b, _p, classes) in enumerate(positions):
        best = max(classes.values())
        optimal = {a for a, c in classes.items() if c == best}
        assert int(chosen[i]) in optimal, (
            f"endgame {i}: search chose class {classes.get(int(chosen[i]))} when class {best} was available"
        )
        idw = _weights_by_id(topk_ids[i], weights[i])
        for a_hi, c_hi in classes.items():
            for a_lo, c_lo in classes.items():
                if c_hi > c_lo and abs(c_hi - c_lo) == 2 and a_hi in idw and a_lo in idw:
                    # only assert the unambiguous win-vs-loss pairs
                    assert idw[a_hi] > idw[a_lo], (
                        f"endgame {i}: winning move {a_hi} at {idw[a_hi]:.4f} <= losing move {a_lo} at {idw[a_lo]:.4f}"
                    )
