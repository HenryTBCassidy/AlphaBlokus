"""Stream B4 — width shadow test (frozen positions, identical randomness).

Control:      top_k=64,  gumbel_max_considered=64, 128 sims.
Intervention: top_k=128, gumbel_max_considered=64, 128 sims.

Measures, per frozen dev-cache position, WITHOUT drawing conclusions:

  (a) does the root's chosen action (Sequential-Halving winner) change between
      control and intervention?
  (b) of the actions ranked 65-128 by prior (only visible to the intervention's
      wider top-k pool), how many receive at least one simulation (enter a
      "consequential" child search) under the intervention's Gumbel root
      selection, which considers only 64 of the top_k candidates?
  (c) completed-Q (mctx's ``qtransform_completed_by_mix_value`` at the root,
      matching production's actual default for the gumbel policy path -- see
      ``search.py``'s gumbel branch, which does not override ``qtransform``)
      on the 64 actions common to both configs: intervention minus control,
      per action and for the control-chosen action specifically.
  (d) for positions where the decision changed (a), does the change survive an
      independent, full-action-space, much-deeper python MCTS search (the
      project's python ``search/mcts.py`` reference, K=1, no top-k
      compaction) -- i.e. does the deeper referee's argmax action agree with
      the intervention's pick, the control's pick, or neither?

CRITICAL DESIGN NOTE -- verified empirically, not assumed:
jax's counter-based PRNG draws random values from a flat index over the
output shape. For a *batched* call, shape (B, K) at batch row i>0 occupies a
DIFFERENT flat index range when K=64 vs K=128, so a shared rng_key does NOT
give matching Gumbel noise on the shared 64 actions for any row beyond row 0
if positions are batched together. Verified on this box (2026-08-04):

    g64  = jax.random.gumbel(key, shape=(2, 64))
    g128 = jax.random.gumbel(key, shape=(2, 128))
    g64[0] == g128[0, :64]   -> True
    g64[1] == g128[1, :64]   -> False

So this script deliberately runs **one position at a time** (batch size 1)
for both control and intervention, with the same per-position folded rng key,
which does verifiably align the noise on the shared 64 actions (checked for
B=1 above). This costs wall-clock (no cross-position batching) but is what
makes "identical randomness" true rather than assumed. The script asserts
in-loop that the intervention's first-64 root ids equal control's root ids
(both are exact top-k-by-prior-logit, so this should hold exactly) and warns
if not.

Does not modify anything under src/ -- imports only. Duplicates the small
amount of ``search.py`` gumbel-branch logic needed to retain ``search_tree``
(``make_search`` intentionally discards it via the ``SearchResult`` NamedTuple).

Usage::

    uv run python scripts/width_shadow_probe.py \
        --checkpoint temp/runs/blokus/blokus_cloud_v3/Nets/accepted_32.pth.tar \
        --filters <F> --blocks <B> --positions 100 --sims 128 --considered 64 \
        --seed 0 --deeper-sims 3200 --dtype bfloat16 --out temp/benchmarks/streamb4/probe.json

CPU dev/smoke run (no GPU contention with a concurrent B1 ladder job)::

    JAX_PLATFORMS=cpu uv run python scripts/width_shadow_probe.py \
        --checkpoint <ckpt> --filters <F> --blocks <B> --positions 3 --sims 32 \
        --considered 16 --top-k-control 32 --top-k-intervention 64 \
        --deeper-sims 64 --dtype float32 --out temp/benchmarks/streamb4/smoke.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent.parent
DEV_CACHE_PATH = REPO_ROOT / "tests" / "fixtures" / "blokus_duo_positions" / "dev_5000.npz"


def _empty_counts() -> dict[str, Any]:
    return {
        "n_positions": 0,
        "a_chosen_changed": 0,
        "b_rank65_128_any_entered": 0,
        "b_rank65_128_slots_entered": 0,
        "b_rank65_128_slots_total": 0,
        "c_completed_q_delta_chosen": [],
        "c_completed_q_delta_all_common": [],
        "d_changed_positions": 0,
        "d_deeper_agrees_intervention": 0,
        "d_deeper_agrees_control": 0,
        "d_deeper_agrees_neither": 0,
        "topk_prefix_mismatch_warnings": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--filters", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--positions", type=int, default=100)
    parser.add_argument("--sims", type=int, default=128)
    parser.add_argument("--considered", type=int, default=64, help="gumbel_max_considered, fixed for both arms")
    parser.add_argument("--top-k-control", type=int, default=64)
    parser.add_argument("--top-k-intervention", type=int, default=128)
    parser.add_argument("--cpuct", type=float, default=2.5)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deeper-sims", type=int, default=3200, help="python full-action-space referee sim count")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    import jax
    import jax.numpy as jnp
    import mctx

    from alphablokus.config import MCTSConfig, NetConfig, RunConfig
    from alphablokus.games.blokusduo.game import BlokusDuoGame
    from alphablokus.games.blokusduo.jax.bridge import numpy_state_from_board
    from alphablokus.games.blokusduo.jax.checkpoint import convert_torch_checkpoint, params_to_device
    from alphablokus.games.blokusduo.jax.kernels import GameState, make_kernels
    from alphablokus.games.blokusduo.jax.net import encode_states, forward
    from alphablokus.games.blokusduo.jax.search import SearchConfig, make_search
    from alphablokus.games.blokusduo.jax.tables import build_jax_tables
    from alphablokus.games.blokusduo.nn.wrapper import NNetWrapper
    from alphablokus.games.blokusduo.pieces import default_pieces_path
    from alphablokus.search.mcts import MCTS
    from alphablokus.testing.positions import iter_cached_positions

    pieces_path = default_pieces_path()
    game = BlokusDuoGame(pieces_config_path=pieces_path)
    game.enable_optimised_movegen()

    boards, players, seqs = [], [], []
    for _, (board, player, sequence) in enumerate(iter_cached_positions(game, DEV_CACHE_PATH)):
        if 4 <= len(sequence) <= 26 and game.get_game_ended(board, player) == 0:
            boards.append(board)
            players.append(player)
            seqs.append(sequence)
        if len(boards) >= args.positions:
            break
    logger.info("loaded {} frozen dev-cache positions", len(boards))

    kernels = make_kernels(build_jax_tables(game))
    params = params_to_device(
        convert_torch_checkpoint(args.checkpoint.resolve(), args.blocks),
        dtype=args.dtype,
    )
    dtype = jnp.float32 if args.dtype == "float32" else jnp.bfloat16

    # Python full-action-space MCTS referee (K=1, exact search, no top-k
    # compaction anywhere) -- same reference class validate_jax_search.py
    # uses, just at a much higher sim count for part (d).
    run_config = RunConfig(
        game="blokusduo",
        run_name="width_shadow_probe",
        num_generations=1,
        num_eps=1,
        temp_threshold=12,
        update_threshold=0.55,
        num_arena_matches=2,
        root_directory=REPO_ROOT / "temp",
        load_model=False,
        mcts_config=MCTSConfig(num_mcts_sims=args.deeper_sims, cpuct=args.cpuct),
        net_config=NetConfig(
            learning_rate=1e-3,
            dropout=0.0,
            epochs=1,
            batch_size=8,
            cuda=False,
            num_filters=args.filters,
            num_residual_blocks=args.blocks,
        ),
    )
    nnet = NNetWrapper(game, run_config)
    nnet.load_checkpoint(filename=str(args.checkpoint.resolve()))

    control_search = make_search(
        kernels,
        SearchConfig(
            num_simulations=args.sims,
            top_k=args.top_k_control,
            cpuct=args.cpuct,
            dtype=args.dtype,
            policy="gumbel",
            gumbel_max_considered=args.considered,
        ),
    )
    intervention_search = make_search(
        kernels,
        SearchConfig(
            num_simulations=args.sims,
            top_k=args.top_k_intervention,
            cpuct=args.cpuct,
            dtype=args.dtype,
            policy="gumbel",
            gumbel_max_considered=args.considered,
        ),
    )

    # --- Duplicated (not imported -- make_search discards the tree) gumbel
    # search that also returns the mctx search_tree, so completed-Q can be
    # read off directly via the SAME qtransform production actually uses for
    # the gumbel path (search.py's gumbel branch calls
    # mctx.gumbel_muzero_policy with no qtransform override -> library
    # default qtransform_completed_by_mix_value). B=1 only (see module
    # docstring for why batching breaks the noise-alignment guarantee).
    game_result_per_state = jax.vmap(kernels.game_result)

    def make_tree_search(top_k: int):
        def policy_value(p: dict, states: GameState) -> tuple[jnp.ndarray, jnp.ndarray]:
            planes = encode_states(states.ppb, states.current_player, dtype=dtype)
            return forward(p, planes)

        def topk_legal(logits: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            values, ids = jax.lax.top_k(logits, top_k)
            ids = jnp.where(jnp.isneginf(values), kernels.pass_index, ids)
            return values, ids.astype(jnp.int32)

        def recurrent_fn(p: dict, rng_key: jnp.ndarray, action: jnp.ndarray, embedding: tuple):
            states, topk_ids = embedding
            batch = jnp.arange(action.shape[0])
            global_action = topk_ids[batch, action]
            movers = states.current_player
            new_states = kernels.step_batch(states, global_action)
            result_for_mover = game_result_per_state(new_states, movers)
            terminated = result_for_mover != 0.0
            reward = jnp.where(terminated, result_for_mover, 0.0)
            discount = jnp.where(terminated, 0.0, -1.0)
            log_pi, value = policy_value(p, new_states)
            value = jnp.where(terminated, 0.0, value)
            masks = kernels.legal_mask_batch(new_states)
            child_logits, child_ids = topk_legal(jnp.where(masks, log_pi, -jnp.inf))
            output = mctx.RecurrentFnOutput(reward=reward, discount=discount, prior_logits=child_logits, value=value)
            return output, (new_states, child_ids)

        def run(p: dict, rng_key: jnp.ndarray, states: GameState):
            log_pi, root_value = policy_value(p, states)
            masks = kernels.legal_mask_batch(states)
            root_log_pi = jnp.where(masks, log_pi, -jnp.inf)
            root_logits, root_ids = topk_legal(root_log_pi)
            root = mctx.RootFnOutput(prior_logits=root_logits, value=root_value, embedding=(states, root_ids))
            policy_output = mctx.gumbel_muzero_policy(
                params=p,
                rng_key=rng_key,
                root=root,
                recurrent_fn=recurrent_fn,
                num_simulations=args.sims,
                invalid_actions=jnp.isneginf(root_logits),
                max_num_considered_actions=args.considered,
            )
            tree = policy_output.search_tree
            completed_q = jax.vmap(mctx.qtransform_completed_by_mix_value, in_axes=[0, None])(
                tree, tree.ROOT_INDEX
            )
            return root_ids, policy_output, completed_q

        return jax.jit(run)

    tree_search_control = make_tree_search(args.top_k_control)
    tree_search_intervention = make_tree_search(args.top_k_intervention)

    counts = _empty_counts()
    per_position: list[dict[str, Any]] = []
    base_key = jax.random.PRNGKey(args.seed)
    t0 = time.perf_counter()

    for i, (board, player) in enumerate(zip(boards, players, strict=True)):
        rows = [numpy_state_from_board(board, player)]
        state = GameState(*(np.stack(r) for r in zip(*rows, strict=True)))
        pos_key = jax.random.fold_in(base_key, i)

        rc = control_search(params, pos_key, state)
        ri = intervention_search(params, pos_key, state)

        ids_c = np.asarray(rc.topk_ids[0])
        ids_i = np.asarray(ri.topk_ids[0])
        prefix_ok = bool(np.array_equal(ids_c, ids_i[: args.top_k_control]))
        if not prefix_ok:
            counts["topk_prefix_mismatch_warnings"] += 1
            logger.warning("position {}: control top-k is not intervention's prefix (unexpected)", i)

        chosen_c = int(rc.chosen_global[0])
        chosen_i = int(ri.chosen_global[0])
        changed = chosen_c != chosen_i

        rank65_128_ids = ids_i[args.top_k_control :]
        rank65_128_visits = np.asarray(ri.visit_counts[0])[args.top_k_control :]
        entered_mask = rank65_128_visits > 0
        counts["b_rank65_128_slots_entered"] += int(entered_mask.sum())
        counts["b_rank65_128_slots_total"] += int(len(rank65_128_visits))
        if entered_mask.any():
            counts["b_rank65_128_any_entered"] += 1

        # (c) completed-Q on the 64 common actions.
        _, _, cq_c = tree_search_control(params, pos_key, state)
        _, _, cq_i = tree_search_intervention(params, pos_key, state)
        cq_c = np.asarray(cq_c[0])
        cq_i = np.asarray(cq_i[0])
        id_to_slot_i = {int(a): s for s, a in enumerate(ids_i)}
        deltas = []
        chosen_c_delta = None
        for slot_c, gid in enumerate(ids_c):
            slot_i = id_to_slot_i.get(int(gid))
            if slot_i is None:
                continue  # should not happen given prefix_ok, but don't assume
            delta = float(cq_i[slot_i] - cq_c[slot_c])
            deltas.append(delta)
            if int(gid) == chosen_c:
                chosen_c_delta = delta
        counts["c_completed_q_delta_all_common"].extend(deltas)
        if chosen_c_delta is not None:
            counts["c_completed_q_delta_chosen"].append(chosen_c_delta)

        # (d) deeper full-action-space referee, only for changed decisions.
        deeper_action = None
        if changed:
            counts["d_changed_positions"] += 1
            mcts = MCTS(game, nnet, MCTSConfig(num_mcts_sims=args.deeper_sims, cpuct=args.cpuct))
            canonical = game.get_canonical_form(board, player)
            # temp=0 is handled specially by get_action_prob as exact argmax-of-
            # visit-counts (ties broken by random choice among ties) -- NOT a
            # small-but-nonzero temperature, which overflows raising large visit
            # counts to a ~1000th power. Confirmed by reading mcts.py directly.
            probs = np.asarray(mcts.get_action_prob(canonical, temp=0))
            deeper_action = int(np.argmax(probs))
            # chosen_c / chosen_i are canonical-perspective global ids (root
            # search always runs on the state as given, i.e. canonical), so
            # comparable directly against the python referee's canonical
            # action id.
            if deeper_action == chosen_i:
                counts["d_deeper_agrees_intervention"] += 1
            elif deeper_action == chosen_c:
                counts["d_deeper_agrees_control"] += 1
            else:
                counts["d_deeper_agrees_neither"] += 1

        counts["n_positions"] += 1
        counts["a_chosen_changed"] += int(changed)
        per_position.append(
            {
                "index": i,
                "seq_len": len(seqs[i]),
                "chosen_control": chosen_c,
                "chosen_intervention": chosen_i,
                "changed": changed,
                "rank65_128_entered": int(entered_mask.sum()),
                "rank65_128_total": int(len(rank65_128_visits)),
                "completed_q_delta_chosen": chosen_c_delta,
                "deeper_action": deeper_action,
            }
        )
        logger.info(
            "[{}/{}] changed={} rank65-128 entered={}/{} dQ(chosen)={}",
            i + 1,
            len(boards),
            changed,
            int(entered_mask.sum()),
            len(rank65_128_visits),
            f"{chosen_c_delta:.4f}" if chosen_c_delta is not None else "n/a",
        )

    elapsed = time.perf_counter() - t0

    def _summ(xs: list[float]) -> dict[str, float | None]:
        if not xs:
            return {"n": 0, "mean": None, "abs_mean": None, "min": None, "max": None}
        arr = np.asarray(xs)
        return {
            "n": len(arr),
            "mean": float(arr.mean()),
            "abs_mean": float(np.abs(arr).mean()),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    n = counts["n_positions"]
    report = {
        "config": {
            "checkpoint": str(args.checkpoint),
            "positions": n,
            "sims": args.sims,
            "considered_fixed": args.considered,
            "top_k_control": args.top_k_control,
            "top_k_intervention": args.top_k_intervention,
            "seed": args.seed,
            "deeper_sims": args.deeper_sims,
            "dtype": args.dtype,
        },
        "a_chosen_action_changed": {
            "count": counts["a_chosen_changed"],
            "n": n,
            "fraction": counts["a_chosen_changed"] / n if n else None,
        },
        "b_rank65_128_entered_child_search": {
            "positions_with_any_entered": counts["b_rank65_128_any_entered"],
            "positions_n": n,
            "fraction_positions_any": counts["b_rank65_128_any_entered"] / n if n else None,
            "slots_entered": counts["b_rank65_128_slots_entered"],
            "slots_total": counts["b_rank65_128_slots_total"],
            "fraction_slots": (
                counts["b_rank65_128_slots_entered"] / counts["b_rank65_128_slots_total"]
                if counts["b_rank65_128_slots_total"]
                else None
            ),
        },
        "c_completed_q_delta_intervention_minus_control": {
            "chosen_action_only": _summ(counts["c_completed_q_delta_chosen"]),
            "all_64_common_actions": _summ(counts["c_completed_q_delta_all_common"]),
        },
        "d_deeper_referee_on_changed_decisions": {
            "changed_positions": counts["d_changed_positions"],
            "agrees_intervention": counts["d_deeper_agrees_intervention"],
            "agrees_control": counts["d_deeper_agrees_control"],
            "agrees_neither": counts["d_deeper_agrees_neither"],
        },
        "sanity": {
            "topk_prefix_mismatch_warnings": counts["topk_prefix_mismatch_warnings"],
            "note": "should be 0 -- intervention's top-64-by-prior must equal control's top-64 exactly",
        },
        "wall_clock_seconds": elapsed,
        "per_position": per_position,
    }

    out = args.out or REPO_ROOT / "temp" / "benchmarks" / "width_shadow_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    logger.info("report written to {}", out)
    logger.info(
        "a: {}/{} changed | b: {}/{} positions had a rank65-128 action enter search | d: of {} changed, "
        "{} agree intervention, {} agree control, {} agree neither",
        counts["a_chosen_changed"],
        n,
        counts["b_rank65_128_any_entered"],
        n,
        counts["d_changed_positions"],
        counts["d_deeper_agrees_intervention"],
        counts["d_deeper_agrees_control"],
        counts["d_deeper_agrees_neither"],
    )


if __name__ == "__main__":
    main()
